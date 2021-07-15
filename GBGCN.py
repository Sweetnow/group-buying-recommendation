#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from librecframework.argument.manager import HyperparamManager
from librecframework.pipeline import DefaultLeaveOneOutPipeline
from librecframework.data import DatasetFuncs
from librecframework.data.dataset import TrainDataset, LeaveOneOutTestDataset
import librecframework.data.functional as fdf
from librecframework.model import EmbeddingBasedModel
from librecframework.loss import BPRLoss, MaskedMSELoss, L2Loss
from librecframework.utils.graph_generation import complete_graph_from_pq
from librecframework.utils.convert import name_to_activation, scisp_to_torch
from librecframework.trainhook import ValueMeanHook

# To make code short
# Define:
# item = item
# prtc = participant
# init = initiator


class GCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph: dgl.DGLGraph, embeddings: torch.Tensor) -> torch.Tensor:
        # pylint: disable=E1101
        graph = graph.local_var()

        graph.ndata['h'] = embeddings

        graph.update_all(fn.copy_src(src='h', out='m'),
                         fn.mean(msg='m', out='h'))
        embeddings = graph.ndata['h']

        return embeddings


class GBGCN(EmbeddingBasedModel):
    def __init__(
            self,
            info,
            dataset: TrainDataset,
            prtc_item_graph: dgl.DGLGraph,
            init_item_graph: dgl.DGLGraph,
            prtc_to_init_graph: dgl.DGLGraph,
            init_to_prtc_graph: dgl.DGLGraph,
            social_graph: torch.Tensor):
        super().__init__(info, dataset, create_embeddings=True)
        self._bpr_loss = BPRLoss('none')

        self._SocialL2 = L2Loss(info.SL2)
        self.social_graph = social_graph.cuda()
        device = self.social_graph.device

        self.prtc_item_graph = prtc_item_graph.to(device)
        self.init_item_graph = init_item_graph.to(device)
        self.prtc_to_init_graph = prtc_to_init_graph.to(device)
        self.init_to_prtc_graph = init_to_prtc_graph.to(device)
        self.gcn = GCNLayer()
        self.layer = self.info.layer

        self.init_view_layers = [lambda x:x for _ in range(self.layer)]
        self.prtc_view_layers = [lambda x:x for _ in range(self.layer)]

        self.post_embedding_size = (1 + self.layer) * self.embedding_size
        self.init_to_item_layers = nn.ModuleList([nn.Linear(
            self.post_embedding_size, self.post_embedding_size
        ) for _ in range(1)])
        self.prtc_to_item_layers = nn.ModuleList([nn.Linear(
            self.post_embedding_size, self.post_embedding_size
        ) for _ in range(1)])
        self.item_to_init_layers = nn.ModuleList([nn.Linear(
            self.post_embedding_size, self.post_embedding_size
        ) for _ in range(1)])
        self.prtc_to_init_layers = nn.ModuleList([nn.Linear(
            self.post_embedding_size, self.post_embedding_size
        ) for _ in range(1)])
        self.item_to_prtc_layers = nn.ModuleList([nn.Linear(
            self.post_embedding_size, self.post_embedding_size
        ) for _ in range(1)])
        self.init_to_prtc_layers = nn.ModuleList([nn.Linear(
            self.post_embedding_size, self.post_embedding_size
        ) for _ in range(1)])

        self.act = name_to_activation(self.info.act)
        self.alpha = self.info.alpha
        self.beta = self.info.beta
        self.eps = 1e-8

    def load_pretrain(self, pretrain_info: Dict[str, Any]) -> None:
        path = pretrain_info['GBMF']
        pretrain = torch.load(path, map_location='cpu')
        self.ps_feature.data = pretrain['ps_feature']
        self.qs_feature.data = pretrain['qs_feature']

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        init_feature, prtc_feature = self.ps_feature, self.ps_feature
        item_feature_for_init, item_feature_for_prtc = self.qs_feature, self.qs_feature

        # bi-graph
        prtc_item_feature = torch.cat(
            (prtc_feature, item_feature_for_prtc), dim=0)
        init_item_feature = torch.cat(
            (init_feature, item_feature_for_init), dim=0)
        prtc_item_features = [prtc_item_feature]
        init_item_features = [init_item_feature]

        for k in range(self.layer):
            prtc_item_feature = self.gcn(
                self.prtc_item_graph, prtc_item_feature)
            prtc_item_feature = self.act(
                self.prtc_view_layers[k](prtc_item_feature))
            init_item_feature = self.gcn(
                self.init_item_graph, init_item_feature)
            init_item_feature = self.act(
                self.init_view_layers[k](init_item_feature))
            prtc_item_features.append(F.normalize(prtc_item_feature))
            init_item_features.append(F.normalize(init_item_feature))
        prtc_item_features = torch.cat(prtc_item_features, dim=1)
        init_item_features = torch.cat(init_item_features, dim=1)
        prtc_feature, item_feature_for_prtc = torch.split(
            prtc_item_features, (self.num_ps, self.num_qs), dim=0)
        init_feature, item_feature_for_init = torch.split(
            init_item_features, (self.num_ps, self.num_qs), dim=0)

        # cross
        init_features = [init_feature]
        prtc_features = [prtc_feature]
        item_features_for_init = [item_feature_for_init]
        item_features_for_prtc = [item_feature_for_prtc]

        for k in range(1):
            # G1
            prtc_and_item = torch.cat(
                (prtc_feature, item_feature_for_prtc), dim=0)
            prtc_and_item = self.gcn(self.prtc_item_graph, prtc_and_item)
            item_to_prtc, prtc_to_item = torch.split(
                prtc_and_item, (self.num_ps, self.num_qs), dim=0)
            item_to_prtc = self.act(self.item_to_prtc_layers[k](item_to_prtc))
            prtc_to_item = self.act(self.prtc_to_item_layers[k](prtc_to_item))
            # G2
            init_and_item = torch.cat(
                (init_feature, item_feature_for_init), dim=0)
            init_and_item = self.gcn(self.init_item_graph, init_and_item)
            item_to_init, init_to_item = torch.split(
                init_and_item, (self.num_ps, self.num_qs), dim=0)
            item_to_init = self.act(self.item_to_init_layers[k](item_to_init))
            init_to_item = self.act(self.init_to_item_layers[k](init_to_item))
            # G3
            init_to_prtc = self.gcn(self.init_to_prtc_graph, init_feature)
            init_to_prtc = self.act(self.init_to_prtc_layers[k](init_to_prtc))
            prtc_to_init = self.gcn(self.prtc_to_init_graph, prtc_feature)
            prtc_to_init = self.act(self.prtc_to_init_layers[k](prtc_to_init))
            # Reduce
            item_feature_for_init = init_to_item
            item_features_for_init.append(item_feature_for_init)
            item_feature_for_prtc = prtc_to_item
            item_features_for_prtc.append(item_feature_for_prtc)
            init_feature = (item_to_init + prtc_to_init) / 2
            init_features.append(init_feature)
            prtc_feature = (item_to_prtc + init_to_prtc) / 2
            prtc_features.append(prtc_feature)

        init_features = torch.cat(init_features, dim=1)
        prtc_features = torch.cat(prtc_features, dim=1)
        item_features_for_init = torch.cat(item_features_for_init, dim=1)
        item_features_for_prtc = torch.cat(item_features_for_prtc, dim=1)
        return init_features, prtc_features, item_features_for_init, item_features_for_prtc

    def _forward(
            self,
            ps: torch.Tensor,
            qs: torch.Tensor,
            prtcs_or_friends: torch.Tensor,
            propagate_result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if propagate_result is not None:
            init_features, prtc_features, item_features_for_init, item_features_for_prtc = propagate_result
        else:
            init_features, prtc_features, item_features_for_init, item_features_for_prtc = self.propagate()

        init_embeddings = init_features[ps]
        item_embeddings_for_init = item_features_for_init[qs]
        inits = torch.matmul(
            init_embeddings, item_embeddings_for_init.transpose(1, 2))
        prtc_embeddings = prtc_features[prtcs_or_friends]
        item_embeddings_for_prtc = item_features_for_prtc[qs]
        prtcs = torch.matmul(
            prtc_embeddings, item_embeddings_for_prtc.transpose(1, 2))
        return {
            'inits': inits,
            'prtcs': prtcs
        }, [init_embeddings, item_embeddings_for_init, prtc_embeddings, item_embeddings_for_prtc]

    def forward(
            self,
            ps: torch.Tensor,
            qs: torch.Tensor,
            prtcs_or_friends: torch.Tensor,
            masks: torch.Tensor,
            is_valid: torch.Tensor,
            propagate_result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if propagate_result is not None:
            init_features, prtc_features, item_features_for_init, item_features_for_prtc = propagate_result
        else:
            init_features, prtc_features, item_features_for_init, item_features_for_prtc = self.propagate()
        # size batch template
        counter = masks.sum(1)
        indice = []
        results = {
            # [B, 1, #qs]
            'inits': [],
            # [B, ?, #qs]
            'prtcs': []
        }
        L2 = []
        for n in range(counter.min(), counter.max() + 1):
            index = torch.where(counter == n)[0]
            if len(index) <= 0:
                continue
            indice.append(index)
            # ============ DO BATCH =============
            result, embedding = self._forward(
                ps[index],
                qs[index],
                prtcs_or_friends[index, :n],
                (init_features, prtc_features,
                 item_features_for_init, item_features_for_prtc)
            )
            result['prtcs'] = F.pad(
                result['prtcs'], (0, 0, 0, masks.shape[1] - n))
            for k in results.keys():
                v = result.pop(k)
                results[k].append(v)
            L2 += embedding
            # ============ DO BATCH =============
        indice = torch.cat(indice, dim=0)
        sorted_order = torch.sort(indice)[1]
        for k, v in results.items():
            v = torch.cat(v, dim=0)
            v = v[sorted_order]
            results[k] = v
        # ============ AFTER ============
        if not self.training:
            masks = masks.float()
            # [B, #qs]
            init_score = results['inits'].squeeze(1)
            friend_size = masks.sum(1, keepdim=True)
            # [B, #fs, #qs]
            friend_score = results['prtcs'] * masks.unsqueeze(2)
            friend_score = torch.sum(
                friend_score, dim=1) / (friend_size+self.eps)
            results = (1 - self.alpha) * init_score + self.alpha * friend_score
        # ============ AFTER ============

        # social reg
        ps_feature = init_features[:, :self.embedding_size]
        ps_embedding = ps_feature[ps].expand(
            -1, qs.shape[1], -1)
        p_from_f = torch.matmul(self.social_graph, ps_feature)
        p_from_f = p_from_f[ps].expand_as(ps_embedding)
        delta = ps_embedding - p_from_f

        return results, (masks, is_valid), (L2, delta)

    def calculate_loss(
            self,
            modelout,
            batch_size: int) -> torch.Tensor:
        results, (masks, is_valid), (L2, delta) = modelout
        masks = masks.float()
        B, P, Q = results['prtcs'].shape
        init_loss = self._bpr_loss(
            results['inits'].squeeze(1)).mean()

        valid_result = results['prtcs'][is_valid]
        valid_masks = masks[is_valid]
        valid_loss = self._bpr_loss(
            valid_result.view(-1, Q)).view_as(valid_masks)
        valid_loss = valid_loss * valid_masks
        valid_loss = valid_loss.sum(1) / (valid_masks.sum(1) + self.eps)
        valid_loss = valid_loss.sum()

        invalid_result = results['prtcs'][~is_valid]
        invalid_masks = masks[~is_valid]
        invalid_loss = self._bpr_loss(
            - invalid_result.view(-1, Q)).view_as(invalid_masks)
        invalid_loss = invalid_loss * invalid_masks
        invalid_loss = invalid_loss.sum(1) / (invalid_masks.sum(1) + self.eps)
        invalid_loss = invalid_loss.sum()

        loss = init_loss + \
            (valid_loss + self.beta * invalid_loss) / is_valid.shape[0]
        if L2 is not None:
            L2loss = self._L2(*L2, batch_size=batch_size)
            self.trainhooks['L2'](L2loss.item())
            loss = loss + L2loss
        if delta is not None:
            SocialL2loss = self._SocialL2(delta, batch_size=batch_size)
            self.trainhooks['SocialL2'](SocialL2loss.item())
            loss = loss + SocialL2loss
        return loss

    def before_evaluate(self):
        return self.propagate()

    def evaluate(
            self,
            before: Tuple[torch.Tensor, torch.Tensor],
            ps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


MODEL = GBGCN


def hyperparameter() -> HyperparamManager:
    hpm = HyperparamManager('Hyperparameter Arguments',
                            None, f'{MODEL.__name__}Info')
    hpm.register(
        'embedding_size',
        ['-EB', '--embedding-size'],
        dtype=int,
        validator=lambda x: x > 0,
        helpstr='model embedding size',
        default=32
    )
    hpm.register(
        'lr',
        multi=True,
        dtype=float,
        validator=lambda x: x > 0,
        helpstr='learning rate'
    )
    hpm.register(
        'L2',
        ['--L2'],
        multi=True,
        dtype=float,
        validator=lambda x: x >= 0,
        helpstr='model L2 normalization'
    )
    hpm.register(
        'SL2',
        ['--SL2'],
        multi=True,
        dtype=float,
        validator=lambda x: x >= 0,
        helpstr='model Social L2 normalization'
    )
    hpm.register(
        'layer',
        ['-L', '--layer'],
        multi=True,
        dtype=int,
        validator=lambda x: x >= 0,
        helpstr='model layers'
    )
    hpm.register(
        'alpha',
        ['-A', '--alpha'],
        multi=True,
        dtype=float,
        validator=lambda x: 1 >= x >= 0,
        helpstr='model (0) initiator and friend (1) weight'
    )
    hpm.register(
        'beta',
        ['-B', '--beta'],
        multi=True,
        dtype=float,
        validator=lambda x: x >= 0,
        helpstr='model invalid friend loss weight'
    )
    hpm.register(
        'act',
        ['--act'],
        multi=False,
        dtype=str,
        default='sigmoid',
        helpstr='model activation'
    )
    hpm.register(
        'pretrain',
        dtype=bool,
        default=True,
        helpstr='pretrain'
    )
    return hpm


def get_prtc_list_mask(self: TrainDataset) -> None:
    self.max_friend = max(map(len, self.friend_dict.values()))
    self.max_prtc = max(map(len, self.records)) - 2
    self.max_len = max(self.max_friend, self.max_prtc)
    self.prtc_list = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    self.prtc_mask = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    for i, record in enumerate(self.records):
        init, others = record[0], record[2:]
        mask = np.zeros([self.max_len], dtype=np.int32)
        if len(others) > 0:
            length = len(others)
            self.prtc_list[i, :length] = others
        else:
            length = len(self.friend_dict[init])
            friends = list(self.friend_dict[init])
            self.prtc_list[i, :length] = friends
        self.prtc_mask[i, :length] = 1


def get_prtc_list_mask_for_test(self: TrainDataset) -> None:
    self.max_len = max(map(len, self.friend_dict.values()))
    self.prtc_list = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    self.prtc_mask = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    for i, record in enumerate(self.records):
        init = record[0]
        mask = np.zeros([self.max_len], dtype=np.int32)
        length = len(self.friend_dict[init])
        friends = list(self.friend_dict[init])
        self.prtc_list[i, :length] = friends
        self.prtc_mask[i, :length] = 1


def train_getitem(self: TrainDataset, index: int):
    p, q_pos = self.pos_pairs[index]
    neg_q = self.neg_qs[index][self.epoch]
    prtc_or_friend = self.prtc_list[index]
    mask = self.prtc_mask[index]
    is_valid = len(self.records[index]) > 2
    return {
        'ps': torch.LongTensor([p]),
        'qs': torch.LongTensor([q_pos, neg_q]),
        'prtcs_or_friends': torch.LongTensor(prtc_or_friend),
        'masks': torch.LongTensor(mask),
        'is_valid': is_valid
    }


def test_getitem(self: LeaveOneOutTestDataset, index: int):
    p, q_pos = self.pos_pairs[index]
    neg_qs = self.neg_qs[index]
    prtc_or_friend = self.prtc_list[index]
    mask = self.prtc_mask[index]
    gt = torch.zeros(len(neg_qs)+1, dtype=torch.float)
    gt[-1] = 1
    return {
        'ps': torch.LongTensor([p]),
        'qs': torch.LongTensor(np.r_[neg_qs, q_pos]),
        'prtcs_or_friends': torch.LongTensor(prtc_or_friend),
        'masks': torch.LongTensor(mask),
        'is_valid': True
    }, {'train_mask': 0, 'ground_truth': gt}


if __name__ == "__main__":
    pipeline = DefaultLeaveOneOutPipeline(
        description=MODEL.__name__,
        supported_datasets=['BeiBei'],
        train_funcs=DatasetFuncs(
            record=fdf.modify_nothing,
            postinit=get_prtc_list_mask,
            sample=fdf.itemrec_sample,
            getitem=train_getitem,
            length=fdf.default_train_length
        ),
        test_funcs=DatasetFuncs(
            record=fdf.modify_nothing,
            postinit=get_prtc_list_mask_for_test,
            sample=None,
            getitem=test_getitem,
            length=fdf.default_leave_one_out_test_length
        ),
        hyperparam_manager=hyperparameter(),
        other_arg_path='config/config.json',
        pretrain_path='config/pretrain.json',
        sample_tag='default',
        pin_memory=True,
        min_memory=7,
        test_batch_size=128)
    pipeline.parse_args()
    pipeline.before_running()

    num_ps = pipeline.train_data.num_ps
    num_qs = pipeline.train_data.num_qs
    init_item_graph = dgl.from_scipy(complete_graph_from_pq(
        pipeline.train_data.ground_truth,
        sp.coo_matrix(([], ([], [])), shape=(num_ps, num_ps)),
        sp.coo_matrix(([], ([], [])), shape=(num_qs, num_qs)),
        dtype=np.float32,
        return_sparse=True,
        return_scipy=True,
        normalize='none'
    ))

    pos_pairs = []
    for one in pipeline.train_data.records:
        item = one[1]
        for f in one[2:]:
            pos_pairs.append((f, item))
    indice = np.array(pos_pairs, dtype=np.int32)
    values = np.ones(len(pos_pairs), dtype=np.float32)
    participant_ground_truth = sp.coo_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=(num_ps, num_qs))
    prtc_item_graph = dgl.from_scipy(complete_graph_from_pq(
        participant_ground_truth,
        sp.coo_matrix(([], ([], [])), shape=(num_ps, num_ps)),
        sp.coo_matrix(([], ([], [])), shape=(num_qs, num_qs)),
        dtype=np.float32,
        return_sparse=True,
        return_scipy=True,
        normalize='none'
    ))

    prtc_to_init_graph = dgl.graph(([],[]))
    init_to_prtc_graph = dgl.graph(([], []))
    prtc_to_init_graph.add_nodes(num_ps)
    init_to_prtc_graph.add_nodes(num_ps)
    for one in pipeline.train_data.records:
        init, prtc = one[0], one[2:]
        if len(prtc) > 0:
            prtc_to_init_graph.add_edges(prtc, init)
            init_to_prtc_graph.add_edges(init, prtc)

    social_graph_sp = pipeline.train_data.social_graph
    n = social_graph_sp.shape[0]
    social_graph_sp = social_graph_sp + sp.eye(n)
    social_graph_sp = social_graph_sp.multiply(
        1 / (social_graph_sp.sum(1) + 1e-8))
    social_graph_th = scisp_to_torch(social_graph_sp).float()

    pipeline.during_running(
        MODEL,
        {
            'prtc_item_graph': prtc_item_graph,
            'init_item_graph': init_item_graph,
            'prtc_to_init_graph': prtc_to_init_graph,
            'init_to_prtc_graph': init_to_prtc_graph,
            'social_graph': social_graph_th},
        {
            'L2': ValueMeanHook('L2loss'),
            'SocialL2': ValueMeanHook('SocialL2loss')
        },
        torch.optim.SGD)
    pipeline.after_running()
