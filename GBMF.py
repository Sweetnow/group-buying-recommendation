#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librecframework.argument.manager import HyperparamManager
from librecframework.pipeline import DefaultLeaveOneOutPipeline
from librecframework.data import DatasetFuncs
from librecframework.data.dataset import TrainDataset, LeaveOneOutTestDataset
import librecframework.data.functional as fdf
from librecframework.model import EmbeddingBasedModel
from librecframework.loss import BPRLoss


class GBMF(EmbeddingBasedModel):
    def __init__(self, info, dataset: TrainDataset):
        super().__init__(info, dataset, create_embeddings=True)
        self._bpr_loss = BPRLoss('none')
        self.alpha = info.alpha
        self.eps = 1e-8

    def load_pretrain(self, pretrain_info: Dict[str, Any]) -> None:
        path = pretrain_info['MF']
        pretrain = torch.load(path, map_location='cpu')
        self.ps_feature.data = pretrain['ps_feature']
        self.qs_feature.data = pretrain['qs_feature']

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ps_feature, self.qs_feature

    def _forward(
            self,
            ps: torch.Tensor,
            qs: torch.Tensor,
            participants_or_friends: torch.Tensor,
            propagate_result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if propagate_result is not None:
            ps_feature, qs_feature = propagate_result
        else:
            ps_feature, qs_feature = self.propagate()
        ps = torch.cat((ps, participants_or_friends), dim=1)
        ps_embeddings = ps_feature[ps]
        qs_embeddings = qs_feature[qs]
        score = torch.matmul(ps_embeddings, qs_embeddings.transpose(1, 2))
        # [B, ?, #qs]
        initiators, participants = torch.split(
            score, (1, participants_or_friends.shape[1]), dim=1)
        return {
            'initiators': initiators,
            'participants': participants
        }, [ps_embeddings, qs_embeddings]

    def forward(
            self,
            ps: torch.Tensor,
            qs: torch.Tensor,
            participants_or_friends: torch.Tensor,
            masks: torch.Tensor,
            is_valid: torch.Tensor,
            propagate_result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if propagate_result is not None:
            ps_feature, qs_feature = propagate_result
        else:
            ps_feature, qs_feature = self.propagate()
        # size batch template
        counter = masks.sum(1)
        indice = []
        results = {
            # [B, 1, #qs]
            'initiators': [],
            # [B, ?, #qs]
            'participants': []
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
                participants_or_friends[index, :n],
                (ps_feature, qs_feature)
            )
            result['participants'] = F.pad(
                result['participants'], (0, 0, 0, masks.shape[1] - n))
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
            initiator_score = results['initiators'].squeeze(1)
            friend_size = masks.sum(1, keepdim=True)
            # [B, #fs, #qs]
            friend_score = results['participants'] * masks.unsqueeze(2)
            friend_score = torch.sum(
                friend_score, dim=1) / (friend_size+self.eps)
            results = (1 - self.alpha) * initiator_score + self.alpha * friend_score
        # ============ AFTER ============
        return results, (masks, is_valid), L2

    def calculate_loss(
            self,
            modelout,
            batch_size: int) -> torch.Tensor:
        results, (masks, is_valid), tensors = modelout
        masks = masks.float()
        B, P, Q = results['participants'].shape
        initiator_loss = self._bpr_loss(
            results['initiators'].squeeze(1)).mean()

        valid_result = results['participants'][is_valid]
        valid_masks = masks[is_valid]
        valid_loss = self._bpr_loss(
            valid_result.view(-1, Q)).view_as(valid_masks)
        valid_loss = valid_loss * valid_masks
        valid_loss = valid_loss.sum(1) / (valid_masks.sum(1) + self.eps)
        valid_loss = valid_loss.sum()

        loss = initiator_loss + valid_loss / is_valid.shape[0]

        if tensors is not None:
            loss = loss + self._L2(*tensors, batch_size=batch_size)
        return loss

    def before_evaluate(self):
        return self.propagate()

    def evaluate(
            self,
            before: Tuple[torch.Tensor, torch.Tensor],
            ps: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


MODEL = GBMF


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
        'alpha',
        ['-A', '--alpha'],
        multi=True,
        dtype=float,
        validator=lambda x: 1 >= x >= 0,
        helpstr='model (0) initiator and friend (1) weight'
    )
    return hpm


def get_participant_list_mask(self: TrainDataset) -> None:
    self.max_friend = max(map(len, self.friend_dict.values()))
    self.max_participant = max(map(len, self.records)) - 2
    self.max_len = max(self.max_friend, self.max_participant)
    self.participant_list = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    self.participant_mask = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    for i, record in enumerate(self.records):
        initiator, others = record[0], record[2:]
        mask = np.zeros([self.max_len], dtype=np.int32)
        if len(others) > 0:
            length = len(others)
            self.participant_list[i, :length] = others
        else:
            length = len(self.friend_dict[initiator])
            friends = list(self.friend_dict[initiator])
            self.participant_list[i, :length] = friends
        self.participant_mask[i, :length] = 1


def get_participant_list_mask_for_test(self: TrainDataset) -> None:
    self.max_len = max(map(len, self.friend_dict.values()))
    self.participant_list = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    self.participant_mask = np.zeros(
        (len(self.records), self.max_len), dtype=np.int32)
    for i, record in enumerate(self.records):
        initiator = record[0]
        mask = np.zeros([self.max_len], dtype=np.int32)
        length = len(self.friend_dict[initiator])
        friends = list(self.friend_dict[initiator])
        self.participant_list[i, :length] = friends
        self.participant_mask[i, :length] = 1


def train_getitem(self: TrainDataset, index: int):
    p, q_pos = self.pos_pairs[index]
    neg_q = self.neg_qs[index][self.epoch]
    participant_or_friend = self.participant_list[index]
    mask = self.participant_mask[index]
    is_valid = len(self.records[index]) > 2
    return {
        'ps': torch.LongTensor([p]),
        'qs': torch.LongTensor([q_pos, neg_q]),
        'participants_or_friends': torch.LongTensor(participant_or_friend),
        'masks': torch.LongTensor(mask),
        'is_valid': is_valid
    }


def test_getitem(self: LeaveOneOutTestDataset, index: int):
    p, q_pos = self.pos_pairs[index]
    neg_qs = self.neg_qs[index]
    participant_or_friend = self.participant_list[index]
    mask = self.participant_mask[index]
    gt = torch.zeros(len(neg_qs)+1, dtype=torch.float)
    gt[-1] = 1
    return {
        'ps': torch.LongTensor([p]),
        'qs': torch.LongTensor(np.r_[neg_qs, q_pos]),
        'participants_or_friends': torch.LongTensor(participant_or_friend),
        'masks': torch.LongTensor(mask),
        'is_valid': True
    }, {'train_mask': 0, 'ground_truth': gt}


if __name__ == "__main__":
    pipeline = DefaultLeaveOneOutPipeline(
        description=MODEL.__name__,
        supported_datasets=['BeiBei'],
        train_funcs=DatasetFuncs(
            record=fdf.modify_nothing,
            postinit=get_participant_list_mask,
            sample=fdf.itemrec_sample,
            getitem=train_getitem,
            length=fdf.default_train_length
        ),
        test_funcs=DatasetFuncs(
            record=fdf.modify_nothing,
            postinit=get_participant_list_mask_for_test,
            sample=None,
            getitem=test_getitem,
            length=fdf.default_leave_one_out_test_length
        ),
        hyperparam_manager=hyperparameter(),
        other_arg_path='config/config.json',
        pretrain_path='config/pretrain.json',
        sample_tag='default',
        pin_memory=True,
        min_memory=6,
        test_batch_size=128)
    pipeline.parse_args()
    pipeline.before_running()
    pipeline.during_running(MODEL, {})
    pipeline.after_running()
