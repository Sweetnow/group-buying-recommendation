## Group-Buying Recommendation for Social E-Commerce

This is the official implementation of the paper ***Group-Buying Recommendation for Social E-Commerce*** [(PDF)](https://arxiv.org/abs/2010.06848) accepted by ICDE'2021.


### Group-Buying Dataset

**Group buying**, as an emerging form of purchase
in social e-commerce websites, such as Pinduoduo.com
, has recently
achieved great success. In this new business model, users, *initiator*,
can launch a group and share products to their social networks,
and when there are enough friends, *participants*, join it, the
deal is clinched. Group-buying recommendation for social ecommerce, which recommends an item list when users want to
launch a group, plays an important role in the group success ratio
and sales.

The information about the dataset can be found in `BeiBei/readme.txt`.

### Code

We separate model definition from the framework `librecframework` for easily understanding.

You can find the framework `librecframework` in https://github.com/Sweetnow/librecframework.

Both modules mentioned in `requirements.txt` and `librecframework` should be installed before running the code.

More details about our codes will be added soon.

### Usage

1. Download both `librecframework` and this repo
```bash
git clone git@github.com:Sweetnow/librecframework.git
git clone git@github.com:Sweetnow/group-buying-recommendation.git
```

2. Install `librecframework` (**Python >= 3.8**)
```bash
cd librecframework/
bash install.sh
```

3. Install [dgl](https://www.dgl.ai/pages/start.html)

3. Download `negative.zip` from [Release](https://github.com/Sweetnow/group-buying-recommendation/releases/download/v1.0/negative.zip), unzip it and copy `*.negative.txt` to `datasets/BeiBei/`
```bash
wget https://github.com/Sweetnow/group-buying-recommendation/releases/download/v1.0/negative.zip
unzip negative.zip
cp negative/* ${PATH-TO-GROUP-BUYING-RECOMMENDATION}/datasets/BeiBei
```

*PS: negative sampling file is used for testing. More details can be found in [Datasets README](./datasets/readme.md)*

4. Set `config/config.json` and `config/pretrain.json` following [Docs](https://github.com/Sweetnow/librecframework#example).

5. Run the following command to know the CLI and check python environment:
```bash
python3 GBGCN train -h
# or
# python3 GBGCN test -h
```

*PS: If you set hyperparameters that support multi input to multi values, the framework will automatically do grid-search accroding to your input. That is, use the Cartesian product of the hyperparameters for training and testing. For example, set `--lr 0.1 0.01 -L 1 2`, the codes will train and test model with hyperparameters \[(0.1, 1), (0.1, 2), (0.01, 1), (0.01, 2)\].*

### Citation

If you want to use our codes or dataset in your research, please cite:



```
@inproceedings{zhang2021group,
  title={Group-Buying Recommendation for Social E-Commerce},
  author={Zhang, Jun and Gao, Chen and Jin, Depeng and Li, Yong},
  booktitle={2021 IEEE 37th International Conference on Data Engineering (ICDE)},
  year={2021},
  organization={IEEE}
}
```



### Acknowledgement
