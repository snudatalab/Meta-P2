# Fast and Accurate Domain Adaptation for Irregular Tensor Decomposition

This repository is the official implementation of 
"Fast and Accurate Domain Adaptation for Irregular Tensor Decomposition"
(KDD 2024).



## Prerequisites

- Python 3.5+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [sciket-learn](https://scikit-learn.org/)

## Datasets

Preprocessed data are included in the `data` directory.
You can use your own data if it is a 3-way irregular tensors in multiple domains.

| Name        | # Domain |   Max $I_k$ |  $(J, K)$ | # Non-zero | Summary | Download                                                        |
|-------------|---------:|------------:|----------:|-----------:|--------:|:----------------------------------------------------------------|
| Nasdaq      |        3 |      12,709 |   (6, 11) |     2,742K |   stock | https://kaggle.com/datasets/paultimothymooney/stock-market-data |
| SP500       |       11 |      13,321 |   (6, 13) |     7,318K |   stock | https://kaggle.com/datasets/paultimothymooney/stock-market-data |
| Korea stock |       11 |       3,089 |   (6, 10) |     2,038K |   stock | https://github.com/jungijang/KoreaStockData                     |
| NATOPS      |       20 |       2,009 |  (77, 24) |    42,720K |     HAR | https://github.com/yalesong/natops                              |
| Cricket     |       12 |       1,197 |   (6, 10) |     1,292K |     HAR | https://timeseriesclassification.com                            |

## Usage

```
python main.py --data {dataset} --abnormal-ratio {abnormal_ratio} --task {task} --rank {target rank}
```
- {dataset} : Dataset to use. One of fingermovement, nasdaq, sp500, kor-stock, natops or cricket.
- {abnormal_ratio}: Ratio of abnormal values (e.g., 0.1, 0.3).
- {task}: Task to perform. One of missing-value-prediction or anomaly-detection
- {rank}: Target rank of decomposition

## Run demo

You can run a demo script `run.sh` that reproduces the experimental results in the paper by the following command.
```
bash run.sh
```