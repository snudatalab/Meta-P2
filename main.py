"""
Fast and Accurate Domain Adaptation for Irregular Tensor Decomposition (submitted to KDD 2024)
This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""

import argparse
import numpy as np
import torch

import models
import data

def to_device(gpu):
    """
    make torch to use GPU
    """
    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')

def parse_args():
    """
    parser arguments to run program in cmd
    """
    parser = argparse.ArgumentParser()

    # Pre-sets before start the program
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # Hyperparameters for training
    parser.add_argument('--data', type=str, default="sp500")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--abnormal-ratio', type=float, default=0.1)
    parser.add_argument('--task', type=str, default="missing-value-prediction",
                        help='missing-value-prediction / anomaly-detection')

    # Hyperparameters for models
    parser.add_argument('--rank', type=int, default=20)
    parser.add_argument('--imputation', type=str, default="y")
    parser.add_argument('--residual', type=str, default="y")

    return parser.parse_args()

def main(iteration):
    # initial setting for torch
    args = parse_args()
    args.seed = iteration
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = to_device(args.gpu)

    trn_X, test_X, test_X_miss, time_len_list, max_time_len, missing_idxs = \
        data.read_data(args.data, args.abnormal_ratio, args.task)

    n_sensors = test_X.size(2)
    n_trn_tasks = trn_X.size(0)
    n_test_tasks = test_X.size(0)
    n_tasks = n_trn_tasks + n_test_tasks

    # Upload data to GPU
    trn_X = trn_X.to(device)
    test_X = test_X.to(device)
    test_X_miss = test_X_miss.to(device)

    # Set-up models
    print("------------------------ Meta-Decomposition ------------------------")
    model = models.meta_decomposition(n_sensors, args.rank, time_len_list, max_time_len, n_tasks,
                                      missing_idxs, args.task, args.imputation, args.residual).to(device)
    X = torch.cat([trn_X, test_X_miss], 0).permute((0, 1, 3, 2))
    X_true = torch.cat([trn_X, test_X], 0).permute((0, 1, 3, 2))
    errs, times = model.train_model(X, X_true, args.epochs)

    return errs, times

if __name__ == '__main__':
    final_metrics = []
    for i in range(1, 11):
        print(f"\n================================= LEARNING #{i} =================================")
        metric, times = main(i)
        final_metrics.append(metric[-1])
        print(f"\nPerformance: {metric[-1]:.4f}")
        print(f"Each iteration takes {np.mean(times):.4f} sec")

    print('\n================================= Final result =================================')

    final_metrics.remove(max(final_metrics))
    final_metrics.remove(min(final_metrics))

    print(f"\nFinal performance: {np.mean(final_metrics):.4f} with standard deviation: {np.std(final_metrics):.4f}")