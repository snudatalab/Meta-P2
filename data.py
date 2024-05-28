"""
This file includes codes for loading and preprocessing data
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing

def split_data(label, trn_ratio, val_ratio=0.1, seed=0):
    """
    Split data into training, validation, and test set
    """
    state = np.random.RandomState(seed)
    all_idx = np.arange(label.shape[0])
    trn_idx = list(state.choice(all_idx, size=int(all_idx.shape[0] * trn_ratio), replace=False))
    test_idx = list(set(all_idx).difference(set(trn_idx)))
    val_idx = list(state.choice(test_idx, size=int(np.shape(test_idx)[0] * val_ratio), replace=False))
    test_idx = list(set(test_idx).difference(set(val_idx)))

    return trn_idx, val_idx, test_idx

def pickleLoad(path):
    """
    load pickle data
    """
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        X = data["value"]
        try:
            Y = data["label"]
            le = preprocessing.LabelEncoder()
            le.fit(Y)
            Y = le.transform(Y)
        except:
            Y = [0, 0]
    else:
        print("No data")
    return X, Y

def max_len(X):
    """
    compute the max length of irregular axis
    """
    max_len, min_len = 0, np.inf
    length = []
    for x in X:
        for t in x:
            temp_len = t.size(1)
            length.append(temp_len)
            max_len = max([max_len, temp_len])
            min_len = min([min_len, temp_len])
    return max_len, min_len, length

def pad_data(X, max_len):
    """
    pad data with zero
    """
    data = []
    for x in X:
        tasks = []
        for t in x:
            padded = F.pad(input=t, pad=(0, max_len-t.size(1), 0, 0), mode='constant', value=0)
            tasks.append(padded)
        data.append(tasks)
    return data

def create_missing(X, missing_ratio, task, seed=0):
    """
    create missing/anomalous entries
    """
    state = np.random.RandomState(seed)
    all_idx = np.arange(X.shape[-1])
    missing_idxs = []

    for i in range(X.size(0)):
        for j in range(X.size(1)):
            for k in range(X.size(2)):
                missing_idx = list(state.choice(all_idx, size=int(all_idx.shape[0] * missing_ratio+1), replace=False))
                if task == 'missing-value-prediction':
                    X[i, j, k][missing_idx] = 0
                    missing_idxs.append(missing_idx)
                elif task == 'anomaly-detection':
                    X[i, j, k][missing_idx] /= 2
                    missing_idxs.append(missing_idx)

    return X, missing_idxs

def read_data(dataset, missing_ratio, task):
    """
    read data from path
    """
    root = './data/'

    if dataset=='sp500':
        root += "stock_market_data/sp500/preprocess/"
        class_list = ['Communication Services.pickle', 'Consumer Discretionary.pickle', 'Consumer Staples.pickle',
                      'Energy.pickle', 'Financials.pickle', 'Health Care.pickle', 'Industrials.pickle',
                      'Information Technology.pickle', 'Materials.pickle', 'Real Estate.pickle', 'Utilities.pickle']

    elif dataset=='nasdaq':
        root += "stock_market_data/nasdaq/preprocess/stock_data/"
        class_list = ['Basic Materials.pickle', 'Communication Services.pickle', 'Consumer Cyclical.pickle',
                      'Financial Services.pickle', 'Healthcare.pickle', 'Real Estate.pickle', 'Utilities.pickle']

    elif dataset=='cricket':
        root += "cricket/preprocess/"
        class_list = [str(b'1.0')+".pickle", str(b'2.0')+".pickle", str(b'3.0')+".pickle", str(b'4.0')+".pickle",
                      str(b'5.0')+".pickle", str(b'6.0')+".pickle", str(b'7.0')+".pickle", str(b'8.0')+".pickle",
                      str(b'9.0') + ".pickle", str(b'10.0')+".pickle", str(b'11.0')+".pickle", str(b'12.0')+".pickle"]

    elif dataset=='natops':
        root += "natops-h/"
        class_list = sorted(os.listdir(root))

    elif dataset=='fingermovement':
        root += "fingermovement/preprocess/"
        class_list = [str(b'left') + ".pickle", str(b'right') + ".pickle"]

    elif dataset=='kor-stock':
        root += "kor_stock/"
        class_list=sorted(os.listdir(root))

    X, Y = [], []

    for file in class_list:
        if dataset in ['kor-stock', 'nasdaq', 'natops']:
            temp_list = []
            for xx in pickleLoad(root+file)[0]:
                temp_list.append(torch.transpose(xx, 0, 1))
            X.append(temp_list)
        else:
            X.append(pickleLoad(root+file)[0])
        Y.append(pickleLoad(root + file)[1])

    # Non-zero elements
    n_non_zero = 0
    for i in range(len(X)):
            x = X[i]
            for j in range(len(x)):
                n_non_zero += X[i][j].shape[0]*X[i][j].shape[1]

    test_Y, trn_Y = torch.tensor(np.array(Y[-1])), torch.tensor(np.array(Y[:-1]))

    trn_X, test_X_list = X[:-1], [X[-1]]
    max_length, min_len, time_length = max_len(X)

    trn_X = pad_data(trn_X, max_length)
    test_X = pad_data(test_X_list, max_length)

    trn_X_list, test_X_list = [], []
    for X in trn_X:
        trn_X_list.append(torch.stack(X, 0).type(torch.float32))
    trn_X = torch.stack(trn_X_list, 0)
    for X in test_X:
        test_X_list.append(torch.stack(X, 0).type(torch.float32))
    test_X = torch.stack(test_X_list, 0).type(torch.float32)

    test_X_miss, missing_idx = create_missing(test_X.clone(), missing_ratio, task)

    # Normalize data
    if dataset in ['nasdaq', 'kor-stock', 'sp500']:
        for i in range(trn_X.shape[0]):
            for j in range(trn_X.shape[1]):
                trn_X[i, j] = F.normalize(trn_X[i, j], dim=1)
        for i in range(test_X_miss.shape[0]):
            for j in range(test_X_miss.shape[1]):
                test_X[i, j] = F.normalize(test_X[i, j], dim=1)
                test_X_miss[i, j] = F.normalize(test_X_miss[i, j], dim=1)

    print(f"------------------------ Data ------------------------\n"
          f"Train domains size: {trn_X.size()}\n"
          f"Test domains size: {test_X.size()}\n"
          f"Number of non-zero elements: {n_non_zero}\n")

    return trn_X, test_X, test_X_miss, time_length, max_length, missing_idx