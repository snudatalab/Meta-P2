"""
This file includes codes for Meta-P2 model
"""

from torch import nn
from .utils import *
import time
import numpy as np

class meta_decomposition(nn.Module):
    """
    Meta Tensor Decomposition
    """
    def __init__(self, n_sensors, rank, time_length, max_time_length, n_tasks, missing_idxs, task, imputation, residual):
        super(meta_decomposition, self).__init__()

        n_instances_all = len(time_length)
        n_instances = int(len(time_length) / n_tasks)

        self.Q = nn.Parameter(torch.rand(n_instances_all, max_time_length, rank, dtype=torch.float32))
        for idx, time_length in enumerate(time_length):
            self.Q.data[idx, time_length:] = 0
        self.Q = nn.Parameter(self.Q.view(n_tasks, n_instances, max_time_length, rank))

        self.W = nn.Parameter(torch.rand(n_tasks, n_instances, rank, dtype=torch.float32))
        self.V = nn.Parameter(torch.rand(n_sensors, rank, dtype=torch.float32))
        self.V_meta = nn.Parameter(torch.rand(n_sensors, rank, dtype=torch.float32))
        self.H = nn.Parameter(torch.rand(n_tasks, rank, rank, dtype=torch.float32))

        self.n_tasks = n_tasks
        self.n_instances = n_instances
        self.missing_idxs = missing_idxs
        self.task = task
        self.imputation = imputation
        self.residual = residual

    def forward(self):
        """
        Return the diagonal matrix
        """
        return self.W[:-1], self.W[-1].unsqueeze(0)

    def train_model(self, X, X_true, epochs):
        """
        train model
        """
        with torch.no_grad():
            print("Training starts ...")

            for epoch in range(1, epochs+1):
                X_pred = self.loop(X)

            print("Test starts ...")
            self.V_meta = nn.Parameter(self.V.data)
            rec_errs, iter_times = [], []

            for epoch in range(1, epochs+1):
                start = time.time()
                X_pred = self.loop(X, True)

                if self.task == "missing-value-prediction":
                    if self.imputation == "y":
                        if epoch == 1:
                            X[-1] = self.update_missing_values(X[-1], X_pred[-1])
                    recovery_err = self.compute_missing_rate(X_true[-1], X_pred[-1])
                elif self.task == 'anomaly-detection':
                    if epoch == 1:
                        X_true = X[-1].clone().detach()
                    prec, anomaly_idxs = self.compute_anomaly_precision(X_true, X_pred[-1])

                    if epoch==1 and self.imputation == "y":
                        X[-1] = self.update_anomaly_values(X[-1], X_pred[-1], anomaly_idxs)

                if epoch % 1 == 0:
                    if self.task == 'anomaly-detection':
                        rec_errs.append(prec)
                    else:
                        rec_errs.append(recovery_err.item())

                end = time.time()
                iter_times.append(end-start)
            return rec_errs, iter_times

    def _update_Q(self, X, dom):
        """
        update Q with truncated SVD
        """
        XV = torch.einsum("bij, jk -> bik", X[dom], self.V)

        S = torch.stack([torch.diag(s) for s in self.W[dom]])
        XVS = torch.einsum("bij, bjk -> bik", XV, S)
        XVSH_t = torch.einsum("bij, jk -> bik", XVS, self.H[dom].T)

        Z, S_d, P = torch.svd(XVSH_t)
        P_t = P.permute(0, 2, 1)

        return torch.einsum("bij, bjk -> bik", Z, P_t)

    def _update_HVW(self, Y, domain, is_test=False, X=None):
        """
        update H, V, and W
        """
        Y1 = Y.permute(0, 2, 1).reshape(Y.size(0), -1)
        Y2 = Y.permute(2, 0, 1).reshape(-1, Y.size(1)).T
        Y3 = Y.permute(1, 0, 2).reshape(-1, Y.size(2)).T

        H_inv = torch.matmul(self.W[domain].T, self.W[domain]) * torch.matmul(self.V.T, self.V)
        self.H[domain] = nn.Parameter(torch.matmul(torch.matmul(Y1, khatri(self.W[domain], self.V)), torch.pinverse(H_inv)))
        W_inv = torch.matmul(self.V.T, self.V) * torch.matmul(self.H[domain].T, self.H[domain])
        self.W[domain] = nn.Parameter(torch.matmul(torch.matmul(Y3, khatri(self.V, self.H[domain])), torch.pinverse(W_inv)))
        V_inv = torch.matmul(self.W[domain].T, self.W[domain]) * torch.matmul(self.H[domain].T, self.H[domain])
        self.V = nn.Parameter(torch.matmul(torch.matmul(Y2, khatri(self.W[domain], self.H[domain])), torch.pinverse(V_inv)))

        if is_test:
            if self.residual == 'y':
                return (self.V + self.V_meta)/2
            else:
                return self.V
        else:
            Q1 = nn.Parameter(self._update_Q(X, domain))
            Y = torch.einsum("bij, bjk -> bik", Q1.permute((0, 2, 1)), X[domain]).permute((1, 2, 0))
            Y1 = Y.permute(0, 2, 1).reshape(Y.size(0), -1)
            Y2 = Y.permute(2, 0, 1).reshape(-1, Y.size(1)).T
            Y3 = Y.permute(1, 0, 2).reshape(-1, Y.size(2)).T

            H_inv = torch.matmul(self.W[domain].T, self.W[domain]) * torch.matmul(self.V.T, self.V)
            H1 = nn.Parameter(torch.matmul(torch.matmul(Y1, khatri(self.W[domain], self.V)), torch.pinverse(H_inv)))
            W_inv = torch.matmul(self.V.T, self.V) * torch.matmul(H1.T, H1)
            W1 = nn.Parameter(torch.matmul(torch.matmul(Y3, khatri(self.V, H1)), torch.pinverse(W_inv)))

            V_inv = torch.matmul(W1.T, W1) * torch.matmul(H1.T, H1)
            V2 = torch.matmul(torch.matmul(Y2, khatri(W1, H1)), torch.pinverse(V_inv))
            return V2

    def inner_loop(self, X, domain, is_test=False):
        """
        inner-loop of meta-learning
        """
        self.Q[domain] = nn.Parameter(self._update_Q(X, domain))
        Y = torch.einsum("bij, bjk -> bik", self.Q[domain].permute((0, 2, 1)), X[domain]).permute((1, 2, 0))
        V_t = self._update_HVW(Y, domain, is_test, X)

        return V_t

    def loop(self, X, is_test=False):
        """
        outer-loop of meta-learning
        """
        if is_test:
            self.V = nn.Parameter(self.inner_loop(X, -1, is_test))
        else:
            n_trn_tasks = self.n_tasks-1

            V = 0
            for i in range(n_trn_tasks):
                V += self.inner_loop(X, i)
            self.V = nn.Parameter(V/n_trn_tasks)
        return self.reconstruct()

    def reconstruct(self):
        """
        reconstruct the original tensor with computed latent factor matrices
        """
        Q = self.Q.flatten(0, 1)
        W = self.W.flatten(0, 1)
        H = torch.cat([h.repeat([self.n_instances, 1, 1]) for h in self.H])

        U = torch.einsum("bij, bjk -> bik", Q, H)
        S = torch.stack([torch.diag(s) for s in W])
        US = torch.einsum("bij, bjk -> bik", U, S)
        X_pred = torch.einsum("bij, jk -> bik", US, self.V.T)

        return X_pred.view(self.n_tasks, -1, X_pred.size(1), X_pred.size(2))

    def compute_missing_rate(self, X, X_pred):
        """
        compute the normalized missing value prediction error
        """
        x = X.permute((0, 2, 1))
        x_pred = X_pred.permute((0, 2, 1))

        err, norm = 0, 0
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                idx = self.missing_idxs[i*x.size(1) + j]
                err += torch.pow(torch.norm(x_pred[i,j][idx] - x[i,j][idx]), 2)
                norm += torch.pow(torch.norm(x[i,j][idx]), 2)

        return err / norm

    def compute_anomaly_precision(self, X, X_pred):
        """
        Compute the precision of anomaly detection task
        """
        _X_pred = X_pred.permute((0, 2, 1))
        _X = X.permute((0, 2, 1))

        anomaly_idxs = []
        for i in range(_X_pred.shape[0]):
            for j in range(_X_pred.shape[1]):
                _, anomaly_idx = torch.topk((_X_pred[i, j] - _X[i, j]), int(len(self.missing_idxs[0])))
                anomaly_idxs.append(anomaly_idx.cpu().detach().numpy())

        anomaly_idxs = np.array(anomaly_idxs)
        missing_idxs = np.array(self.missing_idxs)

        cnt=0
        for i in range(anomaly_idxs.shape[0]):
            intersect = np.intersect1d(anomaly_idxs[i], missing_idxs[i])
            cnt += len(intersect)
        return cnt / (anomaly_idxs.shape[0]*anomaly_idxs.shape[1]), anomaly_idxs

    def update_missing_values(self, X, X_pred):
        """
        Imputation of missing values on the target domain
        """
        x = X.permute((0, 2, 1))
        x_pred = X_pred.permute((0, 2, 1))

        for i in range(x.size(0)):
            for j in range(x.size(1)):
                idx = self.missing_idxs[i*x.size(1) + j]
                x[i,j][idx] = x_pred[i,j][idx]

        return x.permute((0, 2, 1))

    def update_anomaly_values(self, X, X_pred, anomaly_idxs):
        """
        Imputation of the detected anomalies on the target domain
        """
        x = X.permute((0, 2, 1))
        x_pred = X_pred.permute((0, 2, 1))

        for i in range(x.size(0)):
            for j in range(x.size(1)):
                idx = anomaly_idxs[i*x.size(1) + j]
                x[i,j][idx] = x_pred[i,j][idx]

        return x.permute((0, 2, 1))