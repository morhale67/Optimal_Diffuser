import torch
from torch import nn
import math
from Lasso import sparse_encode


def breg_rec(diffuser_batch, bucket_batch, batch_size):
    recs_container = torch.zeros((batch_size, diffuser_batch.shape[2]))
    for rec_ind in range(batch_size):
        niter_out = 1  # 50
        niter_in = 1  # 3
        mu = 10  # 0.01
        lamda = 0.3
        rec = sparse_encode(bucket_batch[rec_ind], diffuser_batch[rec_ind], maxiter=1, niter_inner=1, alpha=lamda,
                            algorithm='split-bregman')

        recs_container = recs_container.clone()
        recs_container[rec_ind] = rec

    return recs_container


class Gen(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(256, n_masks)
        self.bn3 = nn.BatchNorm1d(n_masks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        # x = torch.sign(x)
        out = self.sigmoid(x)
        return out