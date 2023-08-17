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
        self.pic_width = int(math.sqrt(img_dim))
        self.linear1 = nn.Linear(z_dim, img_dim)
        self.bn1 = nn.BatchNorm1d(img_dim)
        self.relu1 = nn.ReLU()
        #
        # self.linear2 = nn.Linear(128, img_dim)
        # self.bn2 = nn.BatchNorm1d(img_dim)
        # self.relu2 = nn.ReLU()

        # self.linear3 = nn.Linear(img_dim, n_masks * img_dim)
        # self.bn3 = nn.BatchNorm1d(n_masks * img_dim)
        # self.sigmoid = nn.Sigmoid()

        # Change the last linear layer to a convolutional layer
        self.conv3 = nn.Conv2d(1, n_masks, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(n_masks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(-1, 1, self.pic_width, self.pic_width)
        x = self.conv3(x)
        x = self.bn3(x)
        out = self.sigmoid(x)
        return out

