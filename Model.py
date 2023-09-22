import torch
from torch import nn
import math
from Lasso import sparse_encode
from testers import experiment_berg_params
from testers import plot_rec_image


def breg_rec(diffuser_batch, bucket_batch, batch_size):
    if diffuser_batch.dim() == 3:
        img_dim = diffuser_batch.shape[2]  # diffuser in shape (batch_size, n_masks, img_dim)
    else:
        img_dim = diffuser_batch.shape[1]  # diffuser in shape (n_masks, img_dim)

    recs_container = torch.zeros(batch_size, img_dim)
    for rec_ind in range(batch_size):
        maxiter = 1  # 1, 50
        niter_inner = 1  # 1, 3
        # mu = 0.01  # 10
        alpha = 0.3

        # experiment_berg_params(bucket_batch[rec_ind], diffuser_batch[rec_ind])

        if diffuser_batch.dim() == 3:
            diffuser = diffuser_batch[rec_ind]
        else:
            diffuser = diffuser_batch

        rec = sparse_encode(bucket_batch[rec_ind], diffuser, maxiter=maxiter, niter_inner=niter_inner, alpha=alpha,
                            algorithm='split-bregman')
        # plot_rec_image(rec, maxiter, niter_inner, alpha)
        recs_container = recs_container.clone()
        recs_container[rec_ind] = rec

    return recs_container


class Gen_no_batch(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, n_masks * img_dim// 2)
        self.bn3 = nn.BatchNorm1d(n_masks * img_dim // 2)
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


class Gen(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, n_masks * img_dim)
        self.bn3 = nn.BatchNorm1d(n_masks * img_dim)
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
        x = self.sigmoid(x)
        return x


class Gen_big_diff(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks, ac_stride):
        super().__init__()
        self.ac_stride = ac_stride
        self.pic_width = int(math.sqrt(img_dim))
        self.linear1 = nn.Linear(z_dim, img_dim)
        self.bn1 = nn.BatchNorm1d(img_dim)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(img_dim, img_dim)
        self.bn2 = nn.BatchNorm1d(img_dim)
        self.relu2 = nn.ReLU()

        self.diff_width = self.pic_width + self.ac_stride * n_masks
        self.linear3 = nn.Linear(img_dim, self.diff_width*self.pic_width)
        self.bn3 = nn.BatchNorm1d(self.diff_width*self.pic_width)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.sigmoid(x)
        return x


class Gen_conv_3(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()
        self.img_dim = img_dim
        self.n_masks = n_masks
        self.pic_width = int(math.sqrt(self.img_dim))
        self.linear1 = nn.Linear(z_dim, self.img_dim)
        self.bn1 = nn.BatchNorm1d(self.img_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(1, self.n_masks // 4, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(self.n_masks//4)

        self.conv3 = nn.Conv2d(self.n_masks // 4, self.n_masks//2, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(self.n_masks//2)

        self.conv4 = nn.Conv2d(self.n_masks//2, self.n_masks, kernel_size=3, padding='same')
        self.bn4 = nn.BatchNorm2d(self.n_masks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(-1, 1, self.pic_width, self.pic_width)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.sigmoid(x)
        return x


class Gen_conv1(nn.Module):
    def __init__(self, z_dim, img_dim, n_masks):
        super().__init__()
        self.img_dim = img_dim
        self.n_masks = n_masks
        self.pic_width = int(math.sqrt(self.img_dim))
        self.linear1 = nn.Linear(z_dim, self.img_dim)
        self.bn1 = nn.BatchNorm1d(self.img_dim)
        self.relu1 = nn.ReLU()
        #
        # self.linear2 = nn.Linear(128, img_dim)
        # self.bn2 = nn.BatchNorm1d(img_dim)
        # self.relu2 = nn.ReLU()

        # self.linear3 = nn.Linear(img_dim, n_masks * img_dim)
        # self.bn3 = nn.BatchNorm1d(n_masks * img_dim)
        # self.sigmoid = nn.Sigmoid()

        # Change the last linear layer to a convolutional layer
        self.conv3 = nn.Conv2d(1, self.n_masks, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm2d(self.n_masks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = x.view(-1, 1, self.pic_width, self.pic_width)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x.reshape(-1, self.n_masks, self.img_dim)
        x = self.sigmoid(x)
        return x

