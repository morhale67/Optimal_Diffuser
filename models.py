from torch import nn
import math
import torch


def before_sign_hook(grad):
    print("before_sign grad ",grad)

    return grad


def after_sign_hook(grad):
    print("after_sign grad ",grad)

    return grad


class Gen(nn.Module):
    def __init__(self, z_dim, img_dim, cr):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, math.floor(img_dim / cr) * img_dim)
        self.bn3 = nn.BatchNorm1d(math.floor(img_dim / cr) * img_dim)
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


class Gen_1(nn.Module):
    def __init__(self, z_dim, img_dim, cr):
        super().__init__()

        self.linear1 = nn.Linear(z_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(256, math.floor(img_dim / cr) * img_dim)
        self.bn3 = nn.BatchNorm1d(math.floor(img_dim / cr) * img_dim)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        before_sign = self.bn3(x)
        before_sign.register_hook(before_sign_hook)
        before_sign.retain_grad()
        after_sign = torch.sign(before_sign)
        after_sign.register_hook(after_sign_hook)
        after_sign.retain_grad()
        out = self.relu3(after_sign)

        return out
