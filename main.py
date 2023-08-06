import torch
from torch import nn
import wandb
import math
from torchviz import make_dot
from torch.autograd import Variable
import os
import torchvision.datasets as dset
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from Traning import train

wandb.login(key='8aec627a04644fcae0f7f72d71bb7c0baa593ac6')

sweep_id = 'ks5k7jhc'
wandb.agent(sweep_id, train, project='Optimal Diffuser', count=20)
