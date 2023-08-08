import torch
from torch import nn
import wandb
import math
import torchvision.datasets as dset
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import torch.utils.data as data
import torch.optim as optim
from Model import Gen
from Model import breg_rec
from LogFunctions import print_training_messages
import time


def train_local(params, log_path):
    data_root = 'home/dsi/chalamo/PycharmProjects/Optimal_Diffuser/data/MNIST'
    loader = build_dataset(params['batch_size'], params['num_workers'], data_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = build_network(params['z_dim'], params['img_dim'], params['cr'], device)
    optimizer = build_optimizer(network, params['optimizer'], params['lr'])

    for epoch in range(params['epochs']):
        start_epoch = time.time()
        avg_loss = train_epoch(network, loader, optimizer, params['batch_size'], params['z_dim'], params['img_dim'],
                               params['cr'], device)
        avg_psnr = 0
        print_training_messages(epoch, avg_loss, avg_psnr, start_epoch, log_path)



def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        data_root = 'home/dsi/chalamo/PycharmProjects/Optimal_Diffuser/data/MNIST'
        loader = build_dataset(config.batch_size, config.num_workers, data_root)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        network = build_network(config.z_dim, config.img_dim, config.cr, device)
        optimizer = build_optimizer(network, config.optimizer, config.lr)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer, config.batch_size, config.z_dim, config.img_dim, config.cr, device)
            wandb.log({"loss": avg_loss, "epoch": epoch})


def build_dataset(batch_size, num_workers, data_root):
    transform = tr.Compose([
                            tr.ToTensor(),
                            #tr.Normalize((0.5), (0.5,)),
                            # tr.Resize((64,64))
                      ])

    train_dataset = dset.MNIST(data_root, train=True, download=True, transform=transform)

    indices = torch.arange(20000)
    mnist_20k = data.Subset(train_dataset, indices)

    train_dataset_size = int(len(mnist_20k) * 0.8)
    val_dataset_size = int(len(mnist_20k) * 0.2)
    # val_dataset_size = int(len(mnist_20k) - train_dataset_size)
    train_set, val_set = data.random_split(mnist_20k, [train_dataset_size, val_dataset_size])

    train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size, shuffle=True, pin_memory=True, num_workers=2)

    return train_loader


def build_network(z_dim, img_dim, cr, device):
    network = Gen(z_dim, img_dim, cr)
    return network.to(device)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer, batch_size, z_dim, img_dim, cr, device):
    cumu_loss = 0
    for batch_index, sim_bucket_tensor in enumerate(loader):
        # with torch.autograd.set_detect_anomaly(True):
        network.train()
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        sim_object = sim_object.to(device)

        noise = torch.randn(int(batch_size), int(z_dim), requires_grad=True).to(device)
        diffuser = network(noise)

        diffuser_reshaped = diffuser.reshape(batch_size, math.floor(img_dim / cr),
                                             img_dim)

        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(diffuser_reshaped, sim_object)

        sim_bucket = torch.transpose(sim_bucket, 1, 2)
        rec = breg_rec(diffuser_reshaped, sim_bucket, batch_size)
        rec = rec.to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(rec, sim_object)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cumu_loss += loss.item()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)