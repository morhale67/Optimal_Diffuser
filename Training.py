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
from LogFunctions import print_and_log_message
from Testing import test_net
from OutputHandler import save_loss_figure
from OutputHandler import save_outputs


def train_local(params, log_path, folder_path):
    data_root = 'home/dsi/chalamo/PycharmProjects/Optimal_Diffuser/data/MNIST'
    train_loader, test_loader = build_dataset(params['batch_size'], params['num_workers'], data_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = build_network(params['z_dim'], params['img_dim'], params['n_masks'], device)
    optimizer = build_optimizer(network, params['optimizer'], params['lr'])
    train_loss, test_loss = [], []
    for epoch in range(params['epochs']):
        start_epoch = time.time()
        train_loss_epoch = train_epoch(network, train_loader, optimizer, params['batch_size'], params['z_dim'], params['img_dim'],
                               params['n_masks'], device, log_path, folder_path)
        print_training_messages(epoch, train_loss_epoch, 0, start_epoch, log_path)
        test_loss_epoch = test_net(epoch, network, test_loader, params, device, log_path, folder_path)
        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)

    save_img_train_test(train_loader, test_loader, network, params, optimizer, device, folder_path, log_path)
    save_loss_figure(train_loss, test_loss, folder_path)
    print_and_log_message('Run Finished Successfully', log_path)


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        data_root = 'home/dsi/chalamo/PycharmProjects/Optimal_Diffuser/data/MNIST'
        n_masks = math.floor(config.img_dim / config.cr)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_loader, test_loader = build_dataset(config.batch_size, config.num_workers, data_root)
        network = build_network(config.z_dim, config.img_dim, n_masks, device)
        optimizer = build_optimizer(network, config.optimizer, config.lr)
        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, train_loader, optimizer, config.batch_size, config.z_dim, config.img_dim,
                                   n_masks, device, '', '')
            wandb.log({"loss": avg_loss, "epoch": epoch})


def build_dataset(batch_size, num_workers, data_root):
    transform = tr.Compose([
        tr.ToTensor(),
        # tr.Normalize((0.5), (0.5,)),
        # tr.Resize((64,64))
    ])

    train_dataset = dset.MNIST(data_root, train=True, download=True, transform=transform)

    indices = torch.arange(20000)
    mnist_20k = data.Subset(train_dataset, indices)

    train_dataset_size = int(len(mnist_20k) * 0.8)
    test_dataset_size = int(len(mnist_20k) * 0.2)
    # val_dataset_size = int(len(mnist_20k) - train_dataset_size)
    train_set, test_set = data.random_split(mnist_20k, [train_dataset_size, test_dataset_size])

    train_loader = DataLoader(train_set, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader


def build_network(z_dim, img_dim, n_masks, device):
    network = Gen(z_dim, img_dim, n_masks)
    return network.to(device)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer, batch_size, z_dim, img_dim, n_masks, device, log_path, folder_path, save_img=False):
    cumu_loss = 0
    network.train()
    for batch_index, sim_bucket_tensor in enumerate(loader):
        # with torch.autograd.set_detect_anomaly(True):
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        sim_object = sim_object.to(device)

        noise = torch.randn(int(batch_size), int(z_dim), requires_grad=True).to(device)
        diffuser = network(noise)

        diffuser_reshaped = diffuser.reshape(batch_size, n_masks, img_dim)

        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(diffuser_reshaped, sim_object)

        sim_bucket = torch.transpose(sim_bucket, 1, 2)
        reconstruct_imgs_batch = breg_rec(diffuser_reshaped, sim_bucket, batch_size)
        reconstruct_imgs_batch = reconstruct_imgs_batch.to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(reconstruct_imgs_batch, sim_object)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cumu_loss += loss.item()

        if save_img:
            save_outputs(reconstruct_imgs_batch, sim_object, int(math.sqrt(img_dim)), folder_path,
                         'train_images')
            break

        try:
            wandb.log({"batch loss": loss.item()})
        except:
            print_and_log_message(f"batch loss {batch_index}/{len(loader.batch_sampler)}: {loss.item()}", log_path)

    return cumu_loss / len(loader)


def save_img_train_test(train_loader, test_loader, network, params, optimizer, device, folder_path, log_path):
    _ = train_epoch(network, train_loader, optimizer, params['batch_size'], params['z_dim'],
                    params['img_dim'], params['cr'], device, log_path, folder_path, save_img=True)
    _ = test_net('end', network, test_loader, params, device, log_path, folder_path, save_img=True)

