import numpy as np
import traceback
import torch.optim as optim
from DataFunctions import build_dataset
import torch
from Training import train_epoch
from Testing import test_net
from LogFunctions import print_and_log_message
from LogFunctions import print_training_messages
from OutputHandler import save_loss_figure
import wandb
import math
import time
import Model


def get_lr(epoch, lr_vec, cum_epochs):
    for i, threshold in enumerate(cum_epochs):
        if epoch < threshold:
            return lr_vec[i]


def train_local(params, log_path, folder_path):
    data_root_medical = params['data_medical']
    train_loader, test_loader = build_dataset(params['batch_size'], params['num_workers'], params['pic_width'],
                                              params['n_samples'], data_root_medical, params['data_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = build_network(params['z_dim'], params['img_dim'], params['n_masks'], device)
    optimizer = build_optimizer(network, params['optimizer'], params['lr'])
    train_loss, test_loss = [], []
    lr = params['lr']
    for epoch in range(params['epochs']):
        if params['learn_vec_lr']:
            lr = get_lr(epoch, params['lr_vec'], params['cum_epochs'])
            optimizer = build_optimizer(network, params['optimizer'], lr)
        start_epoch = time.time()
        train_loss_epoch = train_epoch(epoch, network, train_loader, optimizer, params['batch_size'], params['z_dim'],
                                       params['img_dim'], params['n_masks'], device, log_path, folder_path,
                                       save_img=True)
        print_training_messages(epoch, train_loss_epoch, lr, start_epoch, log_path)
        test_loss_epoch = test_net(epoch, network, test_loader, device, log_path, folder_path, params['batch_size'],
                                   params['z_dim'], params['img_dim'], params['cr'], params['epochs'], save_img=True)
        train_loss.append(train_loss_epoch)
        test_loss.append(test_loss_epoch)

    save_img_train_test(epoch + 1, train_loader, test_loader, network, params, optimizer, device, folder_path, log_path)
    save_loss_figure(train_loss, test_loss, folder_path)
    print_and_log_message('Run Finished Successfully', log_path)


def train(config=None):
    with wandb.init(config=config):
        try:
            config = wandb.config
            data_root = 'data/medical/chunked_256'
            data_name = 'medical'
            n_samples = 20000
            img_dim = config.pic_width ** 2
            n_masks = math.floor(img_dim / config.cr)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            num_workers = 2
            train_loader, test_loader = build_dataset(config.batch_size, num_workers, config.pic_width, n_samples, data_root,
                                                      data_name)
            network = build_network(config.z_dim, img_dim, n_masks, device)
            optimizer = build_optimizer(network, config.optimizer, config.lr)
            for epoch in range(config.epochs):
                train_loss = train_epoch(epoch, network, train_loader, optimizer, config.batch_size, config.z_dim,
                                         img_dim,
                                         n_masks, device, '', '', save_img=True)

                if epoch % 5 == 0:
                    _ = test_net(epoch, network, test_loader, device, '', '', config.batch_size, config.z_dim, img_dim,
                                 config.cr, config.epochs, save_img=True)

            _ = train_epoch(epoch + 1, network, train_loader, optimizer, config.batch_size, config.z_dim, img_dim,
                            n_masks, device, '', '', save_img=True)
            _ = test_net(epoch + 1, network, test_loader, device, '', '', config.batch_size, config.z_dim, img_dim,
                         config.cr, config.epochs, save_img=True)
        except Exception as e:
            print("An error occurred:", str(e))
            traceback.print_exc()  # Print detailed traceback information


def save_img_train_test(epoch, train_loader, test_loader, network, params, optimizer, device, folder_path, log_path):
    _ = train_epoch(epoch, network, train_loader, optimizer, params['batch_size'], params['z_dim'],
                                       params['img_dim'], params['n_masks'], device, log_path, folder_path,
                                       ac_stride=params['ac_stride'], save_img=True)
    _ = test_loss_epoch = test_net(epoch, network, test_loader, device, log_path, folder_path, params['batch_size'],
                                   params['z_dim'], params['img_dim'], params['cr'], params['epochs'], save_img=True)


def build_network(z_dim, img_dim, n_masks, device, ac_stride=5):
    # network = Model.Gen_big_diff(z_dim, img_dim, n_masks, ac_stride)
    network = Model.Gen(z_dim, img_dim, n_masks)
    # # Use DataParallel to wrap your model for multi-GPU training
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     network = nn.DataParallel(network)
    torch.cuda.empty_cache()  # Before starting a new forward/backward pass
    print('Build Model Successfully')
    return network.to(device)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    return optimizer
