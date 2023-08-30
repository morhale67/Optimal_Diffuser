import numpy as np
import traceback
from Training import build_dataset
from Training import build_network
from Training import build_optimizer
import torch
from Training import train_epoch
from Testing import test_net
from LogFunctions import print_and_log_message
import wandb
import math


def train_local(params, log_path, folder_path, Medical=False):
    data_root = 'data/Medical/chunked_128'
    train_loader, test_loader = build_dataset(params['batch_size'], params['num_workers'], params['pic_width'],
                                              data_root, Medical)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = build_network(params['z_dim'], params['img_dim'], params['n_masks'], device, ac_stride=params['ac_stride'])
    optimizer = build_optimizer(network, params['optimizer'], params['lr'])
    train_loss, test_loss = [], []
    for epoch in range(params['epochs']):
        start_epoch = time.time()
        train_loss_epoch = train_epoch(epoch, network, train_loader, optimizer, params['batch_size'], params['z_dim'],
                                       params['img_dim'], params['n_masks'], device, log_path, folder_path,
                                       ac_stride=params['ac_stride'])
        print_training_messages(epoch, train_loss_epoch, 0, start_epoch, log_path)
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
            data_root = 'data/Medical/chunked_128'
            img_dim = config.pic_width ** 2
            n_masks = math.floor(img_dim / config.cr)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            num_workers = 2
            train_loader, test_loader = build_dataset(config.batch_size, num_workers, config.pic_width, data_root,
                                                      Medical=True)
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
                    params['img_dim'], params['cr'], device, log_path, folder_path, save_img=True)
    _ = test_net(epoch, network, test_loader, device, log_path, folder_path, params['batch_size'], params['z_dim'],
                 params['img_dim'], params['cr'], params['epochs'], save_img=True)
