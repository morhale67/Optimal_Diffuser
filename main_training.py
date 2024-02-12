import pickle
import numpy as np
import traceback
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from DataFunctions import build_dataset
import torch
from Lasso import sparse_encode
from Params import get_run_parameters
from Training import train_epoch
from Testing import test_net
from LogFunctions import print_and_log_message
from LogFunctions import print_training_messages
from OutputHandler import save_numerical_figure, save_orig_img, save_all_run_numerical_outputs, PSNR, SSIM, \
    sb_reconstraction_for_all_images, image_results_subplot
import wandb
import math
import time
import Model


def train_local(params, log_path, folder_path):
    train_loader, test_loader = build_dataset(params['batch_size'], params['num_workers'], params['pic_width'],
                                              params['n_samples'], params['data_medical'], params['data_name'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = build_network(params['z_dim'], params['img_dim'], params['n_masks'], device, params['model_name'])
    optimizer = build_optimizer(network, params['optimizer'], params['lr'], params['weight_decay'])
    numerical_outputs = {'train_loss': [], 'test_loss': [], 'train_psnr': [], 'test_psnr': [], 'train_ssim': [], 'test_ssim': []}
    lr = params['lr']
    save_orig_img(train_loader, folder_path, name_sub_folder='train_images')
    save_orig_img(test_loader, folder_path, name_sub_folder='test_images')

    for epoch in range(params['epochs']):
        if params['learn_vec_lr']:
            lr = get_lr(epoch, params['lr_vec'], params['cum_epochs'])
            optimizer = build_optimizer(network, params['optimizer'], lr, params['weight_decay'])
        start_epoch = time.time()
        train_loss_epoch, train_psnr_epoch, train_ssim_epoch = train_epoch(epoch, network, train_loader, optimizer,
                                                         params['batch_size'], params['z_dim'], params['img_dim'],
                                                         params['n_masks'], device, log_path, folder_path,
                                                         params['epsRL1'], save_img=True)
        print_training_messages(epoch, train_loss_epoch, lr, start_epoch, log_path)
        test_loss_epoch, test_psnr_epoch, test_ssim_epoch = test_net(epoch, network, test_loader, device, log_path, folder_path,
                                                    params['batch_size'],
                                                    params['z_dim'], params['img_dim'], params['cr'], params['epochs'],
                                                    save_img=True)
        numerical_outputs = update_numerical_outputs(numerical_outputs, train_loss_epoch, test_loss_epoch, train_psnr_epoch,
                                 test_psnr_epoch, train_ssim_epoch, test_ssim_epoch)

    # save_img_train_test(epoch + 1, train_loader, test_loader, network, params, optimizer, device, folder_path, log_path)
    numerical_outputs['rand_diff_loss'], numerical_outputs['rand_diff_psnr'], numerical_outputs['rand_diff_ssim'] = \
        split_bregman_on_random_for_run(folder_path, params)
    save_all_run_numerical_outputs(numerical_outputs, folder_path)
    sb_reconstraction_for_all_images(folder_path, cr=params['cr'])
    print_and_log_message('Run Finished Successfully', log_path)
    #image_results_subplot(folder_path, data_set='train_images', epochs_to_show=[0, 1, 2, 5, 10, params['epochs']])
    #image_results_subplot(folder_path, data_set='test_images', epochs_to_show=[0, 1, 2, 5, 10, params['epochs']])


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
            train_loader, test_loader = build_dataset(config.batch_size, num_workers, config.pic_width, n_samples,
                                                      data_root,
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
                    save_img=True)
    _ = test_loss_epoch = test_net(epoch, network, test_loader, device, log_path, folder_path, params['batch_size'],
                                   params['z_dim'], params['img_dim'], params['cr'], params['epochs'], save_img=True)


def build_network(z_dim, img_dim, n_masks, device, model_name):
    # network = Model.Gen_big_diff(z_dim, img_dim, n_masks, ac_stride)
    model_class = getattr(Model, model_name)
    network = model_class(z_dim, img_dim, n_masks)
    # # Use DataParallel to wrap your model for multi-GPU training
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     network = nn.DataParallel(network)
    torch.cuda.empty_cache()  # Before starting a new forward/backward pass
    print('Build Model Successfully')
    return network.to(device)


def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def update_numerical_outputs(numerical_outputs, train_loss_epoch, test_loss_epoch, train_psnr_epoch, test_psnr_epoch, train_ssim_epoch, test_ssim_epoch):
    numerical_outputs['train_loss'].append(train_loss_epoch)
    numerical_outputs['test_loss'].append(test_loss_epoch)
    numerical_outputs['train_psnr'].append(train_psnr_epoch)
    numerical_outputs['test_psnr'].append(test_psnr_epoch)
    numerical_outputs['train_ssim'].append(train_ssim_epoch)
    numerical_outputs['test_ssim'].append(test_ssim_epoch)
    return numerical_outputs


def get_lr(epoch, lr_vec, cum_epochs):
    for i, threshold in enumerate(cum_epochs):
        if epoch < threshold:
            return lr_vec[i]


def split_bregman_on_random_for_run(folder_path, params):
    images_tensor = get_test_images(folder_path)
    cum_loss, cum_psnr, cum_ssim = 0, 0, 0
    for orig_image in images_tensor:
        np_orig_image = np.array(orig_image.view(-1, 1))
        sim_diffuser = np.random.normal(0.5, 0.5, [params['n_masks'], params['img_dim']])
        sim_bucket = np.matmul(sim_diffuser, np_orig_image).transpose((1, 0))
        rec_image = sparse_encode(torch.from_numpy(sim_bucket), torch.from_numpy(sim_diffuser), maxiter=1, niter_inner=1, alpha=1, algorithm='split-bregman')

        pic_width = params['pic_width']
        loss = mean_squared_error(rec_image.flatten(), orig_image.flatten())
        psnr = PSNR(rec_image.flatten(), orig_image.flatten(), 1, params['img_dim'])

        rec_image = np.array(rec_image.view(pic_width, pic_width))
        ssim = SSIM(rec_image, np_orig_image.reshape(pic_width, pic_width))
        cum_loss, cum_psnr, cum_ssim = cum_loss + loss, cum_psnr + psnr, cum_ssim + ssim
    avg_loss = cum_loss / len(images_tensor)
    avg_psnr = cum_psnr / len(images_tensor)
    avg_ssim = cum_ssim / len(images_tensor)
    return avg_loss, avg_psnr, avg_ssim


def get_test_images(folder_path):
    orig_img_path = folder_path + '/test_images/orig_imgs_tensors.pt'
    all_images_tensor = torch.load(orig_img_path)
    return all_images_tensor


if __name__ == '__main__':
    folder_path = 'Results/simple_cifar_GEN_bs_2_cr_50_nsamples10_picw_16_lr_0.1'
    params = get_run_parameters()
    avg_loss, avg_psnr, avg_ssim = split_bregman_on_random_for_run(folder_path, params)

