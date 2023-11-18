import numpy as np
import torch
from torch import nn
import wandb
import math
import torchvision.datasets as dset
import torchvision.transforms as tr
from Model import Gen
from Model import breg_rec
from LogFunctions import print_and_log_message
from OutputHandler import save_outputs
from testers import check_diff
from testers import compare_buckets
from Model import Gen_no_batch
import Model
from testers import check_diff_ac
import matplotlib.pyplot as plt
from OutputHandler import save_orig_img
from OutputHandler import save_randomize_outputs


def train_epoch(epoch, network, loader, optimizer, batch_size, z_dim, img_dim, n_masks, device, log_path, folder_path,
                ac_stride=5, save_img=False, big_diffuser=False):
    cumu_loss = 0
    network.train()
    n_batchs = len(loader.batch_sampler)
    save_orig_img(loader, folder_path, name_sub_folder='train_images')

    for batch_index, sim_bucket_tensor in enumerate(loader):
        # with torch.autograd.set_detect_anomaly(True):
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        first_image = sim_object[0, 0, :].view(-1, int(math.sqrt(img_dim)))
        sim_object = sim_object.to(device)
        if big_diffuser:
            pic_width = int(math.sqrt(img_dim))
            noise = torch.randn(int(batch_size), int(z_dim), requires_grad=True).to(device)
            diffuser = network(noise)
            #            print('Forward Net Successfully')
            diffuser = diffuser.reshape(batch_size, pic_width, -1)
            check_diff_ac(np.array(diffuser[0, :, :]))
            diffuser = make_masks_from_big_diff(diffuser, ac_stride)
        else:
            noise = torch.randn(int(batch_size), int(z_dim), requires_grad=True).to(device)
            diffuser = network(noise)
            #            print('Forward Net Successfully')
            #             print(f'bs {batch_size} n_masks {n_masks}  img_dim {img_dim}')
            diffuser = diffuser.reshape(batch_size, n_masks, img_dim)
            # diffuser = diffuser.unsqueeze(0).expand(batch_size, -1, -1)  # new tensor that contains repeated copies of the original tensor's data

        # check_diff(diffuser, sim_object, folder_path='temp/Gan/new')

        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(diffuser, sim_object)
        sim_bucket = torch.transpose(sim_bucket, 1, 2)

        # buckets_same = compare_buckets(sim_bucket[0], diffuser[0], sim_object[0])

        reconstruct_imgs_batch = breg_rec(diffuser, sim_bucket, batch_size).to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(reconstruct_imgs_batch, sim_object)
        #        print('Loss Has Been Calculated')
        optimizer.zero_grad()
        loss.backward()
        #        print('Loss Backward')
        optimizer.step()
        #        print('Optimizer step')
        cumu_loss += loss.item()
        torch.cuda.empty_cache()  # Before starting a new forward/backward pass
        try:
            wandb.log({"batch loss": loss.item()})
            print(f"batch loss {batch_index}/{n_batchs}: {loss.item()}, epoch {epoch}")
        except:
            print_and_log_message(
                f"Epoch number {epoch}, batch number {batch_index}/{n_batchs}:       batch loss {loss.item()}",
                log_path)

        if save_img:
            save_randomize_outputs(epoch, batch_index, reconstruct_imgs_batch, sim_object, int(math.sqrt(img_dim)), folder_path,
                                   'train_images')

    train_loss = cumu_loss / len(loader)
    try:
        wandb.log({'train_loss': train_loss})
    except Exception as e:
        pass

    # and epoch % 5 == 0:
    #        try:
    #           num_images = reconstruct_imgs_batch.shape[0]  # most of the time = batch_size
    #          pic_width = int(math.sqrt(img_dim))
    #         image_reconstructions = [wandb.Image(i.reshape(pic_width, pic_width)) for i in reconstruct_imgs_batch]
    #        sim_object_images = [wandb.Image(i.reshape(pic_width, pic_width)) for i in sim_object]

    #       wandb.log({'sim_diffuser': [wandb.Image(i) for i in diffuser]})
    #      wandb.log({'train image reconstructions': image_reconstructions})
    #     wandb.log({'train original images': sim_object_images})
    # except:

    return train_loss


def make_masks_from_big_diff(diffuser, ac_stride):
    """ diffuser shape: [pic_width, diff_width]"""
    batch_size, pic_width, diff_width = diffuser.shape
    img_dim = pic_width ** 2

    mask_indices = torch.arange(0, diff_width - pic_width, ac_stride, dtype=torch.long).view(-1, 1)
    diffuser_expanded = diffuser[:, :, mask_indices + torch.arange(pic_width)]
    masks = diffuser_expanded.view(batch_size, -1, img_dim)
    return masks
