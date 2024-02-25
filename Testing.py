import torch
from torch import nn
from Model import breg_rec
import wandb
from LogFunctions import print_and_log_message
import math
from OutputHandler import save_orig_img, save_outputs, calc_cumu_ssim_batch, save_randomize_outputs, calc_cumu_psnr_batch
import numpy as np


def test_net(epoch, model, loader, device, log_path, folder_path, batch_size, z_dim, img_dim, cr,
             TV_beta, wb_flag, save_img=False):
    model.eval()
    model.to(device)
    cumu_loss, cumu_psnr, cumu_ssim = 0, 0, 0
    n_batchs = len(loader.batch_sampler)
    n_samples = n_batchs * batch_size
    pic_width = int(math.sqrt(img_dim))

    for batch_index, sim_bucket_tensor in enumerate(loader):
        sim_object, _ = sim_bucket_tensor
        sim_object.to(device)
        noise = torch.randn(batch_size, z_dim, requires_grad=True).to(device)
        output_net = model(noise)
        sim_diffuser = output_net['diffuser_x']
        sim_diffuser_reshaped = sim_diffuser.reshape(batch_size, math.floor(img_dim / cr), img_dim)

        sim_object = sim_object.view(-1, 1, img_dim).to(device)
        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(sim_diffuser_reshaped, sim_object)
        sim_bucket = torch.transpose(sim_bucket, 1, 2)
        reconstruct_imgs_batch = breg_rec(sim_diffuser_reshaped, sim_bucket, batch_size, output_net)

        reconstruct_imgs_batch = reconstruct_imgs_batch.to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(reconstruct_imgs_batch, sim_object)
        cumu_loss += loss.item()
        torch.cuda.empty_cache()  # Before starting a new forward/backward pass
        batch_psnr = calc_cumu_psnr_batch(reconstruct_imgs_batch, sim_object, pic_width)
        batch_ssim = calc_cumu_ssim_batch(reconstruct_imgs_batch, sim_object, pic_width)
        cumu_psnr += batch_psnr
        cumu_ssim += batch_ssim
        print_and_log_message(f"Epoch number {epoch}, batch number {batch_index}/{n_batchs}:"
                              f"       batch loss {loss.item()}", log_path)
        if wb_flag:
            wandb.log({"test_loss_batch": loss.item(), "test_psnr_batch": batch_psnr / batch_size,
                       "test_ssim_batch": batch_ssim / batch_size, "test_batch_index": batch_index})
        if save_img:
            save_randomize_outputs(epoch, batch_index, reconstruct_imgs_batch, sim_object, int(math.sqrt(img_dim)),
                                   folder_path, 'test_images', wb_flag)


    test_loss, test_psnr, test_ssim = cumu_loss / n_samples, cumu_psnr / n_samples, cumu_ssim / n_samples

    if wb_flag:
        wandb.log({"epoch": epoch, 'test_loss': test_loss})

    return test_loss, test_psnr, test_ssim
