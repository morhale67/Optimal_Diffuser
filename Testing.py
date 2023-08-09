import torch
from torch import nn
from Model import breg_rec
import wandb
from LogFunctions import print_and_log_message
import math
from OutputHandler import save_outputs


def test_net(epoch, model, loader, params, device, log_path, folder_path, save_img=False):
    model.eval()
    model.to(device)
    cumu_loss = 0
    for batch_index, sim_bucket_tensor in enumerate(loader):
        sim_object, _ = sim_bucket_tensor
        sim_object.to(device)
        noise = torch.randn(params['batch_size'], params['z_dim'], requires_grad=True).to(device)
        sim_diffuser = model(noise)

        sim_diffuser_reshaped = sim_diffuser.reshape(params['batch_size'], math.floor(params['img_dim'] / params['cr']),
                                             params['img_dim'])

        sim_object = sim_object.view(-1, 1, params['img_dim']).to(device)
        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(sim_diffuser_reshaped, sim_object)
        sim_bucket = torch.transpose(sim_bucket, 1, 2)
        reconstruct_imgs_batch = breg_rec(sim_diffuser_reshaped, sim_bucket, params['batch_size'])

        reconstruct_imgs_batch = reconstruct_imgs_batch.to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(reconstruct_imgs_batch, sim_object)
        cumu_loss += loss.item()

    avg_loss = cumu_loss / len(loader)
    try:
        wandb.log({'sim_diffuser': [wandb.Image(i) for i in sim_diffuser_reshaped]})
        wandb.log({'image reconstructions': [wandb.Image(i.reshape(28, 28)) for i in reconstruct_imgs_batch]})
        wandb.log({'sim_object images': [wandb.Image(i.reshape(28, 28)) for i in sim_object]})
        wandb.log({'val_loss': loss})
        print(f"epoch [{epoch} / {params['epochs']}] \ "
                    f"genValLoss: {loss:.4f}")
    except:
        print_and_log_message('Test Loss: {:.6f}\n'.format(avg_loss), log_path)
        if save_img:
            save_outputs(reconstruct_imgs_batch, sim_object, int(math.sqrt(params['img_dim'])), folder_path, 'test_images')

    return avg_loss


