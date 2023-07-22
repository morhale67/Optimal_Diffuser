import torch
from torch import nn
import math
from utils import breg_rec
import wandb


def eval_net(model, epoch, loader, args):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    for batch_index, sim_bucket_tensor in enumerate(loader):
        sim_object, _ = sim_bucket_tensor
        sim_object.to(device)
        noise = torch.randn(args['batch_size'], args['z_dim'], requires_grad=True).to(device)
        sim_diffuser = model(noise)

        sim_diffuser_reshaped = sim_diffuser.reshape(args['batch_size'], math.floor(args['img_dim'] / args['cr']),
                                             args['img_dim'])

        sim_object = sim_object.view(-1, 1, args['img_dim']).to(device)
        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(sim_diffuser_reshaped, sim_object)
        sim_bucket = torch.transpose(sim_bucket, 1, 2)
        rec = breg_rec(sim_diffuser_reshaped, sim_bucket, args['batch_size'])

        rec = rec.to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(rec, sim_object)

        if batch_index == 0:
            wandb.log({'sim_diffuser': [wandb.Image(i) for i in sim_diffuser_reshaped]})
            wandb.log({'image reconstructions': [wandb.Image(i.reshape(28, 28)) for i in rec]})
            wandb.log({'sim_object images': [wandb.Image(i.reshape(28, 28)) for i in sim_object]})
            wandb.log({'val_loss': loss})
            print(
                f"epoch [{epoch} / {args['epochs']}] \ "
                f"genValLoss: {loss:.4f}"
            )

        return loss


