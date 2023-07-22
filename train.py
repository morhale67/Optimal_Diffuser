import math
import torch
from utils import breg_rec
from torch import nn
import wandb


def train_net(gen,epoch,train_loader,args,opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen.to(device)

    for batch_index, sim_bucket_tensor in enumerate(train_loader):
        # with torch.autograd.set_detect_anomaly(True):
        print("epoch number", epoch, "batch_idx ", batch_index)
        gen.train()
        sim_object, _ = sim_bucket_tensor
        sim_object = sim_object.view(-1, 1, args['img_dim']).to(device)
        sim_object = sim_object.to(device)

        noise = torch.randn(args['batch_size'], args['z_dim'], requires_grad=True).to(device)
        diffuser = gen(noise)

        diffuser_reshaped = diffuser.reshape(args['batch_size'], math.floor(args['img_dim'] / args['cr']),
                                             args['img_dim'])

        sim_object = sim_object.transpose(1, 2)
        sim_bucket = torch.matmul(diffuser_reshaped, sim_object)

        sim_bucket = torch.transpose(sim_bucket, 1, 2)
        rec = breg_rec(diffuser_reshaped, sim_bucket, args['batch_size'])
        rec = rec.to(device)
        sim_object = torch.squeeze(sim_object)
        criterion = nn.MSELoss()
        loss = criterion(rec, sim_object)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch_index == 0:
            wandb.log({'train_loss': loss})

            print(
                f"epoch [{epoch} / {args['epochs']}] \ "
                f"genTrainLoss: {loss:.4f}"
            )
