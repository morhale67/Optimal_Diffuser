import torchvision.datasets as dset
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from train import train_net
from models import Gen
import torch.utils.data as data
from eval_net import eval_net
import wandb


def main():
    args = {'epochs': 16,
            'img_dim': 784,
            'z_dim': 100,
            'lr': 3e-4,
            'optimizer': 'adam',
            'data_root': 'data/mnist' ,
            'num_workers': 2,
            'batch_size': 32,
            'ngpu': 1,
            'cr': 4
            }

    wandb.init(project='diffuser_optimization_nn project', config = args)

    transform = tr.Compose([
                            tr.ToTensor(),
                            #tr.Normalize((0.5), (0.5,)),
                            # tr.Resize((64,64))
                      ])

    train_dataset = dset.MNIST(args['data_root'], train=True, download=True, transform= transform)

    indices = torch.arange(20000)   
    mnist_20k = data.Subset(train_dataset, indices)

    train_dataset_size = int(len(mnist_20k) * 0.8)
    val_dataset_size = int(len(mnist_20k) - train_dataset_size)
    train_set, val_set = data.random_split(mnist_20k, [train_dataset_size, val_dataset_size])

    train_loader = DataLoader(train_set, args['batch_size'], shuffle=True, pin_memory=True, num_workers=args['num_workers'])
    val_loader = DataLoader(val_set, args['batch_size'], shuffle=True, pin_memory=True, num_workers=args['num_workers'])

    gen = Gen(args['z_dim'], args['img_dim'], args['cr'])
    opt = optim.Adam(gen.parameters(), lr=args['lr'])

    wandb.watch(gen, log='all')
    for epoch in range(args['epochs']):
        train_net(gen,epoch,train_loader, args, opt)

        with torch.no_grad():
            eval_net(gen, epoch, val_loader, args)


if __name__ == "__main__":
    main()
