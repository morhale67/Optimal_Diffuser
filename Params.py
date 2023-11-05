import math


def get_run_parameters():
    p = {'data_medical': 'data_DSI/GCP_data',
         'data_name': 'cifar',
         'n_samples': 4,
         'cr': 4,
         'batch_size': 2,
         'lr': 0.01,
         'epochs': 50,
         'pic_width': 64,
         'optimizer': 'adam',
         'big_diffuser': False,
         'ac_stride': 7,
         'num_workers': 4,
         'z_dim': 100}
    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
    return p
