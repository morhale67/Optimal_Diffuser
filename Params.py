import math


def get_run_parameters():
    p = {'data_medical': 'data_DSI/GCP_data',
         'data_name': 'cifar',
         'n_samples': 4,
         'cr': 5,
         'batch_size': 2,
         'lr': 0.001,
         'epochs': 100,
         'pic_width': 32,
         'optimizer': 'adam',
         'big_diffuser': False,
         'ac_stride': 7,
         'num_workers': 4,
         'z_dim': 100}
    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
    return p
