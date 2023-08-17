import math


def get_run_parameters():
    p = {'cr': 2,
         'batch_size': 2,
         'lr': 0.001,
         'epochs': 4,
         'n_fc': 3,
         'num_workers': 1,
         'z_dim': 100,
         'pic_width': 64,
         'optimizer': 'adam'}
    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])

    return p
