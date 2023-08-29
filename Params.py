import math


def get_run_parameters():
    p = {'cr': 1,
         'batch_size': 2,
         'lr': 0.001,
         'epochs': 4,
         'num_workers': 4,
         'z_dim': 100,
         'pic_width': 32,
         'optimizer': 'adam',
         'ac_stride': 7}
    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])

    return p
