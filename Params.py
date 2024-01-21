import math


def get_run_parameters():
    p = {'data_medical': 'data_DSI/GCP_data',
         'data_name': 'simple_cifar',
         'model_name': 'Gen',
         'lr_vec': [0.1, 0.01, 0.001, 0.0005],
         'epochs_vec': [5, 5, 5, 5],
         'learn_vec_lr': True,
         'pic_width': 32,
         'n_samples': 100,
         'cr': 20,
         'batch_size': 2,
         'lr': 0.1,
         'epochs': 3,
         'optimizer': 'adam',
         # 'big_diffuser': False,
         # 'ac_stride': 7,
         'num_workers': 4,
         'z_dim': 100}
    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr'])
    if p['learn_vec_lr']:
        p['epochs'] = sum(p['epochs_vec'])
        p['cum_epochs'] = [sum(p['epochs_vec'][:i + 1]) for i in range(len(p['epochs_vec']))]

    return p
