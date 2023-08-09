import math


def get_run_parameters():
    # p = {'batch_size': 32, 'pic_width': 32, 'prc_patterns': 10, 'n_gray_levels': 16}  # batch size is 5 in paper
    # p['m_patterns'] = (p['pic_width'] ** 2) * p['pic_width'] // 100
    #
    # p['initial_lr'] = 10 ** -2  # paper
    # p['div_factor_lr'] = 1  # paper
    # p['num_dif_lr'] = 1  # paper
    # p['n_epochs'] = 2 #60 * 10**3  # paper? maybe 300*10^3
    #
    # p['lr_vector'] = [10 ** -3, 10 ** -4, 10 - 6]
    # p['epochs_vector'] = [50, 10, 140]
    #
    # p['num_train_samples'], p['num_test_samples'] = 320, 32  # 'all', 'all'
    # p['data_sets'] = ['mnist'] #, 'coco', 'div2k']
    # p['model'] = 'Sparse_Encoder'  # 'ResHolo' or 'Sparse_ResHolo' or 'Sparse_Encoder'
    #
    p = {'epochs': 4, 'n_fc': 1, 'batch_size': 32, 'num_workers': 4, 'z_dim': 100, 'pic_width': 28, 'lr': 0.001, 'optimizer': 'adam', 'cr': 5}
    p['img_dim'] = p['pic_width']*p['pic_width']
    p['n_masks'] = math.floor(p['img_dim'] / p['cr']) * p['img_dim']


    return p
