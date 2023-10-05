import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from Lasso import sparse_encode
import matplotlib.pyplot as plt
import cv2
import Model
from Lasso import sparse_encode


def main_bregman(cr, pic_width=64, load_flag=True):
    xray_folder = 'data/medical/chunked_256/mock_class'
    if load_flag:
        diffuser = np.load("diffuser.npy")
    else:
        diffuser = get_net_diffuser(cr, img_dim=pic_width**2)
        np.save("diffuser.npy", diffuser)
    image_names = get_image_names()
    for image in image_names:
        bucket = create_bucket(diffuser, img_name=image, xray_folder=xray_folder, img_new_width=pic_width)
        experiment_berg_params(torch.from_numpy(bucket), torch.from_numpy(diffuser), pic_width)


def get_net_diffuser(cr, img_dim=64**2, batch_size=2, z_dim=100):
    noise = torch.randn(batch_size, z_dim, requires_grad=True)
    n_masks = img_dim // cr
    network = Model.Gen(z_dim, img_dim, n_masks)
    diffuser = network(noise)
    diffuser = diffuser.reshape(batch_size, n_masks, img_dim)
    diffuser = diffuser[0, :, :].detach().numpy()
    return diffuser


def create_bucket(sim_diffuser, img_name, xray_folder, img_new_width):
    knee_xray = cv2.imread(xray_folder + '/' + img_name, cv2.IMREAD_GRAYSCALE)
    knee_xray = np.array(knee_xray)
    knee_xray_resized = cv2.resize(knee_xray, (img_new_width, img_new_width))
    sim_object = knee_xray_resized.reshape(1, -1)
    sim_object = sim_object.transpose(1, 0)
    sim_bucket = np.matmul(sim_diffuser, sim_object)
    sim_bucket = sim_bucket.transpose((1, 0))
    return sim_bucket


def experiment_berg_params(bucket, diffuser, pic_width, folder_path='temp\\Exp_bregman'):
    for maxiter in range(1, 2, 1):
        title = f'Experiment Bregman - tuning alpha 3'
        img_list = []
        sub_title = []
        for niter_inner in range(1, 2, 1):
            img_list_alpha = []
            sub_title_alpha = []
            for alpha in np.arange(0.8, 2, 0.1):
                print(f'alpha={alpha: .2f}, niter_inner={niter_inner}, maxiter={maxiter}')
                try:
                    # sub_title_alpha.append(f"niter_inner={niter_inner}, alpha={alpha: .2f}")
                    sub_title_alpha.append(f"alpha={alpha: .2f}")
                    rec_image = sparse_encode(bucket, diffuser, maxiter=maxiter,
                                        niter_inner=niter_inner, alpha=alpha, algorithm='split-bregman')
                    img_list_alpha.append(rec_image.view(pic_width, pic_width))
                except torch._C._LinAlgError as e:
                    print(f'params:{maxiter}, {niter_inner}, {alpha}')
                    img_list_alpha.append(torch.zeros(pic_width, pic_width))
            sub_title.append(sub_title_alpha)
            img_list.append(img_list_alpha)
        torch.save(img_list, 'temp\\' + title + '.pt')
        exp_subplot(img_list, sub_title, title=title, folder_path=folder_path)


def exp_subplot(img_list, sub_title, title="Image Subplots", folder_path='temp', num_cols_subplot=4):
    num_rows = len(img_list)
    num_cols = len(img_list[0])
    if num_rows == 1:
        num_rows_subplot = int(np.ceil(num_cols / num_cols_subplot))
        fig, axes = plt.subplots(num_rows_subplot, num_cols_subplot, figsize=(16, 16))
    else:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))

    for row in range(num_rows):
        for col in range(num_cols):
            if num_rows == 1:
                i_image = col
                subplot_row = i_image // num_cols_subplot
                subplot_col = i_image % num_cols_subplot
            else:
                subplot_row, subplot_col = row, col
            image = img_list[row][col].cpu().detach().numpy()

            ax = axes[subplot_row, subplot_col]
            ax.imshow(image, cmap='gray')
            ax.set_title(sub_title[row][col], fontsize=8)
            ax.axis('off')
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)  # Adding a title for the entire set of subplots
    plt.savefig(os.path.join(folder_path, title))
    plt.show()


def get_image_names():
    image_names = ['chunk_middle_part_0016_0320600198_01_WRI-L2_F010.png']
                   # 'chunk_middle_part_0503_1018511008_01_WRI-L1_M012.png',
                    # chunk_middle_part_0033_1098977682_01_WRI-L2_F006.png',
                    # 'chunk_middle_part_0503_1018511008_01_WRI-L1_M012.png',
                    # 'chunk_middle_part_0417_0727170640_02_WRI-R1_F009.png',
                    # 'chunk_middle_part_0503_1018511068_01_WRI-L2_M012.png',
                    # 'chunk_middle_part_0417_0727170681_02_WRI-R2_F009.png',
                    # 'chunk_middle_part_0503_1020470848_02_WRI-L1_M012.png']
    return image_names




if __name__ == '__main__':
    main_bregman(cr=1)

