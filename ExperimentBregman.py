import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from Lasso import sparse_encode
import matplotlib.pyplot as plt
import cv2
import Model
from Lasso import sparse_encode


def main_bregman(cr, pic_width=64):
    xray_folder = 'data/medical/chunked_256/mock_class'
    diffuser = get_net_diffuser(cr, img_dim=pic_width**2)
    image_names = get_image_names()
    for image in image_names:
        bucket = create_bucket(diffuser, img_name=image, xray_folder=xray_folder, img_new_width=pic_width)
        experiment_berg_params(bucket, diffuser)


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
    sim_bucket = torch.from_numpy(sim_bucket)
    return sim_bucket


def experiment_berg_params(bucket, diffuser, folder_path='temp/Exp_bregman'):
    for maxiter in range(1, 20, 1):
        title = f'Experiment Bregman - Outer iteration={maxiter}'
        i, j = 0, 0
        img_tensor = torch.empty(0, 0, 0, 0)
        sub_title = []
        for niter_inner in range(1, 20, 1):
            for alpha in np.arange(1, 10, 1):
                try:
                    rec_image = sparse_encode(bucket, diffuser, maxiter=maxiter,
                                        niter_inner=niter_inner, alpha=alpha, algorithm='split-bregman')
                    img_tensor[i, j, :, :] = rec_image
                    sub_title[i][j] = f"niter_inner={niter_inner}, alpha={alpha}"
                except torch._C._LinAlgError as e:
                    print(f'params:{maxiter}, {niter_inner}, {alpha}')
                    img_tensor[i, :, :] = torch.zeros_like(img_tensor[i-1, :, :])
                j += 1
            i += 1
        exp_subplot(img_tensor, sub_title, title=title, folder_path=folder_path)


def exp_subplot(img_tensor, sub_title, title="Image Subplots", folder_path='temp'):
    num_rows, num_cols, _, _ = img_tensor.shape
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2, 2))
    for row in range(num_rows):
        for col in range(num_cols):
            img_data = img_tensor[row, col, :, :].cpu().detach().numpy()
            ax = axes[row, col]
            ax.imshow(img_data, cmap='gray')
            ax.set_title(sub_title[row][col], fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)  # Adding a title for the entire set of subplots
    plt.savefig(os.path.join(folder_path, title))
    plt.show()


def get_image_names():
    image_names = ['chunk_middle_part_0417_0697542589_01_WRI-R2_F008.png']
                    # 'chunk_middle_part_0503_1018511008_01_WRI-L1_M012.png',
                    # 'chunk_middle_part_0417_0727170640_02_WRI-R1_F009.png',
                    # 'chunk_middle_part_0503_1018511068_01_WRI-L2_M012.png',
                    # 'chunk_middle_part_0417_0727170681_02_WRI-R2_F009.png',
                    # 'chunk_middle_part_0503_1020470848_02_WRI-L1_M012.png']
    return image_names




if __name__ == '__main__':
    main_bregman(cr=1)

