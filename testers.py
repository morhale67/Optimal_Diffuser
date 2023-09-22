import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch
from Lasso import sparse_encode
import time


def check_diff(diffuser, sim_object):
    orj_image, pic_width = plot_orj_image(sim_object)
    masks_to_plot = plot_masks(diffuser)
    plot_img_after_masks(masks_to_plot, orj_image, pic_width)


def plot_masks(diffuser):
    """ diffuser is tensor with size (batch_size, n_masks, img_dim) """
    batch_size, n_masks, img_dim = diffuser.shape
    num_plots = min(n_masks, 6)
    masks_to_plot = diffuser[1, :num_plots, :]
    pic_width = int(math.sqrt(img_dim))
    masks_to_plot = masks_to_plot.view(num_plots, pic_width, pic_width)
    plot_subplot(masks_to_plot, title="Masks")
    return masks_to_plot.cpu().detach().numpy()


def plot_orj_image(sim_object):
    """ sim_object.view(batch_size, 1, img_dim) """
    _, _, img_dim = sim_object.shape
    pic_width = int(math.sqrt(img_dim))
    orj_image = sim_object[0, 0, :]
    orj_image = orj_image.view(pic_width, pic_width).cpu().detach().numpy()

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(orj_image, cmap='gray')
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    save_path = os.path.join('temp', 'Original_Image')
    fig.savefig(save_path)
    plt.show()
    return orj_image, pic_width


def plot_rec_image(rec_image, maxiter, niter_inner, alpha):
    """ rec_image.view(1, img_dim) """
    _, img_dim = rec_image.shape
    pic_width = int(math.sqrt(img_dim))
    rec_image = rec_image.view(pic_width, pic_width).cpu().detach().numpy()

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(rec_image, cmap='gray')
    plt.title('Reconstructed Image', fontsize=12)
    plt.axis('off')
    save_path = os.path.join('temp', f'Reconstructed_Image_{maxiter}_{niter_inner}_{alpha}.png')
    fig.savefig(save_path)
    plt.show()
    return plt.gcf()


def plot_img_after_masks(masks_to_plot, orj_image, pic_width):
    orj_image = orj_image.reshape(1, 1, pic_width, pic_width)
    image_after_masks = masks_to_plot * orj_image
    image_after_masks = torch.tensor(image_after_masks).view(-1, pic_width, pic_width)
    plot_subplot(image_after_masks, title="Image After Masks")


def plot_subplot(img_tensor, title="Image Subplots", num_rows=2, num_cols=3):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2, 2))
    num_plots = num_rows * num_cols
    for i in range(num_plots):
        row = i // num_cols
        col = i % num_cols
        img_data = img_tensor[i, :, :].cpu().detach().numpy()
        ax = axes[row, col]
        ax.imshow(img_data, cmap='gray')
        ax.set_title(f"Mask {i + 1}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)  # Adding a title for the entire set of subplots
    plt.savefig(os.path.join('temp', title))
    plt.show()


def experiment_berg_params(bucket, diffuser):
    for maxiter in range(20, 50, 5):
        for niter_inner in range(3, 10, 2):
            for alpha in np.arange(0.1, 0.3, 0.5):
                try:
                    rec = sparse_encode(bucket, diffuser, maxiter=maxiter,
                                        niter_inner=niter_inner, alpha=alpha, algorithm='split-bregman')
                    figure = plot_rec_image(rec, maxiter, niter_inner, alpha)
                except torch._C._LinAlgError as e:
                    print(f'params:{maxiter}, {niter_inner}, {alpha}')
                    if 'figure' in locals():
                        plt.close(figure)


def compare_buckets(bucket, diffuser, orj_img):
    my_bucket = torch.matmul(diffuser, orj_img)
    if torch.all(my_bucket == bucket):
        return True
    return False


def calculate_autocorrelation(image):
    _, v_range = image.shape
    u_range = 1
    autocorr = np.zeros((u_range, v_range))

    for u in range(u_range):
        for v in range(v_range):
            shifted_image = np.roll(np.roll(image, u, axis=0), v, axis=1)
            autocorr[u, v] = np.sum(image * shifted_image)
    return autocorr.resize(v_range)


def check_diff_ac(diffuser, folder_path='temp'):
    autocorr = calculate_autocorrelation(diffuser)
    random_diffuser = torch.randn_like(diffuser)
    rand_autocorr = calculate_autocorrelation(random_diffuser)
    pic_size, _ = diffuser.shape
    save_autocorr(autocorr, rand_autocorr, pic_size, folder_path)


def save_autocorr(autocorr, rand_autocorr, pic_size, folder_path):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 7))

    plt.plot(autocorr, label='ac output', color='red')
    plt.plot(rand_autocorr, label='ac random', color='blue')


    # Add labels and title
    plt.xlabel('Stride', fontsize=22, fontname='Arial')
    plt.ylabel(f'Autocorrelation_{pic_size}', fontsize=22, fontname='Arial')
    plt.legend()

    # Save the figure to the specified filename
    full_file_path = os.path.join(folder_path, f'ac_{pic_size}')
    plt.savefig(full_file_path)
    plt.show()

