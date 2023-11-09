import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch
from torchvision import datasets, transforms
from Lasso import sparse_encode
import matplotlib.pyplot as plt
import cv2
import time
from DataFunctions import get_simple_images_indices
from torch.utils.data import Subset, DataLoader


def check_diff(diffuser, sim_object, folder_path):
    orj_image, pic_width = plot_orj_image(sim_object, folder_path)
    masks_to_plot = plot_masks(diffuser, folder_path)
    plot_img_after_masks(masks_to_plot, orj_image, pic_width, folder_path)


def plot_masks(diffuser, folder_path):
    """ diffuser is tensor with size (batch_size, n_masks, img_dim) """
    batch_size, n_masks, img_dim = diffuser.shape
    num_plots = min(n_masks, 6)
    masks_to_plot = diffuser[1, :num_plots, :]
    pic_width = int(math.sqrt(img_dim))
    masks_to_plot = masks_to_plot.view(num_plots, pic_width, pic_width)
    plot_subplot(masks_to_plot, title="Masks", folder_path=folder_path)
    return masks_to_plot.cpu().detach().numpy()


def plot_orj_image(sim_object, folder_path='temp/Gan'):
    """ sim_object.view(batch_size, 1, img_dim) """
    _, _, img_dim = sim_object.shape
    pic_width = int(math.sqrt(img_dim))
    orj_image = sim_object[0, 0, :]
    orj_image = orj_image.view(pic_width, pic_width).cpu().detach().numpy()

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(orj_image, cmap='gray')
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    save_path = os.path.join(folder_path, 'Original_Image')
    fig.savefig(save_path)
    plt.show()
    return orj_image, pic_width


def plot_rec_image(rec_image, maxiter, niter_inner, alpha, folder_path='temp/Gan'):
    """ rec_image.view(1, img_dim) """
    _, img_dim = rec_image.shape
    pic_width = int(math.sqrt(img_dim))
    rec_image = rec_image.view(pic_width, pic_width).cpu().detach().numpy()

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(rec_image, cmap='gray')
    plt.title(f'Reconstructed Image : iter={maxiter},{niter_inner}, alpha={alpha}', fontsize=12)
    plt.axis('off')
    save_path = os.path.join(folder_path, f'Reconstructed_Image_{maxiter}_{niter_inner}_{alpha}.png')
    fig.savefig(save_path)
    # plt.show()
    # time.sleep(1)
    plt.close
    # return plt.gcf()


def plot_img_after_masks(masks_to_plot, orj_image, pic_width, folder_path):
    orj_image = orj_image.reshape(1, 1, pic_width, pic_width)
    image_after_masks = masks_to_plot * orj_image
    image_after_masks = torch.tensor(image_after_masks).view(-1, pic_width, pic_width)
    plot_subplot(image_after_masks, title="Image After Masks", folder_path=folder_path)


def plot_subplot(img_tensor, title="Image Subplots", num_rows=2, num_cols=3, folder_path='temp/Gan'):
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
    plt.savefig(os.path.join(folder_path, title))
    plt.show()


def experiment_berg_params(bucket, diffuser, folder_path='temp/Gan'):
    for maxiter in range(1, 10, 1):
        for niter_inner in range(1, 10, 1):
            for alpha in np.arange(0.1, 5, 0.5):
                try:
                    rec = sparse_encode(bucket, diffuser, maxiter=maxiter,
                                        niter_inner=niter_inner, alpha=alpha, algorithm='split-bregman')
                    plot_rec_image(rec, maxiter, niter_inner, alpha, folder_path=folder_path)
                except torch._C._LinAlgError as e:
                    print(f'params:{maxiter}, {niter_inner}, {alpha}')


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


def check_diff_ac(diffuser, folder_path='temp/Gan'):
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





def rec_from_samples(cr, img_new_width):
    xray_folder = 'data/medical/chunked_256/mock_class'
    results_dir = 'temp/split_bregman_xray'
    img_size = img_new_width**2
    realizations_number = math.floor(img_size / cr)
    image_names = get_image_names()

    for i, img_name in enumerate(image_names):
        knee_xray = cv2.imread(xray_folder + '/' + img_name, cv2.IMREAD_GRAYSCALE)
        knee_xray = np.array(knee_xray)
        plt.imshow(knee_xray)
        plt.title(f"{i} ground truth cr={cr}")
        plt.savefig(results_dir + "/" + f"xray-{i}_ground_truth_cr_{cr}.png")
        plt.show()

        knee_xray_resized = cv2.resize(knee_xray, (img_new_width, img_new_width))
        plt.imshow(knee_xray_resized)
        plt.title(f"resiezed {i} ground truth cr={cr}")
        plt.savefig(results_dir + "/" + f"xray-{i}_resized_{img_new_width}_{img_new_width}_cr_{cr}.png")
        plt.show()

        sim_diffuser = create_diffuser(realizations_number, img_new_width**2)
        sim_object = knee_xray_resized.reshape(1, img_size)
        sim_object = sim_object.transpose(1, 0)
        sim_bucket = np.matmul(sim_diffuser, sim_object)
        sim_bucket = sim_bucket.transpose((1, 0))

        sim_diffuser = torch.from_numpy(sim_diffuser)
        sim_bucket = torch.from_numpy(sim_bucket)

        rec = sparse_encode(sim_bucket, sim_diffuser, maxiter=1, niter_inner=1, alpha=1,
                            algorithm='split-bregman')

        plt.imshow(rec.reshape(img_new_width, img_new_width))
        plt.title(f"reconstruction {i} cr={cr}")
        plt.savefig(results_dir + "/" + f"rec_xray-{i}_cr_{cr}.png")
        plt.show()


def create_diffuser(M, N, diffuser_mean=0.5, diffuser_std=0.5):
    diffuser_transmission = np.random.normal(diffuser_mean, diffuser_std, [M, N])
    np.clip(diffuser_transmission, 0, 1, out=diffuser_transmission)  # ensure values within [0, 1]
    return diffuser_transmission


def get_image_names():
    image_names = ['chunk_middle_part_0417_0697542589_01_WRI-R2_F008.png',
                    'chunk_middle_part_0503_1018511008_01_WRI-L1_M012.png',
                    'chunk_middle_part_0417_0727170640_02_WRI-R1_F009.png',
                    'chunk_middle_part_0503_1018511068_01_WRI-L2_M012.png',
                    'chunk_middle_part_0417_0727170681_02_WRI-R2_F009.png',
                    'chunk_middle_part_0503_1020470848_02_WRI-L1_M012.png']
    return image_names



def subplot_simple_images(data_dir, save_dir):
    indices = get_simple_images_indices()
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    cifar_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    custom_dataset = Subset(cifar_dataset, indices)
    dataloader = iter(custom_dataset)

    num_images = len(custom_dataset)
    batch_size = 50

    for batch_start in range(0, num_images, batch_size):
        # Create a new figure for each batch of images
        fig, axes = plt.subplots(5, 10, figsize=(10, 10))

        for i in range(batch_size):
            try:
                image, label = next(dataloader)
            except StopIteration:
                break

            # Convert the image tensor to a NumPy array
            image = image.numpy()

            # Get the corresponding axis for the subplot
            row = i // 10
            col = i % 10
            ax = axes[row, col]

            # Display the image and set the title
            ax.imshow(np.transpose(image, (1, 2, 0)))
            image_index = batch_start + i
            ax.set_title(f"Idx: {image_index}")

        # Remove axis labels and adjust layout
        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()

        # Save or display the figure
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/simple_cifar_images_part_{batch_start}.png")
        else:
            plt.show()
        plt.close()

def subplot_cifar_images(data_dir, save_dir):
    # Define the CIFAR-10 dataset and dataloader
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    cifar_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    dataloader = iter(cifar_dataset)

    num_images = len(cifar_dataset)
    batch_size = 50

    for batch_start in range(0, num_images, batch_size):
        # Create a new figure for each batch of images
        fig, axes = plt.subplots(5, 10, figsize=(10, 10))

        for i in range(batch_size):
            try:
                image, label = next(dataloader)
            except StopIteration:
                break

            # Convert the image tensor to a NumPy array
            image = image.numpy()

            # Get the corresponding axis for the subplot
            row = i // 10
            col = i % 10
            ax = axes[row, col]

            # Display the image and set the title
            ax.imshow(np.transpose(image, (1, 2, 0)))
            image_index = batch_start + i
            ax.set_title(f"Idx: {image_index}")

        # Remove axis labels and adjust layout
        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()

        # Save or display the figure
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(f"{save_dir}/cifar_images_batch_{batch_start}.png")
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    rec_from_samples(5, 64)

