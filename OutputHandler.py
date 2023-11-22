import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import pickle
import re
from Lasso import sparse_encode
from testers import create_diffuser


def make_folder(net_name, p):
    folder_name = f"{p['data_name']}_{net_name}_bs_{p['batch_size']}_cr_{p['cr']}_nsamples{p['n_samples']}_picw_{p['pic_width']}"
    if not p['learn_vec_lr']:
        folder_name = folder_name + f"_lr_{p['lr']}"
    print(folder_name)
    folder_path = 'Results/' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def save_outputs(epoch, output, y_label, pic_width, folder_path, name_sub_folder):
    first_img_path = folder_path + '/' + name_sub_folder + f'/first_image.pth'
    if epoch == 0:
        org_imgs = y_label.view(-1, pic_width, pic_width)
        first_img = org_imgs[0, :, :]
        torch.save(first_img, first_img_path)
    else:
        first_img = torch.load(first_img_path)
    in_out_images = zip(output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    images_dir = folder_path + '/' + name_sub_folder + '/epoch_' + str(epoch)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    for i, (out_image, orig_image) in enumerate(in_out_images):
        same_as_first = torch.equal(first_img, orig_image)
        image_number = int(not same_as_first)
        plt.imsave(images_dir + f'/{name_sub_folder}_{image_number}_out.jpg', out_image.detach().numpy())
        plt.imsave(images_dir + f'/{name_sub_folder}_{image_number}_orig.jpg', orig_image.cpu().detach().numpy())
        if i == 9:
            break


def save_numerical_figure(g1, g2, g1_label, g2_label, filename='loss_figure.png', folder_path='.'):
    # Set custom font and size for the entire plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 7))

    plt.plot(g1, label=g1_label, color='blue')
    plt.plot(g2, label=g2_label, color='orange')

    # Calculate the minimal test loss and its corresponding epoch
    min_g2 = min(g2)
    min_g2_epoch = g2.index(min_g2) + 1  # Adding 1 to convert from 0-based index to epoch number

    # Add labels and title
    plt.xlabel('Epoch', fontsize=22, fontname='Arial')
    plt.ylabel('Loss', fontsize=22, fontname='Arial')
    title = g1_label + ' and ' + g2_label
    subtitle = f'Min {g2_label}: {min_g2:.3f} (Epoch {min_g2_epoch})'
    plt.title(f'{title}\n{subtitle}', fontsize=22, fontname='Times New Roman')

    plt.legend()

    # Save the figure to the specified filename
    full_file_path = os.path.join(folder_path, filename)
    plt.savefig(full_file_path)
    plt.show()


def save_orig_img(loader, folder_path, name_sub_folder):
    all_image_tensors = []
    for index, (batch_images, label) in enumerate(loader):
        for img in batch_images:
            all_image_tensors.append(img)
    all_images_tensor = torch.cat(all_image_tensors, dim=0)
    path_subfolder = os.path.join(folder_path, name_sub_folder)
    if not os.path.exists(path_subfolder):
        os.makedirs(path_subfolder)
    orig_img_path = os.path.join(path_subfolder, 'orig_imgs_tensors.pt')
    torch.save(all_images_tensor, orig_img_path)


def save_randomize_outputs(epoch, batch_index, output, y_label, pic_width, folder_path, name_sub_folder):
    in_out_images = zip(output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    for i, (out_image, orig_image) in enumerate(in_out_images):
        image_number = get_original_image_number(orig_image, folder_path, name_sub_folder, epoch, batch_index)
        if image_number <= 20:
            output_dir = folder_path + '/' + name_sub_folder + f'/image_{image_number}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if epoch == 0:
                plt.imsave(output_dir + f'/{image_number}_orig.jpg', orig_image.cpu().detach().numpy())
            plt.imsave(output_dir + f'/epoch_{epoch}_{image_number}_out.jpg', out_image.detach().numpy())


def get_original_image_number(orig_img, folder_path, name_sub_folder, epoch, batch_index):
    orig_img_path = folder_path + '/' + name_sub_folder + '/orig_imgs_tensors.pt'
    all_images_tensor = torch.load(orig_img_path)
    for index, image_tensor in enumerate(all_images_tensor):
        if torch.equal(orig_img, image_tensor):
            return index
    return -1


def plot_2_images(cur_img, image_tensor, index, folder_path, epoch):
    # Plot the original and matching tensors
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cur_img.numpy(), cmap='gray')
    plt.title('Current Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image_tensor.numpy(), cmap='gray')
    plt.title('Matching Image (Index {})'.format(index))
    save_path = os.path.join(folder_path, f'index_{index}_epoch_{epoch}.png')
    plt.savefig(save_path)


def subplot_epochs_reconstruction(run_folder_path, data_set, image_folder):
    '''
    :param folder_path: the name of the folder with all the image_i folders
    :param num_image: number of image from the train/test set
    :return: no output, save the subplot of the reconstraction process in the main folder
    '''

    images_path = os.path.join(run_folder_path, data_set, image_folder)
    save_folder = os.path.join(run_folder_path, 'reconstruction')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    name_subplot = os.path.join(save_folder, f'{data_set}_{image_folder}_reconstruction.png')
    # Filter images that match the pattern 'epoch_{epoc7h}'
    filtered_images = [img for img in os.listdir(images_path) if f'epoch_' in img]

    filtered_images.sort(key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))

    num_epochs = len(filtered_images)
    num_cols = 10  # Number of columns in the subplot grid
    num_rows = (num_epochs + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 3 * num_rows))

    for i, image_name in enumerate(filtered_images):
        img_path = os.path.join(images_path, image_name)
        img = plt.imread(img_path)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Epoch {i + 1}')

    plt.tight_layout()
    plt.savefig(name_subplot)
    # plt.show()


def subplot_reconstraction_for_all_images(run_folder_path, cr):
    data_set = 'train_images'
    folder_path = os.path.join(run_folder_path, data_set)
    # Get a list of subfolders starting with "image_"
    images_folders = [subfolder for subfolder in os.listdir(folder_path) if subfolder.startswith("image_")]

    # Iterate through each subfolder and call subplot_images_in_folder
    for image_folder in images_folders:
        rec_bregman_for_image(cr, image_folder, data_set, run_folder_path)
        # subplot_epochs_reconstruction(run_folder_path, data_set, image_folder)

    data_set = 'test_images'
    folder_path = os.path.join(run_folder_path, data_set)
    images_folders = [subfolder for subfolder in os.listdir(folder_path) if subfolder.startswith("image_")]
    for image_folder in images_folders:
        rec_bregman_for_image(cr, image_folder, data_set, run_folder_path)
        # subplot_epochs_reconstruction(run_folder_path, data_set, image_folder)


def rec_bregman_for_image(cr, image_folder, data_set, run_folder_path):
    save_folder = os.path.join(run_folder_path, 'reconstruction')

    images_path = os.path.join(run_folder_path, data_set, image_folder)
    org_img_name = [img for img in os.listdir(images_path) if f'_orig' in img]
    image_path = os.path.join(images_path, org_img_name[0])

    org_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    org_image = np.array(org_image)

    plt.imshow(org_image)
    plt.savefig(os.path.join(save_folder, f"{data_set}_{image_folder}_ground_truth.png"))

    pic_width = len(org_image)
    img_dim = pic_width**2
    realizations_number = math.floor(img_dim / cr)

    sim_diffuser = create_diffuser(realizations_number, img_dim)
    sim_object = org_image.reshape(1, img_dim)
    sim_object = sim_object.transpose(1, 0)
    sim_bucket = np.matmul(sim_diffuser, sim_object)
    sim_bucket = sim_bucket.transpose((1, 0))

    sim_diffuser = torch.from_numpy(sim_diffuser)
    sim_bucket = torch.from_numpy(sim_bucket)

    rec = sparse_encode(sim_bucket, sim_diffuser, maxiter=1, niter_inner=1, alpha=1,
                        algorithm='split-bregman')

    plt.imshow(rec.reshape(pic_width, pic_width))
    plt.title(f"reconstruction by random patterns cr={cr}")
    plt.savefig(os.path.join(save_folder, f"Bregman_rec_{data_set}_{image_folder}_cr_{cr}.png"))
    # plt.show()


def PSNR(image1, image2, m, n, max_i=255):
    # max_i is n_gray_levels
    y = torch.add(image1, (-image2))
    y_squared = torch.pow(y, 2)
    mse = torch.sum(y_squared)/(m*n)
    psnr = 10*math.log(max_i**2/mse, 10)
    return psnr


def calc_psnr_batch(output, y_label, pic_width):
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    in_out_images = zip(output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    batch_psnr = 0
    for i, (out_image, orig_image) in enumerate(in_out_images):
        batch_psnr += PSNR(out_image.to(dev), orig_image.to(dev), pic_width, pic_width)
    return batch_psnr


if __name__ == '__main__':
    subplot_reconstraction_for_all_images(r'Results_to_save\by order\simple_cifar_GEN_bs_2_cr_10_nsamples100_picw_32',
                                          cr=10)

