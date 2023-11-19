import os
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import pickle
import re



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


def save_loss_figure(train_loss, test_loss, folder_path='.', filename='loss_figure.png'):
    # Set custom font and size for the entire plot
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 7))

    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(test_loss, label='Test Loss', color='orange')

    # Calculate the minimal test loss and its corresponding epoch
    min_test_loss = min(test_loss)
    min_test_epoch = test_loss.index(min_test_loss) + 1  # Adding 1 to convert from 0-based index to epoch number

    # Add labels and title
    plt.xlabel('Epoch', fontsize=22, fontname='Arial')
    plt.ylabel('Loss', fontsize=22, fontname='Arial')
    title = 'Train and Test Loss'
    subtitle = 'Min Test Loss: {:.6f} (Epoch {})'.format(min_test_loss, min_test_epoch)
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
    plt.show()


def subplot_reconstraction_for_all_images(run_folder_path):
    data_set = 'train_images'
    folder_path = os.path.join(run_folder_path, data_set)
    # Get a list of subfolders starting with "image_"
    images_folders = [subfolder for subfolder in os.listdir(folder_path) if subfolder.startswith("image_")]

    # Iterate through each subfolder and call subplot_images_in_folder
    for image_folder in images_folders:
        subplot_epochs_reconstruction(run_folder_path, data_set, image_folder)

    data_set = 'test_images'
    folder_path = os.path.join(run_folder_path, data_set)
    images_folders = [subfolder for subfolder in os.listdir(folder_path) if subfolder.startswith("image_")]
    for image_folder in images_folders:
        subplot_epochs_reconstruction(run_folder_path, data_set, image_folder)


if __name__ == '__main__':
    subplot_reconstraction_for_all_images(r'Results_to_save\by order\simple_cifar_GEN_bs_2_cr_4_nsamples100_picw_32')

