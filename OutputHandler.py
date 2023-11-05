import os
import matplotlib.pyplot as plt
import torch
import cv2


def make_folder(net_name, p):
    folder_name = f"{p['data_name']}_{net_name}_bs_{p['batch_size']}_cr_{p['cr']}_nsamples{p['n_samples']}_picw_{p['pic_width']}"
    print(folder_name)
    folder_path = 'Results/' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def save_outputs(epoch, output, y_label, pic_width, folder_path, name_sub_folder):
    first_img_path = folder_path + '/' + name_sub_folder + f'first_image.pth'
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
