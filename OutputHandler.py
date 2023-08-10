import os
import matplotlib.pyplot as plt


def make_folder(net_name, p):
    folder_name = f"{net_name}_bs_{p['batch_size']}_cr_{p['cr']}"
    print(folder_name)
    folder_path = 'Results/' + folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def save_outputs(output, y_label, pic_width, folder_path, name_sub_folder):
    in_out_images = zip(output.cpu().view(-1, pic_width, pic_width), y_label.view(-1, pic_width, pic_width))
    images_dir = folder_path + '/' + name_sub_folder
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    for i, (out_image, orig_image) in enumerate(in_out_images):
        plt.imsave(images_dir + f'/test_image_{i}_out.jpg', out_image.detach().numpy())
        plt.imsave(images_dir + f'/test_image_{i}_orig.jpg', orig_image.cpu().detach().numpy())
        if i > 18:
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