from torchvision.datasets import ImageFolder
import cv2
from sklearn.model_selection import train_test_split
import time
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as dset
import random
import torch
import numpy as np
from torchvision import datasets, transforms
import os
from torch.utils.data import Subset, DataLoader


def build_dataset(batch_size, num_workers, pic_width, n_samples, data_root_medical, data_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((pic_width, pic_width)),
        transforms.Grayscale(num_output_channels=1)
    ])
    if data_name.lower() == 'medical':
        data_set = ImageFolder(root=data_root_medical, transform=transform)
    if data_name.lower() == 'simple_cifar':
        data_set = ImageFolder(root='./data_DSI/GCP_data/simple_cifar', transform=transform)
    elif data_name.lower() == 'cifar' or data_name.lower() == 'cifar10':
        data_set = dset.CIFAR10(root='./data/cifar10', train=True, transform=transform, download=True)
    elif data_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((pic_width, pic_width))
        ])
        data_set = dset.MNIST(root='./data', train=True, transform=transform, download=True)

    train_loader, test_loader = create_loader_from_data_set(data_set, n_samples, batch_size, num_workers)
    # save_random_image_from_loader(train_loader, pic_width)
    return train_loader, test_loader


def create_loader_from_data_set(data_set, n_samples, batch_size, num_workers, test_size=0.2):
    indices = list(range(len(data_set)))
    selected_indices = random.sample(indices, n_samples)

    train_indices, test_indices = train_test_split(selected_indices, test_size=test_size,
                              shuffle=False, random_state=42)
    train_indices = adjust_list_length_same_bs(train_indices, batch_size)
    test_indices = adjust_list_length_same_bs(test_indices, batch_size)

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(data_set, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler,
                              shuffle=False)
    test_loader = DataLoader(data_set, batch_size=batch_size, num_workers=num_workers, sampler=test_sampler,
                             shuffle=False)
    return train_loader, test_loader


def extract_mean_std(pic_width, data_root, plot_image=False):
    # mean_std_dict = {28: (0.30109875, 0.18729387),
    #                  64: (0.29853243, 0.188016),
    #                  128: (0.2976776, 0.1878308)}
    # saved_pic_width = list(mean_std_dict.keys())
    # if pic_width in saved_pic_width:
    #     mean, std = mean_std_dict[pic_width]
    #     return mean, std

    data_root = os.path.join(data_root, 'class_dir')
    start = time.time()
    image_paths = [os.path.join(data_root, filename) for filename in os.listdir(data_root)]

    pixel_sum = 0  # Initialize sum of pixel values
    pixel_squared_sum = 0  # Initialize sum of squared pixel values
    num_images = len(image_paths)

    if plot_image and num_images > 0:
        random_image_path = random.choice(image_paths)
        plot_and_save_image(random_image_path, pic_width)

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        image = cv2.resize(image, (pic_width, pic_width))  # Resize the image
        image = image / 255.0  # Normalize pixel values to [0, 1]

        pixel_sum += image.sum()
        pixel_squared_sum += (image ** 2).sum()

    mean = pixel_sum / (num_images * pic_width * pic_width)
    std = np.sqrt(pixel_squared_sum / (num_images * pic_width * pic_width) - mean ** 2)
    print(f'Calculate mean and std to the folder took {time.time()-start} sec')
    print(f'Mean: {mean}')
    print(f'Std : {std}')
    return mean, std


def plot_and_save_image(image_path, pic_width):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (pic_width, pic_width))  # Resize the image

    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_image_path = os.path.join(temp_dir, f'random_image_{pic_width}.png')
    cv2.imwrite(temp_image_path, resized_image)

    # plt.imshow(resized_image, cmap='gray')
    # plt.title('Random Grayscale Image')
    # plt.axis('off')
    # plt.show()


def extract_min_dims(data_root='data/Medical/part1'):
    # part1
    # Calculate minimum image dimensions took 444.76435923576355 sec
    # Minimum Image Width: 227
    # Minimum Image Height: 364
    start = time.time()
    image_paths = [os.path.join(data_root, filename) for filename in os.listdir(data_root)]

    min_width = float('inf')
    min_height = float('inf')

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image as grayscale
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        image_height, image_width = image.shape

        # Update minimum dimensions
        if image_width < min_width:
            min_width = image_width
        if image_height < min_height:
            min_height = image_height

    print(f'Calculate minimum image dimensions took {time.time()-start} sec')
    print(f'Minimum Image Width: {min_width}')
    print(f'Minimum Image Height: {min_height}')
    return min_width, min_height


def adjust_list_length_same_bs(lst, batch_size):
    new_length = len(lst) - (len(lst) % batch_size)
    adjusted_list = lst[:new_length]
    return adjusted_list


def save_random_image_from_loader(loader, pic_width, save_dir='temp'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Choose a random batch from the loader
    batch = next(iter(loader))
    images, _ = batch

    # Choose a random image from the batch
    idx = random.randint(0, images.size(0) - 1)
    image = images[idx]

    image = transforms.ToPILImage()(image)

    # Save the image
    image_path = os.path.join(save_dir, f"loader_image_{pic_width}.png")
    image.save(image_path)
    print(f"Random image saved at: {image_path}")


def chunk_middle_parts(input_folder, pic_width):
    output_folder = f"data/Medical/chunked_middle_parts_{pic_width}"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Modify file extensions if needed
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Get image dimensions
            img_height, img_width, _ = image.shape

            # Calculate crop boundaries to extract the middle part
            left = (img_width - pic_width) // 2
            top = (img_height - pic_width) // 2
            right = left + pic_width
            bottom = top + pic_width

            # Crop and save the middle part of the image
            middle_part = image[top:bottom, left:right]
            chunk_filename = f"chunk_middle_part_{filename}"
            chunk_path = os.path.join(output_folder, chunk_filename)
            cv2.imwrite(chunk_path, middle_part)


def get_simple_images_indices():
    indices = [12, 18, 20, 21, 29, 41, 49, 66, 77, 103, 105, 108, 115, 123, 140, 160, 172, 194, 202, 252, 289, 290, 317,
           323, 330, 335, 341, 348, 349, 371, 402, 409, 415, 441, 461, 479, 504, 535, 538, 539, 588, 605, 650, 655, 663,
           687, 691, 695, 713, 731, 790, 822, 935, 977, 991, 998, 1017, 1064, 1131, 1145, 1168, 1211, 1219, 1231, 1293,
           1328, 1428, 1495, 1522, 1527, 1533, 1588, 1681, 1687, 1816, 1849, 1908, 1912, 1918, 1978, 2063, 2079, 2227,
           2244, 2320, 2328, 2377, 2379, 2446, 2540, 2596, 2639, 2669, 2682, 2690, 2744, 3004, 3020, 3091, 3105, 3184,
           3205, 3291, 3387, 3526, 3536, 3568, 3651, 3692, 3791, 3842, 4078, 4266, 4273, 4331, 4348, 4418, 4444]
    return indices

