import math
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
base_directory = r'Results\cifar_GEN_bs_2_cr_4_nsamples4\train_images'
image_filenames = []
names_subfolders = os.listdir(base_directory)
n_images = min(len(names_subfolders) + 1, 50)

for i in range(1, n_images+1):
    folder_name = f'epoch_{i}'
    folder_path = os.path.join(base_directory, folder_name)
    image_filename = os.path.join(folder_path, 'train_images_1_out.jpg')  # Adjust the file extension as needed
    if os.path.exists(image_filename):
        image_filenames.append(image_filename)


# Define the number of columns for subplots
num_cols = 5  # Adjust this according to your preference

# Calculate the number of rows based on the number of images and columns
num_images = len(image_filenames)
num_rows = math.ceil(num_images / num_cols)

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

# Flatten the axes if needed
if num_images < num_cols:
    axes = axes.reshape(1, -1)
else:
    axes = axes.flatten()

# Iterate through image filenames and display them in subplots
for i, image_filename in enumerate(image_filenames):
    ax = axes[i]
    image = imread(image_filename)
    ax.imshow(image)
    ax.set_title(f'Epoch {i}')
    ax.axis('off')

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.savefig('temp/output_subplot.png')
plt.show()
