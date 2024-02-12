import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import os
import bm3d
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet)
from skimage import io, img_as_float
from PIL import Image


def denoise_image(img_new_width=64):
    xray_folder = 'data/medical/chunked_256/mock_class'
    results_dir = 'data/denoised_images'
    image_names = ['chunk_middle_part_0050_0619640860_01_WRI-R2_M011.png']

    for i, img_name in enumerate(image_names):
        # knee_xray = cv2.imread(xray_folder + '/' + img_name, cv2.IMREAD_GRAYSCALE)
        # knee_xray = np.array(knee_xray)
        knee_xray = img_as_float(io.imread(os.path.join(xray_folder, img_name), as_gray=True))
        plt.imshow(knee_xray, cmap='gray')
        plt.title(f"{i} ground truth")
        plt.savefig(results_dir + "/" + f"xray-{i}_ground_truth.png")
        # plt.show()

        knee_xray_resized = cv2.resize(knee_xray, (img_new_width, img_new_width))
        plt.imshow(knee_xray_resized, cmap='gray')
        plt.title(f"resiezed {i} ground truth")
        plt.savefig(results_dir + "/" + f"xray-{i}_resized_{img_new_width}_{img_new_width}.png")
        # plt.show()

        denoised_image = try_diff_denoising(knee_xray_resized, results_dir, algorithms=['gaussian', 'TV'])
        # plt.imshow(denoised_image, cmap='gray')
        # plt.title(f"denoised after resiezed {i} ")
        # plt.savefig(results_dir + "/" + f"denoised_{i}_resized_{img_new_width}_{img_new_width}.png")
        # plt.show()


def eli_resize(scale_factor, img_path, results_dir):
    input_image = Image.open(img_path).convert("L")
    width, height = input_image.size
    new_width = width // scale_factor
    new_height = height // scale_factor
    resized_img = Image.new("L", (new_width, new_height))

    for x in range(new_width):
        for y in range(new_height):
            # Calculate the corresponding coordinates in the original image
            x1 = x * scale_factor
            y1 = y * scale_factor

            # Initialize variables for calculating the average intensity
            total_intensity = 0

            # Iterate over the pixels in the original image to calculate the average intensity
            for i in range(scale_factor):
                for j in range(scale_factor):
                    intensity = input_image.getpixel((x1 + i, y1 + j))
                    total_intensity += intensity

            average_intensity = total_intensity // (scale_factor * scale_factor)
            resized_img.putpixel((x, y), average_intensity)

    resized_img.save(os.path.join(results_dir, "eli_resize.png"))
    input_image.close()
    return resized_img


def cv2_resize(img_width, img_length, img_path, results_dir):
    org_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    org_img = np.array(org_img)

    plt.imshow(org_img, cmap='gray')
    plt.title(f"original_image")
    plt.savefig(os.path.join(results_dir, "original_image.png"))

    resized_img = cv2.resize(org_img, (img_width, img_length), interpolation=cv2.INTER_NEAREST)
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"INTER_NEAREST")
    plt.savefig(os.path.join(results_dir, "INTER_NEAREST.png"))

    resized_img = cv2.resize(org_img, (img_width, img_length), interpolation=cv2.INTER_AREA)
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"INTER_AREA")
    plt.savefig(os.path.join(results_dir, "INTER_AREA.png"))

    resized_img = cv2.resize(org_img, (img_width, img_length), interpolation=cv2.INTER_LINEAR)
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"INTER_LINEAR")
    plt.savefig(os.path.join(results_dir, "INTER_LINEAR.png"))

    resized_img = cv2.resize(org_img, (img_width, img_length), interpolation=cv2.INTER_CUBIC)
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"INTER_CUBIC")
    plt.savefig(os.path.join(results_dir, "INTER_CUBIC.png"))

    resized_img = cv2.resize(org_img, (img_width, img_length), interpolation=cv2.INTER_LANCZOS4)
    plt.imshow(resized_img, cmap='gray')
    plt.title(f"INTER_LANCZOS4")
    plt.savefig(os.path.join(results_dir, "INTER_LANCZOS4.png"))

    return resized_img


def transforms_resize(img_width, img_length, img_path, results_dir):
    pass


def pillow_resize(img_width, img_path, results_dir):
    image = Image.open(img_path)
    resized_img = image.resize((img_width, img_width))
    resized_img.save(os.path.join(results_dir, "pillow_resize.png"))
    return resized_img


def try_diff_resize(img_width, img_length, img_path, results_dir, algorithms='all'):
    if algorithms == 'all':
        algorithms = ['eli', 'cv2'] #, 'pillow', 'transforms']
    for alg in algorithms:
        if alg == 'eli':
            resized_img = eli_resize(4 , img_path, results_dir)
        if alg == 'cv2':
            resized_img = cv2_resize(img_width, img_length, img_path, results_dir)
        if alg == 'pillow':
            resized_img = pillow_resize(img_width, img_length, img_path, results_dir)
        if alg == 'transforms':
            resized_img = transforms_resize(img_width, img_length, img_path, results_dir)
    return resized_img


def try_diff_denoising(noisy_img, results_dir, algorithms='all'):
    if algorithms == 'all':
        algorithms = ['gaussian', 'bilateral', 'TV', 'wavelet', 'BM3D']
    for algorithm in algorithms:
        if algorithm == 'gaussian':
            denoised_image = nd.gaussian_filter(noisy_img, sigma=0.5)
            plt.imshow(denoised_image, cmap='gray')
            plt.title(f"Gaussian_smoothed")
            plt.savefig(os.path.join(results_dir, "Gaussian_smoothed.png"))
        if algorithm == 'bilateral':
            denoised_image = denoise_bilateral(noisy_img, sigma_spatial=15, multichannel=False)
            plt.imshow(denoised_image, cmap='gray')
            plt.title(f"bilateral_smoothed")
            plt.savefig(os.path.join(results_dir, "bilateral_smoothed.png"))
        if algorithm == 'TV':
            denoised_image = denoise_tv_chambolle(noisy_img, weight=0.01, multichannel=False)
            plt.imshow(denoised_image, cmap='gray')
            plt.title(f"TV_smoothed")
            plt.savefig(os.path.join(results_dir, "TV_smoothed.png"))
        if algorithm == 'wavelet':
            denoised_image = denoise_wavelet(noisy_img, multichannel=False, method='VisuShrink',
                                             mode='soft', rescale_sigma=True)
            plt.imshow(denoised_image, cmap='gray')
            plt.title(f"wavelet_smoothed")
            plt.savefig(os.path.join(results_dir, "wavelet_smoothed.png"))
        if algorithm == 'BM3D':
            denoised_image = bm3d.bm3d(noisy_img, sigma_psd=1, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            plt.imshow(denoised_image, cmap='gray')
            plt.title(f"BM3D_smoothed")
            plt.savefig(os.path.join(results_dir, "BM3D_smoothed.png"))
    return denoised_image


def split_image(image, num_rows, num_cols):
    height, width, _ = image.shape
    smaller_height = height // num_rows
    smaller_width = width // num_cols

    smaller_images = []

    for i in range(num_rows):
        for j in range(num_cols):
            row_start = i * smaller_height
            row_end = (i + 1) * smaller_height
            col_start = j * smaller_width
            col_end = (j + 1) * smaller_width
            smaller_images.append(image[row_start:row_end, col_start:col_end])

    return smaller_images


def folder_image_split(folder_path, num_rows, num_cols, result_folder):
    for j_image, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            original_image = cv2.imread(image_path)
            smaller_images = split_image(original_image, num_rows, num_cols)
            result_image_path = os.path.join(result_folder, f"image{j_image}_original.png")
            cv2.imwrite(result_image_path, original_image)

            for i, small_image in enumerate(smaller_images):
                row = i // num_cols
                col = i % num_cols
                result_image_path = os.path.join(result_folder, f"image{j_image}_row_{row}_col_{col}.png")
                cv2.imwrite(result_image_path, small_image)


if __name__ == '__main__':
    img_path = r'C:\Users\user\Desktop\Projects\research\com_medical_diffuser\data\selected_images\mock_class' \
               r'\chunk_middle_part_0060_0798373378_01_WRI-R2_F016.png '
    # img_path = r'C:\Users\user\Desktop\Projects\research\com_medical_diffuser\data\selected_images\mock_class' \
    #            r'\image7_original.png '
    result_dir = 'temp/resize_exp'
    img_width = 64
    img_length = 64
    try_diff_resize(img_width, img_length, img_path, result_dir, algorithms=['eli', 'cv2'])


