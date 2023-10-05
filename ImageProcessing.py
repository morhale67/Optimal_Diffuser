import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage as nd
import os
import bm3d
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet)
from skimage import io, img_as_float


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
    folder_path = 'data\selected_images\mock_class'
    result_folder = 'data\images_for_sew\mock_class'
    folder_image_split(folder_path, 4, 4, result_folder)


