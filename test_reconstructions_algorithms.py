import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Lasso import sparse_encode


def rec_from_samples(cr, img_new_width):
    xray_processed_base_folder = '/home/amotz/PycharmProjects/medical_experiments_with_rec_algorithms/Optimal_Diffuser/data/medical/processed_med_samples'
    results_dir = "/home/amotz/eli_lab/mor/split_bregman_xray_rec_experiments"
    img_size = img_new_width**2
    realizations_number = math.floor(img_size / cr)

    for i in range(2,7):
        xray_processed_image = f"xray-{i}.png"
        knee_xray = cv2.imread(xray_processed_base_folder + '/' + xray_processed_image)
        knee_xray = np.array(knee_xray)
        knee_xray_color = np.array(knee_xray[:,:,0])
        plt.imshow(knee_xray_color)
        plt.savefig(results_dir + "/" + f"xray-{i}_ground_truth_256_256_cr_{cr}.png")
        plt.show()

        knee_xray_color_64_64 = cv2.resize(knee_xray_color, (img_new_width,img_new_width))
        plt.imshow(knee_xray_color_64_64)
        plt.savefig(results_dir + "/" + f"xray-{i}_resized_{img_new_width}_{img_new_width}_cr_{cr}.png")
        plt.show()

        sim_diffuser = create_diffuser(realizations_number, img_new_width**2)
        sim_object = knee_xray_color_64_64.reshape(1, img_size)

        sim_object = sim_object.transpose(1, 0)
        sim_bucket = np.matmul(sim_diffuser,sim_object)
        sim_bucket = sim_bucket.transpose((1, 0))
        sim_diffuser = torch.from_numpy(sim_diffuser)
        sim_bucket = torch.from_numpy(sim_bucket)

        rec = sparse_encode(sim_bucket, sim_diffuser, maxiter=1, niter_inner=1, alpha=1,
                            algorithm='split-bregman')

        plt.imshow(rec.reshape(img_new_width, img_new_width))
        plt.savefig(results_dir + "/" + f"xray-{i}_resized_rec_cr_{cr}.png")
        plt.show()


def create_diffuser(M, N, diffuser_mean = 0.5, diffuser_std = 0.5, dim = 1): # Create Diffuser
    diffuser_transmission = np.random.normal(diffuser_mean, diffuser_std, [M, N])
    for i in range(M):
        for j in range(N):
            diffuser_transmission[i, j] = max(diffuser_transmission[i, j], 0)
            diffuser_transmission[i, j] = min(diffuser_transmission[i, j], 1)
    return diffuser_transmission


rec_from_samples(2, 64)
rec_from_samples(4, 64)
rec_from_samples(6, 64)
