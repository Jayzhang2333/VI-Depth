import os
import argparse

import torch
import imageio
import numpy as np

from tqdm import tqdm
from PIL import Image
import csv
import pandas as pd

import modules.midas.utils as utils

import pipeline
import metrics

def generate_feature_map(feature_fp, original_height=240, original_width=320, new_height=480, new_width=640):
    # Read the CSV file
    df = pd.read_csv(feature_fp)

    # Initialize a blank depth map for the new image size with zeros
    sparse_depth_map = np.full((new_height, new_width), 0.0, dtype=np.float32)

    # Calculate scaling factors
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    # Iterate through the dataframe and populate the depth map with scaled coordinates
    for index, row in df.iterrows():
        # Scale pixel coordinates to new image size
        pixel_row = int(row['row'] * scale_y)
        pixel_col = int(row['column'] * scale_x)
        depth_value = float(row['depth'])

        # Ensure the scaled coordinates are within the bounds of the new image size
        if 0 <= pixel_row < new_height and 0 <= pixel_col < new_width:
            sparse_depth_map[pixel_row, pixel_col] = depth_value

    return sparse_depth_map


def generate(dataset_path, depth_predictor, nsamples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # ranges for VOID
    min_depth, max_depth = 0.1, 5.0
    min_pred, max_pred = 0.1, 7.0

    # instantiate method
    method = pipeline.TrainingDataGenerator(
        depth_predictor, nsamples, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    # with open(f"{dataset_path}/void_{nsamples}/test_image.txt") as f: 
    #     test_image_list = [line.rstrip() for line in f]
    test_image_list = []
    ground_truth_list = []
    depth_prior_list = []

    # Open and read the CSV file
    with open(f"{dataset_path}/dataset_with_matched_features.csv", newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            test_image_list.append(row[0])  # First column: test image paths
            ground_truth_list.append(row[1])  # Second column: ground truth paths
            depth_prior_list.append(row[2])  # Third column: depth prior paths
        
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    # iterate through inputs list
    for i in tqdm(range(len(test_image_list))):
    # for i in tqdm(range(5)):
        
        # image
        # input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image_fp = test_image_list[i]
        # print(input_image_fp)
        # input_image_fp.replace("/home/auv/FLSea", "/mnt/e/FLSea_latest")
        # input_image_fp.replace("/media/jay/apple/FLSea_latest", "/mnt/e/FLSea_latest")
        input_image_fp = input_image_fp.replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest")
        # print()
        # input_image_fp.replace("imgs", "seaErra")
        input_image = utils.read_image(input_image_fp)

        # sparse depth
        # input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth_fp = depth_prior_list[i]
        # input_sparse_depth_fp.replace("/home/auv/FLSea", "/mnt/e/FLSea_latest")
        # input_sparse_depth_fp.replace("/media/jay/apple/FLSea_latest", "/mnt/e/FLSea_latest")
        input_sparse_depth_fp = input_sparse_depth_fp.replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest")
        input_sparse_depth = generate_feature_map(input_sparse_depth_fp)
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        

        input_sparse_depth_valid = (input_sparse_depth < max_depth) * (input_sparse_depth > min_depth)
        # if np.sum(input_sparse_depth_valid) <= 10:
        #     print("Not enough prior")
        #     continue

        # sparse depth validity map
        # validity_map_fp = input_image_fp.replace("image", "validity_map")
        # validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        # assert(np.all(np.unique(validity_map) == [0, 256]))
        # validity_map[validity_map > 0] = 1
        validity_map = None
        
        # target (ground truth) depth
        # target_depth_fp = input_image_fp.replace("image", "ground_truth")
        target_depth_fp = ground_truth_list[i]
        # target_depth_fp.replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest")
        # target_depth_fp.replace("/media/jay/apple/FLSea_latest", "/mnt/e/FLSea_latest")
        target_depth_fp = target_depth_fp.replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest")
        target_depth = np.array(Image.open(target_depth_fp).resize((640, 480)), dtype=np.float32)
        target_depth[target_depth <= 0] = 0.0
        # print(f"maximum of depth map is {np.max(target_depth)}")

        # target depth valid/mask
        # mask = (target_depth < max_depth)
        # if min_depth is not None:
        #     mask *= (target_depth > min_depth)
        # target_depth[~mask] = np.inf  # set invalid depth
        # target_depth = 1.0 / target_depth

        # run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device)

        ga_path = input_image_fp.replace('imgs', 'ga_result').rsplit('.', 1)[0] + '.npy'
        gt_path = input_image_fp.replace('imgs', 'gt').rsplit('.', 1)[0] + '.npy'
        interpolation_sparse_path = input_image_fp.replace('imgs', 'interpolation_sparse').rsplit('.', 1)[0] + '.npy'

        np.save(ga_path, output['ga_depth'])
        np.save(gt_path, target_depth)
        np.save(interpolation_sparse_path, output['interpolation_sparse'])



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path', type=str, default='/path/to/void_release/',
                        help='Path to VOID release dataset.')
    parser.add_argument('-dp', '--depth-predictor', type=str, default='midas_small', 
                        help='Name of depth predictor to use in pipeline.')
    parser.add_argument('-ns', '--nsamples', type=int, default=150, 
                        help='Number of sparse metric depth samples available.')
    

    args = parser.parse_args()
    print(args)
    
    generate(
        args.dataset_path,
        args.depth_predictor, 
        args.nsamples, 
    )