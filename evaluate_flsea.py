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
import matplotlib.pyplot as plt


def filter_lower_33_percent(sparse_feature_map):
    """
    sparse_feature_map: 2D NumPy array, where non-zero entries represent valid depth/feature values.
    We only keep values in the lower 33% of valid entries; everything else becomes zero.
    """
    # 1. Extract valid (non-zero) values
    valid_values = sparse_feature_map[sparse_feature_map > 0]
    
    # Edge case: if there are no non-zero entries, just return an array of zeros
    if valid_values.size == 0:
        return np.zeros_like(sparse_feature_map)
    
    # 2. Find the threshold corresponding to the 33rd percentile of valid values
    threshold_33 = np.quantile(valid_values, 0.33)  # 0.33 => 33rd percentile

    # 3. Create a mask that keeps only values <= that threshold (and originally non-zero)
    filtered = np.where((sparse_feature_map > 0) & 
                        (sparse_feature_map <= threshold_33),
                        sparse_feature_map,
                        0)
    
    return filtered



def show_images(tensor_images):
    tensor_images = tensor_images.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_images = np.transpose(tensor_images, (0, 2, 3, 1))  # Change from CHW to HWC
    
    # Display the images
    batch_size = tensor_images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))  # Adjust number of subplots as needed
    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]
    
    for idx in range(batch_size):
        axes[idx].imshow(tensor_images[idx])  # Assuming images are normalized [0, 1]
        axes[idx].axis('off')
    
    plt.show()

def generate_feature_map(feature_fp, original_height=240, original_width=320, new_height=608, new_width=968, lizard=False):
    # Detect the delimiter by reading a sample from the file
    with open(feature_fp, 'r') as f:
        sample = f.read(1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ' '])
            delimiter = dialect.delimiter
        except csv.Error:
            # Fallback: default to comma
            delimiter = ','
    
    # Read the CSV file using the detected delimiter
    df = pd.read_csv(feature_fp, sep=delimiter)
    
    # Initialize a blank depth map for the new image size with zeros
    sparse_depth_map = np.full((new_height, new_width), 0.0, dtype=np.float32)
    
    # Calculate scaling factors
    scale_y = new_height / original_height
    scale_x = new_width / original_width
    
    # Iterate through the DataFrame and populate the depth map with scaled coordinates
    for index, row in df.iterrows():
        pixel_row = int(row['row'] * scale_y)
        pixel_col = int(row['column'] * scale_x)
        depth_value = float(row['depth'])
        
        # Ensure the scaled coordinates are within the bounds of the new image size
        if 0 <= pixel_row < new_height and 0 <= pixel_col < new_width:
            # Only assign if the pixel has not been set before (i.e., value is still 0)
            if sparse_depth_map[pixel_row, pixel_col] == 0.0:
                if not lizard:
                    sparse_depth_map[pixel_row, pixel_col] = depth_value
                else:
                    sparse_depth_map[pixel_row, pixel_col] = depth_value / 1.44
                    
    return sparse_depth_map


def evaluate(dataset_path, depth_predictor, nsamples, sml_model_path):
    # sml_model_path = '/home/jay/SML_log/20240912-103625/sml_model-500.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # ranges for VOID
    min_depth, max_depth = 0.1, 10.0
    min_pred, max_pred = 0.1, 25.0

    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    # with open(f"{dataset_path}/void_{nsamples}/test_image.txt") as f: 
    #     test_image_list = [line.rstrip() for line in f]
    test_image_list = []
    ground_truth_list = []
    depth_prior_list = []

    # Open and read the CSV file
    #'/home/jay/ZoeDepth/train_test_inputs/lizard/lizard_image_depth_paths_subsampled_depth.txt'
    file_path = f"{dataset_path}/flsea_preserved_testing_subsampled_10.txt"
    # file_path = f"{dataset_path}/lizard_image_depth_paths_subsampled_depth.txt"
    with open(file_path, 'r') as f:
        sample = f.read(1024)
        f.seek(0)
        if ',' in sample:
            # Process as comma-delimited
            import csv
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                if len(row) < 3:
                    continue  # skip or raise an error if desired
                test_image_list.append(row[0].strip())
                ground_truth_list.append(row[1].strip())
                depth_prior_list.append(row[2].strip())
        else:
            # Process as whitespace-delimited
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = line.split()  # splits on any whitespace
                if len(row) < 3:
                    continue  # skip or raise an error if desired
                test_image_list.append(row[0])
                ground_truth_list.append(row[1])
                depth_prior_list.append(row[2])
    # initialize error aggregators
    avg_error_w_int_depth = metrics.ErrorMetricsAverager()
    avg_error_w_pred = metrics.ErrorMetricsAverager()

    # iterate through inputs list
    for i in tqdm(range(len(test_image_list))):
    # for i in tqdm(range(500)):
        
        # image
        # input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image_fp = test_image_list[i]
        # print(input_image_fp)
        # input_image_fp.replace("/home/auv/FLSea", "/mnt/e/FLSea_latest")
        # input_image_fp.replace("/media/jay/apple/FLSea_latest", "/mnt/e/FLSea_latest")
        # input_image_fp = input_image_fp.replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest")
        # input_image_fp.replace("imgs", "seaErra")
        input_image = utils.read_image(input_image_fp)

        # sparse depth
        # input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth_fp = depth_prior_list[i]
        # input_sparse_depth_fp.replace("/home/auv/FLSea", "/mnt/e/FLSea_latest")
        # input_sparse_depth_fp.replace("/media/jay/apple/FLSea_latest", "/mnt/e/FLSea_latest")
        # input_sparse_depth_fp = input_sparse_depth_fp.replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest")
        input_sparse_depth = generate_feature_map(input_sparse_depth_fp, original_height=240, original_width=320, new_height=608, new_width=968, lizard = False)
        # print(len(input_sparse_depth>0))
        # input_sparse_depth = generate_feature_map(input_sparse_depth_fp, lizard = False)
        input_sparse_depth[input_sparse_depth <= 0] = 0.0
        # input_sparse_depth = filter_lower_33_percent(input_sparse_depth)




        

        input_sparse_depth_valid = (input_sparse_depth < max_pred) * (input_sparse_depth > min_pred)
        if np.sum(input_sparse_depth_valid) <= 4:
            print("Not enough prior")
            continue

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
        # target_depth_fp = target_depth_fp.replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest")
        # target_depth = np.array(Image.open(target_depth_fp).resize((640, 480)), dtype=np.float32)
        target_depth = np.array(Image.open(target_depth_fp), dtype=np.float32)
        target_depth[target_depth <= 0] = 0.0
        input_target_depth_valid = (target_depth < max_depth) * (target_depth > min_depth)
        if np.sum(input_target_depth_valid) <= 0:
            print("Not enough gt")
            continue
        # print(f"maximum of depth map is {np.max(target_depth)}")

        # target depth valid/mask
        mask = (target_depth < max_depth)
        if min_depth is not None:
            mask *= (target_depth > min_depth)
        target_depth[~mask] = np.inf  # set invalid depth
        target_depth = 1.0 / target_depth

        # if np.shape(target_depth)[0] != 399:
        #     print("GT NOT GOOD SHAPE")
        #     continue
        # run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device, target_height = np.shape(target_depth)[0], target_width = np.shape(target_depth)[1])

        # # compute error metrics using intermediate (globally aligned) depth
        # error_w_int_depth = metrics.ErrorMetrics()
        # error_w_int_depth.compute(
        #     estimate = output["ga_depth"], 
        #     target = target_depth, 
        #     valid = mask.astype(bool),
        # )

        # compute error metrics using SML output depth
        # print(np.shape(output["sml_depth"]))
        # print(np.shape(target_depth))
        # print(np.shape(mask))
        # print()
        error_w_pred = metrics.ErrorMetrics()
        error_w_pred.compute(
            estimate = output["sml_depth"], 
            target = target_depth, 
            valid = mask.astype(bool),
        )

        # accumulate error metrics
        # avg_error_w_int_depth.accumulate(error_w_int_depth)
        avg_error_w_pred.accumulate(error_w_pred)


    # compute average error metrics
    # print("Averaging metrics for globally-aligned depth over {} samples".format(
    #     avg_error_w_int_depth.total_count
    # ))
    # avg_error_w_int_depth.average()

    print("Averaging metrics for SML-aligned depth over {} samples".format(
        avg_error_w_pred.total_count
    ))
    avg_error_w_pred.average()

    from prettytable import PrettyTable
    # summary_tb = PrettyTable()
    # summary_tb.field_names = ["metric", "GA Only", "GA+SML"]

    # summary_tb.add_row(["RMSE", f"{avg_error_w_int_depth.rmse_avg:7.2f}", f"{avg_error_w_pred.rmse_avg:7.2f}"])
    # summary_tb.add_row(["RMSE(SILog)", f"{avg_error_w_int_depth.rmse_silog_avg:7.2f}", f"{avg_error_w_pred.rmse_silog_avg:7.2f}"])
    # summary_tb.add_row(["MAE", f"{avg_error_w_int_depth.mae_avg:7.2f}", f"{avg_error_w_pred.mae_avg:7.2f}"])
    # summary_tb.add_row(["AbsRel", f"{avg_error_w_int_depth.absrel_avg:8.3f}", f"{avg_error_w_pred.absrel_avg:8.3f}"])
    # summary_tb.add_row(["iRMSE", f"{avg_error_w_int_depth.inv_rmse_avg:7.2f}", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    # summary_tb.add_row(["iMAE", f"{avg_error_w_int_depth.inv_mae_avg:7.2f}", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    # summary_tb.add_row(["iAbsRel", f"{avg_error_w_int_depth.inv_absrel_avg:8.3f}", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])

    summary_tb = PrettyTable()
    summary_tb.field_names = ["metric", "GA+SML"]

    summary_tb.add_row(["RMSE", f"{avg_error_w_pred.rmse_avg:7.2f}"])
    summary_tb.add_row(["RMSE(SILog)", f"{avg_error_w_pred.rmse_silog_avg:7.2f}"])
    summary_tb.add_row(["MAE", f"{avg_error_w_pred.mae_avg:7.2f}"])
    summary_tb.add_row(["AbsRel", f"{avg_error_w_pred.absrel_avg:8.3f}"])
    summary_tb.add_row(["iRMSE", f"{avg_error_w_pred.inv_rmse_avg:7.2f}"])
    summary_tb.add_row(["iMAE", f"{avg_error_w_pred.inv_mae_avg:7.2f}"])
    summary_tb.add_row(["iAbsRel", f"{avg_error_w_pred.inv_absrel_avg:8.3f}"])
    
    print(summary_tb)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path', type=str, default='/path/to/void_release/',
                        help='Path to VOID release dataset.')
    parser.add_argument('-dp', '--depth-predictor', type=str, default='midas_small', 
                        help='Name of depth predictor to use in pipeline.')
    parser.add_argument('-ns', '--nsamples', type=int, default=150, 
                        help='Number of sparse metric depth samples available.')
    parser.add_argument('-sm', '--sml-model-path', type=str, default='', 
                        help='Path to trained SML model weights.')

    args = parser.parse_args()
    print(args)
    
    evaluate(
        args.dataset_path,
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path,
    )