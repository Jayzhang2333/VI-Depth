import os
import argparse
import time

import torch
import imageio
import numpy as np

from tqdm import tqdm
from PIL import Image

import modules.midas.utils as utils
import pipeline

def evaluate(dataset_path, depth_predictor, nsamples, sml_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # ranges for VOID
    min_depth, max_depth = 0.2, 5.0
    min_pred, max_pred = 0.1, 8.0

    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    with open(f"{dataset_path}/void_{nsamples}/test_image.txt") as f: 
        test_image_list = [line.rstrip() for line in f]
    
    # Start timing
    start_time = time.time()

    # iterate through inputs list
    for i in tqdm(range(len(test_image_list))):
        
        # image
        input_image_fp = os.path.join(dataset_path, test_image_list[i])
        input_image = utils.read_image(input_image_fp)

        # sparse depth
        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
        input_sparse_depth[input_sparse_depth <= 0] = 0.0

        # sparse depth validity map
        validity_map_fp = input_image_fp.replace("image", "validity_map")
        validity_map = np.array(Image.open(validity_map_fp), dtype=np.float32)
        assert(np.all(np.unique(validity_map) == [0, 256]))
        validity_map[validity_map > 0] = 1

        # run pipeline
        output = method.run(input_image, input_sparse_depth, validity_map, device)

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    # Calculate FPS
    num_images = len(test_image_list)
    fps = num_images / total_time
    print(f"Processed {num_images} images in {total_time:.2f} seconds. FPS: {fps:.2f}")

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
