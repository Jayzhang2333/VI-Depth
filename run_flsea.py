import os
import argparse
import glob

import torch
import numpy as np
import pandas as pd

from PIL import Image

import modules.midas.utils as utils

import pipeline
import matplotlib.pyplot as plt


def generate_feature_map(feature_fp, original_height=240, original_width=320, new_height=608, new_width=968):
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

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)


def load_sparse_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth


def run(depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, 
        input_path, output_path, save_output):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # instantiate method
    method = pipeline.VIDepth(
        depth_predictor, nsamples, sml_model_path, 
        min_pred, max_pred, min_depth, max_depth, device
    )

    # get inputs
    img_names = glob.glob(os.path.join(input_path, "image", "*"))
    num_images = len(img_names)

    # create output folders
    if save_output:
        os.makedirs(os.path.join(output_path, 'ga_depth'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'sml_depth'), exist_ok=True)

    for ind, input_image_fp in enumerate(img_names):
        if os.path.isdir(input_image_fp):
            continue

        print("  processing {} ({}/{})".format(input_image_fp, ind + 1, num_images))

        input_image = load_input_image(input_image_fp)

        input_sparse_depth_fp = input_image_fp.replace("image", "sparse_depth")
        input_sparse_depth_fp = input_sparse_depth_fp.replace(".tiff", "_features.csv")
        input_sparse_depth = generate_feature_map(input_sparse_depth_fp)


        input_depth_path = input_image_fp.replace("image", "gt")
        input_depth_path = input_depth_path.replace(".tiff", "_SeaErra_abs_depth.tif")
        # input_depth = load_input_image(input_depth_path)
        input_depth = np.array(Image.open(input_depth_path))
        # gt_mask = np.where(input_depth >0.0)
        gt_mask = input_depth > 0.0

        # plt.imshow(input_sparse_depth, cmap='inferno')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("prior point map")
        # plt.show()


        

        # values in the [min_depth, max_depth] range are considered valid;
        # an additional validity map may be specified
        validity_map = None

        # run method
        output = method.run(input_image, input_sparse_depth, validity_map, device)

        # plt.imshow(input_depth, cmap='inferno')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("input depth")
        # plt.show()

        # masked_difference = np.where(gt_mask, np.abs(output["sml_depth"] - input_depth), 0)
        error_map = np.abs(1/output["sml_depth"] - input_depth)

        # Mask the error map to only show areas where ground truth depth is valid (> 0)
        masked_error_map = np.where(input_depth > 0, error_map, 0)

        # masked_difference = np.where(gt_mask, np.abs(output["sml_depth"] - input_depth), 0)
        ga_error_map = np.abs(1/output["ga_depth"] - input_depth)

        # Mask the error map to only show areas where ground truth depth is valid (> 0)
        ga_masked_error_map = np.where(input_depth > 0, ga_error_map, 0)

        # plt.imshow(1.0/output["ga_depth"], cmap='inferno')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("prior point map")
        # plt.show()
        
        # plt.imshow(1.0/output["sml_depth"], cmap='inferno')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("sml map")
        # plt.show()

        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed

        # # Plot the first image on the first subplot
        # # im1 = axes[0].imshow(1/output["ga_depth"], cmap='viridis')
        # # axes[0].set_title("GA Visualization")
        # # fig.colorbar(im1, ax=axes[0], label='Depth')  # Adds a color bar to the first subplot

        # im1 = axes[0].imshow(1/output["ga_depth"], cmap='viridis')
        # axes[0].set_title("Relative")
        # fig.colorbar(im1, ax=axes[0], label='Depth')  # Adds a color bar to the first subplot

        # # Plot the second image on the second subplot
        # im2 = axes[1].imshow(1/output["ga_depth"], cmap='viridis')  # Replace 'second_image' with your second image
        # axes[1].set_title("GA Visualization")
        # fig.colorbar(im2, ax=axes[1], label='Depth')  # Adds a color bar to the second subplot

        # # Optional: Adjust layout to prevent overlap
        # # plt.tight_layout()

        # # Display the figure
        # # plt.show()



        if save_output:
            basename = os.path.splitext(os.path.basename(input_image_fp))[0]

            # saving depth map after global alignment
            utils.write_depth(
                os.path.join(output_path, 'ga_depth', basename), 
                output["ga_depth"], bits=2
            )

            # saving depth map after local alignment with SML
            utils.write_depth(
                os.path.join(output_path, 'sml_depth', basename), 
                output["sml_depth"], bits=2
            )

            utils.write_depth(
                os.path.join(output_path, 'error_map', basename), 
                masked_error_map, bits=2
            )

            utils.write_depth(
                os.path.join(output_path, 'ga_error_map', basename), 
                ga_masked_error_map, bits=2
            )

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('-dp', '--depth-predictor', type=str, default='dpt_hybrid', 
                            help='Name of depth predictor to use in pipeline.')
    parser.add_argument('-ns', '--nsamples', type=int, default=150, 
                            help='Number of sparse metric depth samples available.')
    parser.add_argument('-sm', '--sml-model-path', type=str, default='', 
                            help='Path to trained SML model weights.')

    # depth parameters
    parser.add_argument('--min-pred', type=float, default=0.1, 
                            help='Min bound for predicted depth values.')
    parser.add_argument('--max-pred', type=float, default=11.0, 
                            help='Max bound for predicted depth values.')
    parser.add_argument('--min-depth', type=float, default=0.1, 
                            help='Min valid depth when evaluating.')
    parser.add_argument('--max-depth', type=float, default=10.0, 
                            help='Max valid depth when evaluating.')

    # I/O paths
    parser.add_argument('-i', '--input-path', type=str, default='./input', 
                            help='Path to inputs.')
    parser.add_argument('-o', '--output-path', type=str, default='./output', 
                            help='Path to outputs.')
    parser.add_argument('--save-output', dest='save_output', action='store_true', 
                            help='Save output depth map.')
    parser.set_defaults(save_output=False)

    args = parser.parse_args()
    print(args)
    
    run(
        args.depth_predictor, 
        args.nsamples, 
        args.sml_model_path, 
        args.min_pred,
        args.max_pred, 
        args.min_depth, 
        args.max_depth,
        args.input_path,
        args.output_path,
        args.save_output
    )