import os
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate a turbid effect based on distance
def apply_turbid_effect(image, depth_map, max_depth):
    image = np.array(image) / 255.0  # Normalize image
    depth_map = np.array(depth_map) / np.max(depth_map)  # Normalize depth map
    
    # Generate turbid effect
    turbid_factor = 1 - (depth_map / max_depth)
    turbid_effect = np.clip(turbid_factor[..., None] * image, 0, 1)  # Apply turbid effect based on distance

    # Convert back to image format
    turbid_image = Image.fromarray((turbid_effect * 255).astype(np.uint8))
    return turbid_image

# Function to process the folders
def process_folders(image_folder, depth_folder, output_folder, max_depth=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.tiff')]
    depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.tif')]

    for image_file, depth_file in zip(image_files, depth_files):
        image_path = os.path.join(image_folder, image_file)
        depth_path = os.path.join(depth_folder, depth_file)
        
        # Open image and corresponding depth map
        image = Image.open(image_path).convert('RGB')
        depth_map = Image.open(depth_path).convert('L')

        # Apply the turbid effect based on depth
        turbid_image = apply_turbid_effect(image, depth_map, max_depth)

        # Save the turbid image
        output_path = os.path.join(output_folder, f'turbid_{image_file}')
        turbid_image.save(output_path)

    return f"Processed images saved in {output_folder}"


# Example usage
image_folder = '/media/jay/apple/FLSea_latest/archive/canyons/u_canyon/u_canyon/imgs_test'
depth_folder = '/media/jay/apple/FLSea_latest/archive/canyons/u_canyon/u_canyon/depth_test'
output_folder = '/media/jay/apple/FLSea_latest/archive/canyons/u_canyon/u_canyon/turbid_u_canyon'

process_folders(image_folder, depth_folder, output_folder)
