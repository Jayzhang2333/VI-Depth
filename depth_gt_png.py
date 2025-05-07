import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def convert_tif_to_png_with_inferno(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all .tif files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            # Construct full file path
            file_path = os.path.join(input_folder, filename)
            
            # Open the .tif image and convert it to a NumPy array
            with Image.open(file_path) as img:
                depth_array = np.array(img)
            
            # mask = np.where(depth_array==0)
            # depth_array[mask] = 12.0
            depth_array[depth_array==0] = 10
            depth_array[depth_array>10] = 10
            
            # Construct output file path
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            
            # Save the depth array as a PNG with the inferno colormap
            plt.imsave(output_path, depth_array, cmap='inferno_r')
            print(f"Converted {filename} to {output_filename} with inferno colormap")

# Example usage
input_folder = '/home/jay/VI-Depth/input/gt'  # Replace with your input folder path
output_folder = '/home/jay/vi_depth_presentation_output/gt'  # Replace with your output folder path
convert_tif_to_png_with_inferno(input_folder, output_folder)
