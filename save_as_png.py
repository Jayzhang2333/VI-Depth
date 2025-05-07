from PIL import Image
import os

def convert_tiff_to_png(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            # Construct full file path
            file_path = os.path.join(input_folder, filename)
            
            # Open the TIFF image
            with Image.open(file_path) as img:
                # Convert to PNG and save
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)
                img.save(output_path, 'PNG')
                print(f"Converted {filename} to {output_filename}")

# Example usage
input_folder = '/home/jay/VI-Depth/input/image'  # Replace with your input folder path
output_folder = '/home/jay/vi_depth_presentation_output/images'  # Replace with your output folder path
convert_tiff_to_png(input_folder, output_folder)
