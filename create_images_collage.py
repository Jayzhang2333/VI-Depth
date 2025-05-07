from PIL import Image
import matplotlib.pyplot as plt
import os

def read_images_from_folder(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
    images = [Image.open(os.path.join(folder_path, f)).convert("RGB") for f in image_files]
    return images

def display_and_save_images(original_folder, gt_folder, ga_prediction_folder, sml_prediction_folder, ga_error_folder, sml_error_folder, output_path):
    folders = [original_folder, gt_folder, ga_prediction_folder, sml_prediction_folder, ga_error_folder, sml_error_folder]
    labels = ["Img", "GT", "GA", "SML", "GA Error", "SML Error"]

    # Read images from each folder
    images_list = [read_images_from_folder(folder) for folder in folders]
    
    # Determine the number of images (assumes all folders have the same number of images)
    num_images = min(len(images) for images in images_list)
    
    fig, axes = plt.subplots(6, num_images-4, figsize=(15, 11))
    
    # Set all padding to zero
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

    # Indices to skip
    skip_indices = {1, 4, 8, 9}

    for row in range(6):
        col_counter = 0
        for col in range(num_images):
            if col in skip_indices:
                continue  # Skip specified indices
            
            axes[row, col_counter].imshow(images_list[row][col])
            axes[row, col_counter].axis('off')
            axes[row, col_counter].set_aspect('auto')  # Ensure images do not maintain aspect ratio constraints
            col_counter += 1
        
        # Add label to the left of each row
        axes[row, 0].text(-0.1, 0.5, labels[row], va='center', ha='right', rotation=90, fontsize=12, transform=axes[row, 0].transAxes)

    # Save the figure as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()

# Example usage
base_folder = 'depth_anything_masking'

# Example usage
original_folder = '/home/jay/vi_depth_presentation_output/images'
gt_folder = '/home/jay/vi_depth_presentation_output/gt'
ga_prediction_folder = f'/home/jay/vi_depth_presentation_output/{base_folder}/ga_depth'
sml_prediction_folder = f'/home/jay/vi_depth_presentation_output/{base_folder}/sml_depth'
ga_error_folder = f'/home/jay/vi_depth_presentation_output/{base_folder}/ga_error_map'
sml_error_folder = f'/home/jay/vi_depth_presentation_output/{base_folder}/error_map'
output_path = f'/home/jay/vi_depth_presentation_output/{base_folder}/combined_image_grid.png'

display_and_save_images(original_folder, gt_folder, ga_prediction_folder, sml_prediction_folder, ga_error_folder, sml_error_folder, output_path)