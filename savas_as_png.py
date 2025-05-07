# import os
# import numpy as np
# import matplotlib.pyplot as plt

# def read_pfm(file):
#     with open(file, "rb") as f:
#         header = f.readline().decode().rstrip()
#         if header == 'PF':
#             color = True
#         elif header == 'Pf':
#             color = False
#         else:
#             raise ValueError("Not a PFM file.")

#         dim_line = f.readline().decode().rstrip()
#         width, height = map(int, dim_line.split())

#         scale = float(f.readline().decode().rstrip())
#         endian = '<' if scale < 0 else '>'
#         scale = abs(scale)

#         data = np.fromfile(f, endian + 'f')
#         data = np.reshape(data, (height, width, 3) if color else (height, width))
#         data = np.flipud(data)
#         return data

# def convert_pfm_to_png(source_dir, dest_dir):
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)

#     for root, dirs, files in os.walk(source_dir):
#         for file in files:
#             if file.endswith('.pfm'):
#                 pfm_path = os.path.join(root, file)
#                 data = read_pfm(pfm_path)

#                 # Prepare the output path
#                 relative_path = os.path.relpath(root, source_dir)
#                 dest_subdir = os.path.join(dest_dir, relative_path)
#                 if not os.path.exists(dest_subdir):
#                     os.makedirs(dest_subdir)

#                 png_filename = os.path.splitext(file)[0] + '.png'
#                 png_path = os.path.join(dest_subdir, png_filename)

#                 # Save the data as a PNG image
#                 if len(data.shape) == 2:  # Grayscale image
#                     plt.imsave(png_path, data, cmap='inferno')
#                 else:  # Color image
#                     plt.imsave(png_path, data)

# # Example usage:
# # Replace '/path/to/source' and '/path/to/destination' with your actual paths
# convert_pfm_to_png('output/error_map', '/home/jay/vi_depth_presentation_output/midas_masking/sml_error')

import os
import numpy as np
import matplotlib.pyplot as plt

def read_pfm(file):
    with open(file, "rb") as f:
        header = f.readline().decode().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_line = f.readline().decode().rstrip()
        width, height = map(int, dim_line.split())

        scale = float(f.readline().decode().rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        data = np.reshape(data, (height, width, 3) if color else (height, width))
        data = np.flipud(data)
        return data

def convert_pfm_to_png(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        # Compute the relative path from the source directory
        relative_path = os.path.relpath(root, source_dir)
        dest_subdir = os.path.join(dest_dir, relative_path)

        # Ensure the destination subdirectory exists
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)

        # Create all subdirectories even if they are empty
        for dir_name in dirs:
            source_subdir = os.path.join(root, dir_name)
            relative_subdir = os.path.relpath(source_subdir, source_dir)
            dest_subdir_path = os.path.join(dest_dir, relative_subdir)
            if not os.path.exists(dest_subdir_path):
                os.makedirs(dest_subdir_path)

        for file in files:
            if file.endswith('.pfm'):
                pfm_path = os.path.join(root, file)
                try:
                    data = read_pfm(pfm_path)
                except ValueError as e:
                    print(f"Skipping file {pfm_path}: {e}")
                    continue

                png_filename = os.path.splitext(file)[0] + '.png'
                png_path = os.path.join(dest_subdir, png_filename)

                # Save the data as a PNG image
                if len(data.shape) == 2:  # Grayscale image
                    plt.imsave(png_path, data, cmap='inferno')
                else:  # Color image
                    plt.imsave(png_path, data)

# Example usage:
# Replace '/path/to/source' with the path to your source directory containing the four folders
# Replace '/path/to/destination' with the path to your desired destination directory
convert_pfm_to_png('/home/jay/VI-Depth/output', '/home/jay/vi_depth_presentation_output/depth_anything_no_masking')
