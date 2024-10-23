import os

# Specify the directory you want to read
folder_path = '/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/relative_vits'
output_file = '/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/relative_vits.txt'

# Open the output file to write the paths
with open(output_file, 'w') as file:
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            # Create the full file path
            file_path = os.path.join(root, name)
            # Write the file path to the text file
            file.write(file_path + '\n')

print(f"File paths saved to {output_file}")
