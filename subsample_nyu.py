import pandas as pd
import os
import random

# Define constants
NUM_SUBSAMPLE_POINTS = 8  # The number of points to subsample
dataset_path = '/home/jay/shortcuts/datasets/nyu_depth_v2/test/test'
dataset_csv_path = "/home/jay/shortcuts/datasets/nyu_depth_v2/test/test/removed_bathroom_nyu_extract_test_sparse_depth.csv"
output_dataset_csv_path = "/home/jay/shortcuts/datasets/nyu_depth_v2/test/test/sub8_removed_bathroom_nyu_extract_test_sparse_depth.csv"

# Read the main dataset CSV file
dataset_df = pd.read_csv(dataset_csv_path, header=None)

# Initialize a list to store updated paths
updated_paths = []

# Loop through each row in the dataset
for index, row in dataset_df.iterrows():
    # Get the original sparse feature CSV path
    sparse_feature_csv_path = row.iloc[-1]
    entire_sparse_feature_csv_path = os.path.join(dataset_path, sparse_feature_csv_path)
    
    # Read the sparse feature CSV file
    feature_df = pd.read_csv(entire_sparse_feature_csv_path, header=0)
    
    # Subsample points if more than required number
    if len(feature_df) > NUM_SUBSAMPLE_POINTS:
        subsampled_df = feature_df.sample(n=NUM_SUBSAMPLE_POINTS, random_state=1)
    else:
        subsampled_df = feature_df  # Use entire data if not enough points
    
    # Construct the path for the new subsampled CSV file
    original_dir = os.path.dirname(entire_sparse_feature_csv_path)
    original_filename = os.path.basename(entire_sparse_feature_csv_path)
    new_filename = f"subsampled_8_{original_filename}"
    subsampled_csv_path = os.path.join(original_dir, new_filename)
    
    # Save the subsampled CSV file
    subsampled_df.to_csv(subsampled_csv_path, index=False)

    relatve_directory = os.path.dirname(sparse_feature_csv_path)
    subsampled_csv_path_to_csv = os.path.join(relatve_directory, new_filename)
    
    # Append the path of the subsampled CSV file to updated paths
    updated_paths.append(subsampled_csv_path_to_csv)

# Update the dataset dataframe with new paths
dataset_df.iloc[:, -1] = updated_paths

# Save the updated dataset CSV
dataset_df.to_csv(output_dataset_csv_path, index=False)
