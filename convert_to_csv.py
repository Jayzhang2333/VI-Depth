import pandas as pd

# Load the text file
input_file = '/home/jay/shortcuts/datasets/nyu_depth_v2/test/test/removed_bathroom_nyu_extract_test_sparse_depth.txt'
output_file = '/home/jay/shortcuts/datasets/nyu_depth_v2/test/test/removed_bathroom_nyu_extract_test_sparse_depth.csv'

# Read the text file into a DataFrame
data = pd.read_csv(input_file, sep=" ", header=None, names=["RGB Path", "Depth Path", "Focal Value", "CSV Path"])

# Save the DataFrame to a CSV file
data.to_csv(output_file, index=False)

print(f"File has been successfully converted to CSV and saved as {output_file}")
