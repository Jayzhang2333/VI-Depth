import data.data_utils as data_utils

test = data_utils.read_paths("/home/jay/Downloads/void_release/void_150/train_image.txt")
print(len(test))
print(test[:10])