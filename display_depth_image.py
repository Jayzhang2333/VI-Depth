import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = '/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/relative_negative_1/16233155481815680_SeaErra_abs_depth.tif'
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with unchanged bit depth

# img_path_2 = '/home/jay/Downloads/16233155481815680.png'
# img_2 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image with unchanged bit depth
print(np.shape(img))

# Check bit depth
# if img.dtype == np.uint8:
#     print("Image is 8-bit.")
#     bit_depth = 8
# elif img.dtype == np.uint16:
#     print("Image is 16-bit.")
#     bit_depth = 16
# else:
#     raise ValueError("Image bit depth not supported")

# Normalize the image to the range [0, 1]
# img_normalized = img.astype(np.float32) / (2**bit_depth - 1)

# Display the normalized image
plt.imshow(img, cmap='gray')
plt.colorbar()
# plt.title(f"Normalized Image ({bit_depth}-bit)")
plt.show()
# import cv2
# import numpy as np

# # Load first image with unchanged bit depth
# img_path = '/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/relative_vits/16233155482756245.png'
# img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# # Load second image with unchanged bit depth
# img_path_2 = '/home/jay/Downloads/16233155482756245.png'
# img_2 = cv2.imread(img_path_2, cv2.IMREAD_UNCHANGED)  # Corrected the path here

# # Check if images are loaded successfully
# if img is None or img_2 is None:
#     print("One or both images could not be loaded.")
# else:
#     # Convert both images to grayscale if needed (if they aren't already)
#     if len(img.shape) == 3:  # Assuming color image, convert to grayscale
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if len(img_2.shape) == 3:  # Assuming color image, convert to grayscale
#         img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

#     # Compare disparity (absolute difference between the images)
#     disparity = np.abs(img.astype(np.float32) - img_2.astype(np.float32))

#     # Check if max disparity is zero to avoid division by zero
#     max_disparity = np.max(disparity)

#     if max_disparity == 0:
#         print("The images are identical. No disparity.")
#     else:
#         # Optionally, display the disparity normalized for display
#         normalized_disparity = disparity / max_disparity
#         cv2.imshow('Disparity', normalized_disparity)  # Display disparity
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     # Print some basic stats
#     print(f"Disparity mean: {np.mean(disparity)}")
#     print(f"Disparity max: {np.max(disparity)}")
#     print(f"Disparity min: {np.min(disparity)}")
