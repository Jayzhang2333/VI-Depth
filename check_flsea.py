from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the .tiff file
# np.array(Image.open(target_depth_fp), dtype=np.float32)
tiff_image = np.array(Image.open('/media/jay/apple/FLSea_latest/archive/red_sea/coral_table_loop/coral_table_loop/depth/16315955232437742_SeaErra_abs_depth.tif'), dtype=np.float32)

# Convert the image to a NumPy array
# depth_array = np.array(tiff_image)

print(np.shape(tiff_image))

# Display the depth map
plt.imshow(tiff_image)  # 'viridis' is a color map that works well for depth
plt.colorbar(label='Depth (meters)')     # Add color bar with a label indicating depth units
plt.title('Depth Map')
plt.show()
