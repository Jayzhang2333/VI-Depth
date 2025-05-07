import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

ground_truth_depth_path = "input/gt/16315129507958105_SeaErra_abs_depth.tif"
ground_truth_depth = np.array(Image.open(ground_truth_depth_path), dtype=np.float32)

plt.imshow(ground_truth_depth, cmap='viridis')
plt.colorbar()
plt.title('Depth Map from TIF')
plt.show()