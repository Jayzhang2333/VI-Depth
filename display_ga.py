import numpy as np
import matplotlib.pyplot as plt

# Load the numpy array (example: ga_depth.npy)
ga_depth = np.load('/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/ga_result/16233155481815680.npy')

# Plot the numpy array
plt.imshow(ga_depth, cmap='viridis')
plt.colorbar()  # Add color bar to the plot
plt.title("GA Depth Visualization")
plt.show()