import numpy as np
import matplotlib.pyplot as plt

def compute_scale_and_shift_ls(prediction, target, mask):
    # tuple specifying with axes to sum
    sum_axes = (0, 1)

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = np.sum(mask * prediction * prediction, sum_axes)
    a_01 = np.sum(mask * prediction, sum_axes)
    a_11 = np.sum(mask, sum_axes)

    # right hand side: b = [b_0, b_1]
    b_0 = np.sum(mask * prediction * target, sum_axes)
    b_1 = np.sum(mask * target, sum_axes)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = np.zeros_like(b_0)
    x_1 = np.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # print(det)
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class LeastSquaresEstimator(object):
    def __init__(self, estimate, target, valid):
        self.estimate = estimate
        self.target = target
        self.valid = valid

        # to be computed
        self.scale = 1.0
        self.shift = 0.0
        self.output = None

    def compute_scale_and_shift(self):
        self.scale, self.shift = compute_scale_and_shift_ls(self.estimate, self.target, self.valid)
        # print(f"shift is {self.shift}")
        # print(f"scale is {self.scale}")

    def apply_scale_and_shift(self):
        self.output = self.estimate * self.scale + self.shift
        self.output2 = self.estimate * self.scale + self.shift

        # fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed

        # # Plot the first image on the first subplot
        # im0 = axes[0].imshow(self.estimate, cmap='inferno')
        # axes[0].set_title("before scale and shift")
        # fig.colorbar(im0, ax=axes[0], label='Depth')  # Adds a color bar to the first subplot

        

        # im1 = axes[1].imshow(1.0/self.output, cmap='inferno')
        # axes[1].set_title("after scale")
        # fig.colorbar(im1, ax=axes[1], label='Depth')  # Adds a color bar to the first subplot

        
        # Plot the second image on the second subplot
        # im2 = axes[0].imshow(1.0/self.output, cmap='inferno')  # Replace 'second_image' with your second image
        # axes[0].set_title("Global Alignment Result")
        # fig.colorbar(im2, ax=axes[0], label='Depth')  # Adds a color bar to the second subplot

        # im3 = axes[1].imshow(1.0/self.target, cmap='inferno')
        # axes[1].set_title("Sparse Prior")
        # fig.colorbar(im3, ax=axes[1], label='Depth')  # Adds a color bar to the first subplot

        # # Optional: Adjust layout to prevent overlap
        # plt.tight_layout()

        # # Display the figure
        # plt.show()

    def clamp_min_max(self, clamp_min=None, clamp_max=None):
        # Ensure the output array is float32
        self.output = self.output.astype(np.float32)
        
        if clamp_min is not None:
            # Convert clamp_min to float32
            clamp_min = np.float32(clamp_min)
            if clamp_min > 0:
                clamp_min_inv = np.float32(1.0) / clamp_min
                # Clamp values above clamp_min_inv
                mask = self.output > clamp_min_inv
                self.output[mask] = clamp_min_inv
                assert np.max(self.output) <= clamp_min_inv
            # If clamp_min is 0 or negative, skip to avoid division by zero
                
        if clamp_max is not None:
            # Convert clamp_max to float32
            clamp_max = np.float32(clamp_max)
            clamp_max_inv = np.float32(1.0) / clamp_max
            # Clamp values below clamp_max_inv
            mask = self.output < clamp_max_inv
            self.output[mask] = clamp_max_inv
            assert np.min(self.output) >= clamp_max_inv
            # check for nonzero range
            # assert np.min(self.output) != np.max(self.output)