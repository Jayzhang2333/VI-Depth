import torch
import numpy as np

from modules.midas.midas_net_custom import MidasNet_small_videpth
from modules.estimator import LeastSquaresEstimator
from modules.interpolator import Interpolator2D

import modules.midas.transforms as transforms
import modules.midas.utils as utils
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import matplotlib.pyplot as plt

def show_images(tensor_images):
    tensor_images = tensor_images.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_images = np.transpose(tensor_images, (0, 2, 3, 1))  # Change from CHW to HWC
    
    # Display the images
    batch_size = tensor_images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))  # Adjust number of subplots as needed
    if batch_size == 1:  # Handle case where batch size is 1
        axes = [axes]
    
    for idx in range(batch_size):
        axes[idx].imshow(tensor_images[idx])  # Assuming images are normalized [0, 1]
        axes[idx].axis('off')
    
    plt.show()



class VIDepth(object):
    def __init__(self, depth_predictor, nsamples, sml_model_path, 
                min_pred, max_pred, min_depth, max_depth, device):

        # # get transforms
        # if depth_predictor == "depth_anything_v2_small":
        #     model_transforms = transforms.get_transforms("dpt_beit_large_512", "void", str(nsamples))
        # else:
        model_transforms = transforms.get_transforms(depth_predictor, "void", str(nsamples))
        self.depth_model_transform = model_transforms["depth_model"]
        self.ScaleMapLearner_transform = model_transforms["sml_model"]

        # define depth model
        if depth_predictor == "dpt_beit_large_512":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512")
        elif depth_predictor == "dpt_swin2_large_384":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_SwinV2_L_384")
        elif depth_predictor == "dpt_large":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        elif depth_predictor == "dpt_hybrid":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        elif depth_predictor == "dpt_swin2_tiny_256":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_SwinV2_T_256")
        elif depth_predictor == "dpt_levit_224":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "DPT_LeViT_224")
        elif depth_predictor == "midas_small":
            self.DepthModel = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        elif depth_predictor == "depth_anything_v2_small":
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
            self.DepthModel = DepthAnythingV2(**model_configs['vits'])
            self.DepthModel.load_state_dict(torch.load('/home/jay/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
            # self.DepthModel.load_state_dict(torch.load('/home/jay/DAV2_log/20240914-192234/model-4460.pth', map_location='cpu'))
            self.DepthModel = self.DepthModel.to(device).eval()
        elif depth_predictor == "depth_anything_v2_base":
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
            self.DepthModel = DepthAnythingV2(**model_configs['vitb'])
            self.DepthModel.load_state_dict(torch.load('Depth_Anything_V2/checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
            # self.DepthModel.load_state_dict(torch.load('/home/jay/Depth-Anything-V2/checkpoints/model-1500.pth', map_location='cpu'))
            self.DepthModel = self.DepthModel.to(device).eval()
        elif depth_predictor == "depth_anything_v2_large":
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
            self.DepthModel = DepthAnythingV2(**model_configs['vitl'])
            self.DepthModel.load_state_dict(torch.load('Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
            # self.DepthModel.load_state_dict(torch.load('/home/jay/Depth-Anything-V2/checkpoints/model-1500.pth', map_location='cpu'))
            self.DepthModel = self.DepthModel.to(device).eval()
        else:
            self.DepthModel = None

        # define SML model
        self.ScaleMapLearner = MidasNet_small_videpth(
            path=sml_model_path,
            min_pred=min_pred,
            max_pred=max_pred,
        )

        # depth prediction ranges
        self.min_pred, self.max_pred = min_pred, max_pred

        # depth evaluation ranges
        self.min_depth, self.max_depth = min_depth, max_depth

        # eval mode
        self.DepthModel.eval()
        self.DepthModel.to(device)

        # eval mode
        self.ScaleMapLearner.eval()
        self.ScaleMapLearner.to(device)


    def run(self, input_image, input_sparse_depth, validity_map, device, target_height = None, target_width = None):

        # if input_height is None and input_width is None:
            
        input_height, input_width = np.shape(input_image)[0], np.shape(input_image)[1]
        # print(input_height)
        # print(input_width)
        # print()

        
        sample = {"image" : input_image}
        sample = self.depth_model_transform(sample)
        im = sample["image"].to(device)

        # print(np.max(input_sparse_depth))
        # print(np.min(input_sparse_depth[input_sparse_depth>0]))
        input_sparse_depth_valid = (input_sparse_depth < self.max_pred) * (input_sparse_depth > self.min_pred)
        # print(len(input_sparse_depth_valid==1))
        if validity_map is not None:
            input_sparse_depth_valid *= validity_map.astype(np.bool)

        input_sparse_depth_valid = input_sparse_depth_valid.astype(bool)
        input_sparse_depth[~input_sparse_depth_valid] = np.inf # set invalid depth
        input_sparse_depth = 1.0 / input_sparse_depth # 1 / depth because it represents scale

        # run depth model
        with torch.no_grad():
            depth_pred = self.DepthModel.forward(im.unsqueeze(0))
            depth_pred = (
                torch.nn.functional.interpolate(
                    depth_pred.unsqueeze(1),
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        

        # plt.imshow(depth_pred, cmap='viridis')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("Relative Depth")
        # plt.show()

        # global scale and shift alignment
        # print(np.shape(input_sparse_depth))
        GlobalAlignment = LeastSquaresEstimator(
            estimate=depth_pred,
            target=input_sparse_depth,
            valid=input_sparse_depth_valid
        )
        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        GlobalAlignment.clamp_min_max(clamp_min=self.min_pred, clamp_max=self.max_pred)
        int_depth = GlobalAlignment.output.astype(np.float32)

        # plt.imshow(1.0/int_depth, cmap='inferno')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("Int Depth Visualization")
        # plt.show()

        # interpolation of scale map
        assert (np.sum(input_sparse_depth_valid) >= 4), "not enough valid sparse points"
        ScaleMapInterpolator = Interpolator2D(
            pred_inv = int_depth,
            sparse_depth_inv = input_sparse_depth,
            valid = input_sparse_depth_valid,
        )
        ScaleMapInterpolator.generate_interpolated_scale_map(
            interpolate_method='linear', 
            fill_corners=False
        )
        int_scales = ScaleMapInterpolator.interpolated_scale_map.astype(np.float32)
        int_scales = utils.normalize_unit_range(int_scales)
        # plt.imshow(int_scales, cmap='inferno')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("Int Scale Visualization")
        # plt.show()

        sample = {"image" : input_image, "int_depth" : int_depth, "int_scales" : int_scales, "int_depth_no_tf" : int_depth}
        sample = self.ScaleMapLearner_transform(sample)
        # print(sample["int_scales"].shape)
        # show_images(sample["int_scales"].unsqueeze(0))
        x = torch.cat([sample["int_depth"], sample["int_scales"]], 0)
        x = x.to(device)
        d = sample["int_depth_no_tf"].to(device)

        # run SML model
        with torch.no_grad():
            sml_pred, sml_scales = self.ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
            sml_pred = (
                torch.nn.functional.interpolate(
                    sml_pred,
                    size=(target_height, target_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()

            
            )
        # print(np.shape(sml_pred))

            # sml_pred = sml_pred.squeeze().cpu().numpy()

        # infite_mask = np.where(int_depth == 1/self.max_pred)
        # sml_pred[infite_mask] = 1/self.max_pred

        # input_sparse_depth = 1.0 / input_sparse_depth # 1 / depth because it represents scale
        # input_sparse_depth[~input_sparse_depth_valid] = 0

        # non_zero_coords = np.argwhere(input_sparse_depth != 0)


        # fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        # # Plot the input image
        # # axes[0].imshow(input_image, cmap='gray')
        # # axes[0].set_title("Input Image")

        

        # # Overlay the 2D array on the input image
        # non_zero_coords = np.argwhere(input_sparse_depth != 0)
        # axes[0].imshow(input_image, cmap='gray')  # Display the original image as background

        # # Normalize the values for the colorbar
        # norm = Normalize(vmin=np.min(input_sparse_depth[input_sparse_depth > 0]), vmax=np.max(input_sparse_depth))
        # cmap = plt.cm.inferno
        # sm = ScalarMappable(cmap=cmap, norm=norm)

        # # Overlay the non-zero values with inferno colormap
        # for coord in non_zero_coords:
        #     y, x = coord
        #     value = input_sparse_depth[y, x]  # Value at the non-zero location
        #     color = cmap(norm(value))  # Normalize and get color
        #     axes[0].scatter(x, y, color=color, s=10)  # Adjust size as needed

        # axes[0].set_title("Sparse Prior Overlay", fontweight="bold", fontsize=14)

        # # Add the colorbar for the overlay
        # cbar = fig.colorbar(sm, ax=axes[0], label='Array Value')
        # cbar.set_label('Non-Zero Value', rotation=270, labelpad=15)

        # cbar.ax.tick_params(labelsize=12, labelcolor='black', width=1.5)  # Set font size and line width
        # for label in cbar.ax.get_yticklabels():
        #     label.set_fontweight("bold")  # Make tick labels bold

        # # Plot the global alignment with inferno colormap
        # im1 = axes[1].imshow(1/int_depth, cmap='inferno')
        # axes[1].set_title("Global Alignment", fontweight="bold", fontsize=14)
        # # fig.colorbar(im1, ax=axes[1], label='Depth')
        # cbar = fig.colorbar(im1, ax=axes[1], label='Depth')
        # cbar.set_label('Non-Zero Value', rotation=270, labelpad=15)

        # cbar.ax.tick_params(labelsize=12, labelcolor='black', width=1.5)  # Set font size and line width
        # for label in cbar.ax.get_yticklabels():
        #     label.set_fontweight("bold")  # Make tick labels bold


        # im2 = axes[2].imshow(depth_pred, cmap='inferno')
        # axes[2].set_title("Relative Depth", fontweight="bold", fontsize=14)
        # # fig.colorbar(im1, ax=axes[1], label='Depth')
        # cbar = fig.colorbar(im2, ax=axes[2], label='Depth')
        # cbar.set_label('Non-Zero Value', rotation=270, labelpad=15)

        # cbar.ax.tick_params(labelsize=12, labelcolor='black', width=1.5)  # Set font size and line width
        # for label in cbar.ax.get_yticklabels():
        #     label.set_fontweight("bold")  # Make tick labels bold

        # plt.tight_layout()
        # plt.show()

        output = {
            "ga_depth"  : int_depth, 
            "sml_depth" : sml_pred, 
        }
        # plt.imshow(1/int_depth, cmap='viridis')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("GA")
        # plt.show()

        # plt.imshow(1/sml_pred, cmap='viridis')  # Use 'viridis' or any other colormap you prefer
        # plt.colorbar(label='Depth')  # Optional: Adds a color bar for reference
        # plt.title("SML")
        # plt.show()
        return output
    

