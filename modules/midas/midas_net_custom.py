"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from torch.nn import functional as F

from .base_model import BaseModel
from .blocks import FeatureFusionBlock_custom, _make_encoder, OutputConv

import numpy as np
import matplotlib.pyplot as plt

def weights_init(m):
    import math
    # initialize from normal (Gaussian) distribution
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class MidasNet_small_videpth(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=64, backbone="efficientnet_lite3", non_negative=False, exportable=True, channels_last=False, align_corners=True,
        blocks={'expand': True}, in_channels=2, regress='r', min_pred=None, max_pred=None):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 64.
            backbone (str, optional): Backbone network for encoder. Defaults to efficientnet_lite3.
        """
        print("Loading weights: ", path)

        super(MidasNet_small_videpth, self).__init__()

        use_pretrained = False if path else True
                
        self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        # for model output
        self.regress = regress
        self.min_pred = min_pred
        self.max_pred = max_pred

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8

        self.first = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.first.apply(weights_init)

        self.pretrained, self.scratch = _make_encoder(self.backbone, features, use_pretrained, groups=self.groups, expand=self.expand, exportable=exportable)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        self.scratch.output_conv = OutputConv(features, self.groups, self.scratch.activation, non_negative)
        
        if path:
            self.load(path)


    def forward(self, x, d):
        """Forward pass.

        Args:
            x (tensor): input data (image)
            d (tensor): unalterated input depth

        Returns:
            tensor: depth
        """
        if self.channels_last==True:
            print("self.channels_last = ", self.channels_last)
            x.contiguous(memory_format=torch.channels_last)

        # print(x.shape)
        layer_0 = self.first(x)
        # print(layer_0.shape)
        layer_1 = self.pretrained.layer1(layer_0)
        # print(layer_1.shape)
        layer_2 = self.pretrained.layer2(layer_1)
        # print(layer_2.shape)
        layer_3 = self.pretrained.layer3(layer_2)
        # print(layer_3.shape)
        layer_4 = self.pretrained.layer4(layer_3)
        # print(layer_4.shape)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        # print(layer_4_rn.shape)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        # print(path_4.shape)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        # print(path_3.shape)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        # print(path_2.shape)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print(path_1.shape)
        
        out = self.scratch.output_conv(path_1)
        # print(out.shape)

        scales = F.relu(1.0 + out)
        # filtered_d = d.clone()  # Clone d to create a new tensor
        # filtered_d[filtered_d == self.max_pred] = 0.0
        lower_bound = 0.8 * (1.0 / self.max_pred)  # Example lower bound for the range
        upper_bound = 1.05 * (1.0 / self.max_pred)  # Example upper bound for the range

        # Create a mask for elements within the specified range
        # print(lower_bound)
        # print(upper_bound)
        mask = (d >= lower_bound) & (d <= upper_bound)

        pred = d * scales
        # pred = filtered_d * scales

        # clamp pred to min and max
        if self.min_pred is not None:
            min_pred_inv = 1.0/self.min_pred
            pred[pred > min_pred_inv] = min_pred_inv
        if self.max_pred is not None:
            max_pred_inv = 1.0/self.max_pred
            # max_pred_inv = 0
            pred[pred < max_pred_inv] = max_pred_inv

            # print(self.max_pred)
            # pred[mask] = max_pred_inv

        # show_images(pred)
        pred_clone = pred.clone()
        pred_clone[mask] = max_pred_inv
        # show_images(d, pred, pred_clone)
        
        # also return scales
        return (pred_clone, scales)
    
def show_images(tensor_image1, tensor_image2, tensor_image3):
    tensor_image1 = tensor_image1.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_image1 = np.transpose(tensor_image1, (0, 2, 3, 1))  # Change from CHW to HWC
    tensor_image2 = tensor_image2.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_image2 = np.transpose(tensor_image2, (0, 2, 3, 1))  # Change from CHW to HWC
    tensor_image3 = tensor_image3.detach().cpu().numpy()  # Convert to numpy if tensor
    tensor_image3 = np.transpose(tensor_image3, (0, 2, 3, 1))  # Change from CHW to HWC
    
    # Display the images
    # batch_size = tensor_images.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust number of subplots as needed
    # if batch_size == 1:  # Handle case where batch size is 1
    #     axes = [axes]
    
    # for idx in range(batch_size):
    im1 =axes[0].imshow(1.0/tensor_image1[0], cmap='viridis')  # Assuming images are normalized [0, 1]
    axes[0].axis('off')
    axes[0].set_title("GA depth")
    fig.colorbar(im1, ax=axes[0], label='Depth')

    im2 = axes[1].imshow(1.0/tensor_image2[0], cmap='viridis')  # Assuming images are normalized [0, 1]
    axes[1].axis('off')
    axes[1].set_title("SML depth")
    fig.colorbar(im2, ax=axes[1], label='Depth')

    im3 = axes[2].imshow(1.0/tensor_image3[0], cmap='viridis')  # Assuming images are normalized [0, 1]
    axes[2].axis('off')
    axes[2].set_title("SML depth masking")
    fig.colorbar(im3, ax=axes[2], label='Depth')
    
    plt.show()