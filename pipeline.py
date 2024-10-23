import torch
import numpy as np

from modules.midas.midas_net_custom import MidasNet_small_videpth
from modules.estimator import LeastSquaresEstimator
from modules.interpolator import Interpolator2D

import modules.midas.transforms as transforms
import modules.midas.utils as utils
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

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


    def run(self, input_image, input_sparse_depth, validity_map, device):

        input_height, input_width = np.shape(input_image)[0], np.shape(input_image)[1]
        
        sample = {"image" : input_image}
        sample = self.depth_model_transform(sample)
        im = sample["image"].to(device)

        input_sparse_depth_valid = (input_sparse_depth < self.max_depth) * (input_sparse_depth > self.min_depth)
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

        # global scale and shift alignment
        GlobalAlignment = LeastSquaresEstimator(
            estimate=depth_pred,
            target=input_sparse_depth,
            valid=input_sparse_depth_valid
        )
        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        GlobalAlignment.clamp_min_max(clamp_min=self.min_pred, clamp_max=self.max_pred)
        int_depth = GlobalAlignment.output.astype(np.float32)

        # interpolation of scale map
        assert (np.sum(input_sparse_depth_valid) >= 3), "not enough valid sparse points"
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

        sample = {"image" : input_image, "int_depth" : int_depth, "int_scales" : int_scales, "int_depth_no_tf" : int_depth}
        sample = self.ScaleMapLearner_transform(sample)
        x = torch.cat([sample["int_depth"], sample["int_scales"]], 0)
        x = x.to(device)
        d = sample["int_depth_no_tf"].to(device)

        # run SML model
        with torch.no_grad():
            sml_pred, sml_scales = self.ScaleMapLearner.forward(x.unsqueeze(0), d.unsqueeze(0))
            sml_pred = (
                torch.nn.functional.interpolate(
                    sml_pred,
                    size=(input_height, input_width),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        output = {
            "ga_depth"  : int_depth, 
            "sml_depth" : sml_pred, 
        }
        return output
    


class TrainingDataGenerator(object):
    def __init__(self, depth_predictor, nsamples, 
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
            self.DepthModel = self.DepthModel.to(device).eval()
        elif depth_predictor == "depth_anything_v2_base":
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
            self.DepthModel = DepthAnythingV2(**model_configs['vitb'])
            self.DepthModel.load_state_dict(torch.load('Depth_Anything_V2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
            self.DepthModel = self.DepthModel.to(device).eval()
        else:
            self.DepthModel = None

       

        # depth prediction ranges
        self.min_pred, self.max_pred = min_pred, max_pred

        # depth evaluation ranges
        self.min_depth, self.max_depth = min_depth, max_depth

        # eval mode
        self.DepthModel.eval()
        self.DepthModel.to(device)



    def run(self, input_image, input_sparse_depth, validity_map, device):

        input_height, input_width = np.shape(input_image)[0], np.shape(input_image)[1]
        
        sample = {"image" : input_image}
        sample = self.depth_model_transform(sample)
        im = sample["image"].to(device)

        input_sparse_depth_valid = (input_sparse_depth < self.max_depth) * (input_sparse_depth > self.min_depth)
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

        # global scale and shift alignment
        GlobalAlignment = LeastSquaresEstimator(
            estimate=depth_pred,
            target=input_sparse_depth,
            valid=input_sparse_depth_valid
        )
        GlobalAlignment.compute_scale_and_shift()
        GlobalAlignment.apply_scale_and_shift()
        GlobalAlignment.clamp_min_max(clamp_min=self.min_pred, clamp_max=self.max_pred)
        int_depth = GlobalAlignment.output.astype(np.float32)

        # interpolation of scale map
        assert (np.sum(input_sparse_depth_valid) >= 3), "not enough valid sparse points"
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

        output = {
            "ga_depth"  : int_depth, 
            "interpolation_sparse" : int_scales, 
        }
        return output