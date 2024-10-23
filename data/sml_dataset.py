import torch.utils.data
import numpy as np
import modules.midas.utils as utils
from PIL import Image

def load_input_image(input_image_fp):
    return utils.read_image(input_image_fp)


# need to check how to load VIOD dataset's sparse depth
# or how to read depth map in general
def load_depth(input_sparse_depth_fp):
    input_sparse_depth = np.array(Image.open(input_sparse_depth_fp), dtype=np.float32) / 256.0
    input_sparse_depth[input_sparse_depth <= 0] = 0.0
    return input_sparse_depth


def load_ga_and_sparse_interpolation(input_fp):
    
    return np.load(input_fp)

class sml_dataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_paths,
                 ga_inverse_paths,
                 interpolate_sparse_inverse_paths,
                 gt_paths,
                 ):

        self.n_sample = len(ga_inverse_paths)

        for paths in [img_paths,ga_inverse_paths, interpolate_sparse_inverse_paths, gt_paths]:
            if paths is not None:
                assert len(paths) == self.n_sample

        self.img_paths = img_paths
        self.ga_inverse_paths = ga_inverse_paths
        self.interpolate_sparse_inverse_paths = interpolate_sparse_inverse_paths
        self.gt_paths = gt_paths


    def __getitem__(self, index):

        
        img = load_input_image(self.img_paths[index].replace("/home/auv/FLSea", "/media/jay/apple/FLSea_latest"))
        ga_inverse = load_ga_and_sparse_interpolation(self.ga_inverse_paths[index])
        interpolate_sparse_inverse = load_ga_and_sparse_interpolation(self.interpolate_sparse_inverse_paths[index])
        gt = load_ga_and_sparse_interpolation(self.gt_paths[index])
        
        # Check whetehr we need any cropping for the dataset
        

        

        return img, ga_inverse, interpolate_sparse_inverse, gt


    def __len__(self):
        return self.n_sample