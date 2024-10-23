import os, datetime
import data.data_utils as data_utils
from sml_train import train
import csv


if __name__ == '__main__':
   
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_path = '/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/imgs/dataset.csv'
    train_image_paths= []
    with open(dataset_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            train_image_paths.append(row[0])  # First column: test image paths
            


    train_ga_inverse_path = data_utils.read_paths("/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/ga_results.txt")
    train_interpolate_sparse_inverse_paths = data_utils.read_paths("/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/interpolation_sparses.txt")
    train_gt_paths = data_utils.read_paths("/media/jay/apple/FLSea_latest/archive/canyons/horse_canyon/horse_canyon/gt.txt")
    # train_ga_result_paths = data_utils.read_paths("/home/jay/Downloads/void_release/void_150/train_ground_truth.txt")

    pre_trained_chk = 'weights/sml_model.dpredictor.dpt_beit_large_512.nsamples.150.ckpt'
    train(
        # data inputs
        train_img_paths = train_image_paths,
        train_ga_inverse_path = train_ga_inverse_path,
        train_interpolate_sparse_inverse_paths = train_interpolate_sparse_inverse_paths,
        train_gt_paths = train_gt_paths,

        # training
        learning_rates = [5e-4,3e-4],
        learning_schedule = [5,10],
        batch_size = 12,
        n_step_per_summary = 20,
        n_step_per_checkpoint = 500,

        # Loss settings
        w_weight_decay = 0.0,
        ground_truth_outlier_removal_kernel_size = -1,
        ground_truth_outlier_removal_threshold = 1.5,
        ground_truth_dilation_kernel_size = -1,

        # model
        restore_path ='', 
        min_pred_depth = 0.1,
        max_pred_depth = 7.0,
        checkpoint_dirpath = os.path.join('/home/jay/SML_log', current_time),
        
        pre_trained_chk = pre_trained_chk,
        n_threads=6,
    )