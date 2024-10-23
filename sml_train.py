import os, time
import cv2
import numpy as np
import torch, torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from modules.midas.midas_net_custom import MidasNet_small_videpth
from modules.estimator import LeastSquaresEstimator
from modules.interpolator import Interpolator2D

import modules.midas.transforms as transforms
import modules.midas.utils as utils

import utils.log_utils as log_utils
import data.sml_dataset as sml_dataset
from utils.net_utils import OutlierRemoval
from utils.loss import compute_loss
from utils.log_utils import log

def train(
        # data input
        train_img_paths,
        train_ga_inverse_path,
        train_interpolate_sparse_inverse_paths,
        train_gt_paths,

        # training
        learning_rates,
        learning_schedule,
        batch_size,
        n_step_per_summary,
        n_step_per_checkpoint,

        # loss
        w_weight_decay,
        ground_truth_outlier_removal_kernel_size,
        ground_truth_outlier_removal_threshold,
        ground_truth_dilation_kernel_size,

        # model
        restore_path,
        min_pred_depth,
        max_pred_depth,
        checkpoint_dirpath,


        pre_trained_chk = None,

        n_threads = 10,
        ):
    
    if not os.path.exists(checkpoint_dirpath):
        os.makedirs(checkpoint_dirpath)

    sml_model_checkpoint_path = os.path.join(checkpoint_dirpath, 'sml_model-{}.pth')

    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'events')

    log_utils.log_params(log_path, locals())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_train_sample = len(train_gt_paths)
    n_train_step = learning_schedule[-1] * np.ceil(n_train_sample / batch_size).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        sml_dataset.sml_dataset(
            img_paths = train_img_paths,
            ga_inverse_paths = train_ga_inverse_path,
            interpolate_sparse_inverse_paths = train_interpolate_sparse_inverse_paths,
            gt_paths = train_gt_paths,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1)
    
    # transform
    # need to check the data's key whtehr theya re matched with the functions
    transform = transforms.get_transforms('dpt_hybrid', 'void', '150')
    ScaleMapLearner_transform = transform['sml_model']


    # Initialize ground truth outlier removal
    if ground_truth_outlier_removal_kernel_size > 1 and ground_truth_outlier_removal_threshold > 0:
        ground_truth_outlier_removal = OutlierRemoval(
            kernel_size=ground_truth_outlier_removal_kernel_size,
            threshold=ground_truth_outlier_removal_threshold)
    else:
        ground_truth_outlier_removal = None

    # Initialize ground truth dilation
    if ground_truth_dilation_kernel_size > 1:
        ground_truth_dilation = torch.nn.MaxPool2d(
            kernel_size=ground_truth_dilation_kernel_size,
            stride=1,
            padding=ground_truth_dilation_kernel_size // 2)
    else:
        ground_truth_dilation = None


    # build model
    ScaleMapLearner = MidasNet_small_videpth(
        path = pre_trained_chk,
        min_pred = min_pred_depth,
        max_pred = max_pred_depth,
    )
    ScaleMapLearner.to(device)
    ScaleMapLearner.train()

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    beta1 = 0.9
    beta2 = 0.999

    # Initialize optimizer with starting learning rate
    parameters_model = list(ScaleMapLearner.parameters())
    optimizer = torch.optim.AdamW([
        {
            'params': parameters_model,
            'weight_decay': w_weight_decay
        }],
        lr=learning_rate,
        betas=(beta1, beta2)
    )

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')

    # Start training
    train_step = 0

    if restore_path is not None and restore_path != '':
        ScaleMapLearner.load(restore_path)

    for g in optimizer.param_groups:
        g['lr'] = learning_rate

    time_start = time.time()

    print('Begin training...', log_path)
    for epoch in range(1, learning_schedule[-1] + 1):
        print('Epoch: ', epoch)
        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates
            for g in optimizer.param_groups:
                g['lr'] = learning_rate

           # Train model for an epoch
        for batch_data in train_dataloader:
            train_step = train_step + 1
            batch_data = [
                in_.to(device) for in_ in batch_data
            ]

            # unpack the saples in the batch
            img, ga_result_inverse, interpolate_sparse_inverse, gt = batch_data

    
            batch_size = ga_result_inverse.shape[0]

            # empty batch
            batch_x = []
            batch_d = []
            batch_image = []
            batch_gt = []

            # each sample in the batch
            for i in range(batch_size):
                # single sample in batch
                int_depth_i = ga_result_inverse[i].squeeze().cpu().numpy()
                int_scales_i = interpolate_sparse_inverse[i].squeeze().cpu().numpy()
        
                # transforms
                sample = {
                        'image':img[i].squeeze().cpu().numpy(),
                        'gt': gt[i].squeeze().cpu().numpy(),
                        'int_depth': int_depth_i,
                        'int_scales': int_scales_i,
                        'int_depth_no_tf': int_depth_i}

                sample = ScaleMapLearner_transform(sample)

                x = torch.cat([sample['int_depth'], sample['int_scales']], 0)
                x = x.to(device)
                d = sample['int_depth_no_tf'].to(device)
                batch_x.append(x)
                batch_d.append(d)
                batch_image.append(sample['image'].to(device))
                batch_gt.append(sample['gt'].to(device))

            x = torch.stack(batch_x, dim=0)
            d = torch.stack(batch_d, dim=0)
            batch_image = torch.stack(batch_image, dim=0)
            batch_gt = torch.stack(batch_gt, dim=0)

            # Forward pass
            sml_pred, sml_scales = ScaleMapLearner.forward(x, d)
            # inverse depth to depth
            d = 1.0 / d
            sml_pred = 1.0 / sml_pred

            # Compute loss function
            if ground_truth_dilation is not None:
                batch_gt = ground_truth_dilation(batch_gt)

            if ground_truth_outlier_removal is not None:
                batch_gt = ground_truth_outlier_removal.remove_outliers(batch_gt)

            validity_map_loss_smoothness = torch.where(
                batch_gt > 0,
                torch.zeros_like(batch_gt),
                torch.ones_like(batch_gt))

            loss, loss_info = compute_loss(sml_pred, batch_gt)
            print('{}/{} epoch:{}: {}'.format(train_step % n_train_step, n_train_step, epoch, loss.item()))

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_step_per_summary) == 0:
                with torch.no_grad():
                    # Log tensorboard summary
                    log_summary(
                        summary_writer=train_summary_writer,
                        tag='train',
                        step=train_step,
                        max_predict_depth=max_pred_depth,
                        image=batch_image,
                        input_depth=d,
                        output_depth=sml_pred,
                        ground_truth=batch_gt,
                        scalars=loss_info,
                        n_display=min(batch_size, 4))

            # Log results and save checkpoints
            if (train_step % n_step_per_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                print('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)
                # Save checkpoints
                ScaleMapLearner.save(sml_model_checkpoint_path.format(train_step))

    # Save checkpoints
    ScaleMapLearner.save(sml_model_checkpoint_path.format(train_step))

            


def log_summary(summary_writer,
                tag,
                step,
                max_predict_depth,
                image=None,
                input_depth=None,
                input_response=None,
                output_depth=None,
                ground_truth=None,
                scalars={},
                n_display=4):

    with torch.no_grad():

        display_summary_image = []
        display_summary_depth = []

        display_summary_image_text = tag
        display_summary_depth_text = tag

        if image is not None:
            image_summary = image[0:n_display, ...]

            display_summary_image_text += '_image'
            display_summary_depth_text += '_image'

            # Add to list of images to log
            display_summary_image.append(
                torch.cat([
                    image_summary.cpu(),
                    torch.zeros_like(image_summary, device=torch.device('cpu'))],
                    dim=-1))

            display_summary_depth.append(display_summary_image[-1])

        if output_depth is not None:
            output_depth_summary = output_depth[0:n_display, ...]

            display_summary_depth_text += '_output_depth'

            # Add to list of images to log
            n_batch, _, n_height, n_width = output_depth_summary.shape

            display_summary_depth.append(
                torch.cat([
                    log_utils.colorize(
                        (output_depth_summary / max_predict_depth).cpu(),
                        colormap='viridis'),
                    torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                    dim=3))

            # Log distribution of output depth
            summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

        if output_depth is not None and input_depth is not None:
            input_depth_summary = input_depth[0:n_display, ...]

            display_summary_depth_text += '_input_depth-error'

            # Compute output error w.r.t. input depth
            input_depth_error_summary = \
                torch.abs(output_depth_summary - input_depth_summary)

            input_depth_error_summary = torch.where(
                input_depth_summary > 0.0,
                input_depth_error_summary / (input_depth_summary + 1e-8),
                input_depth_summary)

            # Add to list of images to log
            input_depth_summary = log_utils.colorize(
                (input_depth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            input_depth_error_summary = log_utils.colorize(
                (input_depth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    input_depth_summary,
                    input_depth_error_summary],
                    dim=3))

            # Log distribution of input depth
            summary_writer.add_histogram(tag + '_input_depth_distro', input_depth, global_step=step)




        if output_depth is not None and input_response is not None:
            response_summary = input_response[0:n_display, ...]

            display_summary_depth_text += '_response'

            # Add to list of images to log
            response_summary = log_utils.colorize(
                response_summary.cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    response_summary,
                    torch.zeros_like(response_summary)],
                    dim=3))

            # Log distribution of input depth
            summary_writer.add_histogram(tag + '_response_distro', input_depth, global_step=step)




        if output_depth is not None and ground_truth is not None:
            ground_truth = ground_truth[0:n_display, ...]
            ground_truth = torch.unsqueeze(ground_truth[:, 0, :, :], dim=1)

            ground_truth_summary = ground_truth[0:n_display]
            validity_map_summary = torch.where(
                ground_truth > 0,
                torch.ones_like(ground_truth),
                torch.zeros_like(ground_truth))

            display_summary_depth_text += '_ground_truth-error'

            # Compute output error w.r.t. ground truth
            ground_truth_error_summary = \
                torch.abs(output_depth_summary - ground_truth_summary)

            ground_truth_error_summary = torch.where(
                validity_map_summary == 1.0,
                (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                validity_map_summary)

            # Add to list of images to log
            ground_truth_summary = log_utils.colorize(
                (ground_truth_summary / max_predict_depth).cpu(),
                colormap='viridis')
            ground_truth_error_summary = log_utils.colorize(
                (ground_truth_error_summary / 0.05).cpu(),
                colormap='inferno')

            display_summary_depth.append(
                torch.cat([
                    ground_truth_summary,
                    ground_truth_error_summary],
                    dim=3))

            # Log distribution of ground truth
            summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_display),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_display),
                global_step=step)



def log_evaluation_results(title,
                           mae,
                           rmse,
                           imae,
                           irmse,
                           abs_rel=None,
                           sq_rel=None,
                           delta1=None,
                           step=-1,
                           log_path=None):

    log(title + ':', log_path)
    log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
        'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE', 'Abs_Rel', 'Sq_Rel', 'Delta1'),
        log_path)
    log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
        step,
        mae,
        rmse,
        imae,
        irmse,
        abs_rel,
        sq_rel,
        delta1),
        log_path)