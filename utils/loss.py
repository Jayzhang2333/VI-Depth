import torch
import torch.nn.functional as F

def compute_loss(pred_depth, gt_depth):
    """
    Computes the loss for depth prediction based on L1 depth loss and multiscale gradient matching.

    Args:
    - pred_depth (torch.Tensor): Predicted depth map (B, C, H, W)
    - gt_depth (torch.Tensor): Ground truth depth map (B, C, H, W)
    
    Returns:
    - loss (torch.Tensor): Total loss (depth loss + 0.5 * gradient loss)
    - loss_info (dict): Detailed information about individual loss components
    """
    # Ensure valid mask for pixels with ground truth
    valid_mask = (gt_depth > 0).float()
    M = valid_mask.sum()

    # L1 Depth loss
    l1_depth_loss = F.l1_loss(pred_depth * valid_mask, gt_depth * valid_mask, reduction='sum') / M

    # Multiscale gradient matching loss
    def compute_gradient_loss(pred, gt):
        diff = gt - pred
        grad_x_pred = torch.abs(diff[:, :, :, :-1] - diff[:, :, :, 1:])
        grad_y_pred = torch.abs(diff[:, :, :-1, :] - diff[:, :, 1:, :])
        return (grad_x_pred.mean() + grad_y_pred.mean()) / M

    grad_loss = 0.0
    for scale in range(3):  # K = 3 levels
        scaled_pred = F.interpolate(pred_depth, scale_factor=1 / (2 ** scale), mode='bilinear', align_corners=False)
        scaled_gt = F.interpolate(gt_depth, scale_factor=1 / (2 ** scale), mode='bilinear', align_corners=False)
        grad_loss += compute_gradient_loss(scaled_pred, scaled_gt)
    
    grad_loss /= 3  # Average over K = 3 scales

    # Total loss
    total_loss = l1_depth_loss + 0.5 * grad_loss

    # Loss info dictionary for logging purposes
    loss_info = {
        'l1_depth_loss': l1_depth_loss.item(),
        'gradient_loss': grad_loss.item(),
        'total_loss': total_loss.item()
    }

    return total_loss, loss_info
