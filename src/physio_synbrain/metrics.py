
import torch
import torch.nn.functional as F

def pearson_correlation(pred, target, dim=1, eps=1e-8):
    """
    Computes Pearson Correlation Coefficient across a specific dimension.
    
    Args:
        pred: Predicted tensor [Batch, Voxels] or [Batch, Voxels, Time]
        target: Target tensor [Batch, Voxels] or [Batch, Voxels, Time]
        dim: Dimension to compute correlation over. 
             If comparison is across voxels (pattern matching), dim=1.
             If comparison is across time (time-series), dim=-1.
    
    Returns:
        corr: Correlation coefficents [Batch] (if dim=1) or averaged scalar.
    """
    pred_mean = pred - pred.mean(dim=dim, keepdim=True)
    target_mean = target - target.mean(dim=dim, keepdim=True)
    
    pred_norm = pred_mean.norm(dim=dim, keepdim=True)
    target_norm = target_mean.norm(dim=dim, keepdim=True)
    
    # Avoid division by zero
    pred_norm = torch.clamp(pred_norm, min=eps)
    target_norm = torch.clamp(target_norm, min=eps)
    
    corr = (pred_mean * target_mean).sum(dim=dim, keepdim=True) / (pred_norm * target_norm)
    
    return corr.squeeze(dim)

def mse_loss(pred, target):
    """
    Computes Mean Squared Error.
    """
    return F.mse_loss(pred, target)

def compute_metrics(pred_amp, target, subject_id=None):
    """
    Wrapper to compute all metrics.
    
    Args:
        pred_amp: Predicted peak amplitude [Batch, Voxels]
        target: Ground truth GLM Betas [Batch, Voxels]
    
    Returns:
        metrics_dict: Dictionary containing 'pearson' and 'mse'.
    """
    # Pearson over Voxels (Spatial Correlation) - How well do we match the spatial pattern?
    # dim=1 is voxels.
    p_corr = pearson_correlation(pred_amp, target, dim=1)
    avg_p_corr = p_corr.mean()
    
    mse = mse_loss(pred_amp, target)
    
    return {
        'pearson': avg_p_corr,
        'mse': mse
    }
