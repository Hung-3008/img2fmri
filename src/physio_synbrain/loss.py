
import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysioSynBrainLoss(nn.Module):
    def __init__(self, lambda_fm=1.0, lambda_recon=0.1, lambda_bold=0.5, constraints=None, norm_mean=None, norm_std=None):
        """
        Args:
            lambda_fm: Weight for Flow Matching loss
            lambda_recon: Weight for reconstruction guidance loss (in x space)
            lambda_bold: Weight for BOLD Pattern Loss (new)
            constraints: BalloonWindkesselConstraints instance
            norm_mean: Mean used for normalization (Tensor [V, 4] or None)
            norm_std: Std used for normalization (Tensor [V, 4] or None)
        """
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_recon = lambda_recon
        self.lambda_bold = lambda_bold
        self.constraints = constraints
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
    def pearson_loss(self, x, y):
        # x, y: [B, V]
        vx = x - torch.mean(x, dim=1, keepdim=True)
        vy = y - torch.mean(y, dim=1, keepdim=True)
        
        cost = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + 1e-8)
        return 1 - torch.mean(cost)

    def forward(self, v_pred, v_target, x_t, x_1, t, c, y_1=None):
        """
        Args:
            v_pred: Predicted velocity [B, V, C] or [B, V*C]
            v_target: Target velocity (x_1 - x_0)
            x_t: Current noisy state
            x_1: Target physio state
            t: Time embedding
            c: Condition embedding
            y_1: Target fMRI GLM Beta [B, V] (Optional)
        """
        
        # 1. Flow Matching Loss
        log_dict = {}
        
        # 1. Flow Matching Loss
        # L_fm = || v_pred - v_target ||^2
        fm_loss = F.mse_loss(v_pred, v_target, reduction='mean')
        log_dict['loss_fm'] = fm_loss.item()
        
        total_loss = self.lambda_fm * fm_loss
        
        # 2. Reconstruction Guidance Loss
        # We can infer the predicted start state from v_pred and x_t.
        # Since x_t = t * x_1 + (1-t) * x_0
        # And v_target = x_1 - x_0
        # We have x_1 = x_t + (1-t) * v_target
        # So we predict x_1_pred = x_t + (1-t) * v_pred
        
        # Reshape t for broadcasting if necessary
        # Assuming x_t has shape [B, V, C] and t has shape [B]
        ndims = x_t.ndim
        view_shape = [x_t.shape[0]] + [1] * (ndims - 1)
        t_expand = t.view(*view_shape)
        
        x_1_pred = x_t + (1.0 - t_expand) * v_pred
        
        recon_loss = F.mse_loss(x_1_pred, x_1, reduction='mean')
        log_dict['loss_recon'] = recon_loss.item()
        
        total_loss += self.lambda_recon * recon_loss
        
        # 3. Hybrid BOLD Pattern Loss
        loss_bold = torch.tensor(0.0, device=v_pred.device)
        
        if self.lambda_bold > 0 and self.constraints is not None and y_1 is not None and not torch.all(y_1 == 0):
            # Unnormalize x_1_pred if stats provided
            if self.norm_std is not None and self.norm_mean is not None:
                x_physio = x_1_pred * self.norm_std + self.norm_mean
            else:
                x_physio = x_1_pred
                
            # Compute BOLD from x_physio
            # constraints.observation expects [..., 4]
            # Assumes x_physio is the 'active' state roughly corresponding to peak response
            y_pred = self.constraints.observation(x_physio) # [B, V]
            
            # Pearson Loss
            p_loss = self.pearson_loss(y_pred, y_1)
            
            loss_bold = p_loss
            log_dict['loss_bold'] = loss_bold.item()
            total_loss += self.lambda_bold * loss_bold
        else:
            log_dict['loss_bold'] = 0.0
            
        return total_loss, log_dict
