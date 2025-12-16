import torch
from model.vqgan_3d import VQGAN
from torch.nn.utils import clip_grad_norm_

class VQGAN_fMRI(VQGAN):
    """
    Wrapper around VQGAN to handle fMRI data batch format from NSDDataset.
    Input batch is a dictionary with 'fmri' key of shape (B, C, T, D, H, W).
    Reshapes to (B*T, C, D, H, W).
    Uses Manual Optimization and Manual Gradient Accumulation.
    """
    def __init__(self, args):
        super().__init__(args)
        self.automatic_optimization = False
        
    def training_step(self, batch, batch_idx):
        # optimizers
        opt_ae, opt_disc = self.optimizers()
        
        # Accumulation steps
        accum_steps = getattr(self.args, 'accumulate_grad_batches', 1)
        
        # Unpack batch
        x = batch['fmri']
        b, c, t, d, h, w = x.shape
        x = x.view(b * t, c, d, h, w)
        
        # -------------------------
        # Train Generator
        # -------------------------
        recon_loss, _, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(x, 0)
        commitment_loss = vq_output['commitment_loss']
        loss_gen = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        
        # Scale loss
        loss_gen = loss_gen / accum_steps
        self.manual_backward(loss_gen)
        
        if (batch_idx + 1) % accum_steps == 0:
            if hasattr(self.args, 'gradient_clip_val') and self.args.gradient_clip_val > 0:
                self.clip_gradients(opt_ae, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
       
            opt_ae.step()
            opt_ae.zero_grad()
        
        # -------------------------
        # Train Discriminator
        # -------------------------
        discloss = self.forward(x, 1)
        
        # Scale loss
        discloss = discloss / accum_steps
        self.manual_backward(discloss)
        
        if (batch_idx + 1) % accum_steps == 0:
            if hasattr(self.args, 'gradient_clip_val') and self.args.gradient_clip_val > 0:
                self.clip_gradients(opt_disc, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            
            opt_disc.step()
            opt_disc.zero_grad()
        
        self.log("train_loss", loss_gen * accum_steps + discloss * accum_steps, prog_bar=True)
            
        return loss_gen * accum_steps

    def validation_step(self, batch, batch_idx):
        x = batch['fmri']
        b, c, t, d, h, w = x.shape
        x = x.view(b * t, c, d, h, w)
        
        recon_loss, x_recon, vq_output, aeloss, perceptual_loss, gan_feat_loss = self.forward(x, 0)
        
        self.log('val/recon_loss', recon_loss, prog_bar=True)
        self.log('val/perceptual_loss', perceptual_loss, prog_bar=True)
        self.log('val/perplexity', vq_output['perplexity'], prog_bar=True)
        self.log('val/commitment_loss', vq_output['commitment_loss'], prog_bar=True)
        
    def configure_optimizers(self):
        return super().configure_optimizers()
