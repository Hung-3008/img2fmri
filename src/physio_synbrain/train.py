
import argparse
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from model import PhysioNeuroFlow
from metrics import compute_metrics

class PhysioDataset(Dataset):
    def __init__(self, fmri_path, stim_idxs_path, clip_path=None, stim_hdf5_path=None):
        self.fmri = np.load(fmri_path, mmap_mode='r') # [N, V]
        self.stim_idxs = np.load(stim_idxs_path)      # [N]
        
        # Determine ROI / Mask size
        self.num_voxels = self.fmri.shape[1]
        
        # Load CLIP
        if clip_path and os.path.exists(clip_path):
            self.clip = np.load(clip_path)
            self.use_precomputed_clip = True
        else:
            self.clip = None
            self.use_precomputed_clip = False
            # Fallback to loading raw images?
            # For now assume clip exists or raise error
            print("Warning: CLIP embeddings not found. Logic to extract on-the-fly not implemented.")
            # In a real scenario, we would load hdf5 here.
            
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, idx):
        # fMRI Beta Map [V]
        y_real = torch.from_numpy(self.fmri[idx].copy()).float()
        
        # CLIP Embedding [768]
        if self.use_precomputed_clip:
            c = torch.from_numpy(self.clip[idx].copy()).float()
        else:
            c = torch.zeros(768) # Placeholder
            
        return {
            'y_real': y_real,
            'c': c,
            'subject_id': torch.tensor(0) # Logic for multi-subject later
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16) # Smaller batch for ODE
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="NSD/data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--dt", type=float, default=0.5)
    args = parser.parse_args()

    # Append timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"sub{args.sub}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers("physio_synbrain_c1", config=vars(args))
    
    # Paths (using processed files)
    processed_dir = os.path.join(args.data_root, 'nsd/processed')
    fmri_path = os.path.join(processed_dir, f'nsd_train_fmri_sub{args.sub}.npy')
    stim_idxs_path = os.path.join(processed_dir, f'nsd_train_stim_idxs_sub{args.sub}.npy')
    # Assuming standard path for clip
    clip_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_train_clip_sub{args.sub}.npy')
    
    if not os.path.exists(fmri_path):
        raise FileNotFoundError(f"Processed fMRI not found at {fmri_path}. Run prepare_nsd_numpy.py first.")

    dataset = PhysioDataset(fmri_path, stim_idxs_path, clip_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Model
    model = PhysioNeuroFlow(
        num_voxels=dataset.num_voxels, 
        context_dim=768,
        time_steps=48, # 24s @ 0.5s TR
        dt=args.dt,
        patch_size=256
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            z_stim = batch['c']     # [B, 768]
            y_real = batch['y_real'] # [B, V] (GLM Beta)
            subj_id = batch['subject_id']
            
            # Forward End-to-End
            # y_pred_seq: [B, T, V]
            # u_pred_seq: [B, V, T]
            y_pred_seq, u_pred_seq = model.forward_train(z_stim, subj_id)
            
            # Loss Calculation
            # Match Peak Response to Beta
            # y_real is approximate amplitude.
            # y_pred_seq max over time approximate amplitude.
            y_pred_amp, _ = torch.max(y_pred_seq, dim=1) # [B, V]
            
            # Basic MSE
            loss_recon = torch.mean((y_pred_amp - y_real) ** 2)
            
            # Regularization (Energy cost of neural activity)
            loss_reg = torch.mean(u_pred_seq ** 2) * 1e-4
            
            loss = loss_recon + loss_reg
            
            accelerator.backward(loss)
            
            # Clip grads for stability in ODE
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 20 == 0:
                # Compute Metrics
                metrics = compute_metrics(y_pred_amp, y_real)
                
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Recon: {loss_recon.item():.4f} | Pearson: {metrics['pearson']:.4f}")
                accelerator.log({
                    "loss": loss.item(), 
                    "recon": loss_recon.item(), 
                    "reg": loss_reg.item(),
                    "pearson": metrics['pearson'].item(),
                    "mse": metrics['mse'].item()
                })
        if epoch % 5 == 0:
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

if __name__ == "__main__":
    main()
