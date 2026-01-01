import argparse
import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from .model import PhysioSiT
from .loss import PhysioSynBrainLoss

class PhysioDataset(Dataset):
    def __init__(self, physio_path, clip_path):

        # Load CLIP Embeddings (load all to RAM, usually small <1GB)
        self.clip = np.load(clip_path)
        
        # Load Physio Data
        try:
            self.physio = np.load(physio_path, mmap_mode='r')
        except (ValueError, OSError):
            # Fallback for raw binary memmap (created by simulate_hemodynamics.py)
            print(f"Warning: Could not load {physio_path} as standard .npy. Trying raw memmap...")
            N = len(self.clip)
            file_size = os.path.getsize(physio_path)
            total_elements = file_size // 4 # float32 = 4 bytes
            
            # Known constants from simulation
            T = 48 # dt=0.5, duration=24.0
            C = 4
            
            # Infer V
            if total_elements % (N * T * C) != 0:
                raise ValueError(f"File size {file_size} does not match expected output shape [N={N}, T={T}, V=?, C={C}]")
                
            V = total_elements // (N * T * C)
            print(f"Inferred shape: [{N}, {T}, {V}, {C}]")
            
            self.physio = np.memmap(physio_path, dtype='float32', mode='r', shape=(N, T, V, C))
            
        # Load Normalization Stats
        stats_path = os.path.join(os.path.dirname(physio_path), f'physio_stats_sub1.npz') # Hardcode sub1 or pass arg?
        # Better infer from physio_path?
        # Let's assume stats file is in same dir as physio_path
        
        # We need to construct stats path.
        # physio_path: .../nsd_train_physio_sub1.npy
        # stats_path: .../physio_stats_sub1.npz
        
        stats_path = physio_path.replace('nsd_train_physio_sub', 'physio_stats_sub').replace('.npy', '.npz')
        
        if os.path.exists(stats_path):
            print(f"Loading normalization stats from {stats_path}...")
            stats = np.load(stats_path)
            self.mean = torch.from_numpy(stats['mean']).float()
            self.std = torch.from_numpy(stats['std']).float()
            self.normalize = True
        else:
            print(f"Warning: Stats not found at {stats_path}. Training without normalization.")
            self.normalize = False

        # Verify alignment
        assert len(self.physio) == len(self.clip), "Data length mismatch!"

        # Load fMRI (GLM Betas) for BOLD Loss
        # Path: data/NSD/nsd/subj01/nsd_train_fmri_sub1.npy (Check exact name)
        # Using data_root from physio_path structure
        # physio_path: .../nsd/physio/nsd_train_physio_sub1.npy
        # fmri_path: .../nsd/subj01/nsd_train_fmri_sub1.npy (Assuming standard NSD structure)
        
        # fmri_path: .../nsd/subj01/nsd_train_fmri_scale_sub1.npy 
        
        # We need to extract subj ID and data root from physio_path or pass them.
        # But here we only have physio_path and clip_path.
        # Let's guess fmri_path from clip_path which is in proper subject dir.
        # clip_path: .../data/NSD/nsd/subj01/nsd_train_clip_sub1.npy
        
        # Correct pattern: nsd_train_clip -> nsd_train_fmri_scale
        fmri_path = clip_path.replace('nsd_train_clip', 'nsd_train_fmri_scale')
        
        if os.path.exists(fmri_path):
            try:
                self.fmri = np.load(fmri_path, mmap_mode='r')
                print(f"Loaded fMRI data from {fmri_path} for BOLD loss.")
            except:
                print(f"Warning: Failed to load fMRI at {fmri_path}. BOLD loss will be disabled.")
                self.fmri = None
        else:
            # Try 'nsd_train_fmriavg_nsdgeneral_sub1.npy' or typical names?
            # Let's assume the user has nsd_train_fmri_sub1.npy as per typical prep
            print(f"Warning: fMRI file not found at {fmri_path}. BOLD loss will be disabled.")
            self.fmri = None
        
    def __len__(self):
        return len(self.physio)
    
    def __getitem__(self, idx):
        # physio: [Time, Voxels, 4]
        # We need to sample a random time point t for Flow Matching?
        # Or do we train on the whole trajectory?
        
        # Standard FM trains on x_1 (Data) and x_0 (Noise).
        # x_1 is a sample from the data distribution.
        # Our "Data" is a Trajectory x(t).
        # We want to generate Trajectories.
        # So x_1 = Whole Trajectory [Time, Voxels, 4]?
        # Or x_1 = Snapshot state [Voxels, 4]?
        
        # If we treat Time as an input to the network `model(x, t_flow, c)`,
        # we are learning a vector field v_t(x).
        # Ideally, we learn to generate the spatial pattern [Voxels, 4].
        # But fMRI changes over time.
        # If we just generate a SNAPSHOT, we lose temporal correlation?
        # SynBrain generates a snapshot (averaged/single).
        # BUT Physio-SynBrain adds Physics to ensure TEMPORAL consistency.
        # So we should train on SNAPSHOTS x(t_physio).
        # And condition on t_physio?
        
        # Proposal:
        # The model generates x(t_physio) given Clip + t_physio.
        # Training Sample:
        # 1. Pick random sample index i.
        # 2. Pick random physio-time point tau in [0, Duration].
        # 3. Target x_1 = Physio[i, tau].
        # 4. Condition = (Clip[i] + Embedding(tau)).
        # 5. FM Task: Flow from Noise -> x_1.
        
        # Wait, if we condition on tau, we don't need ODE solver for time evolution?
        # We want to solve ODE to GENERATE evolution.
        # If we just condition on tau, we are doing independent generation per frame.
        # The "Constraint" connects them.
        
        # Let's settle on:
        # Train to generate frames x_1 ~ p(x | clip, tau).
        
        sample_physio = self.physio[idx] # [Time, Voxels, 4]
        clip_embed = self.clip[idx]      # [768]
        
        # Randomly select a frame (tau)
        T = sample_physio.shape[0]
        tau_idx = np.random.randint(0, T)
        
        target_frame = sample_physio[tau_idx] # [Voxels, 4]
        
        # Normalize target? 
        # Physio states are small: s~0, f~1, v~1, q~1.
        # We should probably center/scale them.
        # For now, raw.
        
        target_frame = torch.from_numpy(sample_physio[tau_idx].copy()).float() # [Voxels, 4]
        
        if self.normalize:
            # Broadcast mean/std?
            # mean/std are [V, C]. target_frame is [V, C].
            target_frame = (target_frame - self.mean) / self.std

        if self.fmri is not None:
             f_sample = self.fmri[idx] # [3, V] or [V]
             if len(f_sample.shape) == 2:
                 # Average over trials to get robust ground truth
                 y_1_val = np.mean(f_sample, axis=0)
             else:
                 y_1_val = f_sample
             y_1_tensor = torch.from_numpy(y_1_val.copy()).float()
        else:
             y_1_tensor = torch.tensor(0.0)

        return {
            'x_1': target_frame,
            'y_1': y_1_tensor,
            'c': torch.from_numpy(clip_embed.copy()).float(),
            'tau': tau_idx / T # Normalized time conditions [0, 1]
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="data/NSD")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # Append timestamp to output_dir to separate runs
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"sub{args.sub}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers("physio_synbrain", config=vars(args))
    
    # Paths
    physio_path = os.path.join(args.data_root, f'nsd/physio/nsd_train_physio_sub{args.sub}.npy')
    clip_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_train_clip_sub{args.sub}.npy')
    
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"CLIP features not found at {clip_path}. Run extract_features.py first.")
        
    # Dataset
    dataset = PhysioDataset(physio_path, clip_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Model
    # Determine num_voxels from data
    num_voxels = dataset.physio.shape[2]
    model = PhysioSiT(num_voxels=num_voxels, in_channels=4, context_dim=768)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    from .constraints import BalloonWindkesselConstraints
    
    # Constraints for BOLD Loss
    # Need to move to device
    T = 48
    dt_physio = 24.0 / T
    constraints = BalloonWindkesselConstraints(dt=dt_physio)
    
    # Loss Function
    # We need to pass valid mean/std if normalized.
    if dataset.normalize:
        norm_mean = dataset.mean.to(accelerator.device)
        norm_std = dataset.std.to(accelerator.device)
    else:
        norm_mean = None
        norm_std = None
        
    criterion = PhysioSynBrainLoss(
        lambda_fm=1.0, 
        lambda_recon=0.1, 
        lambda_bold=0.5, # New BOLD weight
        constraints=constraints,
        norm_mean=norm_mean,
        norm_std=norm_std
    )
    
    # We need to move constraints to device AFTER accelerator prepare?
    # Or just register buffer. BalloonWindkesselConstraints usually has no parameters, just buffers/constants.
    constraints = constraints.to(accelerator.device)
    criterion.constraints = constraints # Ensure device placement
    
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Training Loop - Flow Matching
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            x_1 = batch['x_1'] # Target Data [B, V, 4]
            c = batch['c']     # Condition [B, D]
            tau = batch['tau'] # Physio Time [0, 1]
            
            # Flow Matching Setup
            B = x_1.shape[0]
            x_0 = torch.randn_like(x_1) # Noise
            t = torch.rand(B, device=accelerator.device) # Flow Time t
            
            # Interpolated state x_t
            # Linear Rectified Flow: x_t = t * x_1 + (1-t) * x_0
            t_expand = t.view(B, 1, 1)
            x_t = t_expand * x_1 + (1 - t_expand) * x_0
            
            # Target Vector Field v_target = x_1 - x_0
            v_target = x_1 - x_0
            
            # Predict Vector Field v_pred
            v_pred = model(x_t, t, c, tau)
            
            y_1 = batch['y_1'] if 'y_1' in batch else None # GLM Beta [B, V]
            
            # loss = torch.mean((v_pred - v_target) ** 2)
            loss, log_dict = criterion(v_pred, v_target, x_t, x_1, t, c, y_1)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 100 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | FM: {log_dict['loss_fm']:.4f} | Recon: {log_dict['loss_recon']:.4f} | BOLD: {log_dict.get('loss_bold', 0):.4f}")
                accelerator.log({"loss": loss.item(), **log_dict})
                
        # Save Checkpoint
        if epoch % 5 == 0:
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

if __name__ == "__main__":
    main()
