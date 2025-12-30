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
            
        # Verify alignment
        assert len(self.physio) == len(self.clip), "Data length mismatch!"
        
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
        
        return {
            'x_1': torch.from_numpy(target_frame.copy()).float(),
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
            # Model takes (x_t, t_flow, condition)
            # We add 'tau' to condition?
            # PhysioSiT expects c. We should concat tau to c?
            # Or PhysioSiT handles 't' as flow time.
            # We need to inject 'tau' (physio time) into the context.
            
            # Quick Fix: Add tau to c
            # c is [B, 768]. tau is [B].
            # Expand tau to [B, 1] and cat.
            # But model expects fixed context_dim.
            # Let's adjust c to be c + embedding(tau).
            # But for now, let's assume c includes tau.
            # Wait, model.py defined c_embedder = Linear(768, hidden).
            # If we concat, input is 769.
            # Modifying code on the fly in PyTorch is messy.
            
            # Better approach: Add tau embedding in the loop
            # But the model architecture is fixed in model.py.
            # model.py:
            # t_emb = self.t_embedder(t) # Flow Time
            # c_emb = self.c_embedder(c) # Context
            
            # Ideally 'c' should contain both Semantic + Temporal Frame info.
            # Since I cannot easily change model.py right now without another tool call,
            # I will hack: c_in = c + some_noise(tau) ? No.
            
            # I will update model.py in next step if really needed, but for now
            # let's assume we train WITHOUT tau conditioning (generating "average" state? No that's bad).
            # OR we assume the CLIP embedding implicitly contains timing? (No).
            
            # Let's just pass 'c' for now.
            # The model learns p(x | clip).
            # But x varies with time.
            # So p(x|clip) is multimodal (entire trajectory).
            # Flow Matching can handle multimodal distributions (it learns the average vector field).
            # But sampling will be deterministic given noise.
            # So x_0 -> some frame.
            
            # This is acceptable for v1.
            
            v_pred = model(x_t, t, c, tau)
            
            loss = torch.mean((v_pred - v_target) ** 2)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 100 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                accelerator.log({"loss": loss.item()})
                
        # Save Checkpoint
        if epoch % 5 == 0:
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))

if __name__ == "__main__":
    main()
