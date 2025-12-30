import argparse
import os
import torch
import numpy as np
import wandb
from tqdm import tqdm
from scipy.stats import pearsonr
from accelerate import Accelerator

from .model import PhysioSiT
from .inference import PCFMSampler
from .constraints import BalloonWindkesselConstraints

def evaluate(args):
    accelerator = Accelerator()
    device = accelerator.device
    
    # Paths
    test_clip_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_test_clip_sub{args.sub}.npy')
    try:
        test_clip = np.load(test_clip_path)
    except FileNotFoundError:
        print(f"Test CLIP not found at {test_clip_path}. Please run extract_features.py with --split test")
        return

    test_fmri_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_test_fmri_scale_sub{args.sub}.npy')
    # Use mmap for fMRI ground truth
    try:
        test_fmri = np.load(test_fmri_path, mmap_mode='r')
    except:
        print(f"Could not load test fMRI at {test_fmri_path}")
        return
    
    if len(test_fmri.shape) == 3:
        # [N, 3, V] -> Average over trials
        print("Averaging test fMRI trials...")
        test_fmri = np.mean(test_fmri, axis=1) # [N, V]
        
    print(f"Test Set: {len(test_clip)} samples.")
    
    # Load Model
    num_voxels = test_fmri.shape[1]
    model = PhysioSiT(num_voxels=num_voxels, in_channels=4, context_dim=768)
    
    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        # Accelerator load
        print(f"Loading checkpoint from {ckpt_path}...")
        model = accelerator.prepare(model)
        accelerator.load_state(ckpt_path)
        model = model.module if hasattr(model, 'module') else model
    else:
        print(f"Checkpoint not found or valid directory: {ckpt_path}")
        return
        
    model.eval()
    model.to(device)
    
    # Setup Constraints and Sampler
    # Assuming T=48 (24s) based on training logic simulation
    T = 48 
    dt_physio = 24.0 / T # 0.5s
    
    constraints = BalloonWindkesselConstraints(dt=dt_physio).to(device)
    
    sampler = PCFMSampler(
        model=model,
        constraints=constraints,
        num_voxels=num_voxels,
        shape=(T, num_voxels, 4),
        device=device
    )
    
    print("Starting Evaluation (PCFM Generation)...")
    
    preds_bold = []
    
    # Batched Inference
    batch_size = args.batch_size
    num_samples = len(test_clip)
    
    # Physio Time Grid (Normalized 0-1 for model Input)
    tau_grid = torch.linspace(0, 1, T, device=device)
    
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_clip = torch.from_numpy(test_clip[i:i+batch_size]).float().to(device)
        B = batch_clip.shape[0]
        
        # Initial Noise Trajectory [B, T, V, 4]
        dummy_x = torch.randn(B, T, num_voxels, 4, device=device)
        
        # Generate Trajectory using PCFM
        # steps=20 flow integration steps
        traj_pred = sampler.pcfm_generate(dummy_x, batch_clip, tau_grid, steps=args.flow_steps)
        
        # traj_pred: [B, T, V, 4]
        
        # Convert to BOLD using Constraints Observation
        # observation(x) -> y
        # We need to apply to every time point
        # [B*T*V, 4]
        
        # We can just iterate T or reshape
        bold_traj_list = []
        for t in range(T):
            x_t = traj_pred[:, t, :, :] # [B, V, 4]
            y_t = constraints.observation(x_t) # [B, V]
            bold_traj_list.append(y_t)
            
        bold_traj = torch.stack(bold_traj_list, dim=1) # [B, T, V]
        
        # Extract Peak
        # Peak approx 6s. T=48 (24s). Index = 12.
        peak_idx = 12
        bold_peak = bold_traj[:, peak_idx, :] # [B, V]
        
        preds_bold.append(bold_peak.cpu().numpy())
        
    preds_bold = np.concatenate(preds_bold, axis=0) # [N, V]
    
    # Metrics
    corrs = []
    for i in range(num_samples):
        p = preds_bold[i]
        t = test_fmri[i]
        
        if np.std(p) == 0 or np.std(t) == 0:
            c = 0
        else:
            c, _ = pearsonr(p, t)
        corrs.append(c)
        
    mean_pearson = np.mean(corrs)
    print(f"Mean Pearson Correlation (Pattern Similarity): {mean_pearson:.4f}")
    
    # 2. MSE
    mse = np.mean((preds_bold - test_fmri)**2)
    print(f"MSE: {mse:.4f}")
    
    # Save results
    results = {
        'pearson': mean_pearson,
        'mse': mse,
        'preds': preds_bold,
        # 'gt': test_fmri # Don't save GT to save space if needed
    }
    np.save(os.path.join(args.output_dir, 'evaluation_results.npy'), results)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='data/NSD')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--flow_steps', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)
