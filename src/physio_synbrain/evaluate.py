import argparse
import os
import torch
import numpy as np
import wandb
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from accelerate import Accelerator

from .model import PhysioSiT
from .inference import PCFMSampler
from .constraints import BalloonWindkesselConstraints


def compute_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between two matrices X, Y of shape [N, D]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    dot_XY = torch.norm(X @ Y.T) ** 2
    dot_XX = torch.norm(X @ X.T) ** 2
    dot_YY = torch.norm(Y @ Y.T) ** 2
    return (dot_XY / (torch.sqrt(dot_XX * dot_YY) + 1e-8)).item()


def evaluate_voxel_and_structural_metrics(all_recon_fmri: torch.Tensor,
                                          all_fmri: torch.Tensor):
    """Evaluate voxel-level (MSE, Pearson) and structural-level (CKA, Cosine).

    Args:
        all_recon_fmri: [N, 1, V] tensor of predicted voxel patterns.
        all_fmri:       [N, T, V] tensor of ground-truth voxel patterns
                        (T trials, e.g. T=3 as in SynBrain).

    Returns:
        dict with keys: "MSE", "Pearson", "CKA", "Cosine".
    """
    N, _, V = all_recon_fmri.shape
    all_recon = all_recon_fmri.squeeze(1)  # [N, V]

    mse_vals = []
    pearson_vals = []

    for i in range(N):
        recon = all_recon[i]
        for j in range(all_fmri.shape[1]):
            target = all_fmri[i, j]
            mse = torch.mean((recon - target) ** 2).item()
            p = pearsonr(recon.cpu().numpy(), target.cpu().numpy())[0]
            mse_vals.append(mse)
            pearson_vals.append(p)

    # Flatten across trials for structure-level comparison
    recon_flat = all_recon.repeat_interleave(all_fmri.shape[1], dim=0)  # [N*T, V]
    target_flat = all_fmri.view(-1, V)                                  # [N*T, V]

    # CKA
    cka = compute_cka(recon_flat, target_flat)

    # Cosine similarity (averaged per sample-trial)
    recon_np = recon_flat.cpu().numpy()
    target_np = target_flat.cpu().numpy()
    cos_sim = np.mean([
        cosine_similarity(recon_np[i:i+1], target_np[i:i+1])[0, 0]
        for i in range(recon_np.shape[0])
    ])

    return {
        "MSE": float(np.mean(mse_vals)),
        "Pearson": float(np.mean(pearson_vals)),
        "CKA": float(cka),
        "Cosine": float(cos_sim),
    }

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
        # SynBrain evaluation strategy: Compare against EACH trial individually
        # test_fmri: [N, 3, V]
        num_trials = test_fmri.shape[1]
    else:
        # If already averaged or single trial
        test_fmri = test_fmri[:, np.newaxis, :] # [N, 1, V]
        num_trials = 1
        
    print(f"Test Set: {len(test_clip)} samples. Trials per sample: {num_trials}")
    


    
    # Load Model
    num_voxels = test_fmri.shape[2]
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
    
    # Load Normalization Stats
    stats_path = os.path.join(args.data_root, f'nsd/physio/physio_stats_sub{args.sub}.npz')
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        stats = np.load(stats_path)
        mean = torch.from_numpy(stats['mean']).float().to(device)
        std = torch.from_numpy(stats['std']).float().to(device)
        normalize = True
    else:
        print(f"Warning: Stats not found at {stats_path}. Evaluation assuming raw generation.")
        normalize = False
    
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
    if args.limit is not None:
        num_samples = min(num_samples, args.limit)
        print(f"Limiting evaluation to {num_samples} samples.")
    
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
            
            # Unnormalize if needed
            if normalize:
                x_t = x_t * std + mean
                
            y_t = constraints.observation(x_t) # [B, V]
            bold_traj_list.append(y_t)
            
        bold_traj = torch.stack(bold_traj_list, dim=1) # [B, T, V]
        
        # Extract Peak
        # Peak approx 6s. T=48 (24s). Index = 12.
        peak_idx = 12
        bold_peak = bold_traj[:, peak_idx, :] # [B, V]
        
        preds_bold.append(bold_peak.cpu().numpy())
        
    preds_bold = np.concatenate(preds_bold, axis=0) # [N, V]
    
    # Convert to tensors for voxel + structural metrics (SynBrain-style)
    all_recon_fmri = torch.from_numpy(preds_bold[:num_samples]).float().unsqueeze(1)  # [N,1,V]
    all_fmri = torch.from_numpy(np.asarray(test_fmri[:num_samples])).float()          # [N,T,V]

    metrics = evaluate_voxel_and_structural_metrics(all_recon_fmri, all_fmri)

    print(f"Mean Pearson Correlation (Pattern Similarity): {metrics['Pearson']:.4f}")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"CKA: {metrics['CKA']:.4f}")
    print(f"Cosine similarity: {metrics['Cosine']:.4f}")
    
    # Save results
    results = {
        'pearson': metrics['Pearson'],
        'mse': metrics['MSE'],
        'cka': metrics['CKA'],
        'cosine': metrics['Cosine'],
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
    parser.add_argument('--limit', type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)
