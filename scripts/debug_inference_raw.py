
import argparse
import os
import torch
import numpy as np
from accelerate import Accelerator

from src.physio_synbrain.model import PhysioSiT
from src.physio_synbrain.inference import PCFMSampler
from src.physio_synbrain.constraints import BalloonWindkesselConstraints

def debug_inference():
    # Hardcoded for debugging
    sub = 1
    data_root = 'data/NSD'
    checkpoint = 'checkpoints/sub1_20251231_003257/epoch_195'
    
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load Test CLIP (1 sample)
    test_clip_path = os.path.join(data_root, f'nsd/subj{sub:02d}/nsd_test_clip_sub{sub}.npy')
    test_clip = np.load(test_clip_path)
    # Take first sample
    batch_clip = torch.from_numpy(test_clip[0:1]).float().to(device)
    
    # Load Model structure (infer voxels from clip/fmri path... assuming 5676 from previous debug)
    # CORRECTION: Checkpoint was trained on ~15724 voxels (matching Test Data).
    # The current nsd_train_physio_sub1.npy (5676) seems to be from a different/newer run.
    num_voxels = 15724 
    model = PhysioSiT(num_voxels=num_voxels, in_channels=4, context_dim=768)
    
    print(f"Loading checkpoint {checkpoint}...")
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint)
    model = model.module if hasattr(model, 'module') else model
    model.eval()
    model.to(device)
    
    T = 48
    dt_physio = 24.0 / T
    constraints = BalloonWindkesselConstraints(dt=dt_physio).to(device)
    
    sampler = PCFMSampler(
        model=model,
        constraints=constraints,
        num_voxels=num_voxels,
        shape=(T, num_voxels, 4),
        device=device
    )
    
    print("Generating single trajectory...")
    tau_grid = torch.linspace(0, 1, T, device=device)
    dummy_x = torch.randn(1, T, num_voxels, 4, device=device)
    
    with torch.no_grad():
        traj_pred = sampler.pcfm_generate(dummy_x, batch_clip, tau_grid, steps=20)
        
    # Inspect Raw Physio States
    # traj_pred [1, T, V, 4]
    raw = traj_pred.cpu().numpy()[0] # [T, V, 4]
    
    variables = ['s', 'f', 'v', 'q']
    print("\n--- Raw Physio State Statistics (Generated) ---")
    for i, var in enumerate(variables):
        d = raw[..., i]
        print(f"{var}: Mean={d.mean():.4f}, Std={d.std():.4f}, Min={d.min():.4f}, Max={d.max():.4f}")
        
        # Check T=0 vs T=End to see if it evolves
        print(f"   T=0  Mean={d[0].mean():.4f}")
        print(f"   T=47 Mean={d[-1].mean():.4f}")

    # Inspect Observation (BOLD) output
    bold_traj = []
    for t in range(T):
        x_t = traj_pred[:, t, :, :]
        y_t = constraints.observation(x_t)
        bold_traj.append(y_t)
    bold_traj = torch.stack(bold_traj, dim=1).cpu().numpy()[0] # [T, V]
    
    print("\n--- BOLD Signal Statistics ---")
    print(f"Shape: {bold_traj.shape}")
    print(f"Mean={bold_traj.mean():.4f}, Std={bold_traj.std():.4f}, Min={bold_traj.min():.4f}, Max={bold_traj.max():.4f}")
    
    # Check Peak (t=12)
    peak = bold_traj[12]
    print(f"Peak (t=12): Mean={peak.mean():.4f}, Std={peak.std():.4f}")
    
    # Compare with a dummy observation of "Resting State" (s=0, f=1, v=1, q=1)
    rest_state = torch.tensor([0.0, 1.0, 1.0, 1.0], device=device).view(1, 1, 4).expand(1, num_voxels, 4)
    rest_bold = constraints.observation(rest_state).cpu().numpy()
    print(f"\nResting State BOLD (s=0, f=1...): Mean={rest_bold.mean():.4f}")
    
    # If Generated BOLD is close to Resting BOLD, then the model generates "Resting" (no signal).

if __name__ == "__main__":
    debug_inference()
