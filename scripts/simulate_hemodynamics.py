import argparse
import os
import numpy as np
import torch
import torchdiffeq
from tqdm import tqdm

def balloon_windkessel_ode(t, state, u_t, params):
    """
    Standard Balloon-Windkessel Model ODE
    State: [s, f, v, q]
    u_t: Neural input at time t
    """
    s, f, v, q = state.unbind(-1)
    
    # Parameters
    epsilon = params['epsilon']
    kappa_s = params['kappa_s']
    kappa_f = params['kappa_f']
    tau_0 = params['tau_0']
    alpha = params['alpha']
    E0 = params['E0']
    
    # Safety clamping to avoid instability
    # f, v, q must be positive
    f = torch.clamp(f, min=1e-6)
    v = torch.clamp(v, min=1e-6)
    q = torch.clamp(q, min=1e-6)
    
    ds = epsilon * u_t - kappa_s * s - kappa_f * (f - 1)
    df = s
    
    # dv = (f - v^(1/alpha)) / tau_0
    # Safe power
    dv = (f - torch.pow(v, 1/alpha)) / tau_0
    
    # dq = (f * (1 - (1 - E0)^(1/f)) / E0 - v^(1/alpha - 1) * q) / tau_0
    # Safe power for (1-E0)^(1/f)
    # 1-E0 = 0.6. 1/f can be large if f is small.
    # But f is clamped to 1e-6.
    
    term1 = f * (1 - torch.pow(1 - E0, 1/f)) / E0
    term2 = torch.pow(v, 1/alpha - 1) * q
    dq = (term1 - term2) / tau_0
    
    return torch.stack([ds, df, dv, dq], dim=-1)

class HemodynamicSimulator(torch.nn.Module):
    def __init__(self, dt=0.1, duration=30.0, device='cuda'):
        super().__init__()
        self.device = device
        self.dt = dt
        self.duration = duration
        self.time_steps = int(duration / dt)
        
        # Standard parameters (Friston et al.)
        self.params = {
            'epsilon': 1.0, 
            'kappa_s': 0.65, 
            'kappa_f': 0.41, 
            'tau_0': 0.98,
            'alpha': 0.32,
            'E0': 0.4,
            'V0': 0.02,
            'k1': 7 * 0.4,
            'k2': 2.0,
            'k3': 2 * 0.4 - 0.2
        }
    
    def forward_solve(self, betas):
        B, V = betas.shape
        
        # Initial state: Resting equilibrium [s=0, f=1, v=1, q=1]
        x0 = torch.zeros(B, V, 4, device=self.device)
        x0[..., 0] = 0.0
        x0[..., 1] = 1.0
        x0[..., 2] = 1.0
        x0[..., 3] = 1.0
        
        # Time Points for output (Simulation DT)
        # Note: We use a finer integration step for stability if needed, 
        # but torchdiffeq 'rk4' uses the grid as step if step_size is not specified?
        # Typically fixed methods take steps between t_i and t_{i+1}.
        time_points = torch.linspace(0, self.duration, self.time_steps, device=self.device)
        
        def func(t, x):
            # Impulse u(t) = beta if t < 3.0 else 0
            # Clamp betas [ -5, 5 ]
            safe_betas = torch.clamp(betas, -5.0, 5.0)
            u_val = torch.where(t < 3.0, safe_betas, torch.zeros_like(betas))
            return balloon_windkessel_ode(t, x, u_val, self.params)
        
        # Solve ODE
        # Back to rk4 (fixed step)
        # Adaptive solvers (dopri5) fail on discontinuities (Step function u(t))
        # RK4 is robust enough if equations are stabilized.
        
        # options={'step_size': ...} if we want sub-steps.
        # But let's try standard grid step (dt=0.5). If unstable, we can increase resolution.
        
        traj = torchdiffeq.odeint(func, x0, time_points, method='rk4', rtol=1e-5, atol=1e-5)
             
        # Check for NaNs immediately
        if torch.isnan(traj).any() or torch.isinf(traj).any():
             print("Warning: NaNs/Inf produced in batch simulation. Clamping.")
             traj = torch.nan_to_num(traj, nan=0.0, posinf=10.0, neginf=-10.0)

        # Final Clamp output to biological ranges
        # s in [-inf, inf], f > 0, v > 0, q > 0
        traj[..., 1] = torch.clamp(traj[..., 1], min=0.01, max=10.0)
        traj[..., 2] = torch.clamp(traj[..., 2], min=0.01, max=10.0)
        traj[..., 3] = torch.clamp(traj[..., 3], min=0.01, max=10.0)
        
        return traj.permute(1, 0, 2, 3) # [Batch, Time, Voxels, 4]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='data/NSD')
    parser.add_argument('--output_dir', type=str, default='data/NSD/nsd/physio')
    parser.add_argument('--batch_size', type=int, default=50) # Reduced default batch size
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    input_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_train_fmri_scale_sub{args.sub}.npy')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading data from {input_path}...")
    # Map input file to save RAM if it's large (though 1.7GB is fine, good practice)
    data = np.load(input_path, mmap_mode='r')
    print(f"Data shape: {data.shape}")
    
    # Determine basic dimensions
    if len(data.shape) == 3:
        # [N, 3, V] -> We need to average. mmap + averaging requires reading chunks.
        # We will handle averaging inside the loop to save RAM.
        num_samples = data.shape[0]
        num_voxels = data.shape[2]
    else:
        num_samples = data.shape[0]
        num_voxels = data.shape[1]
        
    print(f"Processing {num_samples} stimuli for {num_voxels} voxels...")
    
    simulator = HemodynamicSimulator(dt=0.5, duration=24.0, device=device)
    time_steps = simulator.time_steps
    output_channels = 4
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'nsd_train_physio_sub{args.sub}.npy')
    
    # Initialize Memmap for Output
    # Shape: [N, Time, Voxels, 4]
    output_shape = (num_samples, time_steps, num_voxels, output_channels)
    print(f"Creating output memmap at {save_path} with shape {output_shape}...")
    
    # 'w+' creates or overwrites file
    output_memmap = np.memmap(save_path, dtype='float32', mode='w+', shape=output_shape)
    
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size)):
            # Load batch
            if len(data.shape) == 3:
                # Read chunk
                batch_data = data[i:i+batch_size] # This reads into RAM
                # Average TRIALS if needed
                betas_np = np.mean(batch_data, axis=1) 
            else:
                betas_np = data[i:i+batch_size]
            
            # To GPU
            batch_betas = torch.tensor(betas_np, dtype=torch.float32, device=device)
            
            # Simulate
            traj = simulator.forward_solve(batch_betas) # [B, T, V, 4]
            
            # Write to disk
            # Determine actual batch size (for last batch)
            curr_batch_size = traj.shape[0]
            output_memmap[i:i+curr_batch_size] = traj.cpu().numpy()
            
            # Flush periodically to ensure write
            if i % (batch_size * 5) == 0:
                output_memmap.flush()
                
            del batch_betas, traj, betas_np
            torch.cuda.empty_cache()

    output_memmap.flush()
    print(f"Done. Saved to {save_path}")
    print(f"Output size: {os.path.getsize(save_path) / 1e9:.2f} GB")

if __name__ == '__main__':
    main()
