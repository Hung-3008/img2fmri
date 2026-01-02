
import torch
import torch.nn as nn
import numpy as np
import torchdiffeq
try:
    from .neural_model import NeuralFlow
    from .constraints import BalloonWindkesselConstraints
except ImportError:
    from neural_model import NeuralFlow
    from constraints import BalloonWindkesselConstraints

class FlowODEWrapper(nn.Module):
    def __init__(self, neural_flow, z_stim):
        super().__init__()
        self.neural_flow = neural_flow
        self.z_stim = z_stim # [B, D]
        
    def forward(self, t, u):
        # t is scalar
        # u is [B, V, T] (Time as Channels)
        # Verify batch size matches
        # NeuralFlow uses u.shape[0]
        t_expand = torch.ones(u.shape[0], device=u.device) * t
        return self.neural_flow(u, t_expand, self.z_stim)

class BalloonODEWrapper(nn.Module):
    def __init__(self, dt, params, constraints, u_trajectory):
        super().__init__()
        self.dt = dt
        # params: alpha, tau_0, epsilon
        self.alpha, self.tau_0, self.epsilon = params
        self.constraints = constraints
        self.u_trajectory = u_trajectory # [B, V, T] (Time channels from Flow)
        # Note: u_trajectory is treated as "constant context" for ADJOINT unless we traverse it.
        # Adjoint method backprops through state y0 and params.
        # u_trajectory is NOT a parameter of this Module (it's input).
        # We need gradients w.r.t u_trajectory.
        # Strict Adjoint usually requires inputs to be args to forward or params.
        # If u_trajectory is closure-captured, standard adjoint might miss it?
        # Actually it works if we add it to `adjoint_params`?
        # OR: u_trajectory is fixed during Integration (it drives dynamics).
        # If we want dL/du, we need u to be part of the state? No.
        # We rely on the fact that 'odeint_adjoint' supports backprop via VJP.
        # If u_trajectory is used in f(t, y), VJP will flow to u_trajectory?
        # Reference: torchdiffeq only computes gradients for: y0, t, and partials w.r.t params.
        # Explicit inputs in closure are tricky.
        # Ideally we pass u_trajectory as a parameter?
        # But u_trajectory is output of previous step.
        # Hack: Concatenate u onto state?
        # Or just use standard odeint for Balloon (step count is fixed and small-ish, memory is low compared to Transformer Flow).
        # Balloon Step: MLP is small (Constraints).
        # Memory for Balloon integration is negligible. 
        # State [B, V, 4]. 4 floats per voxel.
        # 48 steps -> 48 * 4 * V * B.
        # 48 * 4 * 370k * 4 * 4 ~ 1 GB.
        # So standard odeint is FINE for Balloon.
        # Adjoint is CRITICAL for Flow (Transformer).
        pass

    def forward(self, t, state):
        # ... logic ...
        # I will inline this logic in the closure for standard odeint usage in DHB.
        # Flow uses Adjoint.
        pass

class Hypernetwork(nn.Module):
    def __init__(self, num_subjects=8, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_subjects + 1, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 3), # alpha, tau_0, epsilon
            nn.Sigmoid() 
        )
        
    def forward(self, subject_id):
        raw = self.mlp(self.embedding(subject_id))
        alpha = raw[:, 0] * 0.3 + 0.2    
        tau_0 = raw[:, 1] * 1.5 + 0.5    
        epsilon = raw[:, 2] * 1.5 + 0.1  
        return alpha, tau_0, epsilon

class DifferentiableHemodynamicBlock(nn.Module):
    def __init__(self, dt=0.5):
        super().__init__()
        self.dt = dt
        self.constraints = BalloonWindkesselConstraints(dt=dt)
        
    def forward(self, u_trajectory, t_eval, physio_params):
        # Use standard odeint here
        
        u_trajectory = u_trajectory.permute(0, 2, 1) # [B, T, V]
        B, T_steps, V = u_trajectory.shape
        alpha, tau_0, epsilon = physio_params
        
        def dynamics_func(t, state):
            t_idx = t / self.dt
            idx_low = torch.floor(t_idx).long().clamp(0, T_steps-2)
            idx_high = idx_low + 1
            weight = t_idx - idx_low.float()
            u_low = u_trajectory[:, idx_low, :]
            u_high = u_trajectory[:, idx_high, :]
            u_t = (1 - weight).view(-1, 1) * u_low + weight.view(-1, 1) * u_high
            
            s, f, v, q = state.unbind(-1)
            f = torch.clamp(f, min=1e-6)
            v = torch.clamp(v, min=1e-6)
            q = torch.clamp(q, min=1e-6)
            
            eps_b = epsilon.view(B, 1)
            tau_0_b = tau_0.view(B, 1)
            alpha_b = alpha.view(B, 1)
            
            ds = eps_b * u_t - self.constraints.kappa_s * s - self.constraints.kappa_f * (f - 1.0)
            df = s
            dv = (f - v**(1.0/alpha_b)) / tau_0_b
            
            dq_num = f * (1.0 - (1.0 - self.constraints.E0)**(1.0/f)) / self.constraints.E0
            dq_den = v**(1.0/alpha_b - 1.0) * q
            dq = (dq_num - dq_den) / tau_0_b
            
            return torch.stack([ds, df, dv, dq], dim=-1)
            
        x0 = torch.zeros(B, V, 4, device=u_trajectory.device)
        x0[:, :, 1:] = 1.0
        
        # Standard odeint for DHB closure
        trajectory = torchdiffeq.odeint(dynamics_func, x0, t_eval, method='euler')
        trajectory = trajectory.permute(1, 0, 2, 3)
        return trajectory

class PhysioNeuroFlow(nn.Module):
    def __init__(self, num_voxels, context_dim=768, time_steps=48, dt=0.5, patch_size=256):
        super().__init__()
        self.time_steps = time_steps
        self.dt = dt
        self.neural_flow = NeuralFlow(num_voxels, in_channels=time_steps, context_dim=context_dim, patch_size=patch_size)
        self.hypernet = Hypernetwork()
        self.dhb = DifferentiableHemodynamicBlock(dt=dt)
        
    def generate_neural_activity(self, z_stim):
        B = z_stim.shape[0]
        device = z_stim.device
        u_0 = torch.randn(B, self.neural_flow.num_voxels, self.time_steps, device=device)
        
        # Use Wrapper for Adjoint
        wrapper = FlowODEWrapper(self.neural_flow, z_stim)
        
        t_span = torch.tensor([0.0, 1.0], device=device)
        
        # Use Adjoint for Flow
        # Use dopri5 for adaptive step size without storing intermediates
        # or fixed steps if options provided, but we only output t_span points
        traj = torchdiffeq.odeint_adjoint(wrapper, u_0, t_span, method='dopri5')
        
        u_1 = traj[-1]
        return u_1

    def forward_train(self, z_stim, subject_id):
        u_pred = self.generate_neural_activity(z_stim)
        physio_params = self.hypernet(subject_id)
        t_eval = torch.arange(self.time_steps, device=z_stim.device) * self.dt
        state_trajectory = self.dhb(u_pred, t_eval, physio_params)
        y_pred = self.dhb.constraints.observation(state_trajectory)
        return y_pred, u_pred
