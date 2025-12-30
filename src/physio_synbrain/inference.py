import torch
import torch.nn as nn
from torch.func import vmap, jacrev
import gc
import numpy as np

# --- PCFM Core Logic (Adapted from PCFM/pcfm/pcfm_sampling.py) ---

def compute_jacobian(fn, inputs):
    def fn_flat(x):
        return fn(x).flatten()
    J = jacrev(fn_flat)(inputs)
    m = J.shape[0]
    n = inputs.numel()
    return J.reshape(m, n)

class PCFMSampler:
    def __init__(self, model, constraints, num_voxels, shape, device='cuda'):
        """
        model: Trained PhysioSiT model
        constraints: BalloonWindkesselConstraints module
        shape: (T, V, C) or similar.
        """
        self.model = model
        self.constraints = constraints
        self.device = device
        self.T, self.V, self.C = shape
        
    def _model_wrapper(self, x, t, c, tau_grid):
        """
        Wraps the component-wise model to apply to the whole trajectory.
        x: [B, T, V, C] (Trajectory Batch)
        t: scalar (Flow time)
        c: [B, D] (Context)
        tau_grid: [T] (Physio time points)
        
        Returns: v [B, T, V, C]
        """
        B = x.shape[0]
        # Reshape to [B*T, V, C] for batch processing by PhysioSiT
        x_flat = x.view(-1, self.V, self.C)
        
        # Expand conditions
        t_batch = torch.full((B * self.T,), t, device=self.device)
        
        # c is [B, D]. We need to repeat for each time step.
        # c_expand: [B, 1, D] -> [B, T, D] -> [B*T, D]
        c_expand = c.unsqueeze(1).expand(-1, self.T, -1).reshape(-1, c.shape[-1])
        
        # tau is [T]. We need to tile for batch.
        # tau_expand: [1, T] -> [B, T] -> [B*T]
        tau_expand = tau_grid.unsqueeze(0).expand(B, -1).reshape(-1)
        
        # Run Model
        v_flat = self.model(x_flat, t_batch, c_expand, tau_expand)
        
        # Reshape back to [B, T, V, C]
        return v_flat.view(B, self.T, self.V, self.C)

    def _constraint_wrapper(self, u):
        """
        Constraint H(u) = 0.
        u: [B, T, V, C] flattened or shaped?
        PCFM helpers usually expect u to be flat for Jacobian, OR we handle shapes.
        
        For Windkessel:
        x_{i+1} - (x_i + f(x_i)*dt_physio) = 0
        
        Returns residuals: [B, T-1, V, C] (or similar)
        """
        # u is [B, T, V, C]
        # Let's assume u is [T, V*C] for single batch PCFM or we use vmap.
        # Current PCFM implementation in pcfm_sampling.py uses vmap over batch.
        
        # Here we define the function for a SINGLE sample 'u' of shape [T, V, C]
        # Returns [T-1, V, C]
        
        x_curr = u[1:] # t=1..T
        x_prev = u[:-1] # t=0..T-1
        
        # Calc dynamics f(x_prev)
        # constraints.dynamics expects [..., 4]
        # Assuming no input u (decay), or we need to infer u?
        # The user's constraints.py dynamics(x, u=None) defaults to u=0.
        # For generated data, we might assume u=0 (resting state decay?) 
        # OR we assume the model learned to generate 'active' states.
        # If we constrain with u=0, we force decay only? That would kill the signal!
        
        # CRITICAL: The Windkessel model is driven by 'u' (Neural Activity).
        # We don't observe 'u'. We infer it or we model the JOINT (x, u).
        # But our state is 4 channels (s,f,v,q). 'u' is external.
        # If we constrain with u=0, we are saying "Neural activity is zero".
        # This will prevent signal generation.
        
        # SOLUTION: We must allow 'u' to be free?
        # OR, we model 'u' as part of the state?
        # If 'u' is not in the state, constraint is H(x_next, x_prev, u) = 0.
        # Since 'u' is unknown, we cannot strictly enforce the ODE unless we solve for u too.
        
        # Loophole:
        # Maybe 's' (flow inducing signal) effectively captures 'u'?
        # ds/dt = eps*u - k*s - ...
        # If u is free, then s can move anywhere?
        # Not quite. s has inertia.
        # But if we don't know u, we can't constrain ds/dt exactly.
        
        # However, f, v, q purely depend on s, f, v.
        # df/dt = s
        # dv/dt = (f - v^...)/tau
        # dq/dt = ...
        
        # These 3 equations (f, v, q) DO NOT depend on u!
        # They only depend on state variables.
        # So we can enforce constraints on f, v, q!
        # s is "free" (driven by hidden u).
        
        # So H(x) will only return residuals for components 1, 2, 3 (f, v, q).
        # Component 0 (s) is unconstrained (or weakly constrained by smoothness).
        
        # Calculate full dynamics
        dx = self.constraints.dynamics(x_prev, u=None) # u=0 just for 'ds' calc.
        
        # Expected x_next
        x_pred = x_prev + self.constraints.dt * dx
        
        res = x_curr - x_pred
        
        # Mask out 's' channel (channel 0)
        # We only enforce f, v, q consistency.
        # res has shape [T-1, V, 4]
        # We zero out res[..., 0] so it doesn't affect norm/jacobian.
        mask = torch.ones_like(res)
        mask[..., 0] = 0.0
        
        return res * mask

    def pcfm_generate(self, dummy_x, cond, tau_grid, steps=20):
        """
        dummy_x: [B, T, V, C] (Shape reference + Noise start)
        cond: [B, D]
        tau_grid: [T]
        """
        B = dummy_x.shape[0]
        device = self.device
        
        # 1. Initialize Noise
        x0 = torch.randn_like(dummy_x) # Noise
        x1 = x0.clone() # Current State
        
        # Time Grid for Flow
        dt_flow = 1.0 / steps
        t_grid = torch.linspace(0, 1, steps)
        
        # PCFM Projection Logic
        def h_func_single(u_flat):
            # u_flat: [T*V*C]
            u = u_flat.view(self.T, self.V, self.C)
            res = self._constraint_wrapper(u) # [T-1, V, C]
            return res.flatten()
            
        for t in t_grid:
            # Predict Vector Field
            with torch.no_grad():
                v_pred = self._model_wrapper(x1, t, cond, tau_grid)
            
            # Candidate Next Step (Euler)
            # x_next_cand = x1 + v_pred * dt_flow
            
            # PCFM Correction Step
            # We want to find u (x_next) such that H(u) = 0 and u is close to x_next_cand
            # OR, we project v_pred such that x1 + v_proj * dt stays on manifold?
            # PCFM paper: "Project the vector field".
            
            # Wrapper for batched projection
            # We treat each sample in batch independently
            # Since pcfm_sampling.py uses vmap, we can try to implement a simple Newton projection here.
            
            # Simplified PCFM:
            # 1. Take Euler step partial: u_cand = x1 + v * dt
            # 2. Project u_cand onto H(u)=0
            # 3. x_new = u_proj
            
             # Step 1
            u_cand = x1 + v_pred * dt_flow
            
            # Step 2: Project
            # For efficiency, we just do 1 Newton step or so.
            # Jacobian of H w.r.t u is sparse (temporal/spatial).
            # Full Jacobian is too big [TVC, TVC].
            # But constraints are local in time (t, t+1) and voxel-wise independent.
            # So we can project per-voxel!
            
            # Reshape to [B*V, T, C] to isolate independent problems?
            # Yes! Constraints don't mix voxels.
            # H(u) for voxel i only depends on u[..., i, :].
            
            u_cand_reshaped = u_cand.permute(0, 2, 1, 3).reshape(B*self.V, self.T, self.C)
            
            # Define per-voxel constraint function
            def h_voxel(u_traj):
                # u_traj: [T, C]
                # returns [T-1, C]
                x_c = u_traj[1:]
                x_p = u_traj[:-1]
                dx = self.constraints.dynamics(x_p) # u=0
                pred = x_p + self.constraints.dt * dx
                res = x_c - pred
                # Mask s (channel 0)
                mask = torch.ones_like(res)
                mask[..., 0] = 0
                return (res * mask).flatten()

            # Run Manifold Projection (Newton) on batch of voxels
            # Since B*V is large (~1000 * 32), we might need to batch this loop or use vmap.
            
            # Let's assume we skip rigorous projection for now and just rely on Training?
            # User asked to "Combine FCFM". I should try to implement the projection.
            
            # Just applying correction to u_cand:
            # u_new = u_cand - J.T (J J.T)^-1 H(u_cand)
            
            # For now, let's implement a 'soft' correction or just return unconstrained if too slow.
            # But the request is specific.
            
            # Let's do a simplified correction:
            # We just force f, v, q to match s?
            # Forward simulate s?
            # If we trust 's' from the model, we can just re-integrate f, v, q from s!
            # That guarantees H=0 exactly.
            # This is a "Projection" operator. u_proj = Integrate(u_cand[s]).
            
            # This is much faster and precise than Newton for this specific ODE structure.
            # s is free. f,v,q are determined by s and initial conditions.
            
            # REVISED PROJECTION:
            # 1. Extract 's' channel from u_cand.
            # 2. Re-integrate to get compatible f, v, q using the ODE solver.
            # 3. Replace f, v, q channels in u_cand.
            
            # Need Initial Conditions (IC).
            # Assume steady state IC? s=0, f=1, v=1, q=1.
            # Or learn IC?
            # The model predicts the whole trajectory including t=0.
            # We can trust u_cand[0] as IC.
            
            u_proj = self._integrate_projection(u_cand)
            
            x1 = u_proj
            
        return x1

    def _integrate_projection(self, u_traj):
        """
        Projects trajectory onto Windkessel manifold by keeping 's' fixed and reintegrating others.
        u_traj: [B, T, V, 4]
        """
        s = u_traj[..., 0] # [B, T, V]
        
        # Initial State
        x0 = u_traj[:, 0, :, :] # [B, V, 4]
        # Or force steady state: s=0, f=1, v=1, q=1?
        # Let's trust model's IC for s, but reset f,v,q?
        # Better: Trust model's IC.
        
        curr_x = x0.clone()
        
        out_list = [curr_x]
        
        dt = self.constraints.dt
        
        # Iterate time
        for t in range(self.T - 1):
            # We have s[t].
            # Actually, constraint is:
            # ds/dt = ... (ignored)
            # df/dt = s
            
            # So to update f, we use s.
            # f_next = f_curr + s_curr * dt
            
            prev_x = out_list[-1]
            prev_s, prev_f, prev_v, prev_q = prev_x.unbind(-1)
            
            # Use 's' from TRAJECTORY (u_traj) or from integrated state?
            # We keep 's' from u_traj as the "Driver".
            s_driver = s[:, t, :] # [B, V]
            
            # Dynamics
            # df = s
            df = s_driver
            
            # dv = (f - v^...)/tau
            # dq = ...
            # These depend on f, v, q. We use the INTEGRATED values (prev_x).
            
            # Full dynamics call (but override ds/df logic)
            # We can re-use constraints.dynamics but ignore ds return.
            
            # Construct a temporary state using 's_driver' but 'prev_f/v/q'
            # Wait, dynamics(x) uses x.s for df.
            # So we create temp_x = (s_driver, prev_f, prev_v, prev_q)
            temp_x = torch.stack([s_driver, prev_f, prev_v, prev_q], dim=-1)
            
            d_all = self.constraints.dynamics(temp_x, u=None)
            ds, df, dv, dq = d_all.unbind(-1)
            
            # Update
            # s: Keep from u_traj logic?
            # The user's model generated a trajectory for s. We keep it.
            next_s = s[:, t+1, :]
            
            next_f = prev_f + df * dt
            next_v = prev_v + dv * dt
            next_q = prev_q + dq * dt
            
            next_state = torch.stack([next_s, next_f, next_v, next_q], dim=-1)
            out_list.append(next_state)
            
        return torch.stack(out_list, dim=1)

