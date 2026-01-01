import torch
import torch.nn as nn

class BalloonWindkesselConstraints(nn.Module):
    def __init__(self, dt=0.1, params=None):
        super().__init__()
        self.dt = dt
        # Default Parameters (Friston 2000)
        default_params = {
            'epsilon': 1.0, 
            'kappa_s': 0.65, 
            'kappa_f': 0.41, 
            'tau_0': 0.98,
            'alpha': 0.32,
            'E0': 0.4,
            'V0': 0.02
        }
        if params is not None:
            default_params.update(params)
            
        for k, v in default_params.items():
            self.register_buffer(k, torch.tensor(v))
            
    def dynamics(self, x, u=None):
        """
        Calculates the time derivative \dot{x} given state x and input u.
        x: [..., 4] (s, f, v, q)
        u: [..., 1] or scalar. If None, assumes u=0 (decay phase)
        """
        s, f, v, q = x.unbind(-1)
        
        # Enforce positivity for physiological variables to avoid NaN in pow()
        f = torch.clamp(f, min=1e-6)
        v = torch.clamp(v, min=1e-6)
        q = torch.clamp(q, min=1e-6)
        
        if u is None:
            u = torch.zeros_like(s)
            
        ds = self.epsilon * u - self.kappa_s * s - self.kappa_f * (f - 1.0)
        df = s
        dv = (f - v**(1.0/self.alpha)) / self.tau_0
        
        # dq equation
        # q_dot = (f/E0 * (1 - (1-E0)^(1/f)) - v^(1/alpha - 1)*q) / tau0
        dq_num = f * (1.0 - (1.0 - self.E0)**(1.0/f)) / self.E0
        dq_den = v**(1.0/self.alpha - 1.0) * q
        dq = (dq_num - dq_den) / self.tau_0
        
        return torch.stack([ds, df, dv, dq], dim=-1)

    def residual(self, x_prev, x_curr, u=None):
        """
        Calculates the residual of the discretized ODE.
        Forward Euler: x_curr - (x_prev + dt * f(x_prev)) = 0
        Or Implicit: x_curr - x_prev - dt * f(x_curr) = 0
        PCFM usually enforces f(x) = v_flow.
        But here we are in a trajectory optimization / constraint enforcement setting.
        If we are checking if a POINT x satisfies the vector field v at time t?
        No, PCFM enforces that the GENERATED point maps to a valid constraint.
        For ODEs constraint, it usually means the generated trajectory must satisfy the ODE.
        The constraint H(x, u) = 0 usually applies to the temporal evolution.
        
        However, if we treat the Flow Field v(x,t) itself as the thing to constrain:
        v_model(x,t) should approx f(x).
        
        In the "Physio-SynBrain" proposal (Equation 20):
        H(x, u) = 0 is defined as the ODE residuals.
        This implies we are looking at the tuple (x, \dot{x}).
        But in flow matching, we predict v = \dot{x}.
        So the constraint check on a generated 'x' is ambiguous unless we know 'u'.
        
        Proposal 2.3 PCFM Projection Step:
        "At each integration step, the estimated state x_hat is projected... defined by H(x_hat)=0"
        Wait, H(x)=0 usually means x is on a manifold.
        The Balloon-Windkessel creates a manifold in (s,f,v,q) space ONLY if we assume steady state or specific u.
        BUT, if we are integrating: x_next = x_prev + v * dt.
        The constraint is that x_next must be reachable from x_prev via the ODE.
        So H(x_next) = x_next - (x_prev + F(x_prev, u) * dt) = 0.
        This requires x_prev.
        
        So the constraint function must be state-dependent: H(x_next | x_prev).
        """
        dx = self.dynamics(x_prev, u)
        return x_curr - (x_prev + self.dt * dx)

    def observation(self, x):
        """
        BOLD Signal Equation
        y = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
        """
        s, f, v, q = x.unbind(-1)
        
        # Clamp for safety
        v = torch.clamp(v, min=1e-6)
        q = torch.clamp(q, min=1e-6)
        
        k1 = 7.0 * self.E0
        k2 = 2.0
        k3 = 2.0 * self.E0 - 0.2
        
        # Avoid division by zero
        v_safe = v
        
        y = self.V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v_safe) + k3 * (1.0 - v))
        return y
