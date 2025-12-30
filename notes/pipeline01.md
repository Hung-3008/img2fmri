Here is the comprehensive design for the Physio-SynBrain framework, written in English. This proposal integrates the probabilistic semantic mapping of SynBrain with the rigorous physics-based constraints of PCFM to generate biologically plausible fMRI data.
Physio-SynBrain: Physics-Constrained Probabilistic fMRI Synthesis
1. Overview
Physio-SynBrain is a proposed hybrid framework designed to solve the "one-to-many" mapping problem in visual encoding while ensuring physiological validity.
Semantic Consistency: It retains the S2N Mapper and CLIP encoding strategies from SynBrain to capture high-level visual semantics1.


Generative Backbone: It replaces the VAE decoder with a Conditional Flow Matching (CFM) model to generate neural latent states2.


Physiological Validity: It applies Physics-Constrained Flow Matching (PCFM) during inference. The generation trajectory is projected onto the manifold defined by the Balloon-Windkessel hemodynamic model, ensuring the output signal obeys the biological laws of neurovascular coupling333.


2. Mathematical Formulation
2.1. The Balloon-Windkessel Constraint Operator ($\mathcal{H}$)
The core innovation is treating the biological generation of fMRI signals as a hard constraint. Let the system state be vector $\mathbf{x} = [s, f, v, q]^T$ (signal, flow, volume, deoxyhemoglobin) and the input be neural activity $u(t)$.
The constraint operator $\mathcal{H}(\mathbf{x}, u) = 0$ is defined by the residuals of the hemodynamic ODEs:


$$\mathcal{H}(\mathbf{x}, u) = \begin{bmatrix} \dot{s} - (\epsilon u(t) - \kappa_s s - \kappa_f (f - 1)) \\ \dot{f} - s \\ \tau_0 \dot{v} - (f - v^{1/\alpha}) \\ \tau_0 \dot{q} - \left( f \frac{1 - (1 - E_0)^{1/f}}{E_0} - v^{1/\alpha - 1} q \right) \end{bmatrix} = \mathbf{0}$$
The observable BOLD signal $y(t)$ is a non-linear function of these states:


$$y(t) = V_0 \left[ k_1 (1 - q) + k_2 \left( 1 - \frac{q}{v} \right) + k_3 (1 - v) \right]$$
2.2. Conditional Flow Matching (CFM)
We aim to learn a time-dependent vector field $v_t$ that pushes a noise distribution $p_0$ (Gaussian) to the data distribution $p_1$ (neural/hemodynamic states). The flow $\phi_t$ is defined by the ODE6:


$$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x); \theta)$$
Unlike SynBrain which maps to a VAE latent space, we condition the flow on the visual semantic embedding $C$ derived from the S2N Mapper. The objective function follows the Optimal Transport path7777:


$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ || v_t(x_t|C) - (x_1 - x_0) ||^2 \right]$$
2.3. The PCFM Projection Step
During inference (sampling), at each integration step, the estimated state $\hat{x}$ is projected onto the constraint manifold defined by $\mathcal{H}(\hat{x}) = 0$. Using the Gauss-Newton projection derived in PCFM8888, the update rule is:


$$x_{proj} = \hat{x} - J^T (JJ^T)^{-1} \mathcal{H}(\hat{x})$$
Where $J = \nabla \mathcal{H}(\hat{x})$ is the Jacobian of the Balloon-Windkessel equations with respect to the state variables.
3. Physio-SynBrain Algorithm
The following pseudo-code describes the inference process. It combines the semantic injection of SynBrain (Stages 1 & 2) with the constrained generation of PCFM (Stage 3).
Algorithm: Physio-SynBrain Inference
Inputs:
$I$: Visual Stimulus (Image).
$v_\theta$: Pre-trained Conditional Flow Matching Network.
$\text{S2N}$: Pre-trained Semantic-to-Neural Mapper9.


$\text{CLIP}$: Frozen Visual Encoder10.


$\mathcal{H}$: Balloon-Windkessel Constraint Operator11.


$N$: Number of Euler integration steps.
$\lambda$: Relaxed correction penalty weight (optional)12.


Output:
$Y_{BOLD}$: Synthesized fMRI signal.

Python


def Physio_SynBrain_Inference(I, v_theta, S2N, CLIP, N):
    
    # --- Stage 1 & 2: Semantic Conditioning (SynBrain) ---
    # 1. Extract visual semantics using CLIP [cite: 1640]
    z_clip = CLIP.encode(I)
    
    # 2. Map semantics to neural flow condition via S2N [cite: 1656]
    # In Physio-SynBrain, S2N provides the condition C for the flow
    C = S2N(z_clip) 
    
    # --- Stage 3: Physics-Constrained Flow Generation (PCFM) ---
    # 3. Initialize state from prior distribution (Gaussian Noise) [cite: 674]
    # State x includes hemodynamic variables [s, f, v, q]
    x_t = Sample_Gaussian_Noise() 
    
    dt = 1.0 / N
    
    # 4. Integration Loop (Time t from 0 to 1)
    for i in range(N):
        t = i * dt
        
        # A. Predict velocity field conditioned on Semantic Map C
        velocity = v_theta(x_t, t, condition=C)
        
        # B. Forward Euler Step (Shooting) [cite: 141]
        x_next_hat = x_t + velocity * dt
        
        # C. PCFM Projection Step [cite: 142-143]
        # Calculate residual of Balloon-Windkessel equations
        residual = H(x_next_hat)
        
        if norm(residual) > tolerance:
            # Calculate Jacobian of the constraints
            J = Jacobian(H, x_next_hat)
            
            # Project onto the valid hemodynamic manifold
            # x_proj = x - J.T * inv(J * J.T) * residual
            x_next_proj = Solve_Gauss_Newton_Projection(x_next_hat, J, residual)
            
            # D. Correction via OT-Interpolant (Reverse Step) [cite: 153]
            # Estimate reverse trajectory to maintain flow consistency
            x_corrected = OT_Reverse_Step(x_next_proj, x_0=x_t, t_next=t+dt)
            
            x_next = x_corrected
        else:
            x_next = x_next_hat
            
        # Update state
        x_t = x_next

    # 5. Observation Generation
    # Convert final hemodynamic states to BOLD signal 
    s, f, v, q = x_t
    Y_BOLD = BOLD_Observation_Model(v, q)
    
    return Y_BOLD


4. Summary of Improvements
Component
SynBrain (Original)
Physio-SynBrain (Proposed)
Benefit
Latent Mapping
VAE (Probabilistic)
Optimal Transport Flow Matching
Faster, straighter generation paths13.


Guidance
S2N $\to$ VAE Latent
S2N $\to$ Flow Condition
Direct conditioning avoids distribution mismatch14.


Physics
None (Data-driven)
Hard Constraints (PCFM)
Guarantees output satisfies Hemodynamic ODEs15.


Inference
Decoder Forward Pass
Iterative Solver + Manifold Projection
Zero-shot enforcement of biological laws without retraining16.




