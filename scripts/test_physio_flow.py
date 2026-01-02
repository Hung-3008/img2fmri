
import torch
import sys
import os
sys.path.append('src/physio_synbrain')

from model import PhysioNeuroFlow

def test_model():
    print("Initializing Model...")
    # Small config for testing
    num_voxels = 100
    context_dim = 32
    time_steps = 10
    batch_size = 2
    
    model = PhysioNeuroFlow(
        num_voxels=num_voxels, 
        context_dim=context_dim,
        time_steps=time_steps, 
        dt=0.5
    )
    
    # Mock Data
    z_stim = torch.randn(batch_size, context_dim)
    subject_id = torch.zeros(batch_size, dtype=torch.long)
    
    print("Running Forward Pass (End-to-End)...")
    # This invokes odeint twice: Flow and DHB
    y_pred, u_pred = model.forward_train(z_stim, subject_id)
    
    print(f"y_pred shape: {y_pred.shape}") # Expected [B, T, V]
    print(f"u_pred shape: {u_pred.shape}") # Expected [B, V, T]
    
    # Check shapes
    assert y_pred.shape == (batch_size, time_steps, num_voxels)
    assert u_pred.shape == (batch_size, num_voxels, time_steps)
    
    print("Running Backward Pass...")
    target = torch.randn(batch_size, num_voxels)
    
    # Max pooling to simulate peak extraction
    y_peak, _ = torch.max(y_pred, dim=1)
    loss = torch.mean((y_peak - target)**2)
    
    loss.backward()
    
    # Check gradients
    print("Checking gradients...")
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            if param.grad.abs().sum() == 0:
                print(f"Warning: Param {name} has zero gradient.")
        else:
            print(f"Warning: Param {name} has NO gradient.")
            
    if has_grad:
        print("Success: Gradients propagated!")
    else:
        print("Failure: No gradients found.")

if __name__ == "__main__":
    test_model()
