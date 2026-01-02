
import torch
import sys
import os
sys.path.append('src/physio_synbrain')

from model import PhysioNeuroFlow

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Initializing Model...")
    # Small config for testing
    num_voxels = 1000 # Dummy
    context_dim = 768
    time_steps = 10
    batch_size = 2
    
    model = PhysioNeuroFlow(
        num_voxels=num_voxels, 
        context_dim=context_dim,
        time_steps=time_steps, 
        dt=0.5
    ).to(device)
    
    # Mock Data
    z_stim = torch.randn(batch_size, context_dim)
    subject_id = torch.zeros(batch_size, dtype=torch.long)
    z_stim = torch.randn(batch_size, context_dim).to(device) # Move data to device
    subject_id = torch.zeros(batch_size, dtype=torch.long).to(device) # Move data to device
    
    print("Running Forward Pass (End-to-End)...")
    # This invokes odeint twice: Flow and DHB
    y_pred, u_pred = model.forward_train(z_stim, subject_id)
    
    print(f"y_pred shape: {y_pred.shape}") # Expected [B, T, V]
    print("Test 1 Passed: Output shape correct.")
    
    # Test 2: Specific ROI Size (15724 voxels)
    print("Test 2: Specific ROI Size (15724)...")
    model_roi = PhysioNeuroFlow(num_voxels=15724, context_dim=768, time_steps=48, dt=0.5, patch_size=64).to(device)
    # Create mock data for the specific ROI test, ensuring it's on the correct device
    z_stim_roi = torch.randn(batch_size, 768).to(device)
    subject_id_roi = torch.zeros(batch_size, dtype=torch.long).to(device)
    y_pred_roi, u_pred_roi = model_roi.forward_train(z_stim_roi, subject_id_roi)
    assert u_pred_roi.shape == (2, 15724, 48), f"Expected (2, 15724, 48), got {u_pred_roi.shape}"
    print("Test 2 Passed: ROI Size correct.") # Expected [B, V, T]
    
    print(f"u_pred shape: {u_pred.shape}") # Expected [B, V, T]
    
    # Check shapes
    assert y_pred.shape == (batch_size, time_steps, num_voxels)
    assert u_pred.shape == (batch_size, num_voxels, time_steps)
    
    print("Running Backward Pass...")
    target = torch.randn(batch_size, num_voxels).to(device)
    
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
