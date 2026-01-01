
import torch
from accelerate import Accelerator
import os
# from src.physio_synbrain.model import PhysioSiT # Avoid import issues by just checking files

def check_checkpoint(path):
    print(f"Checking {path}...")
    
    if not os.path.exists(path):
        print("Path does not exist")
        return

    print("Files:", os.listdir(path))
    
    # Check for safetensors or bin
    from safetensors.torch import load_file
    try:
        sf_path = os.path.join(path, "model.safetensors")
        if os.path.exists(sf_path):
            state_dict = load_file(sf_path)
            print("Successfully loaded model.safetensors")
            print("Num keys:", len(state_dict))
        else:
            pt_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(pt_path):
                state_dict = torch.load(pt_path, map_location='cpu')
                print("Successfully loaded pytorch_model.bin")
                print("Num keys:", len(state_dict))
            else:
                print("No model file found in checkpoint dir.")
    except Exception as e:
        print(f"Error loading: {e}")

check_checkpoint("/media/hung/data1/codes/imge2fmri/checkpoints/epoch_95")
