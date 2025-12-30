import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args):
    """
    Visualizes BrainVAE input data (fMRI and CLIP embeddings).
    Checks if processed .npy files exist. If so, loads them and plots histograms.
    If not, advises the user to run data preparation.
    """
    
    # 1. Path Resolution
    # Adjust args.data_path if it's relative
    data_path = os.path.abspath(args.data_path)
    subject = args.subject
    
    print(f"Checking for data in: {data_path}")
    
    # Expected file paths based on prepare_nsddata_scale.py
    fmri_file = os.path.join(data_path, f'nsd/subj{subject:02d}/nsd_train_fmri_scale_sub{subject}.npy')
    clip_file = os.path.join(data_path, f'nsd/subj{subject:02d}/nsd_sdxl_clip_train_sub{subject}.npy')
    
    # 2. Check Existence
    fmri_exists = os.path.exists(fmri_file)
    clip_exists = os.path.exists(clip_file)
    
    if not fmri_exists or not clip_exists:
        print("\n\033[91m[WARNING] Processed training data not found!\033[0m")
        if not fmri_exists:
            print(f"  Missing: {fmri_file}")
        if not clip_exists:
            print(f"  Missing: {clip_file}")
            
        print("\nPlease run 'prepare_nsddata_scale.py' to generate these files from raw NSD data.")
        print(f"Raw data should be in: {os.path.join(data_path, 'nsddata')} and {os.path.join(data_path, 'nsddata_betas')}")
        return

    # 3. Data Loading & Visualization
    print(f"\nLoading fMRI data from: {fmri_file}")
    fmri_data = np.load(fmri_file) # Expected: [N, 3, Voxels] or [N, Voxels] depending on processing
    print(f"  Shape: {fmri_data.shape}")
    print(f"  Dtype: {fmri_data.dtype}")
    print(f"  Stats: Mean={np.mean(fmri_data):.4f}, Std={np.std(fmri_data):.4f}, Min={np.min(fmri_data):.4f}, Max={np.max(fmri_data):.4f}")

    print(f"\nLoading CLIP data from: {clip_file}")
    clip_data = np.load(clip_file) # Expected: [N, 256, 1664] probably
    print(f"  Shape: {clip_data.shape}")
    print(f"  Dtype: {clip_data.dtype}")
    print(f"  Stats: Mean={np.mean(clip_data):.4f}, Std={np.std(clip_data):.4f}, Min={np.min(clip_data):.4f}, Max={np.max(clip_data):.4f}")

    # Plotting
    print("\nGenerating histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # fMRI Histogram
    # Flatten just for histogram
    fmri_flat = fmri_data.flatten()
    # Sample if too large to speed up plotting
    if len(fmri_flat) > 100000:
        fmri_flat = np.random.choice(fmri_flat, 100000, replace=False)
        
    axes[0].hist(fmri_flat, bins=50, color='blue', alpha=0.7)
    axes[0].set_title('fMRI Beta Weights Distribution (Sampled)')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)

    # CLIP Histogram
    clip_flat = clip_data.flatten()
    if len(clip_flat) > 100000:
        clip_flat = np.random.choice(clip_flat, 100000, replace=False)

    axes[1].hist(clip_flat, bins=50, color='green', alpha=0.7)
    axes[1].set_title('CLIP Embedding Values Distribution (Sampled)')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_img = 'brainvae_input_dist.png'
    plt.savefig(output_img)
    print(f"\nVisualization saved to: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize BrainVAE Training Data')
    parser.add_argument("--data_path", type=str, required=True, help="Path to the 'data' directory containing 'nsd' folder")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1, 2, 5, 7)")
    
    args = parser.parse_args()
    main(args)
