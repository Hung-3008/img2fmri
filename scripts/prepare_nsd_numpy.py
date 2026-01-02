
import os
import glob
import numpy as np
import nibabel as nib
import scipy.io
import h5py
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/workspace/sdb1/img2fmri/NSD/data')
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/workspace/sdb1/img2fmri/NSD/data/nsd/processed')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing Subject {args.sub}...")
    
    # 1. Load Brain Mask
    # 1. Load Brain Mask (Use nsdgeneral ROI for reduced voxel count)
    mask_path = os.path.join(args.data_root, f'nsddata/ppdata/subj{args.sub:02d}/func1pt8mm/roi/nsdgeneral.nii.gz')
    if not os.path.exists(mask_path):
        print(f"Mask not found at {mask_path}")
        return
    
    print("Loading mask...")
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata() > 0 # Boolean mask
    print(f"Mask shape: {mask.shape}, Voxels: {np.sum(mask)}")
    
    # 2. Process Betas
    beta_dir = os.path.join(args.data_root, f'nsddata_betas/ppdata/subj{args.sub:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR')
    if not os.path.exists(beta_dir):
        print(f"Beta dir not found at {beta_dir}")
        return
        
    # Get session files
    # Typically betas_session01.nii.gz
    # Use glob to find them
    # Check filename pattern provided in "download_data.sh" logs or standard NSD
    # It might be "betas_session01.nii.gz"
    # Wait, earlier 'ls' showed "FRACvalue_sessionXX.nii.gz". 
    # Let's assume files end with .nii.gz and contain "session".
    # Correction: The earlier `ls` output showed `FRACvalue_...` but standard NSD has `betas_sessionXX.nii.gz`.
    # I should check for `betas_session*.nii.gz`.
    
    session_files = sorted(glob.glob(os.path.join(beta_dir, 'betas_session*.nii.gz')))
    if len(session_files) == 0:
        print("No betas_session*.nii.gz found. Checking for other patterns...")
        session_files = sorted(glob.glob(os.path.join(beta_dir, '*.nii.gz')))
        # Filter out FRACvalue
        session_files = [f for f in session_files if 'FRACvalue' not in f]
        
    print(f"Found {len(session_files)} session files.")
    
    all_betas = []
    
    print("Loading and masking betas...")
    for fpath in tqdm(session_files):
        img = nib.load(fpath)
        data = img.get_fdata() # [X, Y, Z, T]
        # Data might be int16, convert to float32
        data = data.astype(np.float32)
        
        # Apply mask
        # Mask is [X, Y, Z]. Data is [X, Y, Z, T].
        # Transpose to [T, X, Y, Z] then mask?
        # Or data[mask, :] ?
        # Masking 4D array with 3D mask:
        # Reshape data to [V_total, T]
        data_flat = data.reshape(-1, data.shape[-1])
        mask_flat = mask.flatten()
        
        # Masked data [V_mask, T]
        masked = data_flat[mask_flat, :]
        
        # Transpose to [T, V_mask]
        masked = masked.T
        all_betas.append(masked)
        
    if len(all_betas) > 0:
        full_betas = np.concatenate(all_betas, axis=0) # [Total_Trials, V]
        
        # Apply Scaling (SynBrain Compatibility: /2000.0)
        print("Scaling fMRI data (/2000.0)...")
        full_betas = full_betas / 2000.0
        
        # Clip outliers
        # SynBrain doesn't explicitly clip after /2000 but standardizing usually follows.
        # We keep clipping to [-5, 5] just to be safe from artifacts.
        full_betas = np.clip(full_betas, -5.0, 5.0)
        
        print(f"Total fMRI shape: {full_betas.shape} | Range: [{full_betas.min():.4f}, {full_betas.max():.4f}]")
        
        out_fmri = os.path.join(args.output_dir, f'nsd_train_fmri_sub{args.sub}.npy')
        np.save(out_fmri, full_betas)
        print(f"Saved fMRI to {out_fmri}")
    else:
        print("No betas loaded!")
        
    # 3. Process Experiment Design (Stimulus Indices)
    exp_path = os.path.join(args.data_root, 'nsddata/experiments/nsd/nsd_expdesign.mat')
    if os.path.exists(exp_path):
        print("Loading Experiment Design...")
        mat = scipy.io.loadmat(exp_path)
        
        if 'masterordering' in mat:
            masterordering = mat['masterordering'] # [N_trials (30000), N_subjects]
            subj_order = masterordering[:, args.sub-1] # [30000]
             
            # Flatten to ensure 1D
            subj_order = subj_order.flatten()
            
            # Since betas are session-based, we need to know how many trials we actually have betas for.
            n_samples = full_betas.shape[0] if len(all_betas) > 0 else 0
            
            # Slice to match available fMRI data
            subj_order = subj_order[:n_samples]
            
            # 1-based to 0-based
            subj_order = subj_order - 1 
            
            out_stim = os.path.join(args.output_dir, f'nsd_train_stim_idxs_sub{args.sub}.npy')
            np.save(out_stim, subj_order)
            print(f"Saved Stim Indices to {out_stim} (Shape: {subj_order.shape})")
            
    else:
        print("ExpDesign not found!")

if __name__ == "__main__":
    main()
