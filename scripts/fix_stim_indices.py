
import os
import argparse
import numpy as np
import scipy.io

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/workspace/sdb1/img2fmri/NSD/data')
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/workspace/sdb1/img2fmri/NSD/data/nsd/processed')
    args = parser.parse_args()
    
    exp_path = os.path.join(args.data_root, 'nsddata/experiments/nsd/nsd_expdesign.mat')
    if not os.path.exists(exp_path):
        print("ExpDesign not found!")
        return

    print("Loading Experiment Design...")
    # struct_as_record=False is better for MATLAB structs, squeeze_me=True simplifies 1xN arrays
    mat = scipy.io.loadmat(exp_path, struct_as_record=False, squeeze_me=True)
    
    if 'masterordering' not in mat or 'subjectim' not in mat:
        print("Missing masterordering or subjectim in mat file.")
        return

    masterordering = mat['masterordering'] # [30000]
    subjectim = mat['subjectim'] # [8, 30000] typically
    
    print(f"Masterordering shape: {masterordering.shape}")
    print(f"Subjectim shape: {subjectim.shape}")
    
    # Logic from SynBrain:
    # nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1
    
    # Vectorized implementation
    subj_idx = args.sub - 1
    
    # masterordering contains 1-based indices into the trial types
    # We subtract 1 to get 0-based indices for python array
    trial_type_indices = masterordering - 1
    
    # Retrieve 1-based Image IDs (1..73000)
    # subjectim[subj_idx, :] gives the map for this subject
    # We index it by trial_type_indices
    image_ids_1based = subjectim[subj_idx, trial_type_indices]
    
    # Convert to 0-based Image IDs (0..72999)
    image_ids_0based = image_ids_1based - 1
    
    # Slice to match existing fMRI data length if needed
    # Check existing fMRI file
    fmri_path = os.path.join(args.output_dir, f'nsd_train_fmri_sub{args.sub}.npy')
    if os.path.exists(fmri_path):
        fmri_data = np.load(fmri_path, mmap_mode='r')
        n_samples = fmri_data.shape[0]
        print(f"Trimming to match fMRI samples: {n_samples}")
        image_ids_0based = image_ids_0based[:n_samples]
    
    out_stim = os.path.join(args.output_dir, f'nsd_train_stim_idxs_sub{args.sub}.npy')
    np.save(out_stim, image_ids_0based)
    print(f"Saved Fixed Stim Indices to {out_stim} (Shape: {image_ids_0based.shape})")
    
    # Verify range
    print(f"Index Range: [{image_ids_0based.min()}, {image_ids_0based.max()}]")

if __name__ == "__main__":
    main()
