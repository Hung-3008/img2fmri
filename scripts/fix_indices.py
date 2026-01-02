
import os
import numpy as np
import scipy.io
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/workspace/sdb1/img2fmri/NSD/data')
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/workspace/sdb1/img2fmri/NSD/data/nsd/processed')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    exp_path = os.path.join(args.data_root, 'nsddata/experiments/nsd/nsd_expdesign.mat')
    if not os.path.exists(exp_path):
        print("ExpDesign not found!")
        return
        
    print("Loading Experiment Design...")
    mat = scipy.io.loadmat(exp_path)
    
    # 1. Get Master Ordering (Local Index 1..10000)
    # Shape (1, 30000)
    masterordering = mat['masterordering'][0] # [30000]
    print(f"Master Ordering shape: {masterordering.shape}")
    
    # 2. Get Subject Image Mapping (Global ID 1..73000)
    # Shape (8, 10000)
    subjectim = mat['subjectim']
    subj_map = subjectim[args.sub-1] # [10000]
    print(f"Subject Map shape: {subj_map.shape}")
    
    # 3. Map Trial -> Global ID
    # indices are 1-based
    
    # Check max/min
    print(f"MO min: {masterordering.min()}, max: {masterordering.max()}")
    print(f"SM min: {subj_map.min()}, max: {subj_map.max()}")
    
    # Map
    # global_id = subj_map[local_idx - 1]
    # result is 1-based Global ID
    # We want 0-based index for HDF5 (assuming HDF5 is 0-indexed 0..72999)
    # So we want (subj_map[...] - 1) or just verify HDF5 indexing.
    # Usually HDF5 is 0-indexed matches ID-1.
    
    # Vectorized map
    # -1 for 0-based index into subj_map
    bs_local_idx = masterordering - 1 
    
    # global_ids_1based
    global_ids_1based = subj_map[bs_local_idx] # [30000]
    
    # global_ids_0based
    global_ids_0based = global_ids_1based - 1
    
    print(f"Mapped IDs (0-based) min: {global_ids_0based.min()}, max: {global_ids_0based.max()}")
    
    out_path = os.path.join(args.output_dir, f'nsd_train_stim_idxs_sub{args.sub}.npy')
    np.save(out_path, global_ids_0based)
    print(f"Saved Corrected Indices to {out_path} (Shape: {global_ids_0based.shape})")

if __name__ == "__main__":
    main()
