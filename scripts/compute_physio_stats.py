
import numpy as np
import os
import argparse

def compute_stats(args):
    physio_path = os.path.join(args.data_root, f'nsd/physio/nsd_train_physio_sub{args.sub}.npy')
    output_path = os.path.join(args.data_root, f'nsd/physio/physio_stats_sub{args.sub}.npz')
    
    if not os.path.exists(physio_path):
        print(f"Error: File not found at {physio_path}")
        return

    print(f"Loading {physio_path}...")
    try:
        # Try standard load
        physio = np.load(physio_path, mmap_mode='r')
    except Exception:
        # Fallback to Shape Inference (copied from train.py logic)
        print("Standard load failed. Using raw memmap...")
        file_size = os.path.getsize(physio_path)
        
        T = 48
        C = 4
        # We know V should be 15724 (from train.py / test_fmri)
        # Or we can check clip file if we want to match train.py exactly.
        # Let's try to deduce V.
        # If we assume N=24930, V=5676.
        # If we assume N=9000, V=15724.
        
        # Let's check if 15724 is a divisor.
        total_float_count = file_size // 4
        possible_V = 15724
        
        if total_float_count % (T * possible_V * C) == 0:
            V = possible_V
            N = total_float_count // (T * V * C)
            print(f"Inferred N={N} based on V={V}")
        else:
            # Fallback to previous logic or error?
            print("Warning: V=15724 does not fit file size. Falling back to default assumption or asking for check.")
            # Let's try to load CLIP to get N if possible
            clip_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_train_clip_sub{args.sub}.npy')
            if os.path.exists(clip_path):
                 clip = np.load(clip_path)
                 N = len(clip)
                 V = total_float_count // (N * T * C)
                 print(f"Inferred V={V} based on CLIP N={N}")
            else:
                 # Original fallback (which caused error)
                 print("CLIP not found. Defaulting to N=24930 (RISKY).")
                 N = 24930
                 V = total_float_count // (N * T * C)

        physio = np.memmap(physio_path, dtype='float32', mode='r', shape=(N, T, V, C))
        
    print(f"Data Shape: {physio.shape}")
    
    # Compute Statistics across (N, T) dimension, keeping (V, C)
    # This gives per-voxel per-channel stats.
    # OR should we share stats across voxels?
    # Physio params (f, v, q) have similar ranges across voxels.
    # But 's' might vary locally.
    # Let's do PER-VOXEL normalization to maximize signal standardization.
    
    # Batch computation to save memory
    print("Computing Mean...")
    # sum and sq_sum
    mean = np.zeros((physio.shape[2], physio.shape[3]), dtype=np.float64)
    var = np.zeros((physio.shape[2], physio.shape[3]), dtype=np.float64)
    
    # Chunk size
    # Chunk size
    chunk_size = 100
    num_samples = physio.shape[0]
    total_frames = num_samples * physio.shape[1]
    
    from tqdm import tqdm

    # 1. Compute Mean
    print("Computing Mean...")
    total_sum = np.zeros_like(mean)
    
    for i in tqdm(range(0, num_samples, chunk_size), desc="Mean"):
        chunk = physio[i:i+chunk_size] # [B, T, V, C]
        chunk_sum = np.sum(chunk, axis=(0, 1)) # [V, C]
        total_sum += chunk_sum
        del chunk
        import gc
        gc.collect()
        
    mean = total_sum / total_frames
    
    # 2. Compute Std
    print("Computing Std...")
    total_sq_diff = np.zeros_like(var)
    
    for i in tqdm(range(0, num_samples, chunk_size), desc="Std"):
        chunk = physio[i:i+chunk_size]
        # (x - mean)^2
        diff = chunk - mean[np.newaxis, np.newaxis, :, :]
        sq_diff = np.sum(diff ** 2, axis=(0, 1))
        total_sq_diff += sq_diff
        del chunk
        del diff
        import gc
        gc.collect()
        
    std = np.sqrt(total_sq_diff / total_frames)
    
    # Avoid zero std
    std[std < 1e-6] = 1.0
    
    print("\n--- Stats Summary ---")
    variables = ['s', 'f', 'v', 'q']
    for i, var in enumerate(variables):
        print(f"{var}: Mean ~ {np.mean(mean[:, i]):.4f}, Std ~ {np.mean(std[:, i]):.4f}")
        
    print(f"Saving to {output_path}...")
    np.savez(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='data/NSD')
    args = parser.parse_args()
    compute_stats(args)
