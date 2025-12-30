
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def load_large_array(path, shape=None, dtype=np.float32):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    try:
        print(f"Attempting np.load with mmap_mode='r' for {path}...")
        data = np.load(path, mmap_mode='r')
        print(f"Success! Shape: {data.shape}, Dtype: {data.dtype}")
        return data
    except Exception as e:
        print(f"np.load failed: {e}")
        if shape is not None:
            print("Attempting manual np.memmap (offset=0, raw binary)...")
            try:
                # Try offset=0 since header inspection showed raw data
                data = np.memmap(path, dtype=dtype, mode='r', shape=shape, offset=0)
                print(f"Manual memmap Success! Shape: {data.shape}")
                return data
            except Exception as e2:
                print(f"Manual memmap failed: {e2}")
        return None

def plot_comparison(idx, stim_data, static_fmri, dynamic_physio):
    print(f"Plotting for sample {idx}...")
    
    # 1. Image
    img = stim_data[idx]
    
    # 2. Static Beta
    static_sample = static_fmri[idx]
    static_activity = np.mean(static_sample, axis=0)
    
    # 3. Dynamic
    top_k = 5
    top_voxels = np.argsort(np.abs(static_activity))[-top_k:]
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # A. Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img.astype(np.uint8) if img.max() > 1 else img)
    ax_img.set_title(f"Stimulus {idx}")
    ax_img.axis('off')
    
    # B. Static Dist
    ax_static = fig.add_subplot(gs[0, 1])
    ax_static.hist(static_activity, bins=100, log=True)
    ax_static.set_title("Static Betas Distribution")
    
    # C. Dynamic
    ax_dyn = fig.add_subplot(gs[1, :])
    time_steps = np.arange(dynamic_physio.shape[1]) * 0.5
    
    for v in top_voxels:
        # Plot 's' state (index 0)
        traj = dynamic_physio[idx, :, v, 0]
        ax_dyn.plot(time_steps, traj, label=f'Voxel {v} (Static={static_activity[v]:.2f})')
        
    ax_dyn.set_title("Dynamic Response (State 's') for Top Active Voxels")
    ax_dyn.set_xlabel("Time (s)")
    ax_dyn.legend()
    
    # D. Full States for Best Voxel
    best_v = top_voxels[-1]
    ax_states = fig.add_subplot(gs[2, :])
    labels = ['s', 'f', 'v', 'q']
    for i in range(4):
        ax_states.plot(time_steps, dynamic_physio[idx, :, best_v, i], label=labels[i])
    
    ax_states.set_title(f"Full States for Voxel {best_v}")
    ax_states.legend()
    
    output_path = 'vis_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    data_root = 'data/NSD/nsd/subj01'
    physio_root = 'data/NSD/nsd/physio'
    
    stim_path = os.path.join(data_root, 'nsd_train_stim_sub1.npy')
    static_path = os.path.join(data_root, 'nsd_train_fmri_scale_sub1.npy')
    physio_path = os.path.join(physio_root, 'nsd_train_physio_sub1.npy')
    
    print("Loading Stimuli...")
    stim = load_large_array(stim_path)
    
    print("Loading Static fMRI...")
    static = load_large_array(static_path) # Assuming float64 from previous output
    
    print("Loading Dynamic Physio...")
    physio = load_large_array(physio_path)
    
    if physio is None and static is not None:
        # Infer shape
        N = static.shape[0]
        V = static.shape[2]
        T = 48
        S = 4
        print(f"Inferring shape from static data: ({N}, {T}, {V}, {S})")
        physio = load_large_array(physio_path, shape=(N, T, V, S), dtype=np.float32)

    if stim is not None and static is not None and physio is not None:
        plot_comparison(0, stim, static, physio)
    else:
        print("Failed to load all data.")

if __name__ == "__main__":
    main()
