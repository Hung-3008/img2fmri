
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to generated training physio data
physio_path = "data/NSD/nsd/physio/nsd_train_physio_sub1.npy"

if not os.path.exists(physio_path):
    print(f"File not found: {physio_path}")
    exit()

print("Loading physio data (memmap)...")
try:
    physio = np.load(physio_path, mmap_mode='r')
except Exception as e:
    print(f"Standard load failed: {e}. Trying shape inference...")
    file_size = os.path.getsize(physio_path)
    N = 24930 # Known N for train
    # Or try to infer?
    # Let's just use raw memmap if standard fails
    T = 48
    C = 4
    total_elements = file_size // 4
    V = total_elements // (N * T * C)
    physio = np.memmap(physio_path, dtype='float32', mode='r', shape=(N, T, V, C))

print(f"Physio Shape: {physio.shape}")

# Check first sample
sample = physio[0] # [T, V, 4]
s = sample[:, :, 0]
f = sample[:, :, 1]
v = sample[:, :, 2]
q = sample[:, :, 3]

print("\nSample 0 Stats:")
print(f"s: min={s.min()}, max={s.max()}, mean={s.mean()}")
print(f"f: min={f.min()}, max={f.max()}, mean={f.mean()}")
print(f"v: min={v.min()}, max={v.max()}, mean={v.mean()}")
print(f"q: min={q.min()}, max={q.max()}, mean={q.mean()}")

# Check Spatial variation at peak (approx t=12)
peak_idx = 12
s_peak = s[peak_idx]
print(f"\nSpatial Std at Peak (t=12): {s_peak.std()}")
if s_peak.std() < 1e-6:
    print("WARNING: Spatial pattern is flat! Simulation might have failed to capture spatial info.")

# Check correlation with original fMRI?
# We assume we don't have it loaded here easily. But if std is 0, that's the smoking gun.
