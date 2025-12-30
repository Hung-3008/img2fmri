
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
base_dir = '/media/hung/data1/codes/imge2fmri/data/NSD/data/nsddata_timeseries/ppdata/subj01/func1pt8mm'
fmri_path = os.path.join(base_dir, 'timeseries/timeseries_session01_run01.nii.gz')
design_path = os.path.join(base_dir, 'design/design_session01_run01.tsv')
output_path = '/media/hung/data1/codes/imge2fmri/output/bold_demo_run01.png'

print(f"Loading fMRI data from {fmri_path}...")
img = nib.load(fmri_path)
data = img.get_fdata() # Shape (X, Y, Z, T)
print(f"Data shape: {data.shape}")

# Load design matrix
print(f"Loading design matrix from {design_path}...")
design_df = pd.read_csv(design_path, header=None)
stim_onsets = design_df[0].values
print(f"Design shape: {stim_onsets.shape}")

# Handle padding: timeseries has 1 extra volume at the end usually
n_vols = min(data.shape[-1], len(stim_onsets))
data = data[..., :n_vols]
stim_onsets = stim_onsets[:n_vols]
print(f"Aligned length: {n_vols}")

# Select a voxel (simple heuristic: middle of the brain, or high variance)
# Middle of the volume
mid_x, mid_y, mid_z = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
voxel_ts = data[mid_x, mid_y, mid_z, :]

# Or find a voxel with high variance to ensure we see "something"
var_map = np.var(data, axis=-1)
# Mask out background (zeros)
var_map[np.mean(data, axis=-1) < 100] = 0
max_loc = np.unravel_index(np.argmax(var_map), var_map.shape)
print(f"Selected voxel at {max_loc} (high variance)")
voxel_ts = data[max_loc[0], max_loc[1], max_loc[2], :]

# Create plot
plt.figure(figsize=(15, 6))

# Plot BOLD signal
plt.plot(voxel_ts, label='BOLD Signal (Raw)', color='black', linewidth=1.5)

# Overlay stimuli events
# Stimulus ID > 0 means a picture was shown
event_indices = np.where(stim_onsets > 0)[0]
# Use a simple vertical line or span
ymin, ymax = plt.ylim()
plt.vlines(event_indices, ymin, ymax, colors='red', alpha=0.2, linewidth=1, label='Stimulus Onset')

plt.title(f"BOLD Signal Sample - Subject 01, Session 01, Run 01\nVoxel: {max_loc}")
plt.xlabel("Time (Volumes / TR)")
plt.ylabel("Signal Intensity")
plt.legend()
plt.tight_layout()

# Save
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
