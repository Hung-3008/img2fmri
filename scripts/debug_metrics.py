
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the saved results
results_path = "/media/hung/data1/codes/imge2fmri/results/test_metric_align/evaluation_results.npy"
if not os.path.exists(results_path):
    print("Results file not found.")
    exit()

data = np.load(results_path, allow_pickle=True).item()
preds = data['preds']
# We need to load GT separately as it wasn't saved in the dict to save space
# Or we can just load the test file again
test_fmri_path = "/media/hung/data1/codes/imge2fmri/data/NSD/nsd/subj01/nsd_test_fmri_scale_sub1.npy"
test_fmri = np.load(test_fmri_path, mmap_mode='r')

# Get the first sample (since we limited to 1)
p = preds[0]
t_trials = test_fmri[0] # [3, V]
t_mean = np.mean(t_trials, axis=0)

print(f"Prediction Shape: {p.shape}")
print(f"GT Trials Shape: {t_trials.shape}")

print("\n--- Statistics ---")
print(f"Pred: Mean={p.mean():.4f}, Std={p.std():.4f}, Min={p.min():.4f}, Max={p.max():.4f}")
print(f"GT (Mean Trial): Mean={t_mean.mean():.4f}, Std={t_mean.std():.4f}, Min={t_mean.min():.4f}, Max={t_mean.max():.4f}")

# Calculate Pearson manually
from scipy.stats import pearsonr
corr, _ = pearsonr(p, t_mean)
print(f"\nPearson (vs Mean GT): {corr:.4f}")

# Calculate MSE manually
mse = np.mean((p - t_mean)**2)
print(f"MSE (vs Mean GT): {mse:.4f}")

# Check individual trials
for i in range(3):
    t = t_trials[i]
    c, _ = pearsonr(p, t)
    m = np.mean((p - t)**2)
    print(f"Trial {i}: Pearson={c:.4f}, MSE={m:.4f}")
