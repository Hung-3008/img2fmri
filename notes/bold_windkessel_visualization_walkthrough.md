# Synthetic fMRI Data Generation & Visualization

## Overview
Successfully created a complete pipeline for generating synthetic 4D fMRI data from neural activity using the **Balloon-Windkessel hemodynamic model**, then visualizing the results.

---

## ✅ What Was Accomplished

### 1. **Understanding the Balloon-Windkessel Model**
- Confirmed that [`BOLDModel`](file:///media/hung/data1/codes/imge2fmri/models/neurolib/neurolib/models/bold/model.py) implements the Balloon-Windkessel model
- Model converts neural activity → BOLD fMRI signal through hemodynamic response
- Based on Friston 2000 & 2003 papers

**Key Model Variables:**
- `X`: Vasodilatory signal
- `F`: Blood flow
- `Q`: Deoxyhemoglobin content
- `V`: Blood volume

### 2. **Created Demo Pipeline Script**
File: [`bold_to_fmri_demo.py`](file:///media/hung/data1/codes/imge2fmri/scripts/bold_to_fmri_demo.py)

**Pipeline Steps:**
1. Load AAL2 brain atlas (120 regions)
2. Generate synthetic neural activity
3. Run Balloon-Windkessel simulation
4. Map BOLD signal to 4D voxel space
5. Save as NIfTI file

**Generated Output:**
- [`simulated_fmri.nii.gz`](file:///media/hung/data1/codes/imge2fmri/data/simulated_fmri.nii.gz) (44 KB)
- Shape: `(91, 109, 91, 5)` - 5 timepoints at 0.5 Hz

### 3. **Created Comprehensive Visualization Script**
File: [`visualize_synthetic_fmri.py`](file:///media/hung/data1/codes/imge2fmri/scripts/visualize_synthetic_fmri.py)

**Visualization Functions:**
- `visualize_neural_and_bold()` - Time series comparison
- `visualize_brain_slices()` - 3D slice views
- `visualize_timeseries_montage()` - All timepoints
- `visualize_regional_bold_heatmap()` - Regional activity heatmap
- `create_comprehensive_summary()` - Multi-panel overview

---

## 📊 Generated Visualizations

All visualizations saved to [`visualizations/`](file:///media/hung/data1/codes/imge2fmri/visualizations)

### 1. Neural Activity & BOLD Signal Comparison
![Neural-BOLD Comparison](file:///media/hung/data1/codes/imge2fmri/visualizations/neural_bold_comparison.png)

**File:** `neural_bold_comparison.png` (210 KB)

Shows the relationship between:
- Left panels: Raw neural activity (firing rates in Hz)
- Right panels: Resulting BOLD signal after hemodynamic convolution
- Demonstrates ~4-6 second delay typical of hemodynamic response

### 2. Brain Slice Visualizations
![Brain Slices](file:///media/hung/data1/codes/imge2fmri/visualizations/brain_slices.png)

**File:** `brain_slices.png` (77 KB)

Three orthogonal views at timepoint 2:
- **Axial** (horizontal): Top-down brain view
- **Coronal** (frontal): Front view
- **Sagittal** (side): Side view

### 3. Time Series Montage
![Time Series Montage](file:///media/hung/data1/codes/imge2fmri/visualizations/timeseries_montage.png)

**File:** `timeseries_montage.png` (35 KB)

Shows all 5 timepoints side-by-side to visualize temporal dynamics.

### 4. Comprehensive Summary
![Comprehensive Summary](file:///media/hung/data1/codes/imge2fmri/visualizations/comprehensive_summary.png)

**File:** `comprehensive_summary.png` (169 KB)

Multi-panel figure including:
- Neural activity traces (sample regions)
- BOLD signal traces
- Regional BOLD heatmap
- Brain slices from all three views
- Time series evolution

---

## 🔬 Technical Details

### Input Specifications
```python
N_regions = 120           # AAL2 atlas regions
T_steps = 10000          # 10 seconds @ 1ms resolution
dt = 1.0                 # Timestep in milliseconds
pattern = 'block'        # Activity pattern type
```

### Processing Flow
```
Neural Activity (120 × 10,000)
        ↓
BOLDModel (Balloon-Windkessel)
        ↓
BOLD Signal (120 × 5) @ 0.5 Hz
        ↓
AAL2 Atlas Mapping
        ↓
4D fMRI (91 × 109 × 91 × 5)
```

### Activity Patterns Available
- `'random'` - Random fluctuations around baseline
- `'oscillatory'` - Brain rhythm oscillations (5-20 Hz)
- `'block'` - Task-based block design
- `'event'` - Brief event-related activations

---

## 📝 Example Usage

### Running the Full Pipeline
```bash
cd /media/hung/data1/codes/imge2fmri
python scripts/bold_to_fmri_demo.py
```

### Generating Visualizations
```bash
python scripts/visualize_synthetic_fmri.py
```

### Using in Jupyter Notebook
```python
import sys
sys.path.insert(0, 'models/neurolib')
from neurolib.models.bold.model import BOLDModel

# Generate neural activity
neural_activity = generate_neural_activity(120, 10000)

# Run BOLD model
bold_model = BOLDModel(N=120, dt=1.0)
bold_model.run(neural_activity)
bold_signal = bold_model.BOLD
```

---

## 🎯 Key Insights

### Can We Reconstruct 4D fMRI from BOLD Signal?
**Answer: YES**, with the following approach:

1. **Regional BOLD Signal** (N regions) + **Brain Atlas** → **Voxel-wise 4D fMRI**
2. Each voxel inherits the BOLD signal of its atlas region
3. Spatial smoothing and noise addition increase realism

### Limitations
- **Uniform signal within regions** - All voxels in same region have identical signal
- **Resolution limited by atlas** - AAL2 has only 120 regions
- **Sharp boundaries** - Requires smoothing for realism
- **No within-region variability** - Unless explicitly added

### Improvements
✓ Use finer atlases (Schaefer 400/1000, Glasser 360)  
✓ Add within-region variability  
✓ Apply spatial smoothing (implemented)  
✓ Add physiological noise (implemented)

---

## 📚 References

**Balloon-Windkessel Model:**
- Friston et al. (2000) - Nonlinear responses in fMRI: The balloon model
- Friston et al. (2003) - Dynamic causal modeling

**Atlas:**
- AAL2 - Automated Anatomical Labeling atlas (120 regions)

---

## ✅ Verification

All scripts executed successfully:
- ✓ BOLD model simulation runs correctly
- ✓ 4D fMRI generation completes without errors
- ✓ All visualization types generated
- ✓ Output files created with expected dimensions

**Total outputs:** 5 files (1 NIfTI + 4 visualizations)
