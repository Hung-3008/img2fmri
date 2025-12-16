import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import nibabel as nib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import glob
import scipy.io as sio

class NSDDataset(Dataset):
    """
    Dataset class for Natural Scenes Dataset (NSD).
    Loads paired fMRI time-series and stimuli images.
    """
    def __init__(self, 
                 config: Dict, 
                 split: str = 'train', 
                 subject_id: str = 'subj01'):
        """
        Args:
            config: Configuration dictionary.
            split: 'train' or 'val'.
            subject_id: Subject ID (e.g., 'subj01').
        """
        self.config = config
        self.fmri_root = config['data']['fmri_root']
        self.mask_root = config['data']['mask_root']
        self.split = split
        self.subject_id = subject_id
        
        self.img_size = config['data']['img_size']
        self.fmri_resolution = tuple(config['data']['fmri_resolution'])
        self.num_frames = config['data']['num_frames']
        
        # Load Brain Mask
        # Path: nsddata/ppdata/subj01/func1pt8mm/brainmask.nii.gz
        mask_path = os.path.join(
            self.mask_root, 
            subject_id, 
            "func1pt8mm", 
            "brainmask.nii.gz"
        )
        if os.path.exists(mask_path):
            print(f"Loading brain mask from {mask_path}")
            self.mask_nii = nib.load(mask_path)
            self.mask = self.mask_nii.get_fdata().astype(bool)
            # Resize mask if needed? 
            # The mask is in 1.8mm space. If we resize fMRI, we should resize mask too.
            # But resizing boolean mask is tricky. Ideally we keep data in native resolution 
            # or crop. For now, assuming we might resize fMRI, we'll deal with it in __getitem__.
        else:
            print(f"Warning: Brain mask not found at {mask_path}")
            self.mask = None

        # Paths to fMRI files
        # Pattern: nsddata_timeseries/ppdata/subjAA/func1pt8mm/timeseries/timeseries_sessionXX_runYY.nii.gz
        self.fmri_files = sorted(glob.glob(os.path.join(
            self.fmri_root, 
            subject_id, 
            "func1pt8mm", 
            "timeseries", 
            "*.nii.gz"
        )))
        
        # Filter for train/val split (simple split by run number for prototype)
        total_runs = len(self.fmri_files)
        if split == 'train':
            self.fmri_files = self.fmri_files[:int(0.9 * total_runs)]
        else:
            self.fmri_files = self.fmri_files[int(0.9 * total_runs):]

        print(f"[{split}] Found {len(self.fmri_files)} fMRI runs for {subject_id}")

    def __len__(self):
        if not hasattr(self, 'index_map'):
            self._build_index()
        return len(self.index_map)

    def _build_index(self):
        self.index_map = []
        stride = max(1, self.num_frames // 2)
        
        for file_idx, fpath in enumerate(self.fmri_files):
            try:
                # We can peek header without loading data
                img = nib.load(fpath)
                n_vols = img.shape[-1]
                
                for t in range(0, n_vols - self.num_frames + 1, stride):
                    self.index_map.append((file_idx, t))
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                
    def __getitem__(self, idx):
        file_idx, t_start = self.index_map[idx]
        fpath = self.fmri_files[file_idx]
        
        # Load fMRI slice
        nii = nib.load(fpath)
        fmri_data = nii.dataobj[..., t_start : t_start + self.num_frames]
        fmri_data = np.array(fmri_data).astype(np.float32) # (X, Y, Z, T)
        
        # Apply mask if available
        if self.mask is not None:
            # Mask is (X, Y, Z). Expand to (X, Y, Z, T)
            # fmri_data[~self.mask] = 0 # Optional: Zero out background
            pass 

        # Transpose to (C, T, D, H, W). PyTorch 3D convs usually want (N, C, D, H, W)
        # Here we have time as 'Depth' if we treat it as 3D? 
        # No, VQGAN_3D is likely (B, C, D, H, W).
        # SC-STD uses 4D VQGAN (Spatiotemporal).
        # Our VQGAN_fMRI wrapper currently reshapes (B, C, T, D, H, W) -> (B*T, C, D, H, W).
        # So we just return (1, T, X, Y, Z). x,y,z map to d,h,w.
        
        fmri_data = np.transpose(fmri_data, (3, 0, 1, 2)) # (T, X, Y, Z)
        
        fmri_tensor = torch.from_numpy(fmri_data).unsqueeze(0) # (1, T, X, Y, Z)
        
        # Interpolate to target resolution [64,64,64]
        if fmri_tensor.shape[2:] != self.fmri_resolution:
             fmri_tensor = torch.nn.functional.interpolate(
                fmri_tensor, 
                size=self.fmri_resolution, 
                mode='trilinear', 
                align_corners=False
            )
            
        # Normalize
        mean = fmri_tensor.mean()
        std = fmri_tensor.std() + 1e-6
        fmri_tensor = (fmri_tensor - mean) / std
        
        # Dummy image for now (Stimuli loading requires HDF5 + Design Matrix mapping)
        dummy_image = torch.zeros((3, self.img_size, self.img_size))
        
        return {
            "fmri": fmri_tensor,
            "image": dummy_image,
            "subject_id": self.subject_id
        }

class NSDDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = config.get('num_workers', 4)
        
    def setup(self, stage=None):
        self.train_dataset = NSDDataset(self.config, split='train')
        self.val_dataset = NSDDataset(self.config, split='val')
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
