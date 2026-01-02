
import os
import argparse
import numpy as np
import torch
import h5py
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

class HDF5IndexedDataset(Dataset):
    def __init__(self, hdf5_path, idxs_path):
        self.hdf5_path = hdf5_path
        self.idxs = np.load(idxs_path)
        self.h5_file = None
        self.dataset = None
        
    def _open_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, 'r')
            if 'imgBrick' in self.h5_file:
                self.key = 'imgBrick'
            elif 'stimuli' in self.h5_file:
                self.key = 'stimuli'
            else:
                self.key = list(self.h5_file.keys())[0]
            self.dataset = self.h5_file[self.key]
            
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, i):
        if self.h5_file is None:
            self._open_file()
            
        real_idx = self.idxs[i]
        img = self.dataset[real_idx] # [425, 425, 3] usually
        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/workspace/sdb1/img2fmri/NSD/data')
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    processed_dir = os.path.join(args.data_root, 'nsd/processed')
    idxs_path = os.path.join(processed_dir, f'nsd_train_stim_idxs_sub{args.sub}.npy')
    hdf5_path = os.path.join(args.data_root, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
    
    if not os.path.exists(idxs_path):
        print(f"Indices not found at {idxs_path}")
        return
        
    print(f"Loading indices for Subj {args.sub}...")
    dataset = HDF5IndexedDataset(hdf5_path, idxs_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    # Load CLIP (HuggingFace)
    print("Loading CLIP (HF)...")
    model_name = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    all_features = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch_imgs in tqdm(dataloader):
            # batch_imgs: tensor [B, 425, 425, 3]
            # Convert to PIL list
            batch_pil = [Image.fromarray(img.numpy().astype('uint8')) for img in batch_imgs]
            
            # Preprocess
            inputs = processor(images=batch_pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Encode
            # get_image_features returns pooled output [B, 768]
            features = model.get_image_features(**inputs)
            
            # Verify normalization? HF usually doesn't normalize output unless specified?
            # Standard CLIP embedding is usually normalized.
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            all_features.append(features.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0) # [N, 768]
    print(f"Features shape: {all_features.shape}")
    
    # Save
    out_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_train_clip_sub{args.sub}.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, all_features)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
