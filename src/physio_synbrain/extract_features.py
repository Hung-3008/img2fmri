import argparse
import os
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class NSDStimuliDataset(Dataset):
    def __init__(self, stim_path):
        self.stim = np.load(stim_path, mmap_mode='r')
        
    def __len__(self):
        return len(self.stim)
    
    def __getitem__(self, idx):
        # Scan is 425x425x3, uint8 0-255
        img = self.stim[idx]
        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='data/NSD')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP
    model, preprocess = clip.load("ViT-L/14", device=device)
    
    # Input Path
    stim_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_{args.split}_stim_sub{args.sub}.npy')
    if not os.path.exists(stim_path):
        print(f"Stimuli file not found: {stim_path}")
        return
        
    print(f"Loading stimuli from {stim_path}...")
    dataset = NSDStimuliDataset(stim_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    
    all_features = []
    
    with torch.no_grad():
        for batch_imgs in tqdm(dataloader):
            # batch_imgs: [B, 425, 425, 3] -> Need to process
            # CLIP preprocess expects PIL images or Tensor [B, 3, 224, 224]
            # Since we have custom batch, we need to apply preprocess manually
            
            # Convert to PIL list
            batch_pil = [Image.fromarray(img.numpy().astype('uint8')) for img in batch_imgs]
            
            # Apply CLIP preprocess
            batch_tensor = torch.stack([preprocess(img) for img in batch_pil]).to(device)
            
            # Encode
            features = model.encode_image(batch_tensor)
            all_features.append(features.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    print(f"Extracted features shape: {all_features.shape}")
    
    output_path = os.path.join(args.data_root, f'nsd/subj{args.sub:02d}/nsd_{args.split}_clip_sub{args.sub}.npy')
    np.save(output_path, all_features)
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
