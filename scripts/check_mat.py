
import scipy.io
import argparse
import os

def main():
    path = '/workspace/sdb1/img2fmri/NSD/data/nsddata/experiments/nsd/nsd_expdesign.mat'
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return
        
    mat = scipy.io.loadmat(path)
    print("Keys in .mat:", mat.keys())
    
    if 'masterordering' in mat:
        mo = mat['masterordering']
        print(f"masterordering shape: {mo.shape}")
        print(f"First 10 of col 0: {mo[:, 0][:10] if len(mo) > 10 else mo}")
        print(f"First 10 of row 0: {mo[0, :10] if len(mo) > 10 else mo}")
        
    if 'subjectim' in mat:
        si = mat['subjectim']
        print(f"subjectim shape: {si.shape}")

if __name__ == "__main__":
    main()
