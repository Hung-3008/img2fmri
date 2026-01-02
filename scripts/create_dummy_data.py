
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs='+', default=[30000, 768])
    args = parser.parse_args()
    
    print(f"Creating dummy data at {args.path} with shape {args.shape}...")
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    data = np.random.randn(*args.shape).astype(np.float32)
    np.save(args.path, data)
    print("Saved.")

if __name__ == "__main__":
    main()
