import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-session", "--session",help="Amount of sessions",default=40)
parser.add_argument("-data_root", "--data_root", help="Root directory for NSD data", default="../../data/NSD")
args = parser.parse_args()
sub=int(args.sub)
session=int(args.session)
data_root = args.data_root
assert sub in [1,2,5,7]

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

stim_order_f = os.path.join(data_root, 'nsddata/experiments/nsd/nsd_expdesign.mat')
if not os.path.exists(stim_order_f):
    raise FileNotFoundError(f"Could not find {stim_order_f}. Please check data_root.")

stim_order = loadmat(stim_order_f)

## Selecting ids for training and test data
sig_train = {}
sig_test = {}
num_trials = session*750
for idx in range(num_trials):
    ''' nsdId as in design csv files'''
    nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1
    if stim_order['masterordering'][idx]>1000:
        if nsdId not in sig_train:
            sig_train[nsdId] = []
        sig_train[nsdId].append(idx)
    else:
        if nsdId not in sig_test:
            sig_test[nsdId] = []
        sig_test[nsdId].append(idx)

train_im_idx = list(sig_train.keys())
test_im_idx = list(sig_test.keys())

roi_dir = os.path.join(data_root, 'nsddata/ppdata/subj{:02d}/func1pt8mm/roi/'.format(sub))
betas_dir = os.path.join(data_root, 'nsddata_betas/ppdata/subj{:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub))

mask_filename = 'nsdgeneral.nii.gz'
mask_path = os.path.join(roi_dir, mask_filename)
if not os.path.exists(mask_path):
     raise FileNotFoundError(f"Could not find mask file at {mask_path}")
mask = nib.load(mask_path).get_fdata()
num_voxel = mask[mask>0].shape[0]

def scale_within_session(betas):
    betas = betas / 2000
    # print('Adjusted data (divided by 2000):')
    # print(betas.dtype, np.min(betas), np.max(betas), betas.shape)
    
    # print('z-scoring beta weights within this session...')
    # mb = np.mean(betas, axis=0, keepdims=True)
    # sb = np.std(betas, axis=0, keepdims=True)
    # # betas = np.nan_to_num((betas - mb) / (sb + 1e-6))
    # betas = np.nan_to_num((betas - mb) / np.clip(sb, 1e-8, 10000))
    # print(np.min(betas), np.max(betas), np.mean(betas), np.std(betas))
    # print ("mean = %.3f, sigma = %.3f" % (np.mean(mb), np.mean(sb)))
    
    return betas

fmri = np.zeros((num_trials, num_voxel)).astype(np.float32)
print("Loading fMRI data...")
for i in range(session):
    beta_filename = "betas_session{0:02d}.nii.gz".format(i+1)
    beta_path = os.path.join(betas_dir, beta_filename)
    if not os.path.exists(beta_path):
        print(f"Warning: {beta_path} not found. Skipping session {i+1}")
        continue
        
    beta_f = nib.load(beta_path).get_fdata().astype(np.float32)
    betas = beta_f[mask>0].transpose()
    fmri[i*750:(i+1)*750] = scale_within_session(betas)
    del beta_f
    del betas
    print(f"Loaded session {i+1}/{session}")
    
print("fMRI Data are loaded: ", fmri.shape)

stim_file_path = os.path.join(data_root, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
if not os.path.exists(stim_file_path):
    raise FileNotFoundError(f"Stimuli file not found at {stim_file_path}")

print("Opening stimuli file (lazy load)...")
f_stim = h5py.File(stim_file_path, 'r')
stim_dset = f_stim['imgBrick']
# stim = f_stim['imgBrick'][:] # REMOVED to save memory

print("Stimuli dataset shape: ", stim_dset.shape)

num_train, num_test = len(train_im_idx), len(test_im_idx)
vox_dim, im_dim, im_c = num_voxel, 425, 3
fmri_array = np.zeros((num_train,3,vox_dim))
stim_array = np.zeros((num_train,im_dim,im_dim,im_c))

print("Processing Training Data...")
for i,idx in enumerate(train_im_idx):
    if i % 100 == 0:
        print(f"Train {i}/{num_train}")
    stim_array[i] = stim_dset[idx] # Direct read from HDF5
    fmri_array[i] = fmri[sorted(sig_train[idx])]  #[3, voxels]
    # fmri_array[i] = fmri[sorted(sig_train[idx])].mean(0)

# Create output directories if they don't exist
output_dir = os.path.join(data_root, 'nsd/subj{:02d}'.format(sub))
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, 'nsd_train_fmri_scale_sub{}.npy'.format(sub)),fmri_array)
np.save(os.path.join(output_dir, 'nsd_train_stim_sub{}.npy'.format(sub)),stim_array)

print("Training data is saved.")

fmri_array = np.zeros((num_test,3,vox_dim))
stim_array = np.zeros((num_test,im_dim,im_dim,im_c))

print("Processing Test Data...")
for i,idx in enumerate(test_im_idx):
    if i % 100 == 0:
        print(f"Test {i}/{num_test}")
    stim_array[i] = stim_dset[idx] # Direct read from HDF5
    fmri_array[i] = fmri[sorted(sig_test[idx])]
    # fmri_array[i] = fmri[sorted(sig_test[idx])].mean(0)

np.save(os.path.join(output_dir, 'nsd_test_fmri_scale_sub{}.npy'.format(sub)),fmri_array)
np.save(os.path.join(output_dir, 'nsd_test_stim_sub{}.npy'.format(sub)),stim_array)

print("Test data is saved.")
f_stim.close() # Close HDF5 file

annots_path = os.path.join(data_root, 'annots/COCO_73k_annots_curated.npy')
if os.path.exists(annots_path):
    annots_cur = np.load(annots_path)
    
    captions_array = np.empty((num_train,5),dtype=annots_cur.dtype)
    print("Processing Training Captions...")
    for i,idx in enumerate(train_im_idx):
        captions_array[i,:] = annots_cur[idx,:]
        
    np.save(os.path.join(output_dir, 'nsd_train_cap_sub{}.npy'.format(sub)),captions_array )
        
    captions_array = np.empty((num_test,5),dtype=annots_cur.dtype)
    print("Processing Test Captions...")
    for i,idx in enumerate(test_im_idx):
        captions_array[i,:] = annots_cur[idx,:]
        
    np.save(os.path.join(output_dir, 'nsd_test_cap_sub{}.npy'.format(sub)),captions_array )
    print("Caption data are saved.")
else:
    print(f"Warning: Annotations file not found at {annots_path}. Skipping captions.")