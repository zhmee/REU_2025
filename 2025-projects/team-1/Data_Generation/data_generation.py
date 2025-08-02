import os
import random
import gc
import numpy as np
import h5py
import boto3
import s3fs
from tqdm import tqdm
from skimage.transform import resize
from tools.general_utils import *
from tools.abi_utils import *

# --- Configuration ---
# AWS S3 Setup (ensure `aws sso login --profile my-sso-profile` is run first)
session = boto3.Session(profile_name='my-sso-profile')
s3 = session.resource('s3')
BUCKET_NAME = 'noaa-goes18'
bucket = s3.Bucket(BUCKET_NAME)

# Sampling and Date Range
SAMPLE_FRACTION = 0.3  # Fraction of timestamps to process
BLOCK_SAMPLE_FRACTION = 1  # Fraction of blocks to process per timestamp
YEAR = 2023
DAY_MIN, DAY_MAX = 83, 113  # Day range (inclusive)
L2_VARIABLES = ['cloud_mask', 'cloud_phase', 'cod', 'cps']
VARIABLES = ['l1b'] + L2_VARIABLES
OUTPUT_DIR = "./2dcloud_scaled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Static Data (Lat/Lon Grids) ---
abi_lats = np.load('./abi_static_lats.npz')
abi_lons = np.load('./abi_static_lons.npz')

# Split into blocks (HKM=0.5km, 1KM, 2KM resolutions)
abi_patches = {
    0.5: divide_lon_lat_into_blocks(abi_lons['_hkm'], abi_lats['_hkm'], block_size=512),
    1: divide_lon_lat_into_blocks(abi_lons['_1km'], abi_lats['_1km'], block_size=256),
    2: divide_lon_lat_into_blocks(abi_lons['_2km'], abi_lats['_2km'], block_size=128)
}

# --- Helper Functions ---
def get_block_coordinates(block_idx: int, resolution: float) -> tuple:
    """Returns (longitude, latitude) for a given block and resolution."""
    lons, lats = abi_patches[resolution]
    return lons[block_idx][2], lats[block_idx][2]

def extract_block(data: np.ndarray, block_idx: int, resolution: float) -> np.ndarray:
    """Extracts a block from data, resizing if needed (to 128x128)."""
    lons, _ = abi_patches[resolution]
    sx, sy = lons[block_idx][0], lons[block_idx][1]
    block_size = 512 if resolution == 0.5 else 256 if resolution == 1 else 128
    block = data[sx:sx+block_size, sy:sy+block_size]
    
    if resolution in (0.5, 1):  # Downsample high-res blocks
        block = safe_resize(block, (128, 128))
    return block

def safe_resize(patch: np.ndarray, target_size: tuple = (128, 128)) -> np.ndarray:
    """Resizes a patch with anti-aliasing, handling NaNs."""
    if patch is None or np.isnan(patch).any():
        return None
    return resize(patch, target_size, anti_aliasing=True, preserve_range=True).astype(np.float32)

# --- Step 1: Collect S3 File List ---
file_list = {var: [] for var in VARIABLES}

print("Scanning S3 for ABI files...")
for var in VARIABLES:
    print(f"Collecting {var} files (Days {DAY_MIN}-{DAY_MAX}, {YEAR})...")
    for day in range(DAY_MIN, DAY_MAX + 1):
        for hour in range(24):
            prefix = f"{abi_file_pattern(var)}/{YEAR}/{day:03d}/{hour:02d}/"
            try:
                objects = list(bucket.objects.filter(Prefix=prefix))
                for obj in objects:
                    if obj.key.endswith('.nc') and f'OR_{abi_file_pattern(var)}' in obj.key:
                        file_list[var].append(obj.key)
            except Exception as e:
                print(f"Error accessing {prefix}: {e}")
    print(f"Found {len(file_list[var])} {var} files.")

# Match L1B and L2 files (only keep timestamps with all data)
valid_file_list = {}
valid_matching_lists = match_l1b_l2_files(*[file_list[var] for var in VARIABLES])
for var, valid_matching_list in zip(VARIABLES, valid_matching_lists):
    valid_file_list[var] = valid_matching_list

# Group files by timestamp
l1b_groups = group_l1b_files(valid_file_list['l1b'])
l2_groups = group_l2_files(*[valid_file_list[var] for var in L2_VARIABLES])
valid_timestamps = [
    s_time for s_time in l1b_groups 
    if len(l1b_groups[s_time]) == 16 and s_time in l2_groups
]
print(f"Found {len(valid_timestamps)} complete timestamps.")

# Randomly sample timestamps to process
sampled_timestamps = random.sample(valid_timestamps, int(SAMPLE_FRACTION * len(valid_timestamps)))

# --- Step 2: Process Sampled Timestamps ---
fs = s3fs.S3FileSystem(anon=True)

for s_time in tqdm(sampled_timestamps, desc="Processing timestamps"):
    # Initialize storage for this timestamp
    rad_data = {}  # Band data (C01-C16)
    rad_dqf = {}   # Data quality flags
    l2_data = {}   # L2 products (cloud_mask, etc.)
    l2_dqf = {}    # L2 quality flags

    # --- Load L1B Data (16 bands) ---
    for band_num, l1b_path in l1b_groups[s_time]:
        with fs.open(f'{BUCKET_NAME}/{l1b_path}') as f:
            rad_data[f'C{band_num:02d}'] = load_abi_l1b(band_num, f)
            with h5py.File(f, 'r') as fd:
                rad_dqf[f'C{band_num:02d}'] = fd['DQF'][:].astype(np.float16)

    # --- Load L2 Data ---
    for var, l2_path in l2_groups[s_time]:
        with fs.open(f'{BUCKET_NAME}/{l2_path}') as f:
            # Use custom loaders for each variable
            if var == 'cloud_mask':
                l2_data[var] = load_abi_cloud_mask(f)
            elif var == 'cloud_phase':
                l2_data[var] = load_abi_cloud_phase(f)
            elif var == 'cod':
                l2_data[var] = load_abi_cod(f, l2_data['cloud_mask'])
            elif var == 'cps':
                l2_data[var] = load_abi_cps(f, l2_data['cloud_mask'])
            else:  # Fallback for other variables
                with h5py.File(f, 'r') as fd:
                    l2_data[var] = fd[abi_var_name(var)][:].astype(np.float32)
            
            # Load quality flags
            with h5py.File(f, 'r') as fd:
                l2_dqf[var] = fd['DQF'][:].astype(np.float16)

    # --- Process Randomly Sampled Blocks ---
    total_blocks = len(abi_patches[2][0])  # 2km resolution blocks
    sampled_blocks = random.sample(range(total_blocks), int(BLOCK_SAMPLE_FRACTION * total_blocks))
    valid_count = 0

    for block_idx in sampled_blocks:
        patch_data = {}
        is_valid = True

        # --- Geolocation & Solar Angles ---
        lons, lats = get_block_coordinates(block_idx, 2)
        SoZeAn, SoAzAn, SeZeAn, SeAzAn = calculate_2d_angles(s_time, lons, lats)
        
        # Skip if solar zenith angle > 72°
        if np.nanmax(SoZeAn) > 72.0:
            continue
        
        patch_data.update({
            'lons': lons,
            'lats': lats,
            'SoZeAn': SoZeAn,
            'SoAzAn': SoAzAn,
            'SeZeAn': SeZeAn,
            'SeAzAn': SeAzAn
        })

        # --- Process L1B Bands ---
        rad_bands, rad_dqfs = [], []
        for band in range(1, 17):
            res = get_abi_l1b_resolution(band)
            band_key = f'C{band:02d}'
            rad_block = extract_block(rad_data[band_key], block_idx, res)
            dqf_block = extract_block(rad_dqf[band_key], block_idx, res)
            
            if rad_block is None or np.isnan(rad_block).any() or dqf_block is None:
                is_valid = False
                break
            rad_bands.append(rad_block)
            rad_dqfs.append(dqf_block)
        
        if not is_valid:
            continue
        
        patch_data['rad'] = np.stack(rad_bands, axis=-1)
        patch_data['rad_dqf'] = np.stack(rad_dqfs, axis=-1)

        # --- Process L2 Variables ---
        for var in L2_VARIABLES:
            l2_block = extract_block(l2_data[var], block_idx, 2)
            l2_dqf_block = extract_block(l2_dqf[var], block_idx, 2)
            
            if l2_block is None or np.isnan(l2_block).any() or l2_dqf_block is None:
                is_valid = False
                break
            
            # Special checks for cloud_phase
            if var == 'cloud_phase':
                unique_phases = np.unique(l2_block[l2_block != 0])  # Exclude background
                if len(unique_phases) <= 3:
                    is_valid = False
                    break
            
            patch_data[f'l2_{var}'] = l2_block
            patch_data[f'l2_dqf_{var}'] = l2_dqf_block

        # --- Save Valid Patch ---
        if is_valid:
            np.savez(
                os.path.join(OUTPUT_DIR, f"ABI_Data_{s_time}_{block_idx}.npz"),
                **patch_data
            )
            valid_count += 1

    print(f"Saved {valid_count} valid patches for {s_time}")
    del rad_data, rad_dqf, l2_data, l2_dqf  # Free memory
    gc.collect()

print("Processing complete!")

'''
Global Min-Max Scaling for 2D Cloud Data
This script applies global min-max scaling to the 
'rad' array in the processed ABI data files
'''
# Define paths
input_dir = OUTPUT_DIR
output_dir = './downstream_2dcloud_minmaxscaled'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Global min-max values for bands 7-16 (band index 6-15 in 0-based)
global_min_max = {
    0: (-0.0009483355097472668, 1.1382938623428345),  # Bands 1
    1: (-0.0012648939155042171, 1.1055210828781128),  # Bands 2
    2: (-0.0011233699042350054, 1.102694034576416),  # Bands 3
    3: (-0.009493589401245117, 0.6176661849021912),  # Bands 4
    4: (-0.0076236603781580925, 0.6795852780342102),  # Bands 5
    6: (196.7938690185547, 325.9274597167969),   # Band 7
    7: (189.560302734375, 261.0274658203125),    # Band 8
    8: (186.3356170654297, 268.9015808105469),   # Band 9
    9: (188.2379150390625, 273.8917541503906),   # Band 10
    10: (186.98793029785156, 310.46893310546875), # Band 11
    11: (117.65159606933594, 281.5234069824219),  # Band 12
    12: (185.39947509765625, 315.96697998046875), # Band 13
    13: (184.56333923339844, 315.795654296875),   # Band 14
    14: (184.06858825683594, 310.3517761230469),  # Band 15
    15: (186.7013397216797, 283.6570739746094)    # Band 16
}

def min_max_scale(data, band_idx):
    """Apply min-max scaling using global min-max values"""
    min_val, max_val = global_min_max[band_idx]
    return (data - min_val) / (max_val - min_val)

# Process each file
for filename in tqdm(random_examples):
    if filename.endswith('.npz'):
        input_path = filename
        output_path = os.path.join(output_dir, os.path.basename(filename))
        
        # Load original data
        with np.load(input_path) as data:
            # Create dictionary to hold all arrays
            save_dict = {}
            
            # Copy all arrays except 'rad'
            for array_name in data.files:
                if array_name != 'rad':
                    save_dict[array_name] = data[array_name]
            # Process 'rad' array
            rad_data = data['rad']
            # Initialize scaled array (same shape and dtype)
            rad_scaled = np.empty_like(rad_data)            
            # Copy bands 1-6 unchanged (if needed)
            rad_scaled[:, :, :6] = rad_data[:, :, :6]           
            # Min-max scale bands 7-16 using global min-max
            for band_idx in range(6, 16):
                band_data = rad_data[:, :, band_idx]
                rad_scaled[:, :, band_idx] = min_max_scale(band_data, band_idx)           
            # Add scaled rad data to save dict
            save_dict['rad'] = rad_scaled           
            # Save to new file
            np.savez(output_path, **save_dict)            
        print(f"Processed and saved: {filename}")
print("All files processed successfully!")