import numpy as np
import os
import glob
from tqdm import tqdm
from collections import defaultdict
import datetime
from datetimerange import DateTimeRange

from tools.abi_utils import *

'''
Calculate means and stds of each channel
'''
def calculate_mean_std(directory, expected_channels=12, file_pattern="abi_l1b_*.npy"):
    channel_sums = np.zeros(expected_channels)
    channel_sq_sums = np.zeros(expected_channels)
    num_elements = 0

    file_list = glob.glob(os.path.join(directory, file_pattern))
    for file_path in tqdm(file_list):
        data = np.load(file_path)

        # Check if data has the expected number of channels
        if data.shape[0] != expected_channels:
            print(f"Skipping file: {file_path} - Unexpected channel count: {data.shape[0]}")
            continue

        # Accumulate sums and squared sums
        channel_sums += np.sum(data, axis=(1, 2))
        channel_sq_sums += np.sum(data**2, axis=(1, 2))
        num_elements += data.shape[1] * data.shape[2]  # H * W per channel

    # Calculate mean and std deviation for each channel
    means = channel_sums / num_elements
    stds = np.sqrt(channel_sq_sums / num_elements - means**2)

    return means, stds

'''
Check the target patches files. If any value out of {0, 1, 2, 3, 4, 5} exists,
delet the target patch and the corresponding input patch file
'''
def validate_and_delete(data_dir='', input_pattern='abi_l1b_', target_pattern='abi_l2_'):
    input_files = sorted([f for f in os.listdir(data_dir) if f.startswith(input_pattern)])
    target_files = sorted([f for f in os.listdir(data_dir) if f.startswith(target_pattern)])

    assert len(input_files) == len(target_files), "Mismatch in L1B and L2 file counts."

    allowed_classes = {0, 1, 2, 3, 4, 5}
    deleted_pairs_count = 0

    for target_file in tqdm(target_files):
        target_path = os.path.join(data_dir, target_file)
        target_data = np.load(target_path)
        
        # Check if all elements in the target data are within the allowed classes
        if not np.all(np.isin(target_data, list(allowed_classes))):
            # Find the corresponding input file
            base_name = target_file.split(target_pattern)[1]
            corresponding_input_file = input_pattern + base_name
            
            # Construct the full path to the input file
            input_path = os.path.join(data_dir, corresponding_input_file)
            
            # Delete the files if the corresponding input file exists
            if os.path.exists(input_path):
                os.remove(input_path)
                os.remove(target_path)
                deleted_pairs_count += 1
                print(f"Deleted: {input_path} and {target_path}")
            else:
                print(f"Corresponding input file not found for {target_path}")

    print(f"Total number of pairs deleted: {deleted_pairs_count}")

'''
Check the damaged file.
Look for files that can not be read by numpy.
'''
def Check_Damage(data_dir, file_pattern = '/*.npy'):
    # Path to the folder containing the .npy files
    data_files = sorted(glob.glob(data_dir + file_pattern))

    # List to store names of damaged files
    damaged_files = []

    # Iterate over all .npy files in the folder
    for data_file in tqdm(data_files):
        file_path = os.path.join(data_dir, data_file)
        try:
            # Attempt to load the .npy file
            data = np.load(file_path)
        except ValueError:
            # If a ValueError occurs, add the filename to the damaged files list
            damaged_files.append(data_file)

    # Print or save the list of damaged files
    print(f"Damaged files: {damaged_files}")
    # Optionally, save the damaged file names to a text file
    with open('damaged_files.txt', 'w') as f:
        for damaged_file in damaged_files:
            f.write(f"{damaged_file}\n")
            
'''
Check matching of ABI L1b and L2 files.
Each L2 file should have 16 L1b files with the same starting time.
'''
def group_l1b_files(l1b_files):
    l1b_dict = defaultdict(list)
    for file in l1b_files:
        basename = os.path.basename(file)
        start_time = basename.split('_s')[-1].split('_')[0]
        band = basename.split('-M')[1].split('C')[1][:2]
        l1b_dict[start_time].append((band, file))
    return l1b_dict

def group_l2_files(*l2_file_lists):
    l2_dict = defaultdict(list)
    for l2_files in l2_file_lists:
        for file in l2_files:
            basename = os.path.basename(file)
            start_time = basename.split('_s')[-1].split('_')[0]
            var = abi_pattern_var(basename.split('OR_')[-1].split('-M')[0])
            l2_dict[start_time].append((var, file))
    return l2_dict
    
def match_l1b_l2_files(l1b_files, *l2_files_lists):
    # Step 1: Group L1b files by start time and band
    l1b_dict = defaultdict(list)
    for file in l1b_files:
        basename = os.path.basename(file)
        start_time = basename.split('_s')[-1].split('_')[0]
        band = basename.split('-M')[1].split('C')[1][:2]
        l1b_dict[start_time].append((band, file))

    # Step 2: Find common start times across all L2 lists
    l2_start_times = []
    for l2_files in l2_files_lists:
        current_l2_times = set()
        for file in l2_files:
            basename = os.path.basename(file)
            start_time = basename.split('_s')[-1].split('_')[0]
            current_l2_times.add(start_time)
        l2_start_times.append(current_l2_times)
    
    common_start_times = set.intersection(*l2_start_times)

    # Step 3: Create VALID start times set (with 16 bands)
    valid_start_times = set()
    for start_time in common_start_times:
        l1b_entries = l1b_dict.get(start_time, [])
        bands = {band for band, _ in l1b_entries}
        if len(bands) == 16 and len(l1b_entries) == 16:
            valid_start_times.add(start_time)

    # Step 4: Filter files using VALID start times
    valid_l1b_files = []
    for start_time in valid_start_times:
        valid_l1b_files.extend([f for _, f in l1b_dict[start_time]])

    valid_l2_results = [[] for _ in l2_files_lists]
    for i, l2_files in enumerate(l2_files_lists):
        for file in l2_files:
            basename = os.path.basename(file)
            start_time = basename.split('_s')[-1].split('_')[0]
            if start_time in valid_start_times:  # Only add if in valid_start_times
                valid_l2_results[i].append(file)
    print(f"After matching: {len(valid_l1b_files)} L1b files, {len(valid_l2_results[0])} files for each L2 variable.")
    return (valid_l1b_files, *valid_l2_results)

'''
The lon-lat for each abi granule are the same
(i, j) is the coordinate of the upper-left point of each chip
Need to find a method to crop each granual more randomly
'''
def divide_lon_lat_into_blocks(lon, lat, block_size):
    blocks_lon = []
    blocks_lat = []
    num_rows, num_cols = lat.shape
    for i in range(0, num_rows, block_size):
        for j in range(0, num_cols, block_size):
            block_lon = lon[i:i+block_size, j:j+block_size]
            block_lat = lat[i:i+block_size, j:j+block_size]
            if (not np.isnan(block_lon).any()) and (not np.isnan(block_lat).any()) :
                blocks_lon.append((i, j, block_lon))
                blocks_lat.append((i, j, block_lat))
    return blocks_lon, blocks_lat

# Usage:
# Uncomment any functions below use the specific function
# data_dir = '/umbc/rs/nasa-access/users/xingyan/fm_finetuning/abi_patches_training/'
# validate_and_delete(data_dir)

# means, stds = calculate_mean_std(l1b_path)
# print("Channel Means:", means)
# print("Channel Standard Deviations:", stds)