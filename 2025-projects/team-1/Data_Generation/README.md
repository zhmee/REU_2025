# GOES-18 ABI Data Processing Pipeline

This pipeline downloads, processes, and extracts blocks of data from GOES-18 Advanced Baseline Imager (ABI) L1B and L2 products stored on AWS S3. The L1 products have 16 spectral bands and are reshaped to 2 km spatial resolution. The L2 cloud products are all at 2 km spatial resolution. The output is a set of spatially aligned patches (128x128 pixels) containing radiance, cloud properties, and geolocation data. Below is a table of ABI products contained in the dataset.

| Level | F/C disk* | Product       | Variable | Explain        | Resolution (km)                                              |
| :---- | :-------- | :------------ | :------- | :------------- | :----------------------------------------------------------- |
| L1B   | F         | ABI-L1b-Rad   | Rad      | Spectral bands | Original: [1, 0.5, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]<br />After processing: 2 |
| L2    | F         | ABI-L2-ACM    | Clear    | Clear sky mask | 2                                                            |
| L2    | F         | ABI-L2-ACTP   | Phase    | Cloud phase    | 2                                                            |
| L2    | F         | ABI-L2-COD2KM | COD      | COD*           | 2                                                            |
| L2    | F         | ABI-L2-CPS    | CPS      | CPS*           | 2                                                            |

## Key Features

- **Data Extraction**: Downloads ABI L1B (16 bands) and L2 (cloud mask, phase, COD, CPS) data from NOAA's public S3 bucket.
- **Spatial Sampling**: Extracts random blocks (patches) from full-disk scans.
- **Preprocessing**: 
  - Conversion to TOA Reflectance and Brightness Temperature: digital numbers (DNs) from the MODIS bands are converted into Top-of-Atmosphere (TOA) reflectance for visible and Near-Infrared (NIR) bands, and brightness temperature (BT) for Thermal Infrared (TIR) bands.
  - Scaling: Min-max normalization for all 16 bands
- **Quality Control**: Filters patches by:
  - Solar zenith angle (<72°)
  - Valid data coverage (no NaN values)
  - Cloud phase diversity (for `cloud_phase` products)
- **Resampling**: Harmonizes all data to 2km resolution (128x128 patches).

## Requirements

- Python 3.8+

- AWS CLI (configured with `my-sso-profile` SSO)

- Required packages:

  ```bash
  pip install numpy h5py netCDF4 xarray s3fs boto3 scikit-image tqdm
  ```

## Setup

1. **Clone this repository**:

   ```bash
   git clone [your-repo-url]
   cd [your-repo]
   ```

2. **Configure AWS SSO**:

   Please read [instructions for configuring AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)

## Usage

### Configuration

Modify these variables in `process_abi_data.py`:

```python
# Sampling
SAMPLE_FRACTION = 0.3    # Fraction of timestamps to process
BLOCK_SAMPLE_FRACTION = 1 # Fraction of blocks per timestamp

# Data Range
YEAR = 2023
DAY_MIN, DAY_MAX = 83, 113  # Day-of-year range (inclusive)

# Output
OUTPUT_DIR = "/path/to/output"  # Where patches will be saved
```

### Run the Pipeline

```bash
python process_abi_data.py
```

### Output Files

Each saved patch (`ABI_Data_[TIMESTAMP]_[BLOCK_ID].npz`) contains:

| **Name**                                                | **Notes**                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------ |
| 'rad'                                                   | The 16 bands of level-1 radiances. Shape (128, 128, 16)      |
| ['l2_cloud_mask', 'l2_cloud_phase', 'l2_cod', 'l2_cps'] | The level-2 cloud properties.                                |
| Variables with 'dqf'                                    | Data quality flags. They are not needed for model training.  |
| 'lons', ‘lats’                                          | Longitute and latitude. They are not needed for model training. |
| 'SoZeAn', 'SoAzAn', 'SeZeAn', 'SeAzAn'                  | Geographical variables. They are not needed for model training. Solar zenith angle, solar azimuth angle, sensor zenith angle, sensor azimuth angle |

### Notes for Users

- **Storage**: Each 128x128 patch is ~0.5MB.
- **Debugging**: Set `SAMPLE_FRACTION=0.01` and `BLOCK_SAMPLE_FRACTION=0.1` for quick tests.
- **Visualization**: Use `tools/plot_patch.py` (example included in repo) to inspect outputs.

## References 

- User guide for ABI L1b products: https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf

- User guide for ABI L2 products: https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
- ABI data preprocessing used by satay: https://github.com/pytroll/satpy/tree/main/satpy/readers