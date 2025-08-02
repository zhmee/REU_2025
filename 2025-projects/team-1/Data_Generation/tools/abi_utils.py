import os
import glob
from datetime import datetime, timedelta
from datetimerange import DateTimeRange
import random
import numpy as np
from netCDF4 import Dataset
import h5py

def print_h5_structure(file, indent=0):
    """Recursively print the structure of an H5 file."""
    for key in file.keys():
        item = file[key]
        print('  ' * indent + key, end='')
        if isinstance(item, h5py.Dataset):
            print(' (Dataset)', item.shape, item.dtype)
        elif isinstance(item, h5py.Group):
            print(' (Group)')
            print_h5_structure(item, indent+1)

def abi_file_pattern(var):
    abi_var_dict = {
        "l1b": "ABI-L1b-RadF",
        "cloud_mask": "ABI-L2-ACMF",
        "cloud_phase": "ABI-L2-ACTPF",
        "cod": "ABI-L2-COD2KMF",
        "cps": "ABI-L2-CPSF"
    }
    pattern = abi_var_dict[f"{var}"]
    return pattern

def abi_pattern_var(pattern):
    abi_pattern_dict = {
        "ABI-L1b-RadF": "l1b",
        "ABI-L2-ACMF": "cloud_mask",
        "ABI-L2-ACTPF": "cloud_phase",
        "ABI-L2-COD2KMF": "cod",
        "ABI-L2-CPSF": "cps"
    }
    var = abi_pattern_dict[f"{pattern}"]
    return var

def abi_var_name(var):
    abi_product_dict = {
        "l1b": "Rad",
        "cloud_mask": "BCM",
        "cloud_phase": "Phase",
        "cod": "COD",
        "cps": "PSD"
    }
    product_name = abi_product_dict[f"{var}"]
    return product_name

def get_abi_l1b_resolution(channel_number):
    if (channel_number<=0):
        return -1
    if (channel_number>16):
        return -1
    resolutions = [1,0.5,1,2,1,2,2,2,2,2,2,2,2,2,2,2]
    return resolutions[channel_number-1]
    
def abi_starting_time(file):
    s_part = filename.split('_s')[1].split('_')[0]
    return s_part

def abi_l1b_nband(file):
    band = basename.split('-M')[1].split('C')[1][:2]
    return band

def abi_parse_timestamp(filename):
    """Extract datetime from filename's 's' timestamp"""
    s_part = abi_starting_time(filename)
    year = int(s_part[:4])
    day_of_year = int(s_part[4:7])
    hour = int(s_part[7:9])
    minute = int(s_part[9:11])
    second = int(s_part[11:13])
    return datetime(year, 1, 1) + timedelta(days=day_of_year-1, hours=hour, minutes=minute, seconds=second)

def load_abi_l1b(channel_number, abi_file=''):
    """Complete ABI L1b calibration with proper scaling and clipping"""
    with h5py.File(abi_file, 'r') as fd:
        # Step 1: Read raw data with scale_factor/add_offset
        rad_var = fd['Rad']
        raw_data = rad_var[:]  # Preserve original dtype
        fill_value = rad_var.attrs['_FillValue']
        
        # Step 2: Apply scaling factors (essential for raw counts)
        scale_factor = rad_var.attrs['scale_factor']
        add_offset = rad_var.attrs['add_offset']
        rad = (raw_data * scale_factor) + add_offset
        
        # Step 3: Mask invalid data
        rad[raw_data == fill_value] = np.nan
        
        channel_num = int(channel_number)
        if channel_num <= 6:  # Visible/NIR bands
            # Step 4a: Convert to TOA reflectance. It's same to kappa0
            # d = fd['earth_sun_distance_anomaly_in_AU'][()]
            # esun = fd['esun'][()]
            # data = (rad * np.pi * d**2) / esun
            data = rad * fd['kappa0'][()]
            
        else:  # IR bands
            # Step 4b: Handle negative radiances (critical!)
            min_rad = _get_minimum_radiance(rad_var)
            rad = np.maximum(rad, min_rad)
            
            # Step 5: Convert to brightness temperature
            fk1 = fd['planck_fk1'][()]
            fk2 = fd['planck_fk2'][()]
            bc1 = fd['planck_bc1'][()]
            bc2 = fd['planck_bc2'][()]
            data = (fk2 / np.log((fk1 / rad) + 1) - bc1) / bc2
            # data = rad
            
            # BT sanity check
            # data = np.clip(data, 180, 340).astype(np.float32)  # Reasonable IR range     
    return data

def _get_minimum_radiance(rad_var):
    """Calculate minimum valid radiance to avoid negative BT"""
    attrs = rad_var.attrs
    scale_factor = attrs.get('scale_factor', 1.0)
    add_offset = attrs.get('add_offset', 0.0)
    
    # From PUG: Minimum radiance occurs at count = -add_offset/scale_factor
    count_zero_rad = -add_offset / scale_factor
    count_pos = np.ceil(count_zero_rad)  # First valid count
    return (count_pos * scale_factor) + add_offset

def load_abi_cloud_mask(abi_file=''):
    with h5py.File(abi_file, 'r') as fd:        
        mask = fd[abi_var_name('cloud_mask')][:].astype(np.float16)
        fill_value = fd[abi_var_name('cloud_mask')].attrs['_FillValue']
        mask[mask == fill_value] = np.nan
        fd.close()
    return mask

def load_abi_cloud_phase(abi_file=''):
    with h5py.File(abi_file, 'r') as fd:
        phase = fd[abi_var_name('cloud_phase')][:].astype(np.float16)
        fill_value = fd[abi_var_name('cloud_phase')].attrs['_FillValue']
        phase[phase == fill_value] = np.nan
        fd.close()
    return phase

def load_abi_cod(abi_file='', cloud_mask=[]):
    with h5py.File(abi_file, 'r') as fd:
        cod_var = fd[abi_var_name('cod')]
        cod_raw = fd[abi_var_name('cod')][:].astype(np.float32)  # Shape: (height, width)
        fill_value = cod_var.attrs['_FillValue']
        scale_factor = cod_var.attrs['scale_factor']
        add_offset = cod_var.attrs['add_offset']
        cod = (cod_raw * scale_factor) + add_offset
        # P221, https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
        cod_min = 0.0
        cod_max = 160.0
        cod[cod == fill_value] = np.nan
        cod[(cod > cod_max) | (cod < cod_min)] = np.nan
        # Clear pixels are allowed, give 0, dep=0
        cod = np.where(cloud_mask == 0, 0.0, cod)
        fd.close()
    return cod.astype(np.float16)

def load_abi_cps(abi_file='', cloud_mask=[]):
    with h5py.File(abi_file, 'r') as fd:
        cps_var = fd[abi_var_name('cps')]
        cod_raw = fd[abi_var_name('cps')][:].astype(np.float32)  # Shape: (height, width)
        fill_value = cps_var.attrs['_FillValue']
        scale_factor = cps_var.attrs['scale_factor']
        add_offset = cps_var.attrs['add_offset']
        cps = (cod_raw * scale_factor) + add_offset
        # P239, https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf
        cps_min = 0.0
        cps_max = 100.0
        cps[cps == fill_value] = np.nan
        cps[(cps > cps_max) | (cps < cps_min)] = np.nan
        # Clear pixels are allowed, give 0
        cps = np.where(cloud_mask == 0, 0.0, cps)
        fd.close()
    return cps.astype(np.float16)

'''
Calculate solar zenith, solar azimuth, sensor zenith, and sensor azimuth
Input:  timeflag: int (Example: 20230831620200)
        lons: 2d np array
        lats: 2d np array
'''
def calculate_2d_angles(timeflag, lons, lats):
    timeflag = str(timeflag)
    doy = int(timeflag[4:7])
    hr = int(timeflag[7:9])
    mi = int(timeflag[9:11])
    sec = int(timeflag[11:13])

    tim = ((sec/60 + mi))/60 + hr
    jday = doy + tim/24
    # print('---------------')
    # print('Fractional Day: ', round(C_jday,4),'#############','Fractional Time: ', round(C_tim,4))
    # print('---------------------------------------------------------')

    SoZeAn, saa1, SoAzAn = Solar_Calculate(lons, lats, jday, tim)
    SeZeAn, SeAzAn = Sensor_Calculate(lons, lats)
    return SoZeAn, SoAzAn, SeZeAn, SeAzAn

def Sensor_Calculate(xlon, xlat):
    pi = 3.14159265
    dtor = pi / 180.0                         # change degree to radians
    
    satlon = -137.0
    satlat = 0.0
    
    # Convert inputs to numpy arrays if they aren't already
    xlon = np.asarray(xlon)
    xlat = np.asarray(xlat)
    
    lon = (xlon - satlon) * dtor   # in radians
    lat = (xlat - satlat) * dtor   # in radians

    beta = np.arccos(np.cos(lat) * np.cos(lon))
    sin_beta = np.sin(beta)

    # zenith angle    
    x = 42164.0 * sin_beta / np.sqrt(1.808e09 - 5.3725e08 * np.cos(beta))
    zenith = np.arcsin(x)
    zenith = zenith / dtor

    # azimuth angle
    azimuth = np.arcsin(np.sin(lon) / sin_beta) / dtor
    
    # Handle the different cases for azimuth calculation
    azimuth = np.where(lat < 0.0, 180.0 - azimuth, azimuth)
    azimuth = np.where(azimuth < 0.0, azimuth + 360.0, azimuth)
    
    return zenith, azimuth

def Solar_Calculate(xlon, xlat, jday, tu):
    pi = 3.14159265
    dtor = pi / 180.0                         # change degree to radians
    
    # Convert inputs to numpy arrays if they aren't already
    xlon = np.asarray(xlon)
    xlat = np.asarray(xlat)
    jday = np.asarray(jday) if np.iterable(jday) else jday
    tu = np.asarray(tu) if np.iterable(tu) else tu
    
    tsm = tu + xlon/15.0                      # mean solar time
    xlo = xlon * dtor
    xla = xlat * dtor
    xj = jday
    
    a1 = (1.00554 * xj - 6.28306) * dtor
    a2 = (1.93946 * xj + 23.35089) * dtor
    et = -7.67825 * np.sin(a1) - 10.09176 * np.sin(a2)  # time equation
    
    tsv = tsm + et/60.0 
    tsv = tsv - 12.0                                    # true solar time
    
    ah = tsv * 15.0 * dtor                              # hour angle
    
    a3 = (0.9683 * xj - 78.00878) * dtor
    delta = 23.4856 * np.sin(a3) * dtor                 # solar declination (in radian)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    cos_ah = np.cos(ah)
    sin_xla = np.sin(xla)
    cos_xla = np.cos(xla)
    
    amuzero = sin_xla * sin_delta + cos_xla * cos_delta * cos_ah
    elev = np.arcsin(amuzero)
    cos_elev = np.cos(elev)
    az = cos_delta * np.sin(ah) / cos_elev
    caz = (-cos_xla * sin_delta + sin_xla * cos_delta * cos_ah) / cos_elev
    
    # Calculate azimuth with proper conditions
    azim = np.arcsin(np.clip(az, -1.0, 1.0))  # clip values to valid range for arcsin
    
    # Handle the different cases for azimuth calculation
    azim = np.where(caz <= 0.0, pi - azim, azim)
    azim = np.where((caz > 0.0) & (az <= 0.0), 2 * pi + azim, azim)
    
    azim = azim + pi
    pi2 = 2 * pi
    azim = np.where(azim > pi2, azim - pi2, azim)
   
    # Conversion in degrees
    elev = elev / dtor
    asol = 90.0 - elev                                 # solar zenith angle in degrees
    
    phis1 = azim / dtor - 180.0                         # solar azimuth angle in degrees
    phis2 = phis1 + 360
    phis = phis2 - 180

    return asol, phis1, phis