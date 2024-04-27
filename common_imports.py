
# common_imports.py

# Import statements for common libraries/modules

import netCDF4 as nc
import importlib
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as dates
import numpy as np
import glob
import sys
import torch
import datetime
import xarray as xr

# append the path to the rpgpy package
sys.path.append('/home/schimmel/code/python/rpgpy')
from rpgpy import read_rpg, spectra2moments


# append the path to the voodoonet package
sys.path.append('/home/schimmel/code/python/voodoonet')
import voodoonet
from voodoonet import utils, loader

# append the path to the voodoonet package
sys.path.append('/home/schimmel/code/python/cloudnetpy')
import cloudnetpy
from cloudnetpy.products import generate_classification
from cloudnetpy.categorize import generate_categorize
from cloudnetpy.plotting.plot_meta import _CLABEL

from matplotlib import cm
from matplotlib.colors import ListedColormap

def get_voodoo_cmap():
    viridis = cm.get_cmap('viridis', 256)
    voodoo_cmap = viridis(np.linspace(0, 1, 256))
    voodoo_cmap[:1, :] = np.array([220/256, 220/256, 220/256, 1])
    voodoo_cmap = ListedColormap(voodoo_cmap)
    return voodoo_cmap

def get_cloudnet_cmap():
    # plot colormaps
    cloudnet_class_labels = [label[0] for label in _CLABEL['target_classification']]
    cloudnet_class_colors = [label[1] for label in _CLABEL['target_classification']]
    cloudnet_class_colors = matplotlib.colors.ListedColormap(cloudnet_class_colors)
    cloudnet_class_labels[0] = 'Clear sky'

    for i, txt in enumerate(cloudnet_class_labels):
        cloudnet_class_labels[i] = txt.replace('&', '\&')

    return cloudnet_class_labels, cloudnet_class_colors

def create_cloudnet_voodoo_categorize(categorize_file, classification_file, model_name, probability_liquid):

    # load the original categorize file
    cat_xr = xr.open_dataset(categorize_file, decode_times=False)
    class_xr = xr.open_dataset(classification_file, decode_times=False)
    status = class_xr['detection_status'].values

    values = np.zeros(probability_liquid.shape)
    voodoo_liquid_mask = probability_liquid > 0.5
    values[voodoo_liquid_mask] = 1.0

    cat_xr['category_bits'].values = _adjust_cloudnetpy_bits(cat_xr, values, status)

    CAT_FILE_NAME = categorize_file[:-3] + f'-voodoo.nc'
    CLS_FILE_NAME = classification_file[:-3] + f'-voodoo.nc'
    cat_xr.attrs['postprocessor'] = f'Voodoo_v2.0, Modelname: {model_name[:-3]}'

    cat_xr[f'liquid_probability'] = cat_xr['Z'].copy()
    cat_xr[f'liquid_probability'].values = probability_liquid
    cat_xr[f'liquid_probability'].attrs = {
        'comment': "This variable contains information about the likelihood of cloud droplet\n"
                    f"availability, predicted by the {cat_xr.attrs['postprocessor']} classifier.",
        'definition': "\nProbability 1 means most likely cloud droplets are present,\n"
                        "probability of 0 means no cloud droplets are available, respectively.\n",
        'units': "1",
        'long_name': f"Predicted likelihood of cloud droplet present."}


    cat_xr[f'noliquid_probability'] = cat_xr['Z'].copy()
    cat_xr[f'noliquid_probability'].values = 1 - probability_liquid
    cat_xr[f'noliquid_probability'].attrs = {
        'comment': "This variable contains information about the likelihood of present cloud droplet,\n"
                    f"predicted by the {cat_xr.attrs['postprocessor']} classifier.",
        'definition': "\nProbability 1 means most likely no cloud droplets are present,\n"
                        "probability of 0 means no cloud droplets are available, respectively.\n",
        'units': "1",
        'long_name': f"Predicted likelihood of no cloud droplet present."}

    # save the new categorize file
    cat_xr.to_netcdf(path=CAT_FILE_NAME, format='NETCDF4', mode='w')
    print(f"\nfile saved: {CAT_FILE_NAME}")

    # generate classification with new bit mask
    generate_classification(CAT_FILE_NAME, CLS_FILE_NAME)
    print(f"\nfile saved: {CLS_FILE_NAME}")





def _adjust_cloudnetpy_bits(cat_xr, values, status):
    n_ts_cloudnet_cat, n_rg_cloudnet_cat = cat_xr['category_bits'].shape
    bits_unit = cat_xr['category_bits'].values.astype(np.uint8)
    new_bits = bits_unit.copy()

    for ind_time in range(n_ts_cloudnet_cat):
        for ind_range in range(n_rg_cloudnet_cat):
            if values[ind_time, ind_range] == 1:
                if status[ind_time, ind_range] in [1, 2]:
                    continue  # skip good radar & lidar echo pixel
                if cat_xr['v'][ind_time, ind_range] < -3:
                    continue
                bit_rep = np.unpackbits(bits_unit[ind_time, ind_range])
                bit_rep[-1] = 1  # set droplet bit
                new_bits[ind_time, ind_range] = np.packbits(bit_rep)
    return new_bits



def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def dh_to_ts(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh*3600))

