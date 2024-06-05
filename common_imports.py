
# common_imports.py

# Import statements for common libraries/modules
import pathlib
script_directory = pathlib.Path(__file__).parent.resolve()

import netCDF4 as nc
import importlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as dates
import numpy as np
import glob
import sys
import datetime
import xarray as xr
import os
import time
import warnings
import csv
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
import scienceplots
from numba import jit
from itertools import groupby, product
import torch

from rpgpy import read_rpg, spectra2moments

import voodoonet
from voodoonet import utils, loader

import cloudnetpy
from cloudnetpy.products import generate_classification
from cloudnetpy.categorize import generate_categorize
from cloudnetpy.plotting.plot_meta import _CLABEL
from cloudnetpy.plotting import generate_figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
warnings.filterwarnings("ignore")

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


def datetime64_to_decimalhour(ts_in):
    ts = []
    for i in range(len(ts_in)):
        ts.append(ts_in[i].astype('datetime64[h]').astype(int) % 24 * 3600 +
                  ts_in[i].astype('datetime64[m]').astype(int) % 60 * 60 +
                  ts_in[i].astype('datetime64[s]').astype(int) % 60)
    return np.array(ts) / 3600


def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def dh_to_ts(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh*3600))


def fetch_liquid_masks(ccl, vcl, cst, rg, lprob, is_raining, v, liquid_threshold=0.5):
    mliqVoodoo = lprob > liquid_threshold
    mliqCLoudnet = (ccl == 1) + (ccl == 3) + (ccl == 5) + (ccl == 7)
    mliqCombination= (vcl == 1) + (vcl == 3) + (vcl == 5) + (vcl == 7)
    _is_lidar_only = cst == 4
    _is_clutter = ccl > 7
    _is_rain = np.array([is_raining] * rg.size, dtype=bool).T
    _is_falling = (ccl == 2) * (v < -3)

    # reclassify all fast-falling hydrometeors and insects/clutter to non-CD
    mliqVoodoo[_is_falling] = False
    mliqVoodoo[_is_clutter] = False

    cloud_mask = mliqCLoudnet + (ccl == 2) + (ccl == 4) + (ccl == 6)
    cloud_mask = remove_cloud_edges(cloud_mask, n=3)
    cloud_mask[_is_lidar_only] = False
    cloud_mask[_is_clutter] = False

    # # create dictionary with liquid pixel masks
    masks = {
        'Voodoo': mliqVoodoo * cloud_mask,
        'Cloudnet': mliqCLoudnet * cloud_mask,
        'Combination': mliqCombination * cloud_mask,
        'cloud_mask': cloud_mask,
    }

    return masks


def remove_cloud_edges(mask, n=2):
    def outline(im):
        ''' Input binary 2D (NxM) image. Ouput array (2xK) of K (y,x) coordinates
            where 0 <= K <= 2*M.
        '''
        topbottom = np.empty((1, 2 * im.shape[1]), dtype=np.uint16)
        topbottom[0, 0:im.shape[1]] = np.argmax(im, axis=0)
        topbottom[0, im.shape[1]:] = (im.shape[0] - 1) - np.argmax(np.flipud(im), axis=0)
        mask = np.tile(np.any(im, axis=0), (2,))
        xvalues = np.tile(np.arange(im.shape[1]), (1, 2))
        return np.vstack([topbottom, xvalues])[:, mask].T

    out_mask = mask.copy()
    for i in range(n):
        cloud_edge = np.full(out_mask.shape, False)
        for boundaries in outline(out_mask):
            cloud_edge[boundaries[0], boundaries[1]] = True
        for boundaries in outline(out_mask.T):
            cloud_edge[boundaries[1], boundaries[0]] = True

        out_mask[cloud_edge] = 0

    return out_mask


def interpolate_T_p_q(ts, rg, model_ts, model_rg, model_T, model_p, model_q):
    def interpolate(values):
        f = interp2d(model_ts, model_rg, check_boundaries(values).T, kind='linear', bounds_error=False, fill_value=None)
        return f(ts, rg)[:, :].T

    def check_boundaries(val_in):
        val = val_in.copy()
        val[-1, :] = np.where(np.isnan(val[-1, :]), val[-2, :], val[-1, :])
        val[:, -1] = np.where(np.isnan(val[:, -1]), val[:, -2], val[:, -1])
        return val

    return interpolate(model_T), interpolate(model_p), interpolate(model_q)



def smooth(y, box_pts, padding='constant'):
    """Smooth a one dimensional array using a rectangular window of box_pts points

    Args:
        y (np.array): array to be smoothed
        box_pts: number of points of the rectangular smoothing window
    Returns:
        y_smooth (np.arrax): smoothed array
    """

    box = np.ones(box_pts) / box_pts
    if padding.lower() == 'constant':
        return np.convolve(y, box, mode='full')[box_pts // 2:-box_pts // 2 + 1]
    else:
        return np.convolve(y, box, mode='same')


def compute_llt_lwp(rg, temperature, pressure, lwp, liquid_masks, n_smoothing=60, idk_factor=1.5):
        rg_res = np.mean(np.diff(rg)) * 0.001

        # liquid water content
        lwc_dict = {}
        for key, mask in liquid_masks.items():
            _lwc = adiabatic_liquid_water_content(
                temperature,
                pressure,
                mask,
                delta_h=float(np.mean(np.diff(rg)))
            )
            _lwc[_lwc > 100] = np.nan
            _lwc[_lwc < 1] = np.nan
            lwc_dict[key] = np.ma.masked_invalid(_lwc)

        _lwp = lwp.copy()
        _lwp[_lwp > 4000] = np.nan
        _lwp[_lwp < 1] = np.nan
        _lwp_pd = pd.Series(_lwp)

        # liquid water path and liquid layer thickness
        llt_dict, lwp_dict = {}, {}
        lwp_dict['mwr'] = _lwp_pd.interpolate(method='nearest').values
        lwp_dict['mwr_s'] = smooth(lwp_dict['mwr'], n_smoothing)
        for key in liquid_masks.keys():
            lwp_dict[key] = np.ma.sum(lwc_dict[key], axis=1) * idk_factor
            lwp_dict[key + '_s'] = smooth(lwp_dict[key], n_smoothing)
            llt_dict[key] = np.count_nonzero(liquid_masks[key], axis=1) * rg_res
            llt_dict[key + '_s'] = smooth(llt_dict[key], n_smoothing)

        return llt_dict, lwp_dict


def correlation_llt_lwp(llt_dict, lwp_dict, valid_lwp_retrieval, methods):

    correlations = {}
    valid_lwp = (0.0 < lwp_dict['mwr_s']) * (lwp_dict['mwr_s'] < 1000.0)

    valid = np.argwhere(valid_lwp * valid_lwp_retrieval)[:, 0]

    for alg in methods:
        correlations.update({
            alg + 'corr(LWP)-s': np.corrcoef(lwp_dict['mwr_s'][valid], lwp_dict[alg + '_s'][valid])[0, 1],
            alg + 'corr(LLT)-s': np.corrcoef(lwp_dict['mwr_s'][valid], llt_dict[alg + '_s'][valid])[0, 1],
        })

    print(correlations)
    return correlations


def fetch_evaluation_metrics(voodoo_status):
    TN, TP, FP, FN = [np.count_nonzero(voodoo_status == i) for i in range(3, 7)]
    b = 0.5
    s = {
        'precision': TP / max(TP + FP, 1.0e-7),
        'npv': TN / max(TN + FN, 1.0e-7),
        'recall': TP / max(TP + FN, 1.0e-7),
        'specificity': TN / max(TN + FP, 1.0e-7),
        'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
        'F1-score': TP / max(TP + 0.5 * (FP + FN), 1.0e-7),
    }
    s['F1/2-score'] = (1 + b*b) * s['precision']*s['recall'] / max(s['recall'] + b*b*s['precision'], 1.0e-7)
    s['acc-balanced'] = (s['precision'] + s['npv']) / 2

    print(TP, FN, FP, TN)
    for key, val in s.items():
        print(f'{val:.3f}', key)
    return s


def first_occurrence_indices(array):

    out = [next((i for i, value in enumerate(col) if value == 1), None) for col in array.T]
    return np.array(out)


def fetch_voodoo_status(rg, liquid_masks, liq_cbh):

    mliqVoodoo = liquid_masks['Voodoo']
    mliqCLoudnet = liquid_masks['Cloudnet']
    cloud_mask = liquid_masks['cloud_mask']
    
    # Compute True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN) masks
    TP_mask =  mliqCLoudnet *  mliqVoodoo
    FP_mask = ~mliqCLoudnet *  mliqVoodoo
    FN_mask =  mliqCLoudnet * ~mliqVoodoo
    TN_mask = ~mliqCLoudnet * ~mliqVoodoo

    # Adjust masks based on the first liquid cloud base height
    for its, icb in enumerate(liq_cbh):
        if icb is not None and icb > 0:
            idx_cb = argnearest(rg, icb)
            TN_mask[its, idx_cb+1:] = False
            FP_mask[its, idx_cb+1:] = False

    # Combine masks into a single mask
    combi_liq_mask = np.zeros(mliqVoodoo.shape)
    non_CD_extension_mask = cloud_mask * ~TP_mask * ~TN_mask * ~FP_mask * ~FN_mask
    CD_extension_mask = mliqVoodoo * ~TP_mask * ~FP_mask

    combi_liq_mask[non_CD_extension_mask] = 1
    combi_liq_mask[CD_extension_mask] = 2
    combi_liq_mask[TN_mask] = 3
    combi_liq_mask[TP_mask] = 4
    combi_liq_mask[FP_mask] = 5
    combi_liq_mask[FN_mask] = 6
    combi_liq_mask[~cloud_mask] = 0

    return combi_liq_mask



def argnearest(array, value):
    """Find the index of the nearest value in a sorted array, such as time or range axis.

    Args:
        array (np.ndarray or list): Sorted array with values, list will be converted to 1D array.
        value: Value to find.

    Returns:
        index
    """
    array = np.asarray(array)
    i = np.searchsorted(array, value) - 1

    if i < array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i += 1
    return i



def change_dir(folder_path, **kwargs):
    """
    This routine changes to a folder or creates it (including subfolders) if it does not exist already.

    Args:
        folder_path (string): path of folder to switch into
    """

    folder_path = folder_path.replace('//', '/', 1)

    if not os.path.exists(os.path.dirname(folder_path)):
        os.makedirs(os.path.dirname(folder_path))
    os.chdir(folder_path)
    print('\ncd to: {}'.format(folder_path))

def interpolate_cbh(ts_cloudnet, ts_ceilo, cbh, icb=0):

    ts_ceilo = datetime64_to_decimalhour(ts_ceilo)

    f = interp1d(
        ts_ceilo,
        cbh[:, icb],
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )
    cbh = f(ts_cloudnet)[:]
    cbh[np.isnan(cbh)] = -1
    return cbh  # ceilo ts needs to be adapted for interpoaltion


def compute_cbh(ceilo_cbh, liquid_masks, range_bins):
    cbh = {'CEILO': ceilo_cbh}
    for key in liquid_masks.keys():
        _tmp = np.argmax(liquid_masks[key] == 1, axis=1)
        cbh[key] = np.ma.masked_less_equal([range_bins[ind_rg] for ind_rg in _tmp], 200)
        cbh[key] = np.ma.masked_invalid(cbh[key])
    return cbh


def find_bases_tops(mask, rg_list):
    """
    This function finds cloud bases and tops for a provided binary cloud mask.
    Args:
        mask (np.array, dtype=bool) : bool array containing False = signal, True=no-signal
        rg_list (list) : list of range values

    Returns:
        cloud_prop (list) : list containing a dict for every time step consisting of cloud bases/top indices, range and width
        cloud_mask (np.array) : integer array, containing +1 for cloud tops, -1 for cloud bases and 0 for fill_value
    """
    cloud_prop = []
    cloud_mask = np.full(mask.shape, 0, dtype=np.int)
    for ind_time in range(mask.shape[0]):  # tqdm(range(mask.shape[0]), ncols=100, unit=' time steps'):
        cloud = [(k, sum(1 for j in g)) for k, g in groupby(mask[ind_time, :])]
        idx_cloud_edges = np.cumsum([prop[1] for prop in cloud])
        bases, tops = idx_cloud_edges[0:][::2][:-1], idx_cloud_edges[1:][::2]
        if tops.size > 0 and tops[-1] == mask.shape[1]:
            tops[-1] = mask.shape[1] - 1
        cloud_mask[ind_time, bases] = -1
        cloud_mask[ind_time, tops] = +1
        cloud_prop.append({'idx_cb': bases, 'val_cb': rg_list[bases],  # cloud bases
                           'idx_ct': tops, 'val_ct': rg_list[tops],  # cloud tops
                           'width': [ct - cb for ct, cb in zip(rg_list[tops], rg_list[bases])]
                           })
    return cloud_prop, cloud_mask


def get_cloud_base_from_liquid_mask(liq_mask, rg):
    """
    Function returns the time series of cloud base height in meter.
    Args:
        liq_mask:
        rg: range values

    Returns: cloud base height

    """
    _, cbct_mask = find_bases_tops(liq_mask * 1, rg)
    n_ts = liq_mask.shape[0]

    CB = np.full(n_ts, np.nan)

    for ind_time in range(n_ts):
        idx = np.argwhere(cbct_mask[ind_time, :] == -1)
        CB[ind_time] = rg[int(idx[0])] if len(idx) > 0 else 0.0
    return CB



def create_csv_file(rg, llt_dict, lwp_dict, cbh_dict, liquid_masks, csv_path, site):
    
    def fetch_bin_edges_lwp_edr(lwp):
        lwp_bin_edges = [
            [0, 2000],
            [0, 25], [25, 50], [50, 100], [100, 150],
            [150, 200], [200, 300], [300, 400], [400, 2000]
        ]

        lwp_masks = [
            (edge[0] < lwp) * (lwp < edge[1]) for edge in lwp_bin_edges
        ]
        return lwp_masks
    
    def all_correlations(llt, lwp, cbh, liquid_masks, bins, hmin=300):
        corr_tmp = {}
        corr = {}

        for alg in liquid_masks.keys():
            corr.update({alg + 'corr(LWP)-s': [], alg + 'corr(LLT)-s': [], alg + 'corr((L)CBH)': []})

            for bin_mask in bins:
                llt_valid = (llt['Voodoo'] > 0.0) * (lwp['mwr_s'] > 0)
                cei_valid = (cbh[alg] > hmin) * (cbh['CEILO'] > hmin)

                mwrlwp = lwp['mwr_s'][np.argwhere(bin_mask * llt_valid)[:, 0]]
                ceilcbh = cbh['CEILO'][np.argwhere(bin_mask * cei_valid)[:, 0]]
                algollt = llt[alg + '_s'][np.argwhere(bin_mask * llt_valid)[:, 0]]
                algolwp = lwp[alg + '_s'][np.argwhere(bin_mask * llt_valid)[:, 0]]
                algocbh = cbh[alg][np.argwhere(bin_mask * cei_valid)[:, 0]]

                corr[alg + 'corr(LLT)-s'].append(np.corrcoef(mwrlwp, algollt)[0, 1])
                corr[alg + 'corr(LWP)-s'].append(np.corrcoef(mwrlwp, algolwp)[0, 1])
                corr[alg + 'corr((L)CBH)'].append(np.corrcoef(ceilcbh, algocbh)[0, 1])

            corr_tmp[alg] = np.array([
                corr[alg + 'corr(LLT)-s'],
                corr[alg + 'corr(LWP)-s'],
                corr[alg + 'corr((L)CBH)'],
            ])
        return corr_tmp

    def performance_metrics2(TP, TN, FP, FN):
        def equitable_thread_score(a, b, c, d):
            n = a+b+c+d
            E = (a+b)*(a+c)/max(1.0e-7, n)
            return (a - E) /max(1.0e-7, (a - E + b + c))
        
        sum_stats = {
            'precision': TP / max(TP + FP, 1.0e-7),
            'far:': FP / max(TP + FP, 1.0e-7),
            'fpr':  FP / max(FP + TN, 1.0e-7),
            'fbi': (TP + FP) / max(TN + FN, 1.0e-7),
            'npv': TN / max(TN + FN, 1.0e-7),
            'recall': TP / max(TP + FN, 1.0e-7),
            'specificity': TN / max(TN + FP, 1.0e-7),
            'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
            'F1-score': TP / max(TP + 0.5 * (FP + FN), 1.0e-7),
            'ets': equitable_thread_score(TP, TN, FP, FN),
        }

        return sum_stats

    
    lwp_masks = fetch_bin_edges_lwp_edr(lwp_dict['mwr_s'])

    correlations_csv = all_correlations(llt_dict, lwp_dict, cbh_dict, liquid_masks, lwp_masks, hmin=300)

    int_columns = ['TP', 'TN', 'FP', 'FN']
    flt_columns = ['precision', 'far', 'fpr', 'fbi', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score', 'ets']
    corr_columns = ['Vr2(LLT)', 'Vr2(LWP)', 'Vr2(LCBH)', 'Cr2(LLT)', 'Cr2(LWP)', 'Cr2(LCBH)']
    extra_columns = ['n_time_steps']  # , 'ETS']
    num_columns = len(int_columns) + len(flt_columns) + len(corr_columns) + len(extra_columns)


    arr0 = np.zeros((len(lwp_masks) + 2, num_columns), dtype=float)
    for i in range(len(lwp_masks) + 1):
        if i < len(lwp_masks):
            _mask = np.array([lwp_masks[i]] * rg.size).T
            _mask = _mask * liquid_masks['cloud_mask']
        elif i == len(lwp_masks):
            _mask = np.zeros(_mask.shape)

        voodoo_status_binned = fetch_voodoo_status(rg, liquid_masks, cbh_dict['CEILO'])
        if np.count_nonzero(_mask) == 0:
            continue

        voodoo_status_binned[~_mask] = 0

        TN = np.count_nonzero(voodoo_status_binned == 3)
        TP = np.count_nonzero(voodoo_status_binned == 4)
        FP = np.count_nonzero(voodoo_status_binned == 5)
        FN = np.count_nonzero(voodoo_status_binned == 6)

        sum_stats = performance_metrics2(TP, TN, FP, FN)
        sum_stats_list = [val for val in sum_stats.values()]

        if i < len(lwp_masks):
            arr0[i, :] = np.array(
                [TP, TN, FP, FN] +
                sum_stats_list +
                list(correlations_csv['Voodoo'][:, i]) +
                list(correlations_csv['Cloudnet'][:, i]) +
                [np.count_nonzero(np.any(_mask, axis=1))],
            )

        elif i == len(lwp_masks):
            # add p value
            arr0[i, 0] = 0.0
        else:
            arr0[i, :] = np.array(
                [TP, TN, FP, FN] +
                sum_stats_list +
                list(correlations_csv['Voodoo'][:, i - len(lwp_masks) - 1]) +
                list(correlations_csv['Cloudnet'][:, i - len(lwp_masks) - 1]) +
                [np.count_nonzero(np.any(_mask, axis=1))],
            )

    # LWP csv
    try:
        stats_list = [
            [['', ] + int_columns + flt_columns + corr_columns + extra_columns] +
            [[f'{site}-lwp-bin{i}', ] + list(val) for i, val in enumerate(arr0[:len(lwp_masks) + 1, :])]
        ]
        with open(csv_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(stats_list[0][0])
            for i, cloud in enumerate(stats_list[0][1:]):
                writer.writerow(cloud)
    except:
        print(f'No lwp file written! {csv_path}')

    return 0




############

grav = 9.80991  # mean earth gravitational acceleration in m s-2
R = 8.31446261815324  # gas constant in kg m2 s−2 K−1 mol−1
eps = 0.62  # ratio of gas constats for dry air and water vapor dimensionless
cp = 1005.  # specific heat of air at constant pressure in J kg-1 K-1
cv = 1860.  # specific heat of air at constant volume in J kg-1 K-1
gamma_d = 9.76786e-3  # dry-adiabatic lapse rate in K m-1
Rair = 287.058e-3  # specific gas constant of air J kg-1 K-1
m_mol = 0.0289644  # molar mass of air in kg mol-1


@jit(nopython=True, fastmath=True)
def saturated_water_vapor_pressure(T: np.array) -> np.array:
    """ Calculates the saturated water vapor pressure.

    Args:
        T: temperature in [C]
    Returns:
        saturated_water_vapor_pressure in [Pa] = [kg m-1 s-2]
    Source:
        https://en.wikipedia.org/wiki/Vapour_pressure_of_water
    """
    return 0.61078 * np.exp(17.27 * T / (T + 237.3))


@jit(nopython=True, fastmath=True)
def mixing_ratio(e_sat: np.array, p: np.array) -> np.array:
    """ Calculates the ratio of the mass of a variable atmospheric constituent to the mass of dry air.

    Args:
        e_sat: saturated water vapor pressure in [kPa]
        p: pressure in [Pa]
    Returns:
        mixing_ratio dimensionless
    Source:
        https://glossary.ametsoc.org/wiki/Mixing_ratio
    """
    return 0.622 * e_sat / (p - e_sat)


@jit(nopython=True, fastmath=True)
def latent_heat_of_vaporization(T: np.array) -> np.array:
    """ Latent heat (also known as latent energy or heat of transformation) is energy released or absorbed,
        by a body or a thermodynamic system, during a constant-temperature process — usually a first-order
        phase transition.

    Args:
        T: temperature in [C]
    Returns:
        latent_heat_of_vaporization in [J kg-1]
    Source:
        https://en.wikipedia.org/wiki/Latent_heat
    """
    return (2500.8 - 2.36 * T + 1.6e-3 * T * T - 6e-5 * T * T * T) * 1.0e-3


@jit(nopython=True, fastmath=True)
def air_density(T: np.array, p: np.array) -> np.array:
    """ Calculates the density using the ideal gas law.

    Args:
        T: temperature in [C]
        p: pressure in [Pa]
    Returns:
        air_density in [kg m-3]
    Source:
        https://en.wikipedia.org/wiki/Barometric_formula
    """
    return p * m_mol / ((T + 273.15) * Rair) * 1.0e-3


def pseudo_adiabatic_lapse_rate(T: np.array, p: np.array, Lv: np.array) -> np.array:
    """ The rate of decrease of temperature with height of a parcel undergoing a pseudoadiabatic process.

    Args:
        T: temperature in [C]
        p: pressure in [Pa]
    Returns:
        pseudo_adiabatic_lapse_rate in [kg m-1]
    Source:
        https:https://glossary.ametsoc.org/wiki/Pseudoadiabatic_lapse_rate
    """
    e_sat = saturated_water_vapor_pressure(T)
    rv = mixing_ratio(e_sat, p)

    numerator = (1 + rv) * ((1 + Lv * rv) / (R + T))
    denominator = (cp + rv * cv + (Lv * Lv * rv * (eps + rv)) / (R * T * T))
    return grav * numerator / denominator


def adiabatic_liquid_water_content(T: np.array, p: np.array, mask: np.array, delta_h: float = 0.035):
    """ Computes the liquid water content under adiabatic assumtion.

    Args:
        T: temperature in [K]
        p: pressure in [Pa]
        mask: liquid cloud droplet mask
        delta_h: mean range resolution in [km]
    Returns:
        pseudo_adiabatic_lapse_rate in [kg m-1]
    Source:
        https://link.springer.com/article/10.1007/BF01030057
    """
    T_cel = T - 273.15
    LWCad = np.zeros(mask.shape)
    for ind_time in range(mask.shape[0]):
        for ind_range in range(mask.shape[1]):
            if mask[ind_time, ind_range]:
                Lv = latent_heat_of_vaporization(T_cel[ind_time, ind_range])
                gamma_s = pseudo_adiabatic_lapse_rate(T_cel[ind_time, ind_range], p[ind_time, ind_range], Lv)
                rho = air_density(T_cel[ind_time, ind_range], p[ind_time, ind_range])
                # formula (1)
                LWCad[ind_time, ind_range] = rho * cp / Lv * (gamma_d - gamma_s) * delta_h

    # correction term formula (2)
    LWCad = LWCad * (1.239 - 0.145 * np.log(delta_h))

    return LWCad