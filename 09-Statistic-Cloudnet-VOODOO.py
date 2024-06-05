from common_imports import *

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import seaborn as sns
from datetime import timezone


# optionally configure the logging
# StreamHandler will print to console
import re
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import depricated.Utils as UT
VOODOO_PATH = os.getcwd()

from matplotlib import cm
from matplotlib.colors import ListedColormap, LogNorm

cw = cm.get_cmap('coolwarm', 8)
hist2dcolors = cw(np.linspace(0, 1, 8))
hist2dcolors = ListedColormap(hist2dcolors)

CA_map = np.array(((171,169,171,255), (96,102,142,255), (21,34,121,255),
(11,44,112,255), (34,84,104,255), (63,120,84,255),
(123,150,58,255), (171,155,52,255), (160,89,32,255),
(158,34,23,255)))/255
CA_dcolors = ListedColormap(CA_map)



#PAPER_PATH = '/Users/willi/Library/Mobile Documents/com~apple~CloudDocs/VOODOO/'
QUICKLOOK_PATH = f'{script_directory}/plots/'
CODE_PATH = f'{script_directory}/'
gfx_path = QUICKLOOK_PATH


#CSV_PATH = f'{PAPER_PATH}/data/csv_nc_v2/'
CSV_PATH = f'{script_directory}/validation_csv/'

EXCLUDE_DAYS_PATH = script_directory + 'excluded_days.csv'


p_method = CSV_PATH[-4:-1]
site = 'Eriswill'

#site = 'LIM'

exclude_crane_days = pd.read_csv(EXCLUDE_DAYS_PATH, dtype=str).iloc[: , 0].to_list()
exclude_crane_info = pd.read_csv(EXCLUDE_DAYS_PATH, dtype=str).iloc[: , 1].to_list()
exclude_crane_dict = {key: val for key, val in zip(exclude_crane_days, exclude_crane_info)}

# split days into correspnding sites

list_csv_PA, list_csv_LE = [], []
dt_list_PA, dt_list_LE = [], []
cnt = 0
for date in glob.glob(CSV_PATH+'*lwp.csv'):

    date_str = date[date.rfind('/')+1:date.rfind('/')+9]
    
    if date_str in exclude_crane_days:
        cnt += 1
        continue
    
    _df = pd.read_csv(date)
    _arr = _df.iloc[: , 1:].to_numpy()
    _arr = np.ma.masked_invalid(_arr)
    
    if 20181101 < int(date_str) < 20190930:
        # punta arenas         
        list_csv_PA.append(_arr)
        _df_PA = _df.copy()
        dt_list_PA.append(datetime.datetime.strptime(date_str, "%Y%m%d"))
    else:
        # leipzig
        list_csv_LE.append(_arr)
        _df_LE = _df.copy()
        dt_list_LE.append(datetime.datetime.strptime(date_str, "%Y%m%d"))

        
print(f'N days exluded {cnt}')
print(f'N_days_PA = {len(list_csv_PA)}')
print(f'N_days_LE = {len(list_csv_LE)}')

np_all_days_PA = np.ma.masked_invalid(np.array(list_csv_PA))
np_all_days_LE = np.ma.masked_invalid(np.array(list_csv_LE))

if site == 'punta-arenas':
    array = np_all_days_PA
    dt_list = dt_list_PA
else:
    array = np_all_days_LE
    dt_list = dt_list_LE
    

np_all_days = array


column_names = ['TP',
 'TN',
 'FP',
 'FN',
 'PPV',
 'FAR',
 'FPR',
 'FBI',
 'NPV',
 'TPR',
 'TNR',
 'ACC',
 'F1S',
 'ets',
 'Vr2(LLT)',
 'Vr2(LWP)',
 'Vr2(LCBH)',
 'Cr2(LLT)',
 'Cr2(LWP)',
 'Cr2(LCBH)',
 'n_time_steps']

print('data.shape = ', np_all_days_PA[1].shape )



# overall statistics
def em_from_df(df, np_arr):
    df_all = df.copy()
    for i_col, v_col in enumerate(column_names[:4]):
        #tmp = np.ma.masked_less_equal(np_arr[:, :, i_col], 0.0)
        tmp = np_arr[:, :, i_col]
        df_all[v_col] = np.ma.sum(tmp, axis=0)
        df_all[v_col][0] = np.ma.sum(tmp[:, 2:])
        #print(df_all[v_col][0])
        
    return df_all

irow, icol = 1, 1
df_all_PA = em_from_df(_df_PA, np_all_days_PA)
EM_PA = np.array(df_all_PA[['TP', 'TN', 'FP', 'FN']])
print('Punta Arenas')
print(f'CD = {EM_PA[irow, 0]+EM_PA[irow, 3]: _.0f}, no-CD = {EM_PA[irow, 1]+EM_PA[irow, 2]:_.0f}')
print(df_all_PA[['TP', 'TN', 'FP', 'FN']])

df_all_LE = em_from_df(_df_LE, np_all_days_LE)
EM_LE = np.array(df_all_LE[['TP', 'TN', 'FP', 'FN']])
print('\nLeipzig')
print(f'CD = {EM_LE[irow, 0]+EM_LE[irow, 3]: _.0f}, no-CD = {EM_LE[irow, 1]+EM_LE[irow, 2]:_.0f}')
print(df_all_LE[['TP', 'TN', 'FP', 'FN']])


print('TP      TN      FP     FN')
total_em_PA = np.ma.sum(np_all_days_PA[:, 2:, :4], axis=(0, 1), dtype=int)
print(total_em_PA)


print('\nperformance PA')
UT.performance_metrics(*total_em_PA)



## Recalculating the performance scores with overall $EM$ values
def performance_scoes_from_df(df):
    df_all = df.copy()
    for i_row, v_row in enumerate(row_names):
        em_list = list(df_all.iloc[i_row, 1:5])
        performance_dict = UT.performance_metrics2(*em_list)
        df_all.iloc[i_row, 5:15] = [val for val in performance_dict.values()]
    return df_all
    
    
df_all_PA = performance_scoes_from_df(df_all_PA)
print('Punta Arenas')
print(df_all_PA[['precision', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score']])
#print(df_all_PA[['precision', 'fpr', 'fbi', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score', 'ets']])


df_all_LE = performance_scoes_from_df(df_all_LE)
print('\n\nLeipzig')
print(df_all_LE[['precision', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score']])
#print(df_all_LE[['precision', 'fpr', 'fbi', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score', 'ets']])


def equitable_thread_score(a, d, b, c):
    n = a+b+c+d
    E = (a+b)*(a+c)/n, 
    return (a - E) / (a - E + b + c)



def false_positive_rate(a, d, b, c):
    return b / (b + d)

def frequency_bias_index(a, d, b, c):
    return (a + b) / (a + c)

def false_alarm_rate(a, d, b, c):
    return b / (a + b)

def probability_of_detections(a, d, b, c):
    return a / (a + c)


print('Equitable Threat Scroe (ETS)')
for i in range(EM_PA.shape[0]):
    ets_PA = equitable_thread_score(*EM_PA[i, :])[0]
    ets_LE = equitable_thread_score(*EM_LE[i, :])[0]
    print(f' ETS-PA({i}) = {ets_PA:3.2f},    ETS-LE({i}) = {ets_LE:3.2f}')

    
print('\nFrequency Bias Index (FBI)')
for i in range(EM_PA.shape[0]):
    ets_PA = frequency_bias_index(*EM_PA[i, :])
    ets_LE = frequency_bias_index(*EM_LE[i, :])
    print(f' FBI-PA({i}) = {ets_PA:3.2f},    FBI-LE({i}) = {ets_LE:3.2f}')
    
        
print('\nProbabiltiy of Detection (POD) aka recall')
for i in range(EM_PA.shape[0]):
    ets_PA = probability_of_detections(*EM_PA[i, :])
    ets_LE = probability_of_detections(*EM_LE[i, :])
    print(f' POD-PA({i}) = {ets_PA:3.2f},    POD-LE({i}) = {ets_LE:3.2f}')
        
        
print('\nFalse Alarm Rate (FAR) aka 1-precision')
for i in range(EM_PA.shape[0]):
    ets_PA = false_alarm_rate(*EM_PA[i, :])
    ets_LE = false_alarm_rate(*EM_LE[i, :])
    print(f' FAR-PA({i}) = {ets_PA:3.2f},    FAR-LE({i}) = {ets_LE:3.2f}')

    
print('\nProbability of False Detections (POFD) aka false prositive rate')
for i in range(EM_PA.shape[0]):
    ets_PA = false_positive_rate(*EM_PA[i, :])
    ets_LE = false_positive_rate(*EM_LE[i, :])
    print(f' FAR-PA({i}) = {ets_PA:3.2f},    FAR-LE({i}) = {ets_LE:3.2f}')


## Recalculating the correlation coefficients as mean over all days
# overall statistics
def print_correlations(df_all, np_all):
    for i_col, v_col in enumerate(column_names[10:], 10):
        df_all[v_col] = np.ma.mean(np_all[:, :, i_col], axis=0)

    print(df_all[['Vr2(LLT)', 'Cr2(LLT)', 'Vr2(LWP)', 'Cr2(LWP)', 'Vr2(LCBH)', 'Cr2(LCBH)']])
    
print_correlations(df_all_PA, np_all_days_PA)
print('\n\n')
print_correlations(df_all_LE, np_all_days_LE)

## prints for statistic latex format
def pmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('pmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{pmatrix}']
    rv += ['  ' + ' & '.join(l.split()) for l in lines]
    rv +=  [r'\end{pmatrix}']
    return ' '.join(rv)

from IPython.display import display, Math

def pmatrix_new(array):
    data = ''
    for line in array:        
        if len(line) == 1:
            data += ' %.3f &'%line + r' \\\n'
            continue
        for element in line:
            data += ' %.3f &'%element
        data += r' \\' + '\n'
    display(Math('\\begin{bmatrix} \n%s\end{bmatrix}'%data))

def tex_table_row(_df_all, col_list):
    return list(_df_all[col_list].iloc[0].to_numpy())

em_PA = df_all_PA[['TP', 'FP', 'FN', 'TN']].iloc[0].to_numpy()
em_LE = df_all_LE[['TP', 'FP', 'FN', 'TN']].iloc[0].to_numpy()

#print(df_all_days[['precision', 'recall', 'accuracy', 'F1-score']].iloc[0].to_numpy())
perf_PA = tex_table_row(_df_PA, ['precision', 'recall', 'accuracy', 'F1-score'])
perf_LE = tex_table_row(_df_LE, ['precision', 'recall', 'accuracy', 'F1-score'])

corr_v_PA = tex_table_row(df_all_PA, ['Vr2(LLT)', 'Vr2(LWP)', 'Vr2(LCBH)'])
corr_c_PA = tex_table_row(df_all_PA, ['Cr2(LLT)', 'Cr2(LWP)', 'Cr2(LCBH)'])
corr_v_LE = tex_table_row(df_all_LE, ['Vr2(LLT)', 'Vr2(LWP)', 'Vr2(LCBH)'])
corr_c_LE = tex_table_row(df_all_LE, ['Cr2(LLT)', 'Cr2(LWP)', 'Cr2(LCBH)'])
for i in range(len(corr_v_PA)):
    corr_v_PA[i] = f'{corr_v_PA[i]:.2f}'
    corr_c_PA[i] = f'{corr_c_PA[i]:.2f}'
    corr_v_LE[i] = f'{corr_v_LE[i]:.2f}'
    corr_c_LE[i] = f'{corr_c_LE[i]:.2f}'
    
    

# \LaTeX error matrix
print(f'Statistics-PA\n')

print(pmatrix(total_em_PA[[0, 2, 3,1]]))
perf_PA = UT.performance_metrics(*total_em_PA)
print()
print('prec, recall ... ', f"{perf_PA['precision']:.2f} & {perf_PA['recall']:.2f} & {perf_PA['accuracy']:.2f} & {perf_PA['F1-score']:.2f} & ", end='')
print(f'{corr_v_PA[0]} / {corr_c_PA[0]} & {corr_v_PA[1]} / {corr_c_PA[1]} & {corr_v_PA[2]} / {corr_c_PA[2]}')


print(f'\n\nStatistics-LE\n')
print(pmatrix(total_em_LE[[0, 2, 3,1]]))
perf_LE = UT.performance_metrics(*total_em_LE)
print()
print('prec, recall ... ', f"{perf_LE['precision']:.2f} & {perf_LE['recall']:.2f} & {perf_LE['accuracy']:.2f} & {perf_LE['F1-score']:.2f} & ", end='')
print(f'{corr_v_LE[0]} / {corr_c_LE[0]} & {corr_v_LE[1]} / {corr_c_LE[1]} & {corr_v_LE[2]} / {corr_c_LE[2]}')


print(f'Statistics-PA new\n')

print(pmatrix(total_em_PA[[0, 2, 3,1]]))
perf_PA = UT.performance_metrics(*total_em_PA)
print()
print('prec, recall ... ', f"{perf_PA['precision']:.2f} & {perf_LE['npv']:.2f} & {perf_PA['recall']:.2f} & {perf_PA['specificity']:.2f} & {perf_PA['accuracy']:.2f} & {perf_LE['F1-score']:.2f} & ", end='')
print(f'{corr_v_PA[0]} / {corr_c_PA[0]} & {corr_v_PA[1]} / {corr_c_PA[1]}')
#print(f'{corr_v_PA[0]} / {corr_c_PA[0]} & {corr_v_PA[1]} / {corr_c_PA[1]} & {corr_v_PA[2]} / {corr_c_PA[2]}')


print(f'\n\nStatistics-LE new\n')
print(pmatrix(total_em_LE[[0, 2, 3,1]]))
perf_LE = UT.performance_metrics(*total_em_LE)
print()
print('prec, recall ... ', f"{perf_LE['precision']:.2f} & {perf_LE['npv']:.2f} & {perf_LE['recall']:.2f} & {perf_PA['specificity']:.2f} & {perf_LE['accuracy']:.2f} & {perf_LE['F1-score']:.2f} & ", end='')
print(f'{corr_v_LE[0]} / {corr_c_LE[0]} & {corr_v_LE[1]} / {corr_c_LE[1]}')
#print(f'{corr_v_LE[0]} / {corr_c_LE[0]} & {corr_v_LE[1]} / {corr_c_LE[1]} & {corr_v_LE[2]} / {corr_c_LE[2]}')




### Performance metric number of occurrence binned, scatter plot
bin_edges = np.array([0, 25, 50, 100, 150, 200, 300, 400, 2000])
bin_edges_edr = np.linspace(-9, 1, 9)

row_names = bin_edges
row_names2 = bin_edges_edr
print('PA shape', np_all_days_PA.shape)
print('LE shape', np_all_days_LE.shape)

n_profiles_PA = np.sum(np_all_days_PA[:, 0, -1])
n_profiles_LE = np.sum(np_all_days_LE[:, 0, -1])
print(f'n_profiles PA = {n_profiles_PA:_}')
print(f'n_profiles LE = {n_profiles_LE:_}')


names = [
     'TP',
     'TN',
     'FP',
     'FN',
     'PPV',
     'FAR',
     'FPR',
     'FBI',
     'NPV',
     'TPR',
     'TNR',
     'ACC',
     'F1S',
     'ets',
     'Vr2(LLT)',
     'Vr2(LWP)',
     'Vr2(LCBH)',
     'Cr2(LLT)',
     'Cr2(LWP)',
     'Cr2(LCBH)',
     'n_time_steps'
]
    
from scipy.ndimage import gaussian_filter

dt_sorted = sorted(dt_list)
ts_sorted = np.array([(_dt - datetime.datetime(1970, 1, 1)).total_seconds() for _dt in dt_sorted])

def histogram_corr_lwp(array: np.array, xlim):
    """
    Args:
        array: two dim. array 0=n_time_steps, 1=n_lwp_bins
    """

    hist_list = []
    sizes_list = []
    len_y = array.shape[1]
    for j in range(len_y):
        tmp = array[:, j]
        hist = np.histogram(tmp[tmp>0], range=tuple(xlim), bins=15,
                            weights=np.zeros_like(tmp[tmp>0]) + 1. / max(tmp[tmp>0].size, 1.0e-7)
                            #density=True
                           )
        hist_list.append(hist[0])
        sizes_list.append(tmp[tmp>0].size)
        
    X = hist[1]
    Y = np.linspace(0, len_y, len_y+1)
    Z = np.ma.masked_less_equal(hist_list, 0)
    
    return X, Y, Z, sizes_list


def _plot_pm_ts_histo(array, ax, i, xlim=[0.01, 1.01], vmin=1.e-3, vmax=0.3, weights=None):

    X, Y, Z, sizes_list = histogram_corr_lwp(array[:, :, i], xlim=xlim) # choose index of metric as last dimension
    
    if weights is not None:
        Z = Z * weights[:, None]
        
    pmesh = ax.pcolormesh(Y, X, Z.T, 
                          #cmap=hist2dcolors,
                          cmap='seismic',
                             #vmin=0, vmax=vmax, 
                             norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
                             shading='auto'
                            )
    
    ax.set_xticks(Y)
    ax.set_title(names[i])
    ax.locator_params(axis='x', nbins=array[:, :, i].shape[1])

    return ax, pmesh


def _plot_2dhisto(array, idx, name='', xlabel='', ylabel='', fontsize=10, vmin=1.0e-3, vmax=1, relative=False):
    
    if 'edr' in name:
        rownames = [f'{name:2.1f}' for name in row_names2[1:-1]]
    else:
        rownames = [f'{name}' for name in row_names[1:]]
        x_lim=None
    
    weights = None
    
    
    with plt.style.context(['science', 'ieee']):
        fig, ax = plt.subplots(nrows=1, ncols=len(idx), figsize=(7, 1.25))
        pmesh = np.array((len(idx), 2))
        
        sum_all = np.sum(array[:, :, :4], axis=(0, 2))
        for i, irow in enumerate(idx):
            ax[i], pmesh = _plot_pm_ts_histo(
                array, ax[i], irow, 
                xlim=[-0.01, 1.01], 
                vmin=vmin, vmax=vmax, 
                weights=weights,
            )
            ax[i].set_xticks(np.arange(0, len(rownames)))
            ax[i].set_xticklabels(rownames, rotation=45)
            if i > 0:
                ax[i].set_yticklabels([])
        
#        for iax in ax[1:]:
#            iax.set_yticklabels([])

        # Set common labels
        n_profiles = np.sum(array[:, :, -1], dtype=int)
        title_string = name[:name.find('-')].replace('_', ' ') + rf', n$_\text{{samples}} = {{{n_profiles:,}}}$'
        fig.suptitle(title_string, y=1.15)
        fig.text(0.5, -0.2, xlabel, ha='center', va='center', fontsize=fontsize)
        fig.text(0.07, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=fontsize)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.825, 0.15, 0.01, 0.7]) # x, y, w, h
        cb = fig.colorbar(pmesh, cax=cbar_ax)
        cb.set_label(label='relative frequency \nof occurrence', fontsize=fontsize, weight='normal')
        fig.subplots_adjust(wspace=0.2, bottom=0.1, top=0.85)
        plt.show()
        fig.savefig(f'{gfx_path}{name}{p_method}.png', dpi=200, facecolor='white')
 


_plot_2dhisto(
    np_all_days_PA[:,2:9, :], 
    [4, 8, 9, 10, 11], 
    name=f'Punta_Arenas-error-matrix-histo-lwp-paper', 
    xlabel = r'liquid water path LWP [g\,m$^{-2}$]', 
    ylabel='performance \n metric score [1]',
    vmin=0.01, vmax=1.0
)
#_plot_2dhisto(
#    np_all_days_PA[:,2:9, :], 
#    [4, 6, 8, 9], 
#    name=f'Punta_Arenas-error-matrix_histo-lwp-relative', 
#    xlabel = r'liquid water path LWP [g\,m$^{-2}$]', 
#    ylabel='performance metric score [1]',
#    vmin=0.001, vmax=0.1,
#    relative=True
#)


_plot_2dhisto(
    np_all_days_LE[:, 2:9, :], 
    [4, 8, 9, 10, 11], 
    name=f'Leipzig-error-matrix-histo-lwp-paper', 
    xlabel = r'liquid water path LWP [g\,m$^{-2}$]', 
    ylabel='performance \nmetric score [1]',
    vmin=0.01, vmax=1
    
)

#_plot_2dhisto(
#    np_all_days_LE[:, 2:9, :], 
#    [4, 8, 9, 10, 11], 
#    name=f'Leipzig-error-matrix_histo-lwp-relative', 
#    xlabel = r'liquid water path LWP [g\,m$^{-2}$]', 
#    ylabel='performance metric score [1]',
#    vmin=0.01, vmax=1,
#    relative=True
#    
#)



# lwp scatisticds
def _plot_pm_ts_mean(np_array, string, k0=0, fontsize=12):
    from scipy.ndimage import gaussian_filter
    idx1 = [-3, -6]
    idx2 = [-2, -5]

    pd.options.display.float_format = '{:.3f}'.format

    dt_sorted = sorted(dt_list)
    ts_sorted = np.array([(_dt - datetime.datetime(1970, 1, 1)).total_seconds() for _dt in dt_sorted])   
    x_range = (-.5, 1.)

    names = ['Cloudnet', 'VOODOO']
    line_style = ['-', '--', '-',  '--']
    colors = ['black','red', ]

    with plt.style.context(['science', 'ieee', 'grid']):
        fig, ax = plt.subplots(ncols=1, figsize=(3, 1.7))

        #dt = dt_sorted
        mean_list = []
        for i, idx in enumerate([idx1]):
            for ii in range(len(idx)):
                Z_PA = np_array[:, 0, idx[ii]]
                _hist_PA = np.histogram(Z_PA, range=x_range, bins=15, density=True)
                _skwew = scipy.stats.skew(_hist_PA[0], nan_policy='propagate')
                
                legend_string = '  $\mu = $ ' + f'{np.ma.median(Z_PA):.2f}'
                
                pmesh = ax.bar(
                    _hist_PA[1][1:], _hist_PA[0],  alpha=0.5,
                    label=names[ii]+legend_string, width=0.1,
                )

            #ax.legend(bbox_to_anchor=(0.25, 0.9, 0.75, 0.3))
            ax.legend(loc='upper left')
            ax.set_xlim(x_range)
#            ax.set_ylim((0,60))
            ax.grid(b=True, which='major', color='grey', linestyle='--', alpha=0.9)
            ax.grid(b=True, which='minor', color='grey', linestyle=':', alpha=0.5)
            ax.set_xlabel('correlation coefficient $r^2_{\mathrm{LWP}}$')
            ax.set_ylabel('relative frequency of occurence')
            bbox = ax.get_yticklabels()[-1].get_window_extent()
            x,_ = ax.transAxes.inverted().transform([bbox.x0, bbox.y0])
            n_profils = np.sum(np_array[:, 0, -1], dtype=int)
            ax.set_title(string.replace('_', ' ') + rf', n$_\text{{profiles}} = {{{n_profils:,}}}$',
                         ha='left', x=x+0.05)

        plt.show()
        fig.savefig(f'performance-metrics-{k0}-lwp_{string}{p_method}.png', dpi=200, facecolor='white')
    
_plot_pm_ts_mean(np_all_days_PA, 'Punta_Arenas', 0)