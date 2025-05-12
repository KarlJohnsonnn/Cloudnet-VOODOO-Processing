# Import common imports
from common_imports import *

date = "20230217"

classification_files        = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification.nc'
categorize_files            = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize.nc'
classification_voodoo_files = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification-voodoo.nc'
categorize_voodoo_files     = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize-voodoo.nc'

rpg_lv0_files = sorted(glob.iglob(f'/data/eriswil/rpg94/Y{date[:4]}/{date[2:]}*.LV0'))

# define device (GPU or CPU) and progress bar
options = utils.VoodooOptions(
    device="cuda:0", 
    progress_bar=True,
    trained_model='eriswil-model-Y2022-2024-10pct-cuda:1.pt'
    )

# inference
probability_liquid = voodoonet.infer(rpg_lv0_files, options=options)

# create the voodoo categorize file
create_cloudnet_voodoo_categorize(
    categorize_files, classification_files, options.trained_model, probability_liquid
    )

# load the classification data
class_xr = xr.open_dataset(classification_files)
class_voodoo_xr = xr.open_dataset(classification_voodoo_files)

# get Cloudnet classification colormap and labels
cloudnet_class_labels, cloudnet_class_cmap = get_cloudnet_cmap()
len_ticks = len(cloudnet_class_labels)
            
z1 = class_xr['target_classification'].values.T
z2 = class_voodoo_xr['target_classification'].values.T
x = class_xr['time'].values
y = class_xr['height'].values

# plot the cloudnet classification
fig, ax = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
imC = ax[0].pcolormesh(x, y, z1, cmap=cloudnet_class_cmap)
imC = ax[1].pcolormesh(x, y, z2, cmap=cloudnet_class_cmap)
cbC = plt.colorbar(
    imC,  ax=ax, ticks=np.linspace(0.5,len_ticks-0.5, len_ticks), 
    orientation="vertical", pad=0.05, shrink=0.85
    )
cbC.ax.set_yticklabels(cloudnet_class_labels)
cbC.ax.tick_params(labelsize=14)

# format major xtick label
for axi in ax:
    axi.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

fig.savefig('/home/schimmel/examples/plots/classification-Cloudnet-VOODOO.png', dpi=300)