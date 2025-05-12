# Import common imports
from common_imports import *
import xarray as xr

date = "20230204"
year = date[:4]
classification_files = sorted(glob.iglob(f'/data/eriswil/cloudnet/classification/Y{year}/{date}*.nc'))[0]
categorize_files = sorted(glob.iglob(f'/data/eriswil/cloudnet/categorize/Y{year}/{date}*.nc'))[0]
rpg_lv0_files = sorted(glob.iglob(f'/data/eriswil/rpg94/Y{year}/{date[2:]}*.LV0'))

# define device (GPU or CPU) and progress bar
options = utils.VoodooOptions(
    device="cuda:0", 
    progress_bar=True,
    trained_model='eriswil-model-Y2022-2024-10pct-cuda:1.pt'
    )

# inference
probability_liquid = voodoonet.infer(rpg_lv0_files[:2], options=options)

# VOODOO cloud droplet likelyhood colorbar (viridis + grey below minimum value)
from matplotlib import cm
from matplotlib.colors import ListedColormap

viridis = cm.get_cmap('viridis', 256)
voodoo_cmap = viridis(np.linspace(0, 1, 256))
voodoo_cmap[:1, :] = np.array([220/256, 220/256, 220/256, 1])
voodoo_cmap = ListedColormap(voodoo_cmap)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.pcolormesh(probability_liquid.T, cmap=voodoo_cmap, vmin=0.5, vmax=1)
fig.savefig('voodoo-liquid-likelihood2.png', dpi=300)
