# Import common imports
from common_imports import *
import xarray as xr

date = "20230217"

classification_files        = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification.nc'
categorize_files            = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize.nc'
classification_voodoo_files = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification-voodoo.nc'
categorize_voodoo_files     = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize-voodoo.nc'

# define device (GPU or CPU) and progress bar
options = utils.VoodooOptions(
    device="cuda:0", 
    progress_bar=True,
    trained_model='eriswil-model-Y2022-2024-10pct-cuda:1.pt'
    )

# load the data
cat_xr = xr.open_dataset(categorize_files, decode_times=False)
cat_voodoo_xr = xr.open_dataset(categorize_voodoo_files, decode_times=False)
class_xr = xr.open_dataset(classification_files, decode_times=False)
class_voodoo_xr = xr.open_dataset(classification_voodoo_files, decode_times=False)

