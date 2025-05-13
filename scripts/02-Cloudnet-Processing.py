from common_imports import *


# eriswil data + leipzig tropos
data_root = '/data/eriswil/cloudnet/input/'
#date = '20230217'
date = '20240108'
y, m, d = str(date[:4]), str(date[4:6]).zfill(2), str(date[6:]).zfill(2)

# Required Input Files
input_files = {
    # see: https://actris-cloudnet.github.io/cloudnetpy/api.html#categorize.generate_categorize
    'lv0_files': glob.glob(f'/data/eriswil/rpg94/Y{date[:4]}/{date[2:]}_*.LV0'), 
    'radar': glob.glob(data_root + f'rpg94/{date}_eriswil_rpg-fmcw-94*.nc')[0],
    'lidar': glob.glob(data_root + f'ceilo/{date}_eriswil_chm15kx*.nc')[0],
    'model': glob.glob(data_root + f'model/{date}_eriswil_ecmwf*.nc')[0],
    'mwr':   glob.glob(data_root + f'hatpro/{date}_eriswil_hatpro*.nc')[0]
}


if 'lv0_files' in input_files:
    categorzie_file = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize_voodoo.nc'
    classification_file = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification_voodoo.nc'
else:   
    categorzie_file = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize.nc'
    classification_file = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification.nc'


uuid = generate_categorize(input_files, categorzie_file)
uuid = generate_classification(categorzie_file, classification_file)