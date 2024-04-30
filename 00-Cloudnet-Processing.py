from common_imports import *


data_root = '/data/eriswil/cloudnet/input/'
date = '20230217'

# Required Input Files
input_files = {
    'lv0_files': glob.glob(f'/data/eriswil/rpg94/Y{date[:4]}/{date[2:]}_*.LV0'), 
    'radar': data_root + f'rpg94/{date}_eriswil_rpg-fmcw-94.nc',
    'lidar': data_root + f'ceilo/{date}_eriswil_chm15kx.nc',
    'model': data_root + f'model/{date}_eriswil_ecmwf.nc',
    'mwr':   data_root + f'hatpro/{date}_eriswil_hatpro.nc'
}

if 'lv0_files' in input_files:
    categorzie_file = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize_voodoo.nc'
    classification_file = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification_voodoo.nc'
else:   
    categorzie_file = f'/data/eriswil/cloudnet/categorize/cloudnet_only/{date}_categorize.nc'
    classification_file = f'/data/eriswil/cloudnet/classification/cloudnet_only/{date}_classification.nc'


uuid = generate_categorize(input_files, categorzie_file)
uuid = generate_classification(categorzie_file, classification_file)