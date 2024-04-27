# Import common imports
from common_imports import *

year = '2024'
classification_files = sorted(glob.iglob(f'/data/eriswil/cloudnet/classification/Y{year}/*.nc'))
rpg_lv0_files = sorted(glob.iglob(f'/data/eriswil/rpg94/Y{year}/*.LV0'))[::10] # use every 10th file

print(len(rpg_lv0_files))
print(f'CUDA available: {torch.cuda.is_available()},  Number of CUDA devices: {torch.cuda.device_count()}')

voodoonet.generate_training_data(rpg_lv0_files, classification_files, f'training-data-set-Y{year}-10pct.pt')
