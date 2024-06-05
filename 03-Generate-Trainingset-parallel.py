import os
import glob
import torch
import toml
import multiprocessing
from common_imports import *
import voodoonet  # Assuming voodoonet is the module you're using for generating training data

def load_config(config_path='03-config.toml'):
    return toml.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path))

def generate_training_data(site, site_config, start_hour, file_interval):
    classification_files = sorted(glob.glob(site_config['classification_path'] + '*classification*.nc'))
    lv0_files =            sorted(glob.glob(site_config['lv0_path'] + '**/*.LV0', recursive=True))[start_hour::file_interval]

    print(f"Site: {site}")
    print(f"Number of LV0 files: {len(lv0_files)}")

    output_file = site_config['output_root'] + f"training-data-set-{site}-{start_hour}-{file_interval}.pt"
    try:
        voodoonet.generate_training_data(lv0_files, classification_files, output_file)
        print(f"Training data set generated: {output_file}")
    except Exception as e:
        print(f"Error generating training data for site {site}: {e}")

def process_site(site, site_config, start_hour, file_interval):
    generate_training_data(site, site_config, start_hour, file_interval)

if __name__ == '__main__':
    config = load_config()

    #sites = list(config.keys())  # Get all site names from the config file
    start_hour = 0
    file_interval = 10  # Use every 10th file, adjust as needed

    #generate_training_data('eriswil', config['eriswil'], 0, 10)
    sites =[
        #'eriswil',
        #'leipzig-tropos',
        'punta-arenas',
        'leipzig-lim',
        ] 
    # Prepare arguments for multiprocessing
    tasks = []
    for site in sites:
        for start_hour in range(file_interval):
            tasks.append((site, config[site], start_hour, file_interval))

    # Run in parallel
    num_processes = min(len(tasks), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_site, tasks)
