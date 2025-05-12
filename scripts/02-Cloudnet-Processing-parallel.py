import os
import glob
import toml
import multiprocessing
import pandas as pd
from common_imports import *

def load_config(config_path='02-config.toml'):
    return toml.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path))

def get_input_files(date, config, site, voodoo=False):
    y, m, d = str(date[:4]), str(date[4:6]).zfill(2), str(date[6:]).zfill(2)
    site_config = config[site]

    input_files = { 
        'radar':    glob.glob(site_config['radar_path'] + f'{date}*.nc')[0],
        'lidar':    glob.glob(site_config['lidar_path'] + f'{date}*.nc')[0],
        'mwr':      glob.glob(site_config['mwr_path']   + f'{date}*.nc')[0],
        'model':    glob.glob(site_config['model_path'] + f'{date}*.nc')[0],
    }
    
    output_root = site_config['output_root']
    output_files = {
        'categorize': f"{output_root}{date}_{site}_categorize.nc",
        'classification': f"{output_root}{date}_{site}_classification.nc"
    }

    if voodoo:
        input_files['lv0_files'] = glob.glob(site_config['lv0_path'].format(y=y, m=m, d=d, date=date))
        output_files['categorize'] = output_files['categorize'].replace('.nc', '_voodoo.nc')
        output_files['classification'] = output_files['classification'].replace('.nc', '_voodoo.nc')

    return input_files, output_files

def process_day(date, config, site, voodoo=False, overwrite=False):
    try:
        input_files, output_files = get_input_files(date, config, site, voodoo)
    except Exception as e:
        print(f"Error processing date {date} for site: {e}")
        return

    if not overwrite and os.path.exists(output_files['categorize']):
        print(f"Output file {output_files['categorize']} already exists, skipping.")
        return

    try:
        uuid_categorize = generate_categorize(input_files, output_files['categorize'])
        print(f"Generated categorize file with UUID: {uuid_categorize}")
    except Exception as e:
        print(f"Error generating categorize file for date {date}: {e}")
        return

    try:
        uuid_classification = generate_classification(output_files['categorize'], output_files['classification'])
        print(f"Generated classification file with UUID: {uuid_classification}")
    except Exception as e:
        print(f"Error generating classification file for date {date}: {e}")

def process_date_range(date_range, config, site,  voodoo=False, num_processes=None, overwrite=False):
    num_processes = num_processes or multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_day, [(date, config, site, voodoo, overwrite) for date in date_range])

if __name__ == '__main__':
    config = load_config()
    
    sites_date_ranges = [
        (['20181127', '20190927'], 'punta-arenas'),
        (['20201216', '20220312'], 'leipzig-lim'),
        (['20221101', '20241130'], 'leipzig-tropos'),
        (['20221101', '20240301'], 'eriswil')
    ]

    voodoo = False  # Set to True if VOODOO processing is required
    overwrite = False
    num_processes = 70  # Specify the number of processes to use, or None for default (CPU count)

    for date_range, site in sites_date_ranges:
        date_list = [str(date) for date in pd.date_range(start=date_range[0], end=date_range[1], freq='D').strftime('%Y%m%d')]
        process_date_range(date_list, config, site, voodoo=voodoo, num_processes=num_processes, overwrite=overwrite)
