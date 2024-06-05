from common_imports import *
import multiprocessing
import toml
import glob
import os
from cloudnetpy.instruments import rpg2nc, ceilo2nc, hatpro2nc

def load_config():
    # open the toml file in the same folder as the python script, add the aboslute path by looking on the path to this script
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '01-config.toml')
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config


def get_raw_input_files(date, site, config):
    y, m, d = str(date[:4]), str(date[4:6]).zfill(2), str(date[6:]).zfill(2)

    if site not in config:
        raise ValueError('Invalid Site')
    
    site_config = config[site]
    lidar_glob = site_config.get('lidar_glob', False)
    lidar_pattern = site_config['lidar_path']

    input_files = { 
        'radar': site_config['radar_path'].format(y=y, m=m, d=d, date=date),
        'lidar': glob.glob(lidar_pattern.format(y=y, m=m, d=d, date=date))[0] if lidar_glob else lidar_pattern.format(y=y, m=m, d=d, date=date),
        'mwr':   site_config['mwr_path'].format(y=y, m=m, d=d, date=date)
    }
    
    output_root = site_config['output_root']
    site_meta  = {'name': site, 'altitude': site_config['altitude']}
    
    output_files = {
        'radar': f"{output_root}radar/{date}_{site}_radar.nc",
        'lidar': f"{output_root}lidar/{date}_{site}_lidar.nc",
        'mwr':   f"{output_root}mwr/{date}_{site}_mwr.nc",
    }
    
    return input_files, output_files, site_meta

def process_file_type(file_type, input_file, output_file, site_meta, conversion_func, overwrite=False):
    if not overwrite and os.path.exists(output_file):
        print(f"{file_type.capitalize()} output file {output_file} already exists, skipping.")
        return None

    try:
        uuid = conversion_func(input_file, output_file, site_meta)
        print(f"{file_type.capitalize()} processed successfully with UUID {uuid}")
        return uuid
    except Exception as e:
        print(f"Error processing {file_type}: {e}")
        return None

def process_day(date, site, config, overwrite=False):
    try:
        input_files, output_files, site_meta = get_raw_input_files(date, site, config)
    except Exception as e:
        print(f"Error fetching input files for date {date} and site {site}: {e}")
        return
    
    uuid_map = {}
    uuid_map['radar'] = process_file_type('radar', input_files['radar'], output_files['radar'], site_meta, rpg2nc, overwrite)
    uuid_map['lidar'] = process_file_type('lidar', input_files['lidar'], output_files['lidar'], site_meta, ceilo2nc, overwrite)
    uuid_map['mwr']   = process_file_type('mwr',   input_files['mwr'],   output_files['mwr'],   site_meta, hatpro2nc, overwrite)
    print(uuid_map)
    return uuid_map

def process_date_range(date_range, site, config, num_processes=None, overwrite=False):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_day, [(date, site, config, overwrite) for date in date_range])

if __name__ == '__main__':
    config = load_config()

    #date_range, site = ['20181127', '20190927'], 'punta-arenas'
    #date_range, site = ['20201216', '20220312'], 'leipzig-lim'
    date_range, site = ['20221101', '20241130'], 'leipzig-tropos'
    #date_range, site = ['20221101', '20240301'], 'eriswil'

    # create a list of dates given the date_range start and end date
    pd_date_range = pd.date_range(
        start=date_range[0], 
        end=date_range[1], 
        freq='D'
    ).strftime('%Y%m%d')
    date_list = [str(date) for date in pd_date_range]

    process_date_range(
        date_list, 
        site, 
        config,
        num_processes   = 70,       # Specify the number of processes to use, or None for default (CPU count),
        overwrite       = False,    # Change to True if you want to overwrite existing files
    )
