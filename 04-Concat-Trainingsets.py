# Import common imports
from common_imports import *
import toml
from tqdm.auto import tqdm

def load_config(config_path='03-config.toml'):
    return toml.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path))

# Function to find all .pt files in a given directory
def list_pt_files(directory):
    return glob.glob(f"{directory}/*.pt")

# Function to load and concatenate .pt files
def load_and_concatenate_pt_files(files, dim=0):
    features, labels = [], []
    for file in tqdm(files, ncols=100, desc='Loading .pt files'):
        tensor = torch.load(file)
        features.append(tensor['features'])
        labels.append(tensor['labels'])

    concatenated_features = torch.cat(features, dim=dim)
    concatenated_labels = torch.cat(labels, dim=dim)
    return {'features': concatenated_features, 'labels': concatenated_labels}


if __name__ == '__main__':

    config = load_config()

    all_pt_files = [list_pt_files(val['output_root']) for _, val in config.items()]
    all_pt_files_flattened = [item for sublist in all_pt_files for item in sublist]

    input_pt_files = all_pt_files_flattened[::3]
    # Print all found .pt files
    for pt_file in input_pt_files:
        print(pt_file)

    concat_dataset = load_and_concatenate_pt_files(input_pt_files)

    # save new tensor y to file
    torch.save(concat_dataset, '/data/voodoo-trainings-data/training-data-set-EW-LE-LE-PA-01.pt')

