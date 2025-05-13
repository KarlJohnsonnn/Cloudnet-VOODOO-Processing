# Cloudnet-VOODOO-Processing

---

Future applications:
- VOODOO for arctic data sets
- LSC BioSmoke data sets
- add reader for other Doppler radar devices
  - MIRA35
  - KAZR
  - MMCR

ToDo:
- add option to setup GPU for cloudnet-processing when calling `generate_categorize`
- better naming convention for model/training parameters, e.g. `split` --> `training_split` for class VoodooOptions
- find where and how to add metric information for `wandb` monitoring

This repository resambles the analysis done by [Schimmel et al. 2022](https://amt.copernicus.org/articles/15/5343/2022/).

Required packages:
  - https://github.com/actris-cloudnet/cloudnetpy/tree/main
  - https://github.com/actris-cloudnet/rpgpy
  - https://github.com/actris-cloudnet/voodoonet



## Installation

### 0. Create a conda/python environment and activate it

For conda environment:
```bash
conda create -n cloudnet-voodoo-processing python=3.10
```
or for python environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
activate the environment:
```bash
conda activate cloudnet-voodoo-processing
```

### 1. Clone the repository and install required packages

```bash
git clone https://github.com/actris-cloudnet/Cloudnet-VOODOO-Processing.git
cd Cloudnet-VOODOO-Processing
pip install -r requirements.txt
```


### 2. Use the jupyter notebooks for testing the code

The following notebooks guide you through the VOODOO processing pipeline:

1. **01-Download-Example-Data.ipynb**: Downloads example data needed for training and testing the model.
2. **02-Generate-Trainingset.ipynb**: Creates the training dataset from the downloaded data.
3. **03-Train-New-Model.ipynb**: Trains a new VOODOO model using the generated training dataset.
