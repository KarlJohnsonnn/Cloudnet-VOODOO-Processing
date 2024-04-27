# Import common imports
from common_imports import *


options = utils.VoodooOptions(
    device="cuda:1", 
    progress_bar=True,
    )


training_options = utils.VoodooTrainingOptions(
    epochs=3,
    batch_size=256,
    wandb=utils.WandbConfig(
        project='eriswil',
        name='voodoonet-training-Y2022-2024-10pct',
        entity='krljhnsn'
    ),
)

# create a lambda function that returns the training data set file for given year
training_data_set = f'../training-data-set-Y2022-2024-10pct.pt'
new_model_file = f'eriswil-model-Y2022-2024-10pct-{options.device}.pt'

voodoonet.train(
    training_data_set, 
    new_model_file, 
    model_options=options, 
    training_options=training_options,
    )