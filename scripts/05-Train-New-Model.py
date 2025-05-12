# Import common imports
from common_imports import *


options = utils.VoodooOptions(
    device="cuda:1", 
    progress_bar=True,
    )


training_options = utils.VoodooTrainingOptions(
    epochs=5,
    batch_size=4096,
    wandb=utils.WandbConfig(
        project='voodoo2.5',
        name='voodoo-4sites-model',
        entity='krljhnsn'
    ),
)

# create a lambda function that returns the training data set file for given year
training_data_set = f'/data/voodoo-trainings-data/training-data-set-EW-LE-LE-PA-01.pt'
new_model_file = f'Voodoo-4sites-model-{options.device}.pt'

voodoonet.train(
    training_data_set, 
    new_model_file, 
    model_options=options, 
    training_options=training_options,
    )