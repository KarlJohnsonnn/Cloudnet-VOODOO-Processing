# Import common imports
from common_imports import *


# create a lambda function that returns the training data set file for given year
training_data_set = lambda year: f'training-data-set-Y{year}-10pct.pt'

y0 = torch.load(training_data_set(2022))
y1 = torch.load(training_data_set(2023))
y2 = torch.load(training_data_set(2024))
y = {}

for key in y0.keys():
    y[key] = torch.cat((y0[key], y1[key], y2[key]), dim=0)

print(y['features'].shape)

# save new tensor y to file
torch.save(y, 'training-data-set-Y2022-2024-10pct.pt')

