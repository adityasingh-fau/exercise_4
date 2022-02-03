import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

batch = 10
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
dt = pd.read_csv('./data.csv', sep=';')
training_data, testing_data = train_test_split(dt, test_size=0.15)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset 
# objects
loader = t.utils.data.DataLoader(ChallengeDataset(training_data, 'train'), batch_size=batch, shuffle=True)
train_data_loading = loader
testing_data_load = t.utils.data.DataLoader(ChallengeDataset(training_data, 'test'), batch_size=batch, shuffle=False)
# create an instance of our ResNet model
model = model.ResNet()
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
lossfunc = t.nn.BCEWithLogitsLoss()
# lossfunc = t.nn.HuberLoss()
# set up the optimizer (see t.optim)
opt = t.optim.Adam(model.parameters(), lr=0.001, weight_decay=0, betas=(0.9, 0.999), amsgrad=False, eps=1e-10)
# create an object of type Trainer and set its early stopping criterion
model_trainer = Trainer(model, lossfunc, opt, train_data_loading, testing_data_load, True, 1e-6)
# go, go, go... call fit on trainer
res = model_trainer.fit(epochs=30)
f1_mean, f1_cracks_mean, f1_inactives_mean = model_trainer.f1_scores()
f1_mean_fit, f1_crack_fit, f1_inactive_fit = model_trainer.f1_scoresFit()

print("Test: F1 Crack {} F1 Inactive {} F1 Mean {}".format(f1_cracks_mean, f1_inactives_mean, f1_mean))
print("Fit:  F1 Crack {} F1 Inactive {} F1 Mean {}".format(f1_mean_fit, f1_inactive_fit, f1_mean_fit))

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
