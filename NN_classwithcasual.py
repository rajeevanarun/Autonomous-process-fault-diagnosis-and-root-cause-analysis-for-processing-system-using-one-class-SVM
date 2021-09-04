# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:48:44 2021

@author: rajee
"""

# Neural Network train model

# -*- coding: utf-8 -*-
import os
import tqdm
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from sklearn.neural_network import MLPClassifier


data = pd.read_excel('cwstep1.xlsx')
feature_names=data.columns[0:5]
X = data.iloc[:,0:5]
Y = data.iloc[:,5]
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.4,random_state = 42)

# scale value
scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)
# NN model and training
model = MLPClassifier(solver='lbfgs',alpha=1e-15,hidden_layer_sizes=(4,),random_state=1)
model.fit(scaled_train,y_train)
# Model test and prediction error calculation
predict_y = model.predict(scaled_test)
MAE = mean_absolute_error(y_test,predict_y)

### COMPUTE PERMUTATION AND SCORING ###

os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

final_score = []
shuff_pred = []

for i,col in enumerate(x_test.columns):

    # shuffle column
    shuff_test = scaled_test.copy()
    shuff_test[:,i] = np.random.permutation(shuff_test[:,i])
    
    # compute score
    score = mean_absolute_error(y_test, model.predict(shuff_test))
    shuff_pred.append(model.predict(shuff_test))
    final_score.append(score)
    
final_score = np.asarray(final_score)

plt.subplots(1, 1, figsize=(12, 8))
plt.barh(range(x_train.shape[1]), abs(final_score - MAE)/MAE*100)
plt.yticks(range(x_train.shape[1]),feature_names)
plt.title('IDV_1 Fault Analysis - NN_Permutation')
np.set_printoptions(False)

# casuality anaysis 
# Simulate mean differance test among prediction

os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

id_ = 20

merge_pred = np.hstack([shuff_pred[id_], predict_y])
observed_diff = abs(shuff_pred[id_].mean() - merge_pred.mean())
extreme_values = []
sample_d = []

for _ in range(10000):
    sample_mean = np.random.choice(merge_pred, size=shuff_pred[id_].shape[0]).mean()
    sample_diff = abs(sample_mean - merge_pred.mean())
    sample_d.append(sample_diff)
    extreme_values.append(sample_diff >= observed_diff)
    
p_value = np.sum(extreme_values)/10000


### PLOT MEAN DIFFERENCE DISTRIBUTIONS ###

plt.subplots(1, 1, figsize=(12, 8))

y, x, _ = plt.hist(sample_d, alpha=0.6)
xx = max(x)
plt.vlines(observed_diff, 0, max(y), colors='red', linestyles='dashed')
plt.title('IDV4_fault condition XMEAS20')
plt.show()

