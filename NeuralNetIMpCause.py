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
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import learning_curve
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import seaborn as sns
import time


data = pd.read_excel('cwstep1.xlsx')
feature_names=data.columns[0:5]
length = len(feature_names)

X = data.iloc[:,0:5]
Y_t = data.iloc[:,5].values

Y = Y_t.reshape(-1,1)
ohe = OneHotEncoder()
Y = ohe.fit_transform(Y).toarray()
cl_num = len(Y.T)

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 50)

# scale value
Xt = X.to_numpy()

scaler = StandardScaler()
scaler.fit(Xt[0:400,1:5])

scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

start_time = time.time()
# Neural network
model = Sequential()
model.add(Dense(length+1, input_dim=length, activation='tanh'))
model.add(Dense(68,activation='tanh'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(72, activation='sigmoid'))
model.add(Dense(cl_num, activation='softmax'))

#model compiler0
opt = keras.optimizers.Adam(lr=0.004)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

history = model.fit(scaled_train,y_train, epochs=300, batch_size=32)
# Model test and prediction error calculation
y_pred = model.predict(scaled_test)

#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)

MAE = mean_absolute_error(y_test,y_pred)

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

x_score = abs(final_score - MAE)/MAE*100
stop_time = time.time()

print(stop_time-start_time)

plt.subplots()
plt.barh(range(x_train.shape[1]), x_score)
plt.yticks(range(x_train.shape[1]),feature_names)
plt.vlines((0.30* max(x_score)),0,5,colors ='red',linestyles='dashdot',label = 'Marginal_score')
plt.title('CSTH F1 Fault Analysis - NN_Permutation')
plt.ylabel('CSTH Variable')
plt.xlabel('Permutation Score')
plt.legend()
np.set_printoptions(False)
plt.show()

corr, _ = spearmanr(x_train)
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names.tolist(), leaf_rotation=90)
# dendro_idx = dendro['ivl']
plt.ylabel('spearmanr correlation score')
plt.title('CSTH F2 Fault Analysis - spearmanr correlatrion')
plt.show()
