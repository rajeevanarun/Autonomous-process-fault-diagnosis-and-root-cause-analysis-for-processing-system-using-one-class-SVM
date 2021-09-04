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
from collections import defaultdict


data = pd.read_excel('IDV15.xlsx')
feature_names=data.columns[0:22]
length = len(feature_names)

X = data.iloc[:,0:22]
Y_t = data.iloc[:,22].values

Y = Y_t.reshape(-1,1)
ohe = OneHotEncoder()
Y = ohe.fit_transform(Y).toarray()
cl_num = len(Y.T)

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)

# scale value
Xt = X.to_numpy()

scaler = StandardScaler()
scaler.fit(Xt[0:400,0:22])

scaler = StandardScaler()
scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

start_time = time.time()
# Neural network
model = Sequential()
model.add(Dense(length+1, input_dim=length))
# model.add(Dense(68,activation='tanh'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(cl_num, activation ='softmax'))

#model compiler0
opt = keras.optimizers.Adam(lr=0.004)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

history = model.fit(scaled_train,y_train, epochs=100, batch_size=100)
# Model test and prediction error calculation
y_pred = model.predict(scaled_test)

#Converting predictions to label
pred = y_pred.argmax(axis=1)
test = y_test.argmax(axis=1)
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

plt.subplots(1,1,figsize=(12, 8))
plt.barh(range(x_train.shape[1]), x_score)
plt.yticks(range(x_train.shape[1]),feature_names)
plt.vlines((0.25* max(x_score)),0,22,colors ='red',linestyles='dashdot',label = 'Marginal_score')
plt.title('TE IDV15 Fault Analysis - NN_Permutation')
plt.ylabel('TE Variables')
plt.xlabel('Permutation Score')
plt.legend()
np.set_printoptions(False)
plt.show()

# # Explain the casuality
# os.environ['PYTHONHASHSEED'] = str(33)
# np.random.seed(33)
# random.seed(33)
# obs_dff_val = []
# for id_ in range (0,22):
#     shuff_pred1 = np.asarray(shuff_pred[id_])
#     shuff_pred1 = shuff_pred1.argmax(axis=1)
#     merge_pred = np.hstack([test, pred])
#     observed_diff = abs(shuff_pred1.mean() - merge_pred.mean())
#     obs_dff_val.append(observed_diff)
#     extreme_values = []
#     sample_d = []

#     for _ in range(10000):
#         sample_mean = np.random.choice(merge_pred, len(shuff_pred1)).mean()
#         sample_diff = abs(sample_mean - merge_pred.mean())
#         sample_d.append(sample_diff)
#         extreme_values.append(sample_diff >= observed_diff)
    
#     np.sum(extreme_values)/10000

#     y, x, _ = plt.hist(sample_d, alpha=0.6)
#     plt.vlines(observed_diff, 0,max(y), colors='red', linestyles='dashed')
#     plt.title('P_value model: T=%i' %(id_+1))
#     plt.show()
# print(obs_dff_val.index(max(obs_dff_val))+1)

# corr, _ = spearmanr(x_train)
# ax = sns.heatmap(corr,linewidths=.5,xticklabels=feature_names, yticklabels=feature_names)
# # corr_linkage = hierarchy.ward(corr)
# # dendro = hierarchy.dendrogram(corr_linkage, labels=feature_names.tolist(), leaf_rotation=90)
# # # dendro_idx = dendro['ivl']
# # plt.ylabel('spearmanr correlation score')
# # plt.title('TE IDV15 Fault Analysis - spearmanr correlatrion')
# # plt.show()

# # cluster_ids = hierarchy.fcluster(corr_linkage, 0.1, criterion='distance')
# # cluster_id_to_feature_ids = defaultdict(list)
# # for idx, cluster_id in enumerate(cluster_ids):
# #     cluster_id_to_feature_ids[cluster_id].append(idx)
# # selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
# # # Generate the p-value