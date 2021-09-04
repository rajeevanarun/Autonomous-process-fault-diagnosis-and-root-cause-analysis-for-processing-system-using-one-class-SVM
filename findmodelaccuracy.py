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

data = pd.read_excel('IDV1.xlsx')
feature_names=data.columns[0:22]
length = len(feature_names)

X = data.iloc[:,0:22]
Y_t = data.iloc[:,22].values

Y = Y_t.reshape(-1,1)
ohe = OneHotEncoder()
Y = ohe.fit_transform(Y).toarray()
cl_num = len(Y.T)


x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.6,random_state = 42)

# scale value
Xt = X.to_numpy()

scaler = StandardScaler()
scaler.fit(Xt[0:400,0:22])

scaled_train = scaler.fit_transform(x_train)
scaled_test = scaler.transform(x_test)

start_time = time.time()
# Neural network
model = Sequential()
model.add(Dense(length+1, input_dim=length,activation='tanh'))
model.add(Dense(64,activation='relu'))
model.add(Dense(cl_num, activation ='softmax'))

#model compiler0
opt = keras.optimizers.Adam(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

history = model.fit(scaled_train,y_train,validation_data=(scaled_test, y_test), epochs=150, batch_size=100)
#Model test and prediction error calculation
y_pred = model.predict(scaled_train)
# mse_train = history.history['mse']
# mse_train = np.mean(mse_train)


#Converting predictions to label
pred = y_pred.argmax(axis=1)
test = y_test.argmax(axis=1)
# from sklearn.metrics import accuracy_score
# a = accuracy_score(y_train,y_pred)
# print('Accuracy is:', a*100)

MAE = mean_absolute_error(y_train,y_pred)

### COMPUTE PERMUTATION AND SCORING ###

os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

final_score = []
shuff_pred = []

for i,col in enumerate(x_test.columns):

    # shuffle column
    shuff_train = scaled_train.copy()
    shuff_train[:,i] = np.random.permutation(shuff_train[:,i])
    
    # model trian
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    history_shuff = model.fit(shuff_train,y_train, validation_data=(scaled_test, y_test), epochs=150, batch_size=100)
    y_pred_shuff = model.predict(shuff_train)
        
    # compute score
    score = mean_absolute_error(y_train, y_pred_shuff)
    shuff_pred.append(model.predict(shuff_train))
    final_score.append(score)
    
final_score = np.asarray(final_score)

x_score = abs(final_score - MAE)/MAE*100
stop_time = time.time()

print(stop_time-start_time)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

# ### COMPUTE PERMUTATION AND SCORING ###

# os.environ['PYTHONHASHSEED'] = str(33)
# np.random.seed(33)
# random.seed(33)

# final_score = []
# shuff_pred = []

# X_test = pd.DataFrame(X_test)

# for i,col in enumerate(X_test.columns):

#     # shuffle column
#     shuff_test = X_test.copy()
#     shuff_test[:,i] = np.random.permutation(shuff_test[:,i])
    
#     # compute score
#     score = mean_absolute_error(Y_test, model.predict(shuff_test))
#     shuff_pred.append(model.predict(shuff_test))
#     final_score.append(score)
    
# final_score = np.asarray(final_score)

# plt.subplots(1, 1, figsize=(12, 8))
# plt.barh(range(X_train.shape[1]), abs(final_score - MAE)/MAE*100)
# plt.yticks(range(X_train.shape[1]),feature_names)
# plt.title('IDV_1 Fault Analysis - NN_Permutation')
# np.set_printoptions(False)

