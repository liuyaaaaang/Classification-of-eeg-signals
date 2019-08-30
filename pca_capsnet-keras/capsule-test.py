# -*- coding: utf-8 -*-

from keras.models import load_model
import keras 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



dataset=pd.read_csv('data.csv',header=None)
data_1=dataset.dropna(axis=1)
features=np.array(data_1.iloc[:,2:].values)
lle=LocallyLinearEmbedding(n_components=100,n_neighbors=10)
new_feature=lle.fit_transform(features)
scaler_1=preprocessing.StandardScaler().fit(new_feature)
X_data_1= scaler_1.transform(new_feature)

label=dataset.iloc[:,0].values
encoder=LabelEncoder()
label=encoder.fit_transform(label)
Y_data=pd.get_dummies(label).values
X_train, X_test,y_train, y_test = train_test_split(X_data_1,Y_data, test_size=0.4, random_state=0)
X_train=X_train.reshape((-1,10,10,1)).astype('float32')
X_test=X_test.reshape((-1,10,10,1)).astype('float32')

model = load_model('trained_model.h5')
model.load
Y_pred=model.predict(X_test)
Y_pred_class=np.argmax(Y_pred,axis=1)
Y_test_class=np.argmax(y_test,axis=1)
report=classification_report(Y_pred_class,Y_test_class)