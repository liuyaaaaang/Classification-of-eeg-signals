import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


dataset=pd.read_csv('data.csv',header=None)
da_1=dataset.dropna(axis=1)

X_data = np.array(da_1.iloc[:,2:].values)


label=dataset.iloc[:,0].values
encoder=LabelEncoder()
label=encoder.fit_transform(label)
Y_data=pd.get_dummies(label).values


model = Sequential()

model.add(Dense(128, input_shape=(500,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
    
model.compile(Adam(lr=0.001), 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_data,Y_data,epochs=1000)
