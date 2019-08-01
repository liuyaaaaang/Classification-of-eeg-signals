from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
#from numpy import nan as NaN
from sklearn import svm
import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as ms
import sklearn.metrics as sm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier



###import data###
data=pd.read_csv('data.csv',header=None)
data_1=data.dropna(axis=1)
#print(data_1)
features=np.array(data_1.iloc[:,2:].values)
marks =data.iloc[:,0].values
#print(features.shape)
labels= np.zeros(len(marks))
labels[marks=='R']=1  
labels[marks=='F']=2


Fold = 10
i=1
skf = StratifiedKFold(n_splits=Fold, random_state=42, shuffle=False)
y_true= np.ones(labels.shape)
y_pred = np.ones(labels.shape)

for idx1, idx2 in skf.split(features, labels):  
    print('\nTesting model in fold : %i\n'%(i))
    i = i+1
    train_data=features[idx1,:]
#    print(train_data.shape)
    train_label=labels[idx1]
#    print(train_label.shape)
    
    test_data = features[idx2,:]
    test_label = labels[idx2]
    
    #  normalization
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)
#    print(train_data)
    train_label =train_label.astype(int)
    
    test_data = scaler.transform(test_data)
    test_data = min_max_scaler.transform(test_data)
    test_label =test_label.astype(int)
    
    ##RF###
    clf=RandomForestClassifier(n_estimators=100, n_jobs=4, oob_score=True)
    clf.fit(train_data,train_label)
    predictions=(clf.predict(test_data))
    acc_test=clf.score(test_data,test_label)
    print('test acc is {}'.format(acc_test))
    
    ###model evalution####    
    y_pred[idx2,] = predictions
    y_true[idx2,] = test_label
    
print('\n  Mean accuracy is {}'.format(accuracy_score(y_true,y_pred)))
