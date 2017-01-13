'''
Created on Jan 13, 2017

@author: runsheng
'''
import sys
sys.path.append("./src") # append to system path

import json
import pandas as pd
import numpy as np

import modeling_tool as mt
from make_training_data import data_sampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import matplotlib.pyplot as plt 


df = pd.read_csv('./data/0109_nine_functional_use_descs.csv',header=0)

this_data = data_sampler()
this_data.sample_data(df, num_test_left=50)

tst_X_raw = this_data.tst_data['descs'] #only the first
N_tst, M_tst = tst_X_raw.shape
tst_X = np.ones((N_tst, M_tst+1))
tst_X[:, 1:] = tst_X_raw
tst_Y = this_data.tst_data['target']

# fit the test data
scaler = joblib.load('./net/tensorflow_classifier_Jan12_scaler.pkl')
tst_X = scaler.transform(tst_X)

another_classifier = mt.create_functional_use_classifier.load_model('./net/tensorflow_classifier_Jan12.json')
thispred = another_classifier.predict(tst_X)

acc_tst = np.mean(np.argmax(tst_Y,axis=1) == thispred)
print acc_tst