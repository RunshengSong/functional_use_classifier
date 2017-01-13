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


def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    return pd.DataFrame.from_dict(report_data)

#static parameters
BATCH_SIZE = 1
RUGULARIZATION = 0.

# load data
df = pd.read_csv('./data/0109_nine_functional_use_descs.csv',header=0)


# sample and split data
this_data = data_sampler()
this_data.sample_data(df, num_test_left=50)

trn_X_raw = this_data.trn_data['descs']

# add bias here
N,M = trn_X_raw.shape
trn_X = np.ones((N, M+1))
trn_X[:, 1:] = trn_X_raw
trn_Y = this_data.trn_data['target']

tst_X_raw = this_data.tst_data['descs']
N_tst, M_tst = tst_X_raw.shape
tst_X = np.ones((N_tst, M_tst+1))
tst_X[:, 1:] = tst_X_raw
tst_Y = this_data.tst_data['target']
target_names = np.unique(this_data.trn_data['class'])


this_classifier = mt.create_functional_use_classifier('./net/tensorflow_classifier_Jan12')
trn_X, tst_X, vec = this_classifier.fit_scaler(StandardScaler(),trn_X, tst_X)

# training
this_classifier.train(trn_X,trn_Y,tst_X,tst_Y, num_epoch=10, num_neroun=128,learning_rate=0.01)
thispred = this_classifier.predict(tst_X)
tst_acc1 = np.mean(np.argmax(tst_Y,axis=1) == thispred)
print tst_acc1

# print out the training results
print classification_report(np.argmax(tst_Y,axis=1), this_classifier.predict(tst_X),target_names=target_names)
print confusion_matrix(np.argmax(tst_Y,axis=1), this_classifier.predict(tst_X))


this_classifier.save_model('./net/tensorflow_classifier_Jan12.json')

this_report =classifaction_report_csv(classification_report(np.argmax(tst_Y,axis=1), this_classifier.predict(tst_X)))
this_confusion_matrix = confusion_matrix(np.argmax(tst_Y,axis=1), this_classifier.predict(tst_X))

this_report.to_csv('./results/classification_report_classifier_Jan12.csv')
np.savetxt('./results/conf_matrix_classifier_Jan12.csv',this_confusion_matrix, delimiter=',')





