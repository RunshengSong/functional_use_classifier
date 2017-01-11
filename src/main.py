#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年1月8日

@author: runsheng
'''
import pandas as pd
import numpy as np
from make_training_data import data_sampler
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import csv
from collections import Counter
import matplotlib.pyplot as plt


def feedforward(X, w1, w2):
    h = tf.nn.sigmoid(tf.matmul(X,w1))
    yhat = tf.matmul(h, w2)
    return yhat

def init_weight(shape):
    weights = tf.random_normal(shape,stddev = 0.1)
    return tf.Variable(weights)

def fit_data(trn,tst,vec):
    vec.fit(trn)
    trn = vec.transform(trn)
    tst = vec.transform(tst)
    return trn, tst

def plot_class(classes):
    letter_counter = Counter(classes)
    df_letter = pd.DataFrame.from_dict(letter_counter, orient='index')
    df_letter.plot(kind='bar')
    plt.show()

def create_classifier():
#     df = pd.ExcelFile('../data/data_tst.xlsx').parse('Sheet1')
#     df = pd.ExcelFile('../data/1103_new_ten_functional_use_descs.xlsx').parse('Sheet1')
    
    df = pd.read_csv('../data/0109_nine_functional_use_descs.csv',header=0)
    this_data = data_sampler()
    this_data.sample_data(df)

#     trn_data_df = pd.read_csv('../data/trn_data.csv',header=0)
#     tst_data_df = pd.read_csv('../data/tst_data.csv',header=0)
#     
#     trn_Y = trn_data_df['Label'].values
#     trn_Y = np.eye(len(np.unique(trn_Y)))[trn_Y] # to one-hot encoding
#     trn_X = trn_data_df.drop(['Class','Label','No.'], axis=1).values
#     
#     tst_Y = tst_data_df['Label'].values
#     tst_Y = np.eye(len(np.unique(tst_Y)))[tst_Y] # to one-hot encoding
#     tst_X = tst_data_df.drop(['Class','Label','No.'], axis=1).values

    # data
    trn_X = this_data.trn_data['descs']
    trn_Y = this_data.trn_data['target']
 
    tst_X = this_data.tst_data['descs']
    tst_Y = this_data.tst_data['target']

    
    # layer sizes
    x_size = trn_X.shape[1]
    h_size = 32 # hidden layer
    y_size = trn_Y.shape[1]
    
    # transform data
    vec = StandardScaler()
    trn_X, tst_X = fit_data(trn_X, tst_X, vec)

    # symbols
    X = tf.placeholder("float",shape=[None, x_size])
    y = tf.placeholder("float",shape=[None, 9])
    
    # weights
    w1 = init_weight((x_size,h_size))
    w2 = init_weight((h_size,y_size))
    
    # feed forward
    yhat = feedforward(X, w1, w2)
    pred = tf.argmax(yhat,dimension=1)

    # backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat,y))
    updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for epoch in range(200):
        for i in range(len(trn_X)):
            sess.run(updates, feed_dict={X:trn_X[i:i+1], y:trn_Y[i:i+1]})
        
        trn_acc = np.mean(np.argmax(trn_Y, axis=1) == sess.run(pred, feed_dict={X:trn_X, y:trn_Y}))
        tst_acc = np.mean(np.argmax(tst_Y, axis=1) == sess.run(pred, feed_dict={X:tst_X, y:tst_Y}))

#         print np.argmax(trn_Y, axis=1)
#         print sess.run(pred, feed_dict={X:trn_X, y:trn_Y})
#         print yhat
#         raw_input()
        print("Epoch = %d, Training Accuracy = %.2f%%, Testing Accuracy = %.2f%%" % (epoch + 1, 100. * trn_acc, 100. * tst_acc))
    
    target_name = np.unique(df['Class'])
    final_pred = sess.run(pred, feed_dict={X:tst_X, y:tst_Y})
    
    this_report = classification_report(np.argmax(tst_Y, axis=1), final_pred, target_names = target_name)
    this_conf = confusion_matrix(np.argmax(tst_Y,axis=1), final_pred)
    
    print this_report
    print this_conf
    
if __name__ == '__main__':
    create_classifier()

    
    