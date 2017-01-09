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

def feedforward(X, w1, w2):
    h = tf.nn.sigmoid(tf.matmul(X,w1))
    yhat = tf.matmul(h, w2)
    return yhat

def init_weight(shape):
    weights = tf.random_normal(shape,stddev = 0.1)
    return tf.Variable(weights)

def create_classifier():
    df = pd.ExcelFile('../data/data_tst.xlsx').parse('Sheet1')
    this_data = data_sampler()
    this_data.sample_data(df)
    
    # layer sizes
    x_size = this_data.trn_data['descs'].shape[1]
    h_size = 256 # hidden layer
    y_size = this_data.trn_data['target'].shape[0]
    
    trn_X = this_data.trn_data['descs']
    trn_Y = this_data.trn_data['target']
    tst_X = this_data.tst_data['descs']
    tst_Y = this_data.tst_data['target']
    
    # symbols
    X = tf.placeholder("float",shape=[None, x_size])
    y = tf.placeholder("float",shape=[None, 1])
    
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
    
    for epoch in range(300):
        for i in range(len(trn_X)):
            sess.run(updates, feed_dict={X:trn_X[i:i+1], y:trn_Y[i:i+1]})
        
        trn_acc = np.mean(np.argmax(trn_Y, axis=1) == sess.run(pred, feed_dict={X:trn_X, y:trn_Y}))
        tst_acc = np.mean(np.argmax(tst_Y, axis=1) == sess.run(pred, feed_dict={X:tst_X, y:tst_Y}))
        
        print("Epoch = %d, Training Accuracy = %.2f%%, Testing Accuracy = %.2f%%" % (epoch + 1, 100. * trn_acc, 100. * tst_acc))

if __name__ == '__main__':
    create_classifier()

    
    