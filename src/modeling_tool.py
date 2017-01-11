'''
Created on Dec 21, 2016

@author: rsong_admin
'''

import numpy as np
import pandas as pd
from make_training_data import data_sampler

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

BATCH_SIZE = 1

class create_functional_use_classifier:
    def __init__(self, sess, scalar):
        '''
        create functional use classifier
        '''
        self.sess = sess
        self.scaler = scalar
    
    def fit_scaler(self, trn_data, tst_data):
        '''
        fit sklearn standard scaler
        '''
        self.scaler.fit(trn_data)

        return self.scaler.transform(trn_data), self.scaler.transform(tst_data)
        
    def _feedforward(self, X, w1, w2):
        '''
        feed forward for one hidden-layers nn
        '''
        h1 = tf.nn.sigmoid(tf.matmul(X, w1))
        y_ = tf.matmul(h1, w2)
        return y_
    
    def _init_weights(self,shape):
        weights = tf.random_normal(shape,stddev = 0.1)
        return tf.Variable(weights)
    
    def train(self, trn_X, trn_Y, tst_X, tst_Y, 
              num_epoch=200, num_neroun = 64, learning_rate = 0.01, 
              verbose=True):
        '''
        train
        '''
        # layer sizes
        x_size = trn_X.shape[1]
        h_size = num_neroun # hidden layer
        y_size = trn_Y.shape[1]
        
        # symbols
        X = tf.placeholder("float",shape=[None, x_size])
        y = tf.placeholder("float",shape=[None, y_size])
        
        # weights
        w1 = self._init_weights((x_size, h_size))
        w2 = self._init_weights((h_size, y_size))
        
        # init feedforward
        y_ = self._feedforward(X, w1, w2)
        pred = tf.argmax(y_, dimension=1)

        
        # init back propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
        updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        # init session
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        for epoch in range(num_epoch):
            for i in range(0, len(trn_X),BATCH_SIZE):
                self.sess.run(updates, feed_dict={X:trn_X[i:i+BATCH_SIZE], y:trn_Y[i:i+BATCH_SIZE]})
            
            trn_acc = np.mean(np.argmax(trn_Y, axis=1) == self.sess.run(pred, feed_dict={X:trn_X, y:trn_Y}))
            tst_acc = np.mean(np.argmax(tst_Y, axis=1) == self.sess.run(pred, feed_dict={X:tst_X, y:tst_Y}))
            print("Epoch = %d, Training Accuracy = %.2f%%, Testing Accuracy = %.2f%%" % (epoch + 1, 100. * trn_acc, 100. * tst_acc))
        
        print classification_report(np.argmax(tst_Y,axis=1), self.sess.run(pred, feed_dict={X:tst_X, y:tst_Y}))
        print confusion_matrix(np.argmax(tst_Y,axis=1), self.sess.run(pred, feed_dict={X:tst_X, y:tst_Y}))
        
    def score(self, network, trn_data ,tst_data):
        pass
        


if __name__ == '__main__':
    
    # load data
    df = pd.read_csv('../data/0109_nine_functional_use_descs.csv',header=0)
    
    this_data = data_sampler()
    this_data.sample_data(df, num_test_left=30)
    trn_X = this_data.trn_data['descs']
    trn_Y = this_data.trn_data['target']
    tst_X = this_data.tst_data['descs']
    tst_Y = this_data.tst_data['target']
    
    from collections import Counter
    print Counter(this_data.trn_data['class'])

    raw_input()
    # init

    this_classifier = create_functional_use_classifier(tf.Session(), StandardScaler())
    trn_X, tst_X = this_classifier.fit_scaler(trn_X, tst_X)
    
    
    this_classifier.train(trn_X,trn_Y,tst_X,tst_Y)
    
    
    
    