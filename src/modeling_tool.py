'''
Created on Dec 21, 2016

@author: rsong_admin
'''
import json
import numpy as np
import pandas as pd
from make_training_data import data_sampler

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

import matplotlib.pyplot as plt 

BATCH_SIZE = 1
RUGULARIZATION = 0.

class create_functional_use_classifier:
    def __init__(self, savefile,D=None, num_neroun=None, K=None):
        '''
        create functional use classifier
        '''

        self.savefile = savefile
        if D and K and num_neroun:
            print 'rebuild'
            self.build(D,num_neroun,K)
    
    def fit_scaler(self, scaler, trn_data, tst_data):
        '''
        fit sklearn standard scaler
        '''
        self.scaler = scaler
        self.scaler.fit(trn_data)
        
        return self.scaler.transform(trn_data), self.scaler.transform(tst_data), self.scaler
        
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
    
    def _error_rate(self,p,t):
        return np.mean(p != t)
    
    def build(self,input_size,num_neroun,output_size, learning_rate=0.01 ):
        '''
        build the structure of the network
        only one hidden layer for now
        '''        
        # symbols
        self.X = tf.placeholder("float",shape=[None, input_size])
        self.y = tf.placeholder("float",shape=[None, output_size])
        
        # weights
        self.w1 = self._init_weights((input_size, num_neroun))
        self.w2 = self._init_weights((num_neroun, output_size))
        
        # init feedforward
        y_ = self._feedforward(self.X, self.w1, self.w2)
        self.pred = tf.argmax(y_, dimension=1)

        # init back propagation
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, self.y))
        
        # add regularization term
        cost += RUGULARIZATION * (tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2))
        
        updates = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        
        # saver
        self.saver = tf.train.Saver({'w1': self.w1, 'w2':self.w2})
        return updates
    
    def train(self, trn_X, trn_Y, tst_X, tst_Y, 
              num_epoch=200, num_neroun = 64, learning_rate = 0.01, 
              verbose=True):
        '''
        train
        '''
        # data dimension
        N, D = trn_X.shape
        K = trn_Y.shape[1]
        
        # layer sizes
        x_size = trn_X.shape[1]
        h_size = num_neroun # hidden layer
        y_size = trn_Y.shape[1]
        
        # init cost/update function
        updates = self.build(input_size=x_size,num_neroun=h_size,output_size = y_size, learning_rate=0.01)
        # init session
        init = tf.global_variables_initializer()
        costs = []
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epoch):
                for i in range(0, len(trn_X),BATCH_SIZE):
                    sess.run(updates, feed_dict={self.X:trn_X[i:i+BATCH_SIZE], self.y:trn_Y[i:i+BATCH_SIZE]})
                
                trn_acc = np.mean(np.argmax(trn_Y, axis=1) == sess.run(self.pred, feed_dict={self.X:trn_X, self.y:trn_Y}))
                tst_acc = np.mean(np.argmax(tst_Y, axis=1) == sess.run(self.pred, feed_dict={self.X:tst_X, self.y:tst_Y}))
                costs.append(tst_acc)
                print("Epoch = %d, Training Accuracy = %.2f%%, Testing Accuracy = %.2f%%" % (epoch + 1, 100. * trn_acc, 100. * tst_acc))
        
            self.saver.save(sess,self.savefile)
      
        self.D = D
        self.K = K
        self.num_neroun = num_neroun
        
        plt.plot(costs)
        plt.show()
        
    def predict(self, tst_X):
        ''' classify input tst_data (X) '''
#         y_ = self._feedforward(X, w1, w2)
        with tf.Session() as sess:
            # restore the model
            self.saver.restore(sess, self.savefile)
            this_pred = sess.run(self.pred, feed_dict={self.X:tst_X})
        return this_pred
    
    def score(self,X,Y):  
        return 1 - self._error_rate(self.predict(X), np.argmax(Y,axis=1))
        
    def save_model(self, model_name):
        ''' save to disk '''
        j = {
          'D': self.D,
          'num_neroun': self.num_neroun,
          'K': self.K,
          'model': self.savefile,
        }
        with open(model_name, 'w') as f:
            json.dump(j, f)
        # save scaler
        if self.scaler:
            joblib.dump(self.scaler, self.savefile+'_scaler.pkl',compress=True)
            
        print("model saved")
            
    @staticmethod    
    def load_model(model_name):
        ''' load sess from file '''
        with open(model_name) as f:
            j = json.load(f)
        return create_functional_use_classifier(j['model'], j['D'],j['num_neroun'], j['K'])
    
class ClassifyChemical:
    def __init__(self, input_model_json_path):
        # load
        self.this_classifier = self._load_model(input_model_json_path)
        
    def _load_model(self,model_name_json):
        with open(model_name_json) as f:
            j = json.load(f)
        return create_functional_use_classifier(j['model'], j['D'],j['num_neroun'], j['K'])
    
    def fit_data(self,scaler_path, tst_X):
        ''' fit the test data using the input scaler '''
        scaler = joblib.load(scaler_path)
        return scaler.transform(tst_X)
    
    def predict(self,tst_X):
        with tf.Session() as sess:
            # restore the model
            self.this_classifier.saver.restore(sess, self.this_classifier.savefile)
            this_pred = sess.run(self.this_classifier.pred, feed_dict={self.this_classifier.X:tst_X})
        return this_pred
        
        

if __name__ == '__main__':
    
    # unit test here
    
    # load data
    df = pd.read_csv('../data/0109_nine_functional_use_descs.csv',header=0)
    scaler_path = '../net/tensorflow_classifier_Jan12_scaler.pkl'
    
    this_data = data_sampler()
    this_data.sample_data(df, num_test_left=50)

    # only load test data and add bias
    tst_X_raw = this_data.tst_data['descs']
    N_tst, M_tst = tst_X_raw.shape
    tst_X = np.ones((N_tst, M_tst+1))
    tst_X[:, 1:] = tst_X_raw
    tst_Y = this_data.tst_data['target']
    target_names = np.unique(this_data.trn_data['class'])
    
    # run test
    thisTest = ClassifyChemical('../net/tensorflow_classifier_Jan12.json')
    tst_X = thisTest.fit_data(scaler_path, tst_X)
    print tst_X
    raw_input()
    
    pred = thisTest.predict(tst_X)
    acc = np.mean(np.argmax(tst_Y) == pred)
    
    print acc
    
    
    
    
    
    
        