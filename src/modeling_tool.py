'''
Created on Dec 21, 2016

@author: rsong_admin
'''

import numpy as np
import pandas as pd
import make_training_data as md

import tensorflow as tf

from sklearn.preprocessing import StandardScaler

class create_functional_use_classifier:
    def __init__(self, trainer, scalar):
        '''
        create functional use classifier
        '''
        self.trainer = trainer
        self.scaler = scalar
    
    def fit_scaler(self, trn_data, tst_data):
        '''
        fit sklearn standard scaler
        '''
        pass
        
    
    def train(self, num_epoch=200, num_neroun = 64, learning_rate = 0.01, weight_decay = 0.001, verbose=True):
        '''
        train
        '''
        # validation set here
        pass
        
    def score(self, network, trn_data ,tst_data):
        pass
        


if __name__ == '__main__':
    pass
    
    
    