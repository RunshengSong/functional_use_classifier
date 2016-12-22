'''
Created on Dec 21, 2016

@author: rsong_admin
'''

import numpy as np
import pandas as pd
import classifier
import make_training_data as md

class create_functional_use_classifier:
    def __init__(self, trainer):
        '''
        create functional use classifier
        '''
        self.trainer = trainer
        
    def train(self, num_epoch=200, num_neroun = 64, learning_rate = 0.01, weight_decay = 0.001, verbose=True):
        '''
        train
        '''
        # validation set here
        self.trainer.set_dataset(0.15,featureNorm=True)
        
        self.trainer.train_net(training_times_input = num_epoch, 
                               num_neroun = num_neroun,
                               learning_rate_input = learning_rate, 
                               weight_decay=weight_decay, 
                               momentum_in=0, 
                               verbose_input=verbose)
        
    def score(self, network, trn_data ,tst_data):
        pass
        


if __name__ == '__main__':
    df = pd.ExcelFile('../data/1103_new_ten_functional_use_descs.xlsx').parse('Sheet1')
    
    this_data = md.data_sampler()
    this_data.sample_data(df, num_trn_each_class=400, num_test_left=15)
    
    print this_data.trn_data['descs']
    print this_data.tst_data['target']
    raw_input()
    this_base_trainer = classifier.training(this_data.trn_data['descs'],this_data.trn_data['target'])
    
    this_classifier = create_functional_use_classifier(trainer=this_base_trainer)
    
    this_classifier.train()
    
    
    