'''
Created on Dec 21, 2016

functions to make training data from 
the web crawler

@author: rsong_admin
'''

import numpy as np
import csv
import pandas as pd


class data_sampler:
    def __init__(self):
        '''
        e.g.: this_data.trn_data['descs']
        '''
        self.trn_data = {'descs':[],'target':[],'class':[]}
        self.tst_data = {'descs':[],'target':[],'class':[]}
        self.all_data = {'descs':[],'target':[],'class':[]}
    
    def sample_data(self,df, num_trn_each_class=400, num_test_left = 15):
        '''
        sample data from pandas dataframe
        
        df: data frame object
        num_trn_each_class: how many training data for each class
        '''
        # convert categorical to integer label
        b, c = np.unique(df['Class'],return_inverse=True)
        df['Label'] = c
        num_class = len(b)
        
        # sample for each class, if less than 'num_trn_each_class'
        # left 'num_test_left' then put the rest to the training data
        for i in range(num_class):
            # shuffle
            this_data = df.loc[(df['Label'] == i),:]
            this_data = this_data.reindex(np.random.permutation(this_data.index))
            
            if len(this_data)>=num_trn_each_class:
                trn_this = this_data[0:num_trn_each_class]
                tst_this = this_data[num_trn_each_class:]
            else:
                trn_this = this_data[:-num_test_left]
                tst_this = this_data[-num_test_left:]
            
            # pack to dictionary
            self.trn_data['target'].append(list(trn_this['Label'].values))
            self.trn_data['class'].append(list(trn_this['Class'].values))
            self.trn_data['descs'].append(list(trn_this.drop(['No.','Class','Label'], axis=1)))

            self.tst_data['target'].append(list(tst_this['Label'].values))
            self.tst_data['class'].append(list(tst_this['Class'].values))
            self.tst_data['descs'].append(list(tst_this.drop(['No.','Class','Label'], axis=1)))
            
            self.all_data['target'].append(list(trn_this['Label'].values))
            self.all_data['target'].append(list(tst_this['Label'].values))
            self.all_data['class'].append(list(trn_this['Class'].values))
            self.all_data['class'].append(list(tst_this['Class'].values))
            self.all_data['descs'].append(list(trn_this.drop(['No.','Class','Label'], axis=1)))
            self.all_data['descs'].append(list(tst_this.drop(['No.','Class','Label'], axis=1)))
        
        
if __name__ == '__main__':
    df = pd.ExcelFile('../data/1103_new_ten_functional_use_descs.xlsx').parse('Sheet1')
    
    this_data = data_sampler()
    this_data.sample_data(df)