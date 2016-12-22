'''
Created on Oct 3, 2016

use all of the collected chemical functional use data and 
create training and testing data

re-run this to create new set of data
@author: rsong_admin
'''
import numpy as np
import csv
import pandas as pd

def dataCleaning(df):
    
    '''convert categorical to integer label'''
    b, c = np.unique(df['Class'],return_inverse=True)
    df['Label'] = c
   
    '''
    Here get 400 sample from each category to be training data
    for class less than 100 data, left 5 to be test data and put all of the 
    rest to be training data
    '''
    df_trn = pd.DataFrame()
    df_tst = pd.DataFrame()
    for i in range(11):
        # shuffle
        this_data = df.loc[(df['Label'] == i),:]
        this_data = this_data.reindex(np.random.permutation(this_data.index))

        if len(this_data)>=150:
            trn_this = this_data[0:150]
            tst_this = this_data[150:]
        else:
            trn_this = this_data[:-50]
            tst_this = this_data[-50:]    
        df_trn = df_trn.append(trn_this)
        df_tst = df_tst.append(tst_this)
    
#     df_trn = df_trn.drop(['Class'],axis=1)
#     df_tst = df_tst.drop(['Class'],axis=1)
#     df = df.drop(['Class'],axis=1)
    # Get the raw value of it. 
    trn_data = df_trn
    tst_data = df_tst
    all_data = df

    print 'Raw data cleaned. Shape: '

    return trn_data,tst_data,all_data


if __name__ == '__main__':
    
    df = pd.ExcelFile('./data/1103_new_ten_functional_use_descs.xlsx').parse('Sheet1')
    
#     df = df.drop(['No.'],axis = 1)
    subset = df.columns[:-1]
    df = df.dropna(subset=subset,how='all') #this drop rows that has all values equal to NA
    df = df.fillna(df.mean()) #fill NA with col mean
    
    trn_data, tst_data, all_data = dataCleaning(df)
    
    trn_data.to_csv('./data/trn_data.csv')
    tst_data.to_csv('./data/tst_data.csv')
    all_data.to_csv('./data/all_data.csv')
    
    
    
    
    
    
