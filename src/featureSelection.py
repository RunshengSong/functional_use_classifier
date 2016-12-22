'''
Created on Aug 18, 2015
A group of functions that serve for feature selection using 'Filter method'
First, it delete those variables that has lower variance than threshold,
Second, it calcuate 'Pair-wise' correlation' between variables. If less than threshold,
delete the second variable.

 
@author: sourunsheng
'''
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pylab as plt
from matplotlib import style
style.use('ggplot')


def calCorrMat(df, varThreshold=10):
    '''
    delete variable that has variance lower than threshold. 10 here
    '''
    sel = VarianceThreshold(varThreshold)
    data = sel.fit_transform(df.values)
    aMask = sel.get_support(True)
    newDf = df.iloc[:,aMask]

    corrMat = newDf.corr(method='pearson')
    
    return corrMat, newDf
#     corrMat.to_csv('./data/corrNew.csv')
#     newDf.to_csv('./data/reducedDescs.csv')  

def calLowCorr(corrMat, corrThreshold=0.6):
    '''
    calculate 'Pair-Wise' variable correlation matrix.
    Delete the second one if it less than 0.6 here. 
    '''
    df = corrMat
    totalRow, totalCol = df.shape    
    tracker = np.zeros([totalCol])
    for num_row in range(totalRow):
#         print num_row
        if tracker[num_row] == 1:
            continue
        for num_col in range(num_row+1,totalCol):
            if tracker[num_col] == 1:
                continue
            if df.iloc[num_row,num_col] >= corrThreshold:
                tracker[num_col] = 1      
#     np.savetxt('./data/corrResultsNew.csv',tracker,delimiter=',')
    return tracker      

def newMat(tracker, newDf):
    '''
    Return selected variables.
    '''
#     theMask = np.loadtxt('./data/corrResultsNew.csv',delimiter=',')
    theMask = tracker
    theMask = theMask==0
#     oldData = pd.read_csv('./data/reducedDescs.csv',header=0,index_col=0)
    oldData = newDf
    newData = oldData.iloc[:,theMask]
#     newData.to_csv('./data/reducedDescs2.csv')
#     np.savetxt('./data/theMask.csv',theMask,delimiter=',')
    return newData

def do_featureselection(df , VarianceThreshold,corrThreshold):
    corrMat, newDf = calCorrMat(df, varThreshold = VarianceThreshold)
    tracker = calLowCorr(corrMat, corrThreshold=corrThreshold)
    thisDescs = newMat(tracker, newDf)
    return thisDescs

def fill_nan(df):
    '''
    fill nan values with the average number of other rows (so it has little effect?)
    '''
    return df.fillna(df.mean())

if __name__ == '__main__':
    print 'loading...'
    df = pd.ExcelFile('./data/1027_descriptors.xlsx')
    df = df.parse('test_out')
    print 'Done'
    df = df.drop('Class',1)
    df = df.fillna(df.mean())

    print 'feature selection...'
    this_descs = do_featureselection(df, 15, 0.6)
    this_descs.to_csv('1028_reduced_descs.csv')



    

