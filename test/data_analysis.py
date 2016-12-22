'''
Created on Oct 3, 2016

data analysis 
there are many functions here to clean up or 
analysis the data
see the descriptions for each function

@author: rsong_admin
'''
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
style.use('ggplot')


def duplicate_pesticdes():
    '''
    clean up the duplicated pesticdes
    '''
    df = pd.ExcelFile('./data/pesticides_CAS.xlsx')
    df = df.parse('Sheet1').set_index('CAS')
    this_dict = df.to_dict(orient='index') # convert the dataframe to dictionary
    
    results_dict = {}
    for eachKey in this_dict:
        if eachKey not in results_dict.keys():
            # new pesticide
            thisValues = this_dict[eachKey]
            results_dict[eachKey] = thisValues
    
    results_df = pd.DataFrame.from_dict(results_dict,orient='index')
    results_df.to_excel('./results/non_duplicated_pesticides.xlsx')

def fill_nan(df):
    '''
    fill nan values with the average number of other rows (so it has little effect?)
    '''
    return df.fillna(df.mean())

def test_overlap():
    '''
    test overlap between categoires.
    '''
  
def class_hist():
    '''
    histogram for the class of functional use
    '''
    import matplotlib.pylab as plt
    from matplotlib import style
    from sklearn import preprocessing

    style.use('ggplot')
    
    df = pd.read_csv('./data/trn_data.csv')
    
    class_name = df['Label'].values

    plt.hist(class_name)
    plt.show()
    
def find_unique_CAS():
    '''
    find unique CAS number of the classifier training data
    9 categories, 6226 chemicals
    '''
    df = pd.ExcelFile('./data/functional_uses_all_data_1031.xlsx',header=0).parse('Sheet1')
    res_dict = {}
    for eachRow in zip(df['CAS'],df['Class']):
        thisCAS = eachRow[0]
        thisClass = eachRow[1]
        res_dict.setdefault(thisCAS,[]).append(thisClass) 

    import csv
    with open('check_duplicates.csv','wb') as myfile:
        thisWriter = csv.writer(myfile)
        for eachKey in res_dict:
            thisWriter.writerow([eachKey]+res_dict[eachKey])
        

if __name__ == '__main__':
    find_unique_CAS()
    
    
