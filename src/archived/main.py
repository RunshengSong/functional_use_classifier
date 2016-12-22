'''
Created on Apr 29, 2015

@author: rsong_admin
'''
import classifier

import numpy as np
import csv

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

def dataCleaning():
    df = pd.read_csv('./data/5299chem_76d_Jan4.csv',header=0)
    df = df.drop(['No.'],axis = 1)
    subset = df.columns[:-1]
    df = df.dropna(subset=subset,how='all') #this drop rows that has all values equal to NA
    df = df.fillna(df.mean()) #fill NA with col mean
    
    '''collapse dye '''
#     df.loc[(df['Class']=='Dyes'),'Class']='Dye'
#     df.loc[(df['Class']=='Colorants'),'Class']='Dye'
#     df.loc[(df['Class']=='DyeAndPigments'),'Class']='Dye'
#     df.loc[(df['Class']=='FlavorsandFragrances'),'Class']='Fragrances'
    
    '''delete class with less number''' 
#     to_del =df[df['Class'].isin(['Polymers','Enzyme', 'Antioxidant','Foodadditives'])].index.tolist()
#     df = df.drop(to_del)
    
    '''convert categorical to integer label'''
    b, c = np.unique(df['Class'],return_inverse=True)
    df['Label'] = c
   
    '''
    Here get 100 sample from each category to be training data
    for class less than 100 data, left 5 to be test data and put all of the 
    rest to be training data
    '''

    df_trn = pd.DataFrame()
    df_tst = pd.DataFrame()
    for i in range(11):
        
        this_data = df.loc[(df['Label'] == i),:]
        if len(this_data)>=400:
            trn_this = this_data[0:400]
            tst_this = this_data[400:]
        else:
            trn_this = this_data[:-10]
            tst_this = this_data[-10:]    
        df_trn = df_trn.append(trn_this)
        df_tst = df_tst.append(tst_this)
    
    print df_trn['Label'].value_counts()
    print df_tst['Label'].value_counts()
    
    '''check the size of the training and test data'''
#     for i in range(10):
#         this_cate = df_trn.loc[(df['Label'] == i),:]
#         this_size = len(this_cate)
#         this_class = this_cate['Class'].head(1)
#         print this_class, this_size
    
    df_trn = df_trn.drop(['Class'],axis=1)
    df_tst = df_tst.drop(['Class'],axis=1)
    df = df.drop(['Class'],axis=1)
    # Get the raw value of it. 
    trn_data = df_trn.values
    tst_data = df_tst.values
    all_data = df.values

    print 'Raw data cleaned. Shape: '
    print trn_data.shape
    print tst_data.shape
    
    return trn_data,tst_data,all_data

def training(descs,target):
#     trnDescs = np.loadtxt('./data/trndata_input_less.csv',delimiter=',')
#     trnTarget = np.loadtxt('./data/trndata_target_less.csv',delimiter=',')
    
    trainer = classifier.training(descs,target)
    trainer.set_dataset(0.15,featureNorm=True)
    raw_input('Start to train?')
    trainer.train_net(training_times_input=200, 
                      num_neroun=64,
                      learning_rate_input=0.01, 
                      weight_decay=0.001, 
                      momentum_in=0, 
                      verbose_input=True)
    
    #use this network to predict all values
    predictedValue = trainer.predict(trainer.network, trainer.tstdata['input'])
    realValue = trainer.tstdata['class']  
    acc = trainer.calc_accuracy(realValue, predictedValue)
    print acc #over all acc

    # accuracy for each section, return a dictionary
    sectionalAcc = trainer.SectionalAcc(realValue, predictedValue)
    
    '''plot acc for each section '''

    raw_input('Save?')
    trainer.save_network('./net/10c_903chem_76d_May30.xml')
    
def predicting(network_name,trnData,tstDescs):
    network_name = network_name
    descs_tst = tstDescs
    descs_trn = trnData[:,:-1]

    predictor = classifier.predicting(network_name,
                                      descs_tst,
                                      descs_trn,
                                      featureNorm=True)
    
    predicted_classes = predictor.predict(predictor.network, predictor.descsForPred_normed)
    return predicted_classes

def sec_plotting():
    # plot the sectional accuracy
    classesApp = ['Antibacterial','Chelating','Colorants','Fragrances',
                  'Herbcide','Oxidation','Pesticide.','Pharma.','Solvents','Surfactants']
    accApp = sectionalAcc.values()
    classAcc = {}
    for i in range(len(accApp)):
        newKey = classesApp[i]
        newValue = accApp[i]
        classAcc[newKey] = newValue
    df = pd.Series(classAcc)
    ax = df.plot(kind = 'bar',rot=0)
    plt.title('Sectional Accuracy of Functional Uses Classification with 78 Selected Descriptors Model on Test Data, Jan. 4th',
              fontsize=20,fontweight='bold')
    plt.ylabel('Accuracy in Percent',fontsize=18,fontweight='bold')
    plt.xlabel('Functional Uses',fontsize=18,fontweight='bold')
    # rc('font',weight=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
        tick.label1.fontweights='bold'
    plt.show()
    
if __name__ == '__main__': 
    # training
    trn_data_all, tst_data_all, all_data = dataCleaning()
    np.savetxt('./data/training_data.csv',trn_data_all,delimiter=',')
    raw_input()
    # the last colunm is the label 
    all_data_descs = all_data[:,:-1]
    all_data_target = all_data[:,-1]
    
    trn_data_descs = trn_data_all[:,:-1]
    tst_data_descs = tst_data_all[:,:-1]

    trn_data_target = trn_data_all[:,-1]
    tst_data_target = tst_data_all[:,-1] 
    
    training(trn_data_descs, trn_data_target)
    predictedValue = predicting('./net/10c_903chem_76d_May30.xml', trn_data_all, tst_data_descs)
    
    tst_acc = classifier.predicting.calc_accuracy(tst_data_target, predictedValue)
    sectionalAcc = classifier.predicting.SectionalAcc(tst_data_target, predictedValue)
    
    print tst_acc
    print sectionalAcc 
    print tst_data_target
    
    with open('mycsvfile.csv', 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, sectionalAcc.keys())
        w.writeheader()
        w.writerow(sectionalAcc)
    # plotting
    sec_plotting()
    
