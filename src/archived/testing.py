'''
Created on Oct 3, 2016
load the trained network and do testing

@author: rsong_admin
'''
import classifier
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
style.use('ggplot')

def sec_plotting(sec_acc):
    # plot the sectional accuracy
    classesApp = ['Antibacterial','Chelating','Colorants','Fragrances',
                  'Oxidation','Pesticide.','Pharma.','Solvents','Surfactants']
    accApp = sec_acc.values()
    classAcc = {}
    for i in range(len(accApp)):
        newKey = classesApp[i]
        newValue = accApp[i]
        classAcc[newKey] = newValue
    df = pd.Series(classAcc)
    ax = df.plot(kind = 'bar',rot=0)
#     plt.title('Sectional Accuracy of Functional Uses Classification with 78 Selected Descriptors Model on Test Data, Jan. 4th',
#               fontsize=20,fontweight='bold')
    plt.ylabel('Accuracy in Percent',fontsize=18,fontweight='bold')
    plt.xlabel('Functional Uses',fontsize=18,fontweight='bold')
    # rc('font',weight=15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
        tick.label1.fontweights='bold'
    plt.show()


def predicting(network_name, trnDescs, tstDescs):
    network_name = network_name
    descs_trn = trnDescs
    descs_tst = tstDescs
    
    predictor = classifier.predicting(network_name,
                                      descs_tst,
                                      descs_trn,
                                      featureNorm=True)
    
    predicted_classes = predictor.predict(predictor.network, predictor.descsForPred_normed)
    
    return predicted_classes

if __name__ == '__main__':
    trn_data = pd.read_csv('./data/trn_data.csv')
    
    ''' replace the content in this file to test new chemicals'''
    tst_data = pd.read_csv('./data/tst_data.csv')
    
    class_name = trn_data['Class'].unique()
    real_label = tst_data['Label']
    real_class = tst_data['Class']
    
    trn_data = trn_data.drop(['Class'],axis=1)
    tst_data = tst_data.drop(['Class'],axis=1)
    
    trnDescs = trn_data.values[:,:-1]
    tstDescs = tst_data.values[:,:-1]
    
    pred = predicting('./net/9c_1121chem_59d_Oct28.xml', trnDescs, tstDescs)
# 
    sectionalAcc = classifier.predicting.SectionalAcc(real_label, pred)

    sec_plotting(sectionalAcc)
#     print(classification_report(real_label, pred, target_names=class_name))
    conf_table = confusion_matrix(real_label, pred)
    np.savetxt('conf_table.csv',conf_table,delimiter=',')
    
