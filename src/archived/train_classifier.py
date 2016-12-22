'''
Created on Oct 3, 2016
train the functional use classifier

and test it on the final testing dataset
@author: rsong_admin
'''
import classifier
import pandas as pd
from sklearn import preprocessing

def train(descs, target):
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
    print acc # over all acc

    # accuracy for each section, return a dictionary
    sectionalAcc = trainer.SectionalAcc(realValue, predictedValue)
    
    '''plot acc for each section '''
    raw_input('Save?')
    trainer.save_network('./net/9c_1121chem_59d_Oct28.xml')

def fill_nan(df):
    '''
    fill nan values with the average number of other rows (so it has little effect?)
    '''
    return df.fillna(df.mean())
    
if __name__ == '__main__':
    # load training data
    df = pd.read_csv('./data/trn_data.csv',header = 0)
    df.drop('Class',axis=1,inplace=True)
    target = df['Label']
    df.drop('Label',axis=1,inplace=True)
    
    descs = df.values
    train(descs, target)
    
    
    
    
    