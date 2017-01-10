#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年1月8日

@author: runsheng
'''
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.cross_validation import train_test_split

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    print target.shape
    raw_input()
    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

if __name__ == '__main__':
    get_iris_data()
    
        