#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年1月8日

@author: runsheng
'''
import numpy as np
import pandas as pd


if __name__ == '__main__':
    thisList = np.empty((0,3))
    
    a = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
    
    b = np.array([[0,5,3],
             [4,5,6]
             ])
    
    for i in (a,b):
        thisList = np.vstack((thisList, i))
    print thisList
        