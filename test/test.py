#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年1月8日

@author: runsheng
'''
import numpy as np
import pandas as pd


if __name__ == '__main__':
    theList = np.empty(0)
    
    a = np.array([0,0,0,0])
    b = np.array([1,1,1,1])
    

    for i in (a,b):
        theList = np.append(theList, [[m]for m in i])
    
    print theList
    
        