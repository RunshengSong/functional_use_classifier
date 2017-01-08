#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2017年1月8日

@author: runsheng
'''
import pandas as pd
from make_training_data import data_sampler


if __name__ == '__main__':
    df = pd.ExcelFile('../data/1103_new_ten_functional_use_descs.xlsx').parse('Sheet1')
    this_data = data_sampler()
    this_data.sample_data(df)
    print this_data.trn_data['descs']