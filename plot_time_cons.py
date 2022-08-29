# coding: utf-8
'''
Created on 2021-10-13 11:23:36
Title: plot the time consuming comparison  
Copyright: Copyright (c) 2021 
School: ECNU
author: Chyi  
version 1.0  
'''

import matplotlib.pyplot as plt
import pickle
import numpy as np
test_modes = ['RMS', 'RMS_alt', 'ConOpt', 'SGA', 'Cen_alt','ACG']
time_records = dict()
titles = ['RMSProp', 'RMSProp-alt', 'ConOpt', 'RMSProp-SGA', 'RMSProp-ACA','SGA-ACG']
for mode in test_modes:
    with open('{}/{}_{}_{}_it_time.pickle'.format("../ACC/mixed_gaussian", mode, 6, 256), 'rb') as f:
        time_records[mode] = pickle.load(f)
barlist = []
for key in test_modes[0:(len(test_modes)-1)]:
    barlist.append((np.diff(np.array(time_records[key]))/1000).mean())
barlist.append((np.diff(np.array(time_records['ACG']))/17600).mean())
plt.bar(x=[0,1.5,3,4.5,6], height=barlist[:5],color = "green" )
plt.bar(x=[3,4.5],height=barlist[2:4],color = "blue")
plt.bar(x=[3, 4.5, 7.5], height=barlist[5],color = "red")
plt.xticks([0,1.5,3,4.5,6,7.5], titles,rotation=-10)
plt.ylabel('sec/1 iteration')
plt.savefig('time_consuming_comparison_Guassians_16.pdf')