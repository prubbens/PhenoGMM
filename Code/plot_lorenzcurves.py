#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:21:39 2018

@author: prubbens
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as pd
import seaborn as sns

def lorenz_curve(X):
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    fig, ax = plt.subplots(figsize=[6,6])
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz, 
               linestyle='-', color='darkgreen')
    ## line plot of equality
    ax.plot([0,1], [0,1], color='k')

comp_train_a0_1 = pd.read_csv('../Files/Comp_train_a=0.1.csv', index_col=0, header=0)
comp_test_a0_1 = pd.read_csv('../Files/Comp_test_a=0.1.csv', index_col=0, header=0)

comp_train_a1 = pd.read_csv('../Files/Comp_train_a=1.csv', index_col=0, header=0)
comp_test_a1 = pd.read_csv('../Files/Comp_test_a=1.csv', index_col=0, header=0)

comp_train_a10 = pd.read_csv('../Files/Comp_train_a=10.csv', index_col=0, header=0)
comp_test_a10 = pd.read_csv('../Files/Comp_test_a=10.csv', index_col=0, header=0)

my_cmap = ListedColormap(sns.color_palette('deep',3))

labels = [r'$a = 0.1$', r'$a=1$',r'$a=10$']

fig, ax = plt.subplots(figsize=[6,6])
for i in np.arange(0,300): 
    columns = np.nonzero(comp_train_a0_1.iloc[i,:].values)[0]
    X = comp_train_a0_1.iloc[i,columns].sort_values(ascending=False).values
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,linestyle='-',color=my_cmap.colors[0], alpha=0.5)

    columns = np.nonzero(comp_train_a1.iloc[i,:].values)[0]
    X = comp_train_a1.iloc[i,columns].sort_values(ascending=False).values
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,linestyle='-',color=my_cmap.colors[1], alpha=0.5)    
    
    columns = np.nonzero(comp_train_a10.iloc[i,:].values)[0]
    X = comp_train_a10.iloc[i,columns].sort_values(ascending=False).values
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,linestyle='-',color=my_cmap.colors[2], alpha=0.35)    
    
## line plot of equality
ax.plot([0,1], [0,1], color='k')
plt.legend(labels,fontsize=14)
plt.xlabel('Cumulative proportion of species', fontsize=16)
plt.ylabel('Cumulative proportion of abundances', fontsize=16)
plt.savefig('../Figures/Supporting Information/LorenzCurve_train.png', dpi=300, bbox_tight=True)

fig, ax = plt.subplots(figsize=[6,6])
for i in np.arange(0,100): 
    columns = np.nonzero(comp_test_a0_1.iloc[i,:].values)[0]
    X = comp_test_a0_1.iloc[i,columns].sort_values(ascending=False).values
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,linestyle='-',color=my_cmap.colors[0], alpha=0.5)

    columns = np.nonzero(comp_test_a1.iloc[i,:].values)[0]
    X = comp_test_a1.iloc[i,columns].sort_values(ascending=False).values
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,linestyle='-',color=my_cmap.colors[1], alpha=0.5)    
    
    columns = np.nonzero(comp_test_a10.iloc[i,:].values)[0]
    X = comp_test_a10.iloc[i,columns].sort_values(ascending=False).values
    X_lorenz = X.cumsum() / X.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0) 
    X_lorenz[0], X_lorenz[-1]
    ## scatter plot of Lorenz curve
    ax.plot(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,linestyle='-',color=my_cmap.colors[2], alpha=0.35)    
    
## line plot of equality
ax.plot([0,1], [0,1], color='k')
plt.legend(labels,fontsize=14)
plt.xlabel('Cumulative proportion of species', fontsize=16)
plt.ylabel('Cumulative proportion of abundances', fontsize=16)
plt.savefig('../Figures/Supporting Information/LorenzCurve_test.png', dpi=300, bbox_tight=True)