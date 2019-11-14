#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 15:19:12 2018

@author: prubbens
"""
#import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from os import listdir
from scipy import stats
from sklearn.metrics import pairwise_distances

from functions import calc_D0, calc_D1, calc_D2
from functions import get_insilico_df
from functions import get_gmm_fitted, get_fcfp_gmm
from functions import transform
from statsmodels.stats.multitest import multipletests

        
''' Metavariables '''
np.random.seed(np.int(2703))
PATH_DATA = '../Data/Individual_Populations/'
FILENAMES = listdir(PATH_DATA)
MIN_SAMPLES_LEAF = 5 
N_ITER = 100
N_REP = 2
S = 20

A = 1
FEATURES = ['FL1-H','SSC-H','FSC-H']#'FL2-H','FL3-H','FL4-H'][0:3]
N_CELLS_REP_c = 2500
N_CELLS_REP_i = 2500
N_MIX = 20
N_SAMPLES = 300
N_TREES = 200
TYPE = 'full'#['full','tied','diag','spherical'][0]

comp_train = pd.read_csv('../Files/Comp_train_a='+str(A)+'.csv', index_col=0, header=0) 
comp_test = pd.read_csv('../Files/Comp_test_a='+str(A)+'.csv', index_col=0, header=0) 
metadata = pd.read_excel('../Data/metadata_insilico.xlsx', index_col=0, header=0)

d0_train = calc_D0(comp_train)   
d1_train = calc_D1(comp_train)
d2_train = calc_D2(comp_train)
d0_test = calc_D0(comp_test)   
d1_test = calc_D1(comp_test)
d2_test = calc_D2(comp_test)
bc_train_comp = pd.DataFrame(pairwise_distances(comp_train, metric='braycurtis'), index=comp_train.index, columns=comp_train.index)
bc_test_comp = pd.DataFrame(pairwise_distances(comp_test, metric='braycurtis'), index=comp_test.index, columns=comp_test.index)

df_train = get_insilico_df(PATH_DATA,FILENAMES,comp_train,N_CELLS_REP_c,N_REP)
df_test = get_insilico_df(PATH_DATA,FILENAMES,comp_test,N_CELLS_REP_c,N_REP)
df_train_trans = transform(df_train,FEATURES)
df_test_trans = transform(df_test,FEATURES)

gmm = get_gmm_fitted(df_train_trans, FEATURES, N_MIX, TYPE, False)
fcfp_train_gmm = get_fcfp_gmm(PATH_DATA,FILENAMES,comp_train,FEATURES,gmm,N_CELLS_REP_i,N_REP,N_MIX,True)
fcfp_test_gmm = get_fcfp_gmm(PATH_DATA,FILENAMES,comp_test,FEATURES,gmm,N_CELLS_REP_i,N_REP,N_MIX,True) 
    
corr = np.zeros((N_MIX,20))
p_vals = np.zeros((N_MIX,20))
for i in np.arange(0,N_MIX): 
    for j in np.arange(0,20): 
        corr[i,j], p_vals[i,j] = stats.kendalltau(fcfp_test_gmm.iloc[:,i],comp_test.iloc[:,j])
corr_vals = multipletests(p_vals.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)
thr = p_vals.reshape(-1)[corr_vals[1] <=0.05].max()
mask = p_vals > thr
corr = pd.DataFrame(corr, index=np.arange(0,N_MIX), columns=metadata.Species)
p_vals = pd.DataFrame(p_vals, index=np.arange(0,N_MIX), columns=metadata.Species)
plt.figure(figsize=(4,4))
sns.clustermap(corr.T, mask=mask.T)
plt.savefig('Figures/Clustermap_a=1_default_NMIX=20.png', dpi=500, bbox_tight=False)

g = sns.distplot(p_vals[p_vals <= thr].count(),kde=False, bins=11)
g.set_xlabel('Number of significant correlations', size=16)
g.set_ylabel('Number of populations', size=16)
plt.savefig('Figures/Hist_clustermap_a=1_default_NMIX=20.png', dpi=500, bbox_tight=True)