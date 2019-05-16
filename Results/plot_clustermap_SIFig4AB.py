#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:46:19 2017

@author: prubbens
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats

from sklearn.metrics import pairwise_distances

from functions_rw import calc_D0, calc_D1, calc_D2
from functions_rw import concat_df
from functions_rw import get_gmm_fitted, get_fcfp_gmm_rw
from functions_rw import transform
from statsmodels.stats.multitest import multipletests

from os import listdir

''' Metavariables '''
PATH_DATA = '../Data/Lakes/Muskegon/'
FILENAMES = sorted(listdir(PATH_DATA))
MIN_SAMPLES_LEAF = 5 
N_ITER = 100

N_REP = 3
N_TREES = 200#[10,100,200,400,600,800,1000][2]
TYPE = 'full'#['full','tied','diag','spherical'][0]
N_CELLS_REP_c = 2000 #Cycle II
N_CELLS_REP_i = 2000 #Cycle II
N_MIX = 128
N_BINS = 128
N_RUNS = 1
FEATURES = ['FL1-H','FSC-H','SSC-H']#,'FL2-H','FL3-H','SSC-H','FL4-H'][0:job] #
FILENAMES_norep = [x[:-9] for x in FILENAMES]
VAR_GMM = 'Lake'#['Depth','Lake','Season','Site','Year'][1]

target = pd.read_csv('../Data/Lakes/ByLake_Filtering/1in3/muskegon/muskegon_sampledata_1in3.tsv', sep=' ', index_col=None, header=0, float_precision='high')
otu_rel =pd.read_csv('../Data/Lakes/ByLake_Filtering/1in3/muskegon/muskegon_relative_otu_1in3.tsv', sep=' ', index_col=None, header=0, float_precision='high')
otus = list(otu_rel.columns)

rep1_names = []
rep2_names = []
rep3_names = []
for index in target.index: 
    rep1_names.append(index + '_rep1.csv')
    rep2_names.append(index + '_rep2.csv')
    rep3_names.append(index + '_rep3.csv')
target.loc[target.index,'rep1_names'] = rep1_names
target.loc[target.index,'rep2_names'] = rep2_names
target.loc[target.index,'rep3_names'] = rep3_names

d0_train = calc_D0(otu_rel)   
d1_train = calc_D1(otu_rel)
d2_train = calc_D2(otu_rel)

bc_comp = pd.DataFrame(pairwise_distances(otu_rel, metric='braycurtis'), index=otu_rel.index, columns=otu_rel.index)

fcfp_final_gmm = pd.DataFrame() 
fcfp_final_grid = pd.DataFrame() 
target_train = target
groups = target_train.loc[:,VAR_GMM].unique()
    
for group in groups: 
    idx_gmm = target_train.loc[target_train.loc[:,VAR_GMM] == group].index
    df_train = concat_df(target.loc[idx_gmm,:], ['rep1_names','rep2_names','rep3_names'], N_CELLS_REP_c, PATH_DATA)
    df_train_trans = transform(df_train,FEATURES)
    gmm = get_gmm_fitted(df_train_trans, FEATURES, N_MIX, TYPE, False)
    fcfp_gmm = get_fcfp_gmm_rw(target.loc[:,:], ['rep1_names','rep2_names','rep3_names'], N_REP, N_CELLS_REP_i, N_MIX, FEATURES, gmm, True, PATH_DATA)
    fcfp_final_gmm = pd.concat([fcfp_gmm,fcfp_final_gmm], axis=1, ignore_index=False)
    
corr = np.zeros((N_MIX,128))
p_vals = np.zeros((N_MIX,128))
for i in np.arange(0,N_MIX): 
    for j in np.arange(0,128): 
        corr[i,j], p_vals[i,j] = stats.kendalltau(fcfp_final_gmm.iloc[:,i],otu_rel.iloc[:,j])
corr_vals = multipletests(p_vals.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)
thr = p_vals.reshape(-1)[corr_vals[1] <=0.05].max()
mask = p_vals > thr
corr = pd.DataFrame(corr, index=np.arange(0,N_MIX), columns=otus[0:128])
p_vals = pd.DataFrame(p_vals, index=np.arange(0,N_MIX), columns=otus[0:128])
plt.figure(figsize=(4,4))
sns.clustermap(corr.T, mask=mask.T)
plt.savefig('Figures/SI/Clustermap_MUS_default_NMIX=128.png', dpi=500, bbox_tight=False)

g = sns.distplot(p_vals[p_vals <= thr].count(),kde=False, bins=11)
g.set_xlabel('Number of significant correlations', size=16)
g.set_ylabel('Number of populations', size=16)
plt.savefig('Figures/SI/Hist_clustermap_MUS_default_NMIX=128.png', dpi=500, bbox_tight=True)