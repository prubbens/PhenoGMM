#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:54:48 2018

@author: prubbens
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from statsmodels.stats.multitest import multipletests

results_SOR_a01 = pd.read_csv('OTU/a=0.1_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
results_SOR_a1 = pd.read_csv('OTU/a=1_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
results_SOR_a10 = pd.read_csv('OTU/a=10_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)

results_SOR_a01['Dataset'] = r'$a = 0.1$'
results_SOR_a01.loc[:,r'Q-value'] = multipletests(results_SOR_a01.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results_SOR_a1['Dataset'] = r'$a = 1$'
results_SOR_a1.loc[:,r'Q-value'] = multipletests(results_SOR_a1.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results_SOR_a10['Dataset'] = r'$a = 10$'
results_SOR_a10.loc[:,r'Q-value'] = multipletests(results_SOR_a10.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results_cycle1 = pd.read_csv('OTU/CycleI_GMM_SUP_ABUN_FEAT=3_N_CELLS=2000_N_MIX=128_N_TREES=200_TYPE=full_OTU_OOB_asinh.csv', index_col=0, header=0)
results_cycle1 = results_cycle1.dropna(inplace=False)
results_cycle1['Dataset'] = 'Survey I'
results_cycle1.loc[:,r'Q-value'] = multipletests(results_cycle1.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results_cycle2 = pd.read_csv('OTU/CycleII_GMM_SUP_ABUN_FEAT=3_N_CELLS=2000_N_MIX=128_N_TREES=200_TYPE=full_OTU_OOB_asinh.csv', index_col=0, header=0)
results_cycle2['Dataset'] = 'Cycle II'
results_cycle2.loc[:,r'Q-value'] = multipletests(results_cycle2.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results_inl = pd.read_csv('OTU/INL_GMM_SUP_ABUN_FEAT=3_N_CELLS=2000_N_MIX=128_N_TREES=200_TYPE=full_OTU_OOB_asinh.csv', index_col=0, header=0)
#p_inl = pd.DataFrame(results_inl.pivot(columns='OTU', values=r'$P-value$').mean())
#results_inl = pd.DataFrame(results_inl.pivot(columns='OTU', values=r'$R^2$').mean())
#results_inl.columns = [r'$R^2$']
results_inl['Dataset'] = 'Inland'
results_inl.loc[:,r'Q-value'] = multipletests(results_inl.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results_mich = pd.read_csv('OTU/MICH_GMM_SUP_ABUN_FEAT=3_N_CELLS=2000_N_MIX=128_N_TREES=200_TYPE=full_OTU_OOB_asinh.csv', index_col=0, header=0)
results_mich['Dataset'] = 'Michigan'
results_mich.loc[:,r'Q-value'] = multipletests(results_mich.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results_mus = pd.read_csv('OTU/MUS_GMM_SUP_ABUN_FEAT=3_N_CELLS=2000_N_MIX=128_N_TREES=200_TYPE=full_OTU_OOB_asinh.csv', index_col=0, header=0)
results_mus['Dataset'] = 'Muskegon'
results_mus.loc[:,r'Q-value'] = multipletests(results_mus.loc[:,r'$P-value$'].values.reshape(-1),alpha=0.05,method='fdr_bh', is_sorted=False, returnsorted=False)[1]

results = pd.concat([results_SOR_a01,results_SOR_a1,results_SOR_a10,results_cycle1,results_cycle2,results_inl,results_mich,results_mus], axis=0, ignore_index=False)

#for df in [results_SOR_a01,results_SOR_a1,results_SOR_a10,results_cycle1,results_cycle2,results_inl,results_mich,results_mus]: 
#    df = df.loc[df.loc[:,r'Q-value'] < 0.05]
#    df = df.loc[df.loc[:,r'$\kappa$'] > 0.]
#    print(df.shape[0])
#results.rename_axis('OTU',inplace=True)
#results.reset_index(inplace=True)

g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
#h = sns.swarmplot(x='Dataset', y=r'$R^2$', data=results, dodge=True, palette=pal, alpha=0.3)
g.set_xlabels(fontsize=18)
g.set_ylabels(fontsize=18)
plt.savefig('Figures/R2_abun_OOB.png',bbox_inches='tight', dpi=500)
plt.show()