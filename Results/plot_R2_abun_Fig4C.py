#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 09:54:48 2018

@author: prubbens
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_SOR_a01 = pd.read_csv('OTU/a=0.1_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
results_SOR_a1 = pd.read_csv('OTU/a=1_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
results_SOR_a10 = pd.read_csv('OTU/a=10_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)

results_SOR_a01 = pd.DataFrame(results_SOR_a01.pivot(columns='OTU', values=r'$R^2$').mean())
results_SOR_a01.columns = [r'$R^2$']
results_SOR_a01['Dataset'] = r'$a = 0.1$'
results_SOR_a1 = pd.DataFrame(results_SOR_a1.pivot(columns='OTU', values=r'$R^2$').mean())
results_SOR_a1.columns = [r'$R^2$']
results_SOR_a1['Dataset'] = r'$a = 1$'
results_SOR_a10 = pd.DataFrame(results_SOR_a10.pivot(columns='OTU', values=r'$R^2$').mean())
results_SOR_a10.columns = [r'$R^2$']
results_SOR_a10['Dataset'] = r'$a = 10$'

results_cycle1 = pd.read_csv('asinh/CycleI_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
results_cycle1 = results_cycle1.dropna(inplace=False)
p_cycle1 = pd.DataFrame(results_cycle1.pivot(columns='OTU', values=r'$P-value$').mean())
results_cycle1 = pd.DataFrame(results_cycle1.pivot(columns='OTU', values=r'$R^2$').mean())
results_cycle1.columns = [r'$R^2$']
results_cycle1['Dataset'] = 'Survey I'
results_cycle2 = pd.read_csv('asinh/CycleII_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
p_cycle2 = pd.DataFrame(results_cycle2.pivot(columns='OTU', values=r'$P-value$').mean())
results_cycle2 = pd.DataFrame(results_cycle2.pivot(columns='OTU', values=r'$R^2$').mean())
results_cycle2.columns = [r'$R^2$']
results_cycle2['Dataset'] = 'Cycle II'

results_inl = pd.read_csv('asinh/INL_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
p_inl = pd.DataFrame(results_inl.pivot(columns='OTU', values=r'$P-value$').mean())
results_inl = pd.DataFrame(results_inl.pivot(columns='OTU', values=r'$R^2$').mean())
results_inl.columns = [r'$R^2$']
results_inl['Dataset'] = 'Inland'
results_mich = pd.read_csv('asinh/MICH_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
p_mich = pd.DataFrame(results_mich.pivot(columns='OTU', values=r'$P-value$').mean())
results_mich = pd.DataFrame(results_mich.pivot(columns='OTU', values=r'$R^2$').mean())
results_mich.columns = [r'$R^2$']
results_mich['Dataset'] = 'Michigan'
results_mus = pd.read_csv('asinh/MUS_GMM_SUP_ABUN_FEAT=3_N_CELLS=2500_N_MIX=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
p_mus = pd.DataFrame(results_mus.pivot(columns='OTU', values=r'$P-value$').mean())
results_mus = pd.DataFrame(results_mus.pivot(columns='OTU', values=r'$R^2$').mean())
results_mus.columns = [r'$R^2$']
results_mus['Dataset'] = 'Muskegon'


results = pd.concat([results_SOR_a01,results_SOR_a1,results_SOR_a10,results_cycle1,results_cycle2,results_inl,results_mich,results_mus], axis=0, ignore_index=False)

results.rename_axis('OTU',inplace=True)
results.reset_index(inplace=True)

g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.swarmplot(x='Dataset', y=r'$R^2$', data=results, dodge=True, palette=pal)
g.set_xlabels(fontsize=18)
g.set_ylabels(fontsize=18)
plt.savefig('Figures/R2_abun.png',bbox_inches='tight', dpi=500)
plt.show()