#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:36:05 2018

@author: prubbens
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_a0_1_gmm_uns = pd.read_csv('In Silico/Test_GMM_UNS_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a0_1_gmm_uns['Dataset'] = r'$a = 0.1$'
df_a0_1_gmm_uns['Method'] = 'PhenoGMM'
df_a1_gmm_uns = pd.read_csv('In Silico/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_gmm_uns['Dataset'] = r'$a = 1$'
df_a1_gmm_uns['Method'] = 'PhenoGMM'
df_a10_gmm_uns = pd.read_csv('In Silico/Test_GMM_UNS_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_gmm_uns['Dataset'] = r'$a = 10$'
df_a10_gmm_uns['Method'] = 'PhenoGMM'

df_a0_1_grid_uns = pd.read_csv('In Silico/Test_GRID_UNS_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a0_1_grid_uns['Dataset'] = r'$a = 0.1$'
df_a0_1_grid_uns['Method'] = 'PhenoGrid'
df_a1_grid_uns = pd.read_csv('In Silico/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_grid_uns['Dataset'] = r'$a = 1$'
df_a1_grid_uns['Method'] = 'PhenoGrid'
df_a10_grid_uns = pd.read_csv('In Silico/Test_GRID_UNS_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_grid_uns['Dataset'] = r'$a = 10$'
df_a10_grid_uns['Method'] = 'PhenoGrid'

df_c1_gmm_uns = pd.read_csv('Cycles/CycleI_GMM_UNS_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c1_gmm_uns['Dataset'] = 'Survey I'
df_c1_gmm_uns['Method'] = 'PhenoGMM'
df_c2_gmm_uns = pd.read_csv('Cycles/CycleII_GMM_UNS_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c2_gmm_uns['Dataset'] = 'Survey II'
df_c2_gmm_uns['Method'] = 'PhenoGMM'


df_c1_grid_uns = pd.read_csv('Cycles/CycleI_GRID_UNS_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c1_grid_uns['Dataset'] = 'Survey I'
df_c1_grid_uns['Method'] = 'PhenoGrid'
df_c2_grid_uns = pd.read_csv('Cycles/CycleI_GRID_UNS_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c2_grid_uns['Dataset'] = 'Survey II'
df_c2_grid_uns['Method'] = 'PhenoGrid'

df_inl_gmm_uns = pd.read_csv('Lakes/INL_GMM_UNS_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_inl_gmm_uns['Dataset'] = 'Inland'
df_inl_gmm_uns['Method'] = 'PhenoGMM'
df_mich_gmm_uns = pd.read_csv('Lakes/MICH_GMM_UNS_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mich_gmm_uns['Dataset'] = 'Michigan'
df_mich_gmm_uns['Method'] = 'PhenoGMM'
df_mus_gmm_uns = pd.read_csv('Lakes/MUS_GMM_UNS_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mus_gmm_uns['Dataset'] = 'Muskegon'
df_mus_gmm_uns['Method'] = 'PhenoGMM'

df_inl_grid_uns = pd.read_csv('Lakes/INL_GRID_UNS_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_inl_grid_uns['Method'] = 'PhenoGrid'
df_inl_grid_uns['Dataset'] = 'Inland'
df_mich_grid_uns = pd.read_csv('Lakes/MICH_GRID_UNS_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mich_grid_uns['Method'] = 'PhenoGrid'
df_mich_grid_uns['Dataset'] = 'Michigan'
df_mus_grid_uns = pd.read_csv('Lakes/MUS_GRID_UNS_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mus_grid_uns['Method'] = 'PhenoGrid'
df_mus_grid_uns['Dataset'] = 'Muskegon'


FEAT_UNS = [r'$\rho(\beta)$','Method','Dataset']
df_uns = pd.concat([df_a0_1_gmm_uns.loc[:,FEAT_UNS],df_a0_1_grid_uns.loc[:,FEAT_UNS],df_a1_gmm_uns.loc[:,FEAT_UNS],df_a1_grid_uns.loc[:,FEAT_UNS],df_a10_gmm_uns.loc[:,FEAT_UNS],df_a10_grid_uns.loc[:,FEAT_UNS],df_c1_gmm_uns.loc[:,FEAT_UNS],df_c1_grid_uns.loc[:,FEAT_UNS],df_c2_gmm_uns.loc[:,FEAT_UNS],df_c2_grid_uns.loc[:,FEAT_UNS],df_inl_gmm_uns.loc[:,FEAT_UNS],df_inl_grid_uns.loc[:,FEAT_UNS],df_mich_gmm_uns.loc[:,FEAT_UNS],df_mich_grid_uns.loc[:,FEAT_UNS],df_mus_gmm_uns.loc[:,FEAT_UNS],df_mus_grid_uns.loc[:,FEAT_UNS]])
df_uns = pd.melt(df_uns, id_vars = ['Method','Dataset'], var_name = 'Diversity index', value_name = r'$\rho$')

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Dataset',y=r'$\rho$',data=df_uns, legend=False, hue = 'Method', kind='box', height=4, aspect=2, sharey=True, palette=pal,linewidth=2.5)
#g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Dataset',y=r'$\rho$',data=df_uns, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
#plt.title('Survey I', size=18)
#g.set_axis_labels('Classifier','AUC')
g.set_xlabels('Dataset',fontsize=16)
g.set_ylabels(r'$\rho_{mantel}$',fontsize=16)
g.set_xticklabels([r'$a = 0.1$',r'$a = 1$',r'$a = 10$','Survey I', 'Survey II', 'Inland', 'Michigan','Muskegon'],fontsize=15, rotation=30)
g.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=15)
g.set(ylim=(0.0, 1.0))
for ax in g.axes.flat:
    plt.setp(ax.get_legend().get_texts()[0:2], fontsize=0) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=0)
handles, labels = h.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=0)
#handles, labels = g.get_legend_handles_labels()
plt.tight_layout()
#plt.hlines(y = 0.228, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.savefig('Figures/bDIV.png',dpi=500,bbox_tight=True)