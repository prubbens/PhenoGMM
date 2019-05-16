#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:47:32 2018

@author: prubbens
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("ticks")

def patch_violinplot():
    """Patch seaborn's violinplot in current axis to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.1, 0.1, 0.1))

df_gmm_uns1 = pd.read_csv('Time/GMM/UNS/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=4_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_uns1[r'$K$'] = 4
df_gmm_uns1['Method'] = 'GMM'
df_gmm_uns2 = pd.read_csv('Time/GMM/UNS/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=8_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_uns2[r'$K$'] = 8
df_gmm_uns2['Method'] = 'GMM'
df_gmm_uns3 = pd.read_csv('Time/GMM/UNS/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=16_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_uns3[r'$K$'] = 16
df_gmm_uns3['Method'] = 'GMM'
df_gmm_uns4 = pd.read_csv('Time/GMM/UNS/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=32_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_uns4[r'$K$'] = 32
df_gmm_uns4['Method'] = 'GMM'
df_gmm_uns5 = pd.read_csv('Time/GMM/UNS/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=64_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_uns5[r'$K$'] = 64
df_gmm_uns5['Method'] = 'GMM'
df_gmm_uns6 = pd.read_csv('Time/GMM/UNS/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_uns6[r'$K$'] = 128
df_gmm_uns6['Method'] = 'GMM'
df_gmm_uns7 = pd.read_csv('Time/GMM/UNS/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=256_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_uns7[r'$K$'] = 256
df_gmm_uns7['Method'] = 'GMM'
df_gmm_uns = pd.concat([df_gmm_uns1,df_gmm_uns2,df_gmm_uns3,df_gmm_uns4,df_gmm_uns5,df_gmm_uns6,df_gmm_uns7], axis=0)
df_gmm_uns.rename(index=str, columns={"t(gmm)": "t(s)"}, inplace=True)

df_grid_uns1 = pd.read_csv('Time/Grid/UNS/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=4_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_uns1[r'$K$'] = df_gmm_uns1.loc[:,'n_feat']
df_grid_uns1['Method'] = 'Grid'
df_grid_uns2 = pd.read_csv('Time/Grid/UNS/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=8_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_uns2[r'$K$'] = df_gmm_uns2.loc[:,'n_feat']
df_grid_uns2['Method'] = 'Grid'
df_grid_uns3 = pd.read_csv('Time/Grid/UNS/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=16_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_uns3[r'$K$'] = df_gmm_uns3.loc[:,'n_feat']
df_grid_uns3['Method'] = 'Grid'
df_grid_uns4 = pd.read_csv('Time/Grid/UNS/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=32_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_uns4[r'$K$'] = df_gmm_uns4.loc[:,'n_feat']
df_grid_uns4['Method'] = 'Grid'
df_grid_uns5 = pd.read_csv('Time/Grid/UNS/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=64_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_uns5[r'$K$'] = df_gmm_uns5.loc[:,'n_feat']
df_grid_uns5['Method'] = 'Grid'
df_grid_uns6 = pd.read_csv('Time/Grid/UNS/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_uns6[r'$K$'] = df_gmm_uns6.loc[:,'n_feat']
df_grid_uns6['Method'] = 'Grid'
df_grid_uns7 = pd.read_csv('Time/Grid/UNS/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=256_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_uns7[r'$K$'] = df_gmm_uns7.loc[:,'n_feat']
df_grid_uns7['Method'] = 'Grid'
df_grid_uns = pd.concat([df_grid_uns1,df_grid_uns2,df_grid_uns3,df_grid_uns4,df_grid_uns5,df_grid_uns6,df_grid_uns7], axis=0)
df_grid_uns.rename(index=str, columns={"t(grid)": "t(s)"}, inplace=True)

df_gmm_sup1 = pd.read_csv('Time/GMM/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=4_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_sup1[r'$K$'] = 4
df_gmm_sup1['Method'] = 'RF(GMM)'
df_gmm_sup2 = pd.read_csv('Time/GMM/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=8_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_sup2[r'$K$'] = 8
df_gmm_sup2['Method'] = 'RF(GMM)'
df_gmm_sup3 = pd.read_csv('Time/GMM/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=16_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_sup3[r'$K$'] = 16
df_gmm_sup3['Method'] = 'RF(GMM)'
df_gmm_sup4 = pd.read_csv('Time/GMM/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=32_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_sup4[r'$K$'] = 32
df_gmm_sup4['Method'] = 'RF(GMM)'
df_gmm_sup5 = pd.read_csv('Time/GMM/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=64_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_sup5[r'$K$'] = 64
df_gmm_sup5['Method'] = 'RF(GMM)'
df_gmm_sup6 = pd.read_csv('Time/GMM/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_sup6[r'$K$'] = 128
df_gmm_sup6['Method'] = 'RF(GMM)'
df_gmm_sup7 = pd.read_csv('Time/GMM/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=256_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_gmm_sup7[r'$K$'] = 256
df_gmm_sup7['Method'] = 'RF(GMM)'
df_gmm_sup = pd.concat([df_gmm_sup1,df_gmm_sup2,df_gmm_sup3,df_gmm_sup4,df_gmm_sup5,df_gmm_sup6,df_gmm_sup7], axis=0)
df_gmm_sup.rename(index=str, columns={"t(rf)": "t(s)"}, inplace=True)

df_grid_sup1 = pd.read_csv('Time/Grid/SUP/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=4_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_sup1[r'$K$'] = df_gmm_uns1.loc[:,'n_feat']
df_grid_sup1['Method'] = 'RF(Grid)'
df_grid_sup2 = pd.read_csv('Time/Grid/SUP/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=8_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_sup2[r'$K$'] = df_gmm_uns2.loc[:,'n_feat']
df_grid_sup2['Method'] = 'RF(Grid)'
df_grid_sup3 = pd.read_csv('Time/Grid/SUP/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=16_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_sup3[r'$K$'] = df_gmm_uns3.loc[:,'n_feat']
df_grid_sup3['Method'] = 'RF(Grid)'
df_grid_sup4 = pd.read_csv('Time/Grid/SUP/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=32_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_sup4[r'$K$'] = df_gmm_uns4.loc[:,'n_feat']
df_grid_sup4['Method'] = 'RF(Grid)'
df_grid_sup5 = pd.read_csv('Time/Grid/SUP/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=64_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_sup5[r'$K$'] = df_gmm_uns5.loc[:,'n_feat']
df_grid_sup5['Method'] = 'RF(Grid)'
df_grid_sup6 = pd.read_csv('Time/Grid/SUP/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_sup6[r'$K$'] = df_gmm_uns6.loc[:,'n_feat']
df_grid_sup6['Method'] = 'RF(Grid)'
df_grid_sup7 = pd.read_csv('Time/Grid/SUP/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=256_N_SAMPLES=300_N_TREES=200.csv',index_col=0,header=0)
df_grid_sup7[r'$K$'] = df_gmm_uns7.loc[:,'n_feat']
df_grid_sup7['Method'] = 'RF(Grid)'
df_grid_sup = pd.concat([df_grid_sup1,df_grid_sup2,df_grid_sup3,df_grid_sup4,df_grid_sup5,df_grid_sup6,df_grid_sup7], axis=0)
df_grid_sup.rename(index=str, columns={"t(rf)": "t(s)"}, inplace=True)

df_sup = pd.concat([df_gmm_sup,df_grid_sup], axis=0)
df_sup.index = np.arange(0,df_sup.shape[0])
df_uns = pd.concat([df_gmm_uns,df_grid_uns], axis=0).reindex()
df_uns.index = np.arange(0,df_sup.shape[0])
df_sum = df_sup.replace(['RF(GMM)','RF(Grid)'],['PhenoGMM','PhenoGrid'])
df_sum.loc[:,'t(s)'] = df_uns.loc[:,'t(s)'].add(df_sup.loc[:,'t(s)'], axis=0)
df_gmm_sum = df_sum.loc[df_sum.loc[:,'Method'] == 'PhenoGMM']
df_grid_sum = df_sum.loc[df_sum.loc[:,'Method'] == 'PhenoGrid']

df = pd.concat([df_sum,df_uns,df_sup], axis=0)
pal = sns.color_palette('deep')
g = sns.lineplot(x=r'$K$', y='t(s)', data=df_gmm_uns, style='Method', dashes=[(2, 2), (2, 2)], markers=True, color='b', legend=False)#, scatter_kws={'alpha':0.75})
ax2 = plt.twinx()
h = sns.lineplot(x=r'$K$', y=r'$\kappa(D_1)$', data=df_gmm_uns, style='Method', dashes=[(2, 2), (2, 2)], markers=True, color='orange', legend=False)
g.set(ylim=(1,4000))
g.set_xlabel(r'$K$', fontsize=18)
g.set_ylabel('time (s)', fontsize=18)
h.set_ylabel(r'$\tau_B(D_1)$', fontsize=18)
h.set(ylim=(0,0.75))
plt.savefig('Figures/Time_complexity_PhenoGMM_UNS.png',dpi=500, bbox_tight=True)
plt.show()

df = pd.concat([df_sum,df_uns,df_sup], axis=0)
pal = sns.color_palette('deep')
g = sns.lineplot(x=r'$K$', y='t(s)', data=df_gmm_sum, style='Method', dashes=[(2, 2), (2, 2)], markers=True, color='b', legend=False)#, scatter_kws={'alpha':0.75})
plt.legend(['time (s)'], fontsize=16, loc=(0.65,0.2))
ax2 = plt.twinx()
h = sns.lineplot(x=r'$K$', y=r'$\kappa(D_1)$', data=df_gmm_sup, style='Method', dashes=[(2, 2), (2, 2)], markers=True, color='orange', legend=False)
g.set(ylim=(1,4000))
g.set_xlabel(r'$K$', fontsize=18)
g.set_ylabel('time (s)', fontsize=18)
h.set_ylabel(r'$\tau_B(D_1)$', fontsize=18)
h.set(ylim=(0,0.75))
plt.legend([r'$\tau_B(D_1)$'], fontsize=16, loc='lower right')#, bbox_to_anchor=(0.5, 1.05))
plt.savefig('Figures/Time_complexity_PhenoGMM_SUP.png',dpi=500, bbox_tight=True)
plt.show()

g = sns.lmplot(x='t(s)', y=r'$R^2(D1)$', data=df_sum, hue='Method', fit_reg=False, palette=pal, scatter_kws={'alpha':0.75})
g.set_xlabels('time (s)', fontsize=18)
g.set_ylabels(r'$R^2(D_1)$',fontsize=18)
g.set(ylim=(0.3,0.75))
plt.xscale('log')
plt.savefig('Figures/SI/Time_PhenoGMM_vs_PhenoGrid.png',dpi=300, bbox_tight=True)