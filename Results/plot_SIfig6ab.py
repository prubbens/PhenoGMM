#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:36:05 2018

@author: prubbens
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_c1_gmm_uns = pd.read_csv('Cycles/CycleI_GMM_UNS_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c1_gmm_uns['Method'] = 'PhenoGMM'
df_c2_gmm_uns = pd.read_csv('Cycles/CycleII_GMM_UNS_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c2_gmm_uns['Method'] = 'PhenoGMM'

df_c1_grid_uns = pd.read_csv('Cycles/CycleI_GRID_UNS_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c1_grid_uns['Method'] = 'PhenoGrid'
df_c2_grid_uns = pd.read_csv('Cycles/CycleI_GRID_UNS_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_c2_grid_uns['Method'] = 'PhenoGrid'


FEAT_UNS = [r'$\tau_B(D_0)$',r'$\tau_B(D_1)$',r'$\tau_B(D_2)$','Method']
df_c1_uns = pd.concat([df_c1_gmm_uns.loc[:,FEAT_UNS],df_c1_grid_uns.loc[:,FEAT_UNS]])
df_c1_uns = pd.melt(df_c1_uns, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')
df_c2_uns = pd.concat([df_c2_gmm_uns.loc[:,FEAT_UNS],df_c2_grid_uns.loc[:,FEAT_UNS]])
df_c2_uns = pd.melt(df_c2_uns, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')


pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_c1_uns, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
#g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_c1_uns, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title('Survey I', size=18)
#g.set_axis_labels('Classifier','AUC')
g.set_xlabels('Diversity index',fontsize=16)
g.set_ylabels(r'$\tau_B$',fontsize=16)
g.set_xticklabels([r'$D_0$',r'$D_1$',r'$D_2$'],fontsize=15)
g.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=15)
g.set(ylim=(0.0, 1.0))
for ax in g.axes.flat:
    plt.setp(ax.get_legend().get_texts()[0:2], fontsize=0) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=0)
handles, labels = h.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=0)
#handles, labels = g.get_legend_handles_labels()
plt.tight_layout()
plt.hlines(y = 0.228, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.savefig('Figures/SI/SurveyI_UNS.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_c2_uns, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
#g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_c2_uns, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title('Survey II', size=18)
#g.set_axis_labels('Classifier','AUC')
g.set_xlabels('Diversity index',fontsize=16)
g.set_ylabels(r'$\tau_B$',fontsize=16)
g.set_xticklabels([r'$D_0$',r'$D_1$',r'$D_2$'],fontsize=15)
g.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=15)
g.set(ylim=(0.0, 1.0))
for ax in g.axes.flat:
    plt.setp(ax.get_legend().get_texts()[0:2], fontsize=0) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=0)
handles, labels = h.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=0)
#handles, labels = g.get_legend_handles_labels()
plt.hlines(y = 0.213, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.tight_layout()
plt.savefig('Figures/SI/SurveyII_UNS.png',dpi=500,bbox_tight=True)