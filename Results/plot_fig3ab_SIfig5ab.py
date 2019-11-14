#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:36:05 2018

@author: prubbens
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_c1_gmm_sup = pd.read_csv('Cycles/CycleI_GMM_SUP_OOB_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_blocked_asinh.csv', index_col=0, header=0)
df_c1_gmm_sup['Method'] = 'PhenoGMM'
df_c2_gmm_sup = pd.read_csv('Cycles/CycleII_GMM_SUP_OOB_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_blocked_asinh.csv', index_col=0, header=0)
df_c2_gmm_sup['Method'] = 'PhenoGMM'


df_c1_grid_sup = pd.read_csv('Cycles/CycleI_Grid_SUP_OOB_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_blocked_asinh.csv', index_col=0, header=0)
df_c1_grid_sup['Method'] = 'PhenoGrid'
df_c2_grid_sup = pd.read_csv('Cycles/CycleII_Grid_SUP_OOB_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_blocked_asinh.csv', index_col=0, header=0)
df_c2_grid_sup['Method'] = 'PhenoGrid'

FEAT_UNS = [r'$\tau_B(D_0)$',r'$\tau_B(D_1)$',r'$\tau_B(D_2)$','Method']
df_c1_sup = pd.concat([df_c1_gmm_sup.loc[:,FEAT_UNS],df_c1_grid_sup.loc[:,FEAT_UNS]])
df_c1_sup = pd.melt(df_c1_sup, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')
df_c2_sup = pd.concat([df_c2_gmm_sup.loc[:,FEAT_UNS],df_c2_grid_sup.loc[:,FEAT_UNS]])
df_c2_sup = pd.melt(df_c2_sup, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_c1_sup, legend=False, hue = 'Method', kind='box', size=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
#g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_c1_sup, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
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
plt.savefig('Figures/SurveyI_SUP_bl.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_c2_sup, legend=False, hue = 'Method', kind='box', size=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
#g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_c2_sup, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
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
plt.savefig('Figures/SurveyII_SUP_bl.png',dpi=500,bbox_tight=True)

FEAT_UNS = [r'$R^2(D0)$',r'$R^2(D1)$',r'$R^2(D2)$','Method']
df_c1_r2 = pd.concat([df_c1_gmm_sup.loc[:,FEAT_UNS],df_c1_grid_sup.loc[:,FEAT_UNS]])
df_c1 = pd.melt(df_c1_r2, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$R^2$')
df_c2_r2 = pd.concat([df_c2_gmm_sup.loc[:,FEAT_UNS],df_c2_grid_sup.loc[:,FEAT_UNS]])
df_c2 = pd.melt(df_c2_r2, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$R^2$')

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$R^2$',data=df_c1, legend=False, hue = 'Method', kind='box', size=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
#g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$R^2$',data=df_c1, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title('Survey I', size=18)
#g.set_axis_labels('Classifier','AUC')
g.set_xlabels('Diversity index',fontsize=16)
g.set_ylabels(r'$R^2$',fontsize=16)
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
plt.savefig('Figures/SurveyI_SUP_bl_r2.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$R^2$',data=df_c2, legend=False, hue = 'Method', kind='box', size=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
#g = sns.catplot(x='Dataset',y=r'$R^2$',data=results, kind='box', size=4, aspect=1.6, sharey=True, color='white',linewidth=3)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$R^2$',data=df_c2, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title('Survey II', size=18)
#g.set_axis_labels('Classifier','AUC')
g.set_xlabels('Diversity index',fontsize=16)
g.set_ylabels(r'$R^2$',fontsize=16)
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
plt.savefig('Figures/SurveyII_SUP_bl_r2.png',dpi=500,bbox_tight=True)