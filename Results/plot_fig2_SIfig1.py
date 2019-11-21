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
df_a0_1_gmm_uns['Method'] = 'PhenoGMM'
df_a1_gmm_uns = pd.read_csv('In Silico/Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_gmm_uns['Method'] = 'PhenoGMM'
df_a10_gmm_uns = pd.read_csv('In Silico/Test_GMM_UNS_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_gmm_uns['Method'] = 'PhenoGMM'

df_a0_1_grid_uns = pd.read_csv('In Silico/Test_GRID_UNS_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a0_1_grid_uns['Method'] = 'PhenoGrid'
df_a1_grid_uns = pd.read_csv('In Silico/Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_grid_uns['Method'] = 'PhenoGrid'
df_a10_grid_uns = pd.read_csv('In Silico/Test_GRID_UNS_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_grid_uns['Method'] = 'PhenoGrid'

FEAT_UNS = [r'$\tau_B(D_0)$',r'$\tau_B(D_1)$',r'$\tau_B(D_2)$','Method']
df_a0_1_uns = pd.concat([df_a0_1_gmm_uns.loc[:,FEAT_UNS],df_a0_1_grid_uns.loc[:,FEAT_UNS]])
df_a0_1_uns = pd.melt(df_a0_1_uns, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')
df_a1_uns = pd.concat([df_a1_gmm_uns.loc[:,FEAT_UNS],df_a1_grid_uns.loc[:,FEAT_UNS]])
df_a1_uns = pd.melt(df_a1_uns, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')
df_a10_uns = pd.concat([df_a10_gmm_uns.loc[:,FEAT_UNS],df_a10_grid_uns.loc[:,FEAT_UNS]])
df_a10_uns = pd.melt(df_a10_uns, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_a0_1_uns, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_a0_1_uns, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 0.1$', size=18)
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
plt.tight_layout()
plt.hlines(y = 0.133, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.savefig('Figures/sync_uns_a01_1.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_a1_uns, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_a1_uns, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 1$', size=18)
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
plt.hlines(y = 0.133, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.tight_layout()
plt.savefig('Figures/sync_uns_a1.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_a10_uns, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_a10_uns, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 10$', size=18)
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
plt.hlines(y = 0.133, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.tight_layout()
plt.savefig('Figures/sync_uns_a10.png',dpi=500,bbox_tight=True)

df_a0_1_gmm = pd.read_csv('In Silico/Test_GMM_SUP_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a0_1_gmm['Method'] = 'PhenoGMM'
df_a1_gmm = pd.read_csv('In Silico/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_gmm['Method'] = 'PhenoGMM'
df_a10_gmm = pd.read_csv('In Silico/Test_GMM_SUP_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_gmm['Method'] = 'PhenoGMM'

df_a0_1_grid = pd.read_csv('In Silico/Test_GRID_SUP_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a0_1_grid['Method'] = 'PhenoGrid'
df_a1_grid = pd.read_csv('In Silico/Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_grid['Method'] = 'PhenoGrid'
df_a10_grid = pd.read_csv('In Silico/Test_GRID_SUP_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_grid['Method'] = 'PhenoGrid'

FEAT_UNS = [r'$\tau_B(D_0)$',r'$\tau_B(D_1)$',r'$\tau_B(D_2)$','Method']
df_a0_1 = pd.concat([df_a0_1_gmm.loc[:,FEAT_UNS],df_a0_1_grid.loc[:,FEAT_UNS]])
df_a0_1 = pd.melt(df_a0_1, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')
df_a1 = pd.concat([df_a1_gmm.loc[:,FEAT_UNS],df_a1_grid.loc[:,FEAT_UNS]])
df_a1 = pd.melt(df_a1, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')
df_a10 = pd.concat([df_a10_gmm.loc[:,FEAT_UNS],df_a10_grid.loc[:,FEAT_UNS]])
df_a10 = pd.melt(df_a10, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$\tau_B$')

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_a0_1, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_a0_1, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 0.1$', size=18)
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
plt.hlines(y = 0.133, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.tight_layout()
plt.savefig('Figures/sync_sup_a01_1.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_a1, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_a1, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 1$', size=18)
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
plt.hlines(y = 0.133, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.tight_layout()
plt.savefig('Figures/sync_sup_a1.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$\tau_B$',data=df_a10, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$\tau_B$',data=df_a10, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 10$', size=18)
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
plt.hlines(y = 0.133, xmin = -1, xmax = 3, linestyles='--', colors='grey')
plt.tight_layout()
plt.savefig('Figures/sync_sup_a10.png',dpi=500,bbox_tight=True)

FEAT_UNS = [r'$R^2(D_0)$',r'$R^2(D_1)$',r'$R^2(D_2)$','Method']
df_a0_1_r2 = pd.concat([df_a0_1_gmm.loc[:,FEAT_UNS],df_a0_1_grid.loc[:,FEAT_UNS]])
df_a0_1 = pd.melt(df_a0_1_r2, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$R^2$')
df_a1_r2 = pd.concat([df_a1_gmm.loc[:,FEAT_UNS],df_a1_grid.loc[:,FEAT_UNS]])
df_a1 = pd.melt(df_a1_r2, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$R^2$')
df_a10_r2 = pd.concat([df_a10_gmm.loc[:,FEAT_UNS],df_a10_grid.loc[:,FEAT_UNS]])
df_a10 = pd.melt(df_a10_r2, id_vars = ['Method'], var_name = 'Diversity index', value_name = r'$R^2$')

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$R^2$',data=df_a0_1, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$R^2$',data=df_a0_1, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 0.1$', size=18)
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
plt.tight_layout()
plt.savefig('Figures/SI/sync_sup_a01_1_r2.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$R^2$',data=df_a1, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$R^2$',data=df_a1, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 1$', size=18)
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
plt.tight_layout()
plt.savefig('Figures/SI/sync_sup_a1_r2.png',dpi=500,bbox_tight=True)

pal = sns.color_palette("colorblind")
g = sns.catplot(x='Diversity index',y=r'$R^2$',data=df_a10, legend=False, hue = 'Method', kind='box', height=5, aspect=1, sharey=True, palette=pal,linewidth=2.5)
pal = sns.color_palette('cubehelix')[0:1]
h = sns.stripplot(x='Diversity index',y=r'$R^2$',data=df_a10, hue = 'Method', dodge=True, palette=pal, alpha=0.8)   
plt.title(r'$a = 10$', size=18)
g.set_xlabels('Diversity index',fontsize=16)
g.set_ylabels(r'$R^2$',fontsize=16)
g.set_xticklabels([r'$D_0$',r'$D_1$',r'$D_2$'],fontsize=15)
g.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=15)
g.set(ylim=(0.0, 1.0))
for ax in g.axes.flat:
    plt.setp(ax.get_legend().get_texts()[0:2], fontsize=0)
    plt.setp(ax.get_legend().get_title(), fontsize=0)
handles, labels = h.get_legend_handles_labels()
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=0)
plt.tight_layout()
plt.savefig('Figures/SI/sync_sup_a10_r2.png',dpi=500,bbox_tight=True)