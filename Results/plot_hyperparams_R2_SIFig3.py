#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:47:32 2018

@author: prubbens
"""

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_color_codes()
tips = sns.load_dataset("tips")

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
sns.set_style("ticks")


def patch_violinplot():
    """Patch seaborn's violinplot in current axis to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.1, 0.1, 0.1))

def plot_R2_cat(df,title,cat): 
    df.rename(columns={r'$\kappa(D_1)$':r'$\tau_B(D_1)$'}, inplace=True)
    g = sns.catplot(x=title,y=r'$R^2(D_1)$', data=df, kind='box', height=5, aspect=1, sharey=True, color='white',linewidth=2)
    g.set_titles(size=20)
    g.set_xlabels(fontsize=18)
    g.set_ylabels(fontsize=18)
    g.set(ylim=(0,1))
    plt.savefig('Figures/SI/D1_R2_'+title+'_'+cat+'.png',bbox_inches='tight', dpi=300)
    plt.show()
    return df

df_sup1 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=1_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup1[r'$D$'] = 1
df_sup1['Analysis'] = 'Supervised'
df_sup2 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=2_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup2[r'$D$'] = 2
df_sup2['Analysis'] = 'Supervised'
df_sup3 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEfull_asinh.csv',index_col=0,header=0)
df_sup3[r'$D$'] = 3
df_sup3['Analysis'] = 'Supervised'
df_sup4 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=4_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup4[r'$D$'] = 4
df_sup4['Analysis'] = 'Supervised'
df_sup5 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=5_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup5[r'$D$'] = 5
df_sup5['Analysis'] = 'Supervised'
df_sup6 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=6_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup6[r'$D$'] = 6
df_sup6['Analysis'] = 'Supervised'
df_sup = pd.concat([df_sup1,df_sup2,df_sup3,df_sup4,df_sup5,df_sup6], axis=0)
df = plot_R2_cat(df_sup,r'$D$','Analysis')

df_sup1 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup1['N_CELLS_i'] = 500
df_sup1['Analysis'] = 'Supervised'
df_sup2 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=1000_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup2['N_CELLS_i'] = 1000
df_sup2['Analysis'] = 'Supervised'
df_sup3 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=1500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup3['N_CELLS_i'] = 1500
df_sup3['Analysis'] = 'Supervised'
df_sup4 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2000_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup4['N_CELLS_i'] = 2000
df_sup4['Analysis'] = 'Supervised'
df_sup5 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEfull_asinh.csv',index_col=0,header=0)
df_sup5['N_CELLS_i'] = 2500
df_sup5['Analysis'] = 'Supervised'
df_sup6 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=3000_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup6['N_CELLS_i'] = 3000
df_sup6['Analysis'] = 'Supervised'
df_sup7 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=3500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup7['N_CELLS_i'] = 3500
df_sup7['Analysis'] = 'Supervised'
df_sup8 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=4000_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup8['N_CELLS_i'] = 4000
df_sup8['Analysis'] = 'Supervised'
df_sup9 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=4500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup9['N_CELLS_i'] = 4500
df_sup9['Analysis'] = 'Supervised'
df_sup10 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=5000_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup10['N_CELLS_i'] = 5000
df_sup10['Analysis'] = 'Supervised'
df_sup = pd.concat([df_sup1,df_sup2,df_sup3,df_sup4,df_sup5,df_sup6,df_sup7,df_sup8,df_sup9,df_sup10], axis=0)
df = plot_R2_cat(df_sup,'N_CELLS_i','Analysis')

df_sup1 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup1['N_CELLS_c'] = 500
df_sup1['Analysis'] = 'Supervised'
df_sup2 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=1000_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup2['N_CELLS_c'] = 1000
df_sup2['Analysis'] = 'Supervised'
df_sup3 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=1500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup3['N_CELLS_c'] = 1500
df_sup3['Analysis'] = 'Supervised'
df_sup4 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2000_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup4['N_CELLS_c'] = 2000
df_sup4['Analysis'] = 'Supervised'
df_sup5 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEfull_asinh.csv',index_col=0,header=0)
df_sup5['N_CELLS_c'] = 2500
df_sup5['Analysis'] = 'Supervised'
df_sup6 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=3000_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup6['N_CELLS_c'] = 3000
df_sup6['Analysis'] = 'Supervised'
df_sup7 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=3500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup7['N_CELLS_c'] = 3500
df_sup7['Analysis'] = 'Supervised'
df_sup8 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=4000_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup8['N_CELLS_c'] = 4000
df_sup8['Analysis'] = 'Supervised'
df_sup9 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=4500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup9['N_CELLS_c'] = 4500
df_sup9['Analysis'] = 'Supervised'
df_sup10 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=5000_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup10['N_CELLS_c'] = 5000
df_sup10['Analysis'] = 'Supervised'
df_sup = pd.concat([df_sup1,df_sup2,df_sup3,df_sup4,df_sup5,df_sup6,df_sup7,df_sup8,df_sup9,df_sup10], axis=0)

df = plot_R2_cat(df_sup,'N_CELLS_c','Analysis')

df_sup1 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=25_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup1['N_SAMPLES'] = 25
df_sup1['Analysis'] = 'Supervised'
df_sup2 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=50_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup2['N_SAMPLES'] = 50
df_sup2['Analysis'] = 'Supervised'
df_sup3 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=75_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup3['N_SAMPLES'] = 75
df_sup3['Analysis'] = 'Supervised'
df_sup4 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=100_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup4['N_SAMPLES'] = 100
df_sup4['Analysis'] = 'Supervised'
df_sup5 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=125_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup5['N_SAMPLES'] = 125
df_sup5['Analysis'] = 'Supervised'
df_sup6 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=150_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup6['N_SAMPLES'] = 150
df_sup6['Analysis'] = 'Supervised'
df_sup7 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=175_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup7['N_SAMPLES'] = 175
df_sup7['Analysis'] = 'Supervised'
df_sup8 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=200_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup8['N_SAMPLES'] = 200
df_sup8['Analysis'] = 'Supervised'
df_sup9 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=225_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup9['N_SAMPLES'] = 225
df_sup9['Analysis'] = 'Supervised'
df_sup10 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=250_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup10['N_SAMPLES'] = 250
df_sup10['Analysis'] = 'Supervised'
df_sup11 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=275_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup11['N_SAMPLES'] = 275
df_sup11['Analysis'] = 'Supervised'
df_sup12 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEfull_asinh.csv',index_col=0,header=0)
df_sup12['N_SAMPLES'] = 300
df_sup12['Analysis'] = 'Supervised'
df_sup = pd.concat([df_sup1,df_sup2,df_sup3,df_sup4,df_sup5,df_sup6,df_sup7,df_sup8,df_sup9,df_sup10,df_sup11,df_sup12], axis=0)
df = plot_R2_cat(df_sup,'N_SAMPLES','Analysis')

df_sup1 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=4_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup1[r'$K$'] = 4
df_sup1['Analysis'] = 'Supervised'
df_sup2 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=8_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup2[r'$K$'] = 8
df_sup2['Analysis'] = 'Supervised'
df_sup3 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=16_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup3[r'$K$'] = 16
df_sup3['Analysis'] = 'Supervised'
df_sup4 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=32_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup4[r'$K$'] = 32
df_sup4['Analysis'] = 'Supervised'
df_sup5 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=64_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup5[r'$K$'] = 64
df_sup5['Analysis'] = 'Supervised'
df_sup6 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEfull_asinh.csv',index_col=0,header=0)
df_sup6[r'$K$'] = 128
df_sup6['Analysis'] = 'Supervised'
df_sup7 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=256_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup7[r'$K$'] = 256
df_sup7['Analysis'] = 'Supervised'
df_sup8 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=512_N_SAMPLES=300_N_TREES=200_seed2703_asinh.csv',index_col=0,header=0)
df_sup8[r'$K$'] = 512
df_sup8['Analysis'] = 'Supervised'
df_sup = pd.concat([df_sup1,df_sup2,df_sup3,df_sup4,df_sup5,df_sup6,df_sup7,df_sup8], axis=0)

df = plot_R2_cat(df_sup,r'$K$','Analysis')

df_sup1 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEdiag_asinh.csv',index_col=0,header=0)
df_sup1['TYPE'] = 'diag'
df_sup1['Analysis'] = 'Supervised'
df_sup2 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEfull_asinh.csv',index_col=0,header=0)
df_sup2['TYPE'] = 'full'
df_sup2['Analysis'] = 'Supervised'
df_sup3 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEspherical_asinh.csv',index_col=0,header=0)
df_sup3['TYPE'] = 'spherical'
df_sup3['Analysis'] = 'Supervised'
df_sup4 = pd.read_csv('Hyperparams/SUP/Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_TYPEtied_asinh.csv',index_col=0,header=0)
df_sup4['TYPE'] = 'tied'
df_sup4['Analysis'] = 'Supervised'
df_sup = pd.concat([df_sup1,df_sup2,df_sup3,df_sup4], axis=0)

df = plot_R2_cat(df_sup,'TYPE','Analysis')