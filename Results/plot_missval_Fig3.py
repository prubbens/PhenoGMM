#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:20:48 2018

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

target_FCM = pd.read_csv('../Data/Metadata_FCM_Cycles.csv', index_col='Sample name')

pred_D0_cycleI = pd.read_csv('Missing Values/CycleI_GMM_D0_FEAT=3_N_CELLS_c=2000_N_MIX=128_N_TREES=200_TYPE=full.csv', index_col=0)
pred_D1_cycleI = pd.read_csv('Missing Values/CycleI_GMM_D1_FEAT=3_N_CELLS_c=2000_N_MIX=128_N_TREES=200_TYPE=full.csv', index_col=0)
pred_D2_cycleI = pd.read_csv('Missing Values/CycleI_GMM_D2_FEAT=3_N_CELLS_c=2000_N_MIX=128_N_TREES=200_TYPE=full.csv', index_col=0)
pred_D0_cycleII = pd.read_csv('Missing Values/CycleII_GMM_D0_FEAT=3_N_CELLS_c=2000_N_MIX=128_N_TREES=200_TYPE=full.csv', index_col=0)
pred_D1_cycleII = pd.read_csv('Missing Values/CycleII_GMM_D1_FEAT=3_N_CELLS_c=2000_N_MIX=128_N_TREES=200_TYPE=full.csv', index_col=0)
pred_D2_cycleII = pd.read_csv('Missing Values/CycleII_GMM_D2_FEAT=3_N_CELLS_c=2000_N_MIX=128_N_TREES=200_TYPE=full.csv', index_col=0)
booleanDictionary = {True: 'FCM', False: '16S'}
target_FCM.loc[:,'Sample']  = pd.isnull(target_FCM.loc[:,'D0.tax'])
target_FCM.loc[:,'Sample'] = target_FCM.loc[:,'Sample'].replace(booleanDictionary)

target_FCM.loc[pred_D0_cycleI.index,'D0.tax'] = pred_D0_cycleI.values.mean(axis=1)
target_FCM.loc[pred_D1_cycleI.index,'D1.tax'] = pred_D1_cycleI.values.mean(axis=1)
target_FCM.loc[pred_D2_cycleI.index,'D2.tax'] = pred_D2_cycleI.values.mean(axis=1)
target_FCM.loc[pred_D0_cycleII.index,'D0.tax'] = pred_D0_cycleII.values.mean(axis=1)
target_FCM.loc[pred_D1_cycleII.index,'D1.tax'] = pred_D1_cycleII.values.mean(axis=1)
target_FCM.loc[pred_D2_cycleII.index,'D2.tax'] = pred_D2_cycleII.values.mean(axis=1)

def patch_violinplot():
    """Patch seaborn's violinplot in current axis to workaround matplotlib's bug ##5423."""
    from matplotlib.collections import PolyCollection
    ax = plt.gca()
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_edgecolor((0.1, 0.1, 0.1))

def plot_R2cv_DIV_cycleI(df, pred): 
    df = df.loc[df.loc[:,'Survey'] == 1]
    pred.loc[:,'Time point (d)'] = df.loc[pred.index,'Time point (d)']
    pred = pd.melt(pred,id_vars=['Time point (d)'], value_name='D2_pred')
    pred.loc[:,'Survey'] = 1
    
    fig, ax = plt.subplots()
    sns.regplot(x='Time point (d)',y='D2.tax', data=df.loc[df.loc[:,'Sample'] == '16S'], ax=ax, fit_reg=False)
    sns.regplot(x='Time point (d)', y='D2_pred', data=pred, ax=ax, color='orange', fit_reg=False, scatter_kws={'alpha':0.35})
    ax.set_xlabel('Time point (d)',fontsize=18)
    ax.set_ylabel(r'$D_2$',fontsize=18)
    plt.ylim(0,14)
    plt.savefig('Figures/D2_missVal_box_CycleI.png',bbox_inches='tight', dpi=500)
    return df

def plot_R2cv_DIV_cycleII(df, pred): 
    df = df.loc[df.loc[:,'Survey'] == 2]
    pred.loc[:,'Time point (d)'] = df.loc[pred.index,'Time point (d)']
    pred = pd.melt(pred,id_vars=['Time point (d)'], value_name='D2_pred')
    pred.loc[:,'Survey'] = 2
    
    fig, ax = plt.subplots()
    sns.regplot(x='Time point (d)',y='D2.tax', data=df.loc[df.loc[:,'Sample'] == '16S'], ax=ax, fit_reg=False)
    sns.regplot(x='Time point (d)', y='D2_pred', data=pred, ax=ax, color='orange', fit_reg=False, scatter_kws={'alpha':0.35})
    ax.set_xlabel('Time point (d)',fontsize=18)
    ax.set_ylabel(r'$D_2$',fontsize=18)
    plt.ylim(0,14)
    plt.savefig('Figures/D2_missVal_box_CycleII.png',bbox_inches='tight', dpi=500)
    return df

target_FCM.rename(columns={'Cycle':'Survey'}, inplace=True)

df_cycleI = plot_R2cv_DIV_cycleI(target_FCM, pred_D2_cycleI)
df_cycleII = plot_R2cv_DIV_cycleII(target_FCM, pred_D2_cycleII)