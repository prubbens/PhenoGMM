#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:36:05 2018

@author: prubbens
"""

import pandas as pd
from scipy.stats import ttest_ind

df_cycleI_gmm_uns = pd.read_csv('CycleI_GMM_UNS_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleII_gmm_uns = pd.read_csv('CycleII_GMM_UNS_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleI_II_gmm_uns = pd.read_csv('CycleI+II_GMM_UNS_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
mean_cycleI_gmm_uns = df_cycleI_gmm_uns.mean()
mean_cycleII_gmm_uns = df_cycleII_gmm_uns.mean()
mean_cycleI_II_gmm_uns = df_cycleI_II_gmm_uns.mean()
std_cycleI_gmm_uns = df_cycleI_gmm_uns.std()
std_cycleII_gmm_uns = df_cycleII_gmm_uns.std()
std_cycleI_II_gmm_uns = df_cycleI_II_gmm_uns.std()

df_cycleI_gmm_sup = pd.read_csv('CycleI_GMM_SUP_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleII_gmm_sup = pd.read_csv('CycleII_GMM_SUP_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleI_II_gmm_sup = pd.read_csv('CycleI+II_GMM_SUP_FEAT=3_N_CELLS_c=2000_N_MIX=256_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
mean_cycleI_gmm_sup = df_cycleI_gmm_sup.mean()
mean_cycleII_gmm_sup = df_cycleII_gmm_sup.mean()
mean_cycleI_II_gmm_sup = df_cycleI_II_gmm_sup.mean()
std_cycleI_gmm_sup = df_cycleI_gmm_sup.std()
std_cycleII_gmm_sup = df_cycleII_gmm_sup.std()
std_cycleI_II_gmm_sup = df_cycleI_II_gmm_sup.std()

df_cycleI_grid_uns = pd.read_csv('CycleI_GRID_UNS_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleII_grid_uns = pd.read_csv('CycleII_GRID_UNS_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleI_II_grid_uns = pd.read_csv('CycleI+II_GRID_UNS_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
mean_cycleI_grid_uns = df_cycleI_grid_uns.mean()
mean_cycleII_grid_uns = df_cycleII_grid_uns.mean()
mean_cycleI_II_grid_uns = df_cycleI_II_grid_uns.mean()
std_cycleI_grid_uns = df_cycleI_grid_uns.std()
std_cycleII_grid_uns = df_cycleII_grid_uns.std()
std_cycleI_II_grid_uns = df_cycleI_II_grid_uns.std()

df_cycleI_grid_sup = pd.read_csv('CycleI_GRID_SUP_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleII_grid_sup = pd.read_csv('CycleII_GRID_SUP_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
df_cycleI_II_grid_sup = pd.read_csv('CycleI+II_GRID_SUP_FEAT=3_N_CELLS_c=2000_N_BINS=128_N_TREES=200_TYPE=full_asinh.csv', index_col=0, header=0)
mean_cycleI_grid_sup = df_cycleI_grid_sup.mean()
mean_cycleII_grid_sup = df_cycleII_grid_sup.mean()
mean_cycleI_II_grid_sup = df_cycleI_II_grid_sup.mean()
std_cycleI_grid_sup = df_cycleI_grid_sup.std()
std_cycleII_grid_sup = df_cycleII_grid_sup.std()
std_cycleI_II_grid_sup = df_cycleI_II_grid_sup.std()

t_cycleI_d0_sup, p_cycleI_d0_sup = ttest_ind(df_cycleI_gmm_sup.loc[:,r'$R^2(D0)$'],df_cycleI_grid_sup.loc[:,r'$R^2(D0)$'], equal_var=False)
t_cycleI_d1_sup, p_cycleI_d1_sup = ttest_ind(df_cycleI_gmm_sup.loc[:,r'$R^2(D1)$'],df_cycleI_grid_sup.loc[:,r'$R^2(D1)$'], equal_var=False)
t_cycleI_d2_sup, p_cycleI_d2_sup = ttest_ind(df_cycleI_gmm_sup.loc[:,r'$R^2(D2)$'],df_cycleI_grid_sup.loc[:,r'$R^2(D2)$'], equal_var=False)

t_cycleII_d0_sup, p_cycleII_d0_sup = ttest_ind(df_cycleII_gmm_sup.loc[:,r'$R^2(D0)$'],df_cycleII_grid_sup.loc[:,r'$R^2(D0)$'], equal_var=False)
t_cycleII_d1_sup, p_cycleII_d1_sup = ttest_ind(df_cycleII_gmm_sup.loc[:,r'$R^2(D1)$'],df_cycleII_grid_sup.loc[:,r'$R^2(D1)$'], equal_var=False)
t_cycleII_d2_sup, p_cycleII_d2_sup = ttest_ind(df_cycleII_gmm_sup.loc[:,r'$R^2(D2)$'],df_cycleII_grid_sup.loc[:,r'$R^2(D2)$'], equal_var=False)

t_cycleI_II_d0_sup, p_cycleI_II_d0_sup = ttest_ind(df_cycleI_II_gmm_sup.loc[:,r'$R^2(D0)$'],df_cycleI_II_grid_sup.loc[:,r'$R^2(D0)$'], equal_var=False)
t_cycleI_II_d1_sup, p_cycleI_II_d1_sup = ttest_ind(df_cycleI_II_gmm_sup.loc[:,r'$R^2(D1)$'],df_cycleI_II_grid_sup.loc[:,r'$R^2(D1)$'], equal_var=False)
t_cycleI_II_d2_sup, p_cycleI_II_d2_sup = ttest_ind(df_cycleI_II_gmm_sup.loc[:,r'$R^2(D2)$'],df_cycleI_II_grid_sup.loc[:,r'$R^2(D2)$'], equal_var=False)

t_cycleI_k_d0_sup, p_cycleI_k_d0_sup = ttest_ind(df_cycleI_gmm_sup.loc[:,r'$\tau_B(D_0)$'],df_cycleI_grid_sup.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_cycleI_k_d1_sup, p_cycleI_k_d1_sup = ttest_ind(df_cycleI_gmm_sup.loc[:,r'$\tau_B(D_1)$'],df_cycleI_grid_sup.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_cycleI_k_d2_sup, p_cycleI_k_d2_sup = ttest_ind(df_cycleI_gmm_sup.loc[:,r'$\tau_B(D_2)$'],df_cycleI_grid_sup.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_cycleII_k_d0_sup, p_cycleII_k_d0_sup = ttest_ind(df_cycleII_gmm_sup.loc[:,r'$\tau_B(D_0)$'],df_cycleII_grid_sup.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_cycleII_k_d1_sup, p_cycleII_k_d1_sup = ttest_ind(df_cycleII_gmm_sup.loc[:,r'$\tau_B(D_1)$'],df_cycleII_grid_sup.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_cycleII_k_d2_sup, p_cycleII_k_d2_sup = ttest_ind(df_cycleII_gmm_sup.loc[:,r'$\tau_B(D_2)$'],df_cycleII_grid_sup.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_cycleI_II_k_d0_sup, p_cycleI_II_k_d0_sup = ttest_ind(df_cycleI_II_gmm_sup.loc[:,r'$\tau_B(D_0)$'],df_cycleI_II_grid_sup.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_cycleI_II_k_d1_sup, p_cycleI_II_k_d1_sup = ttest_ind(df_cycleI_II_gmm_sup.loc[:,r'$\tau_B(D_1)$'],df_cycleI_II_grid_sup.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_cycleI_II_k_d2_sup, p_cycleI_II_k_d2_sup = ttest_ind(df_cycleI_II_gmm_sup.loc[:,r'$\tau_B(D_2)$'],df_cycleI_II_grid_sup.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_cycleI_k_d0_uns, p_cycleI_k_d0_uns = ttest_ind(df_cycleI_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_cycleI_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_cycleI_k_d1_uns, p_cycleI_k_d1_uns = ttest_ind(df_cycleI_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_cycleI_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_cycleI_k_d2_uns, p_cycleI_k_d2_uns = ttest_ind(df_cycleI_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_cycleI_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)
t_cycleI_b_uns, p_cycleI_b_uns = ttest_ind(df_cycleI_gmm_uns.loc[:,r'$\rho(\beta)$'],df_cycleI_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)

t_cycleII_k_d0_uns, p_cycleII_k_d0_uns = ttest_ind(df_cycleII_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_cycleII_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_cycleII_k_d1_uns, p_cycleII_k_d1_uns = ttest_ind(df_cycleII_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_cycleII_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_cycleII_k_d2_uns, p_cycleII_k_d2_uns = ttest_ind(df_cycleII_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_cycleII_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)
t_cycleII_b_uns, p_cycleII_b_uns = ttest_ind(df_cycleII_gmm_uns.loc[:,r'$\rho(\beta)$'],df_cycleII_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)

t_cycleI_II_k_d0_uns, p_cycleI_II_k_d0_uns = ttest_ind(df_cycleI_II_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_cycleI_II_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_cycleI_II_k_d1_uns, p_cycleI_II_k_d1_uns = ttest_ind(df_cycleI_II_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_cycleI_II_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_cycleI_II_k_d2_uns, p_cycleI_II_k_d2_uns = ttest_ind(df_cycleI_II_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_cycleI_II_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)
t_cycleI_II_b_uns, p_cycleI_II_b_uns = ttest_ind(df_cycleI_II_gmm_uns.loc[:,r'$\rho(\beta)$'],df_cycleI_II_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)