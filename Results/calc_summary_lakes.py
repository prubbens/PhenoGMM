#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:36:05 2018

@author: prubbens
"""

import pandas as pd
from scipy.stats import ttest_ind

df_inl_gmm_uns = pd.read_csv('INL_GMM_UNS_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mich_gmm_uns = pd.read_csv('MICH_GMM_UNS_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mus_gmm_uns = pd.read_csv('MUS_GMM_UNS_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_all_gmm_uns = pd.read_csv('ALL_GMM_UNS_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)

mean_inl_gmm_uns = df_inl_gmm_uns.mean()
mean_mich_gmm_uns = df_mich_gmm_uns.mean()
mean_mus_gmm_uns = df_mus_gmm_uns.mean()
mean_all_gmm_uns = df_all_gmm_uns.mean()
std_inl_gmm_uns = df_inl_gmm_uns.std()
std_mich_gmm_uns = df_mich_gmm_uns.std()
std_mus_gmm_uns = df_mus_gmm_uns.std()
std_all_gmm_uns = df_all_gmm_uns.std()

df_inl_gmm_sup = pd.read_csv('INL_GMM_SUP_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mich_gmm_sup = pd.read_csv('MICH_GMM_SUP_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mus_gmm_sup = pd.read_csv('MUS_GMM_SUP_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_all_gmm_sup = pd.read_csv('ALL_GMM_SUP_FEAT=3_N_CELLS=2000_N_MIX=256_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)

mean_inl_gmm_sup = df_inl_gmm_sup.mean()
mean_mich_gmm_sup = df_mich_gmm_sup.mean()
mean_mus_gmm_sup = df_mus_gmm_sup.mean()
mean_all_gmm_sup = df_all_gmm_sup.mean()
std_inl_gmm_sup = df_inl_gmm_sup.std()
std_mich_gmm_sup = df_mich_gmm_sup.std()
std_mus_gmm_sup = df_mus_gmm_sup.std()
std_all_gmm_sup = df_all_gmm_sup.std()


df_inl_grid_uns = pd.read_csv('INL_GRID_UNS_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mich_grid_uns = pd.read_csv('MICH_GRID_UNS_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mus_grid_uns = pd.read_csv('MUS_GRID_UNS_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_all_grid_uns = pd.read_csv('ALL_GRID_UNS_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
mean_inl_grid_uns = df_inl_grid_uns.mean()
mean_mich_grid_uns = df_mich_grid_uns.mean()
mean_mus_grid_uns = df_mus_grid_uns.mean()
mean_all_grid_uns = df_all_grid_uns.mean()
std_inl_grid_uns = df_inl_grid_uns.std()
std_mich_grid_uns = df_mich_grid_uns.std()
std_mus_grid_uns = df_mus_grid_uns.std()
std_all_grid_uns = df_all_grid_uns.std()

df_inl_grid_sup = pd.read_csv('INL_GRID_SUP_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mich_grid_sup = pd.read_csv('MICH_GRID_SUP_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_mus_grid_sup = pd.read_csv('MUS_GRID_SUP_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)
df_all_grid_sup = pd.read_csv('ALL_GRID_SUP_FEAT=3_N_CELLS=2000_N_BINS=128_N_TREES=200_TYPE=full_OTU_asinh.csv', index_col=0, header=0)

mean_inl_grid_sup = df_inl_grid_sup.mean()
mean_mich_grid_sup = df_mich_grid_sup.mean()
mean_mus_grid_sup = df_mus_grid_sup.mean()
mean_all_grid_sup = df_all_grid_sup.mean()
std_inl_grid_sup = df_inl_grid_sup.std()
std_mich_grid_sup = df_mich_grid_sup.std()
std_mus_grid_sup = df_mus_grid_sup.std()
std_all_grid_sup = df_all_grid_sup.std()

t_inl_d0_sup, p_inl_d0_sup = ttest_ind(df_inl_gmm_sup.loc[:,r'$R^2(D_0)$'],df_inl_grid_sup.loc[:,r'$R^2(D_0)$'], equal_var=False)
t_inl_d1_sup, p_inl_d1_sup = ttest_ind(df_inl_gmm_sup.loc[:,r'$R^2(D_1)$'],df_inl_grid_sup.loc[:,r'$R^2(D_1)$'], equal_var=False)
t_inl_d2_sup, p_inl_d2_sup = ttest_ind(df_inl_gmm_sup.loc[:,r'$R^2(D_2)$'],df_inl_grid_sup.loc[:,r'$R^2(D_2)$'], equal_var=False)

t_mich_d0_sup, p_mich_d0_sup = ttest_ind(df_mich_gmm_sup.loc[:,r'$R^2(D_0)$'],df_mich_grid_sup.loc[:,r'$R^2(D_0)$'], equal_var=False)
t_mich_d1_sup, p_mich_d1_sup = ttest_ind(df_mich_gmm_sup.loc[:,r'$R^2(D_1)$'],df_mich_grid_sup.loc[:,r'$R^2(D_1)$'], equal_var=False)
t_mich_d2_sup, p_mich_d2_sup = ttest_ind(df_mich_gmm_sup.loc[:,r'$R^2(D_2)$'],df_mich_grid_sup.loc[:,r'$R^2(D_2)$'], equal_var=False)

t_mus_d0_sup, p_mus_d0_sup = ttest_ind(df_mus_gmm_sup.loc[:,r'$R^2(D_0)$'],df_mus_grid_sup.loc[:,r'$R^2(D_0)$'], equal_var=False)
t_mus_d1_sup, p_mus_d1_sup = ttest_ind(df_mus_gmm_sup.loc[:,r'$R^2(D_1)$'],df_mus_grid_sup.loc[:,r'$R^2(D_1)$'], equal_var=False)
t_mus_d2_sup, p_mus_d2_sup = ttest_ind(df_mus_gmm_sup.loc[:,r'$R^2(D_2)$'],df_mus_grid_sup.loc[:,r'$R^2(D_2)$'], equal_var=False)

t_all_d0_sup, p_all_d0_sup = ttest_ind(df_all_gmm_sup.loc[:,r'$R^2(D_0)$'],df_all_grid_sup.loc[:,r'$R^2(D_0)$'], equal_var=False)
t_all_d1_sup, p_all_d1_sup = ttest_ind(df_all_gmm_sup.loc[:,r'$R^2(D_1)$'],df_all_grid_sup.loc[:,r'$R^2(D_1)$'], equal_var=False)
t_all_d2_sup, p_all_d2_sup = ttest_ind(df_all_gmm_sup.loc[:,r'$R^2(D_2)$'],df_all_grid_sup.loc[:,r'$R^2(D_2)$'], equal_var=False)

t_inl_k_d0_sup, p_inl_k_d0_sup = ttest_ind(df_inl_gmm_sup.loc[:,r'$\tau_B(D_0)$'],df_inl_grid_sup.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_inl_k_d1_sup, p_inl_k_d1_sup = ttest_ind(df_inl_gmm_sup.loc[:,r'$\tau_B(D_1)$'],df_inl_grid_sup.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_inl_k_d2_sup, p_inl_k_d2_sup = ttest_ind(df_inl_gmm_sup.loc[:,r'$\tau_B(D_2)$'],df_inl_grid_sup.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_mich_k_d0_sup, p_mich_k_d0_sup = ttest_ind(df_mich_gmm_sup.loc[:,r'$\tau_B(D_0)$'],df_mich_grid_sup.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_mich_k_d1_sup, p_mich_k_d1_sup = ttest_ind(df_mich_gmm_sup.loc[:,r'$\tau_B(D_1)$'],df_mich_grid_sup.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_mich_k_d2_sup, p_mich_k_d2_sup = ttest_ind(df_mich_gmm_sup.loc[:,r'$\tau_B(D_2)$'],df_mich_grid_sup.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_mus_k_d0_sup, p_mus_k_d0_sup = ttest_ind(df_mus_gmm_sup.loc[:,r'$\tau_B(D_0)$'],df_mus_grid_sup.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_mus_k_d1_sup, p_mus_k_d1_sup = ttest_ind(df_mus_gmm_sup.loc[:,r'$\tau_B(D_1)$'],df_mus_grid_sup.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_mus_k_d2_sup, p_mus_k_d2_sup = ttest_ind(df_mus_gmm_sup.loc[:,r'$\tau_B(D_2)$'],df_mus_grid_sup.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_all_k_d0_sup, p_all_k_d0_sup = ttest_ind(df_all_gmm_sup.loc[:,r'$\tau_B(D_0)$'],df_all_grid_sup.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_all_k_d1_sup, p_all_k_d1_sup = ttest_ind(df_all_gmm_sup.loc[:,r'$\tau_B(D_1)$'],df_all_grid_sup.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_all_k_d2_sup, p_all_k_d2_sup = ttest_ind(df_all_gmm_sup.loc[:,r'$\tau_B(D_2)$'],df_all_grid_sup.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_inl_k_d0_uns, p_inl_k_d0_uns = ttest_ind(df_inl_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_inl_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_inl_k_d1_uns, p_inl_k_d1_uns = ttest_ind(df_inl_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_inl_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_inl_k_d2_uns, p_inl_k_d2_uns = ttest_ind(df_inl_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_inl_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)
t_inl_b_uns, p_inl_b_uns = ttest_ind(df_inl_gmm_uns.loc[:,r'$\rho(\beta)$'],df_inl_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)

t_mich_k_d0_uns, p_mich_k_d0_uns = ttest_ind(df_mich_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_mich_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_mich_k_d1_uns, p_mich_k_d1_uns = ttest_ind(df_mich_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_mich_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_mich_k_d2_uns, p_mich_k_d2_uns = ttest_ind(df_mich_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_mich_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)
t_mich_b_uns, p_mich_b_uns = ttest_ind(df_mich_gmm_uns.loc[:,r'$\rho(\beta)$'],df_mich_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)

t_mus_k_d0_uns, p_mus_k_d0_uns = ttest_ind(df_mus_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_mus_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_mus_k_d1_uns, p_mus_k_d1_uns = ttest_ind(df_mus_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_mus_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_mus_k_d2_uns, p_mus_k_d2_uns = ttest_ind(df_mus_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_mus_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)
t_mus_b_uns, p_mus_b_uns = ttest_ind(df_mus_gmm_uns.loc[:,r'$\rho(\beta)$'],df_mus_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)

t_all_k_d0_uns, p_all_k_d0_uns = ttest_ind(df_all_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_all_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_all_k_d1_uns, p_all_k_d1_uns = ttest_ind(df_all_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_all_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_all_k_d2_uns, p_all_k_d2_uns = ttest_ind(df_all_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_all_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)
t_all_b_uns, p_all_b_uns = ttest_ind(df_all_gmm_uns.loc[:,r'$\rho(\beta)$'],df_all_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)