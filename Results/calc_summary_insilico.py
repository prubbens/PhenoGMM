#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:36:05 2018

@author: prubbens
"""

import pandas as pd
from scipy.stats import ttest_ind

df_a0_1_gmm_uns = pd.read_csv('Test_GMM_UNS_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_gmm_uns = pd.read_csv('Test_GMM_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_gmm_uns = pd.read_csv('Test_GMM_UNS_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)

mean_a0_1_gmm_uns = df_a0_1_gmm_uns.mean()
mean_a1_gmm_uns = df_a1_gmm_uns.mean()
mean_a10_gmm_uns = df_a10_gmm_uns.mean()
std_a0_1_gmm_uns = df_a0_1_gmm_uns.std()
std_a1_gmm_uns = df_a1_gmm_uns.std()
std_a10_gmm_uns = df_a10_gmm_uns.std()

df_a0_1_grid_uns = pd.read_csv('Test_GRID_UNS_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_grid_uns = pd.read_csv('Test_GRID_UNS_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_grid_uns = pd.read_csv('Test_GRID_UNS_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)

mean_a0_1_grid_uns = df_a0_1_grid_uns.mean()
mean_a1_grid_uns = df_a1_grid_uns.mean()
mean_a10_grid_uns = df_a10_grid_uns.mean()
std_a0_1_grid_uns = df_a0_1_grid_uns.std()
std_a1_grid_uns = df_a1_grid_uns.std()
std_a10_grid_uns = df_a10_grid_uns.std()

df_a0_1_gmm = pd.read_csv('Test_GMM_SUP_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_gmm = pd.read_csv('Test_GMM_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_gmm = pd.read_csv('Test_GMM_SUP_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)

mean_a0_1_gmm = df_a0_1_gmm.mean()
mean_a1_gmm = df_a1_gmm.mean()
mean_a10_gmm = df_a10_gmm.mean()
std_a0_1_gmm = df_a0_1_gmm.std()
std_a1_gmm = df_a1_gmm.std()
std_a10_gmm = df_a10_gmm.std()

df_a0_1_grid = pd.read_csv('Test_GRID_SUP_a=0.1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a1_grid = pd.read_csv('Test_GRID_SUP_a=1_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)
df_a10_grid = pd.read_csv('Test_GRID_SUP_a=10_FEAT=3_N_CELLS_c=2500_N_CELLS_i=2500_N_MIX=128_N_SAMPLES=300_N_TREES=200_asinh.csv', index_col=0, header=0)

mean_a0_1_grid = df_a0_1_grid.mean()
mean_a1_grid = df_a1_grid.mean()
mean_a10_grid = df_a10_grid.mean()
std_a0_1_grid = df_a0_1_grid.std()
std_a1_grid = df_a1_grid.std()
std_a10_grid = df_a10_grid.std()

t_r2_d0_a0_1, p_r2_d0_a0_1 = ttest_ind(df_a0_1_gmm.loc[:,r'$R^2(D_0)$'],df_a0_1_grid.loc[:,r'$R^2(D_0)$'], equal_var=False)
t_r2_d1_a0_1, p_r2_d1_a0_1 = ttest_ind(df_a0_1_gmm.loc[:,r'$R^2(D_1)$'],df_a0_1_grid.loc[:,r'$R^2(D_1)$'], equal_var=False)
t_r2_d2_a0_1, p_r2_d2_a0_1 = ttest_ind(df_a0_1_gmm.loc[:,r'$R^2(D_2)$'],df_a0_1_grid.loc[:,r'$R^2(D_2)$'], equal_var=False)

t_r2_d0_a1, p_r2_d0_a1 = ttest_ind(df_a1_gmm.loc[:,r'$R^2(D_0)$'],df_a1_grid.loc[:,r'$R^2(D_0)$'], equal_var=False)
t_r2_d1_a1, p_r2_d1_a1 = ttest_ind(df_a1_gmm.loc[:,r'$R^2(D_1)$'],df_a1_grid.loc[:,r'$R^2(D_1)$'], equal_var=False)
t_r2_d2_a1, p_r2_d2_a1 = ttest_ind(df_a1_gmm.loc[:,r'$R^2(D_2)$'],df_a1_grid.loc[:,r'$R^2(D_2)$'], equal_var=False)

t_r2_d0_a10, p_r2_d0_a10 = ttest_ind(df_a10_gmm.loc[:,r'$R^2(D_0)$'],df_a10_grid.loc[:,r'$R^2(D_0)$'], equal_var=False)
t_r2_d1_a10, p_r2_d1_a10 = ttest_ind(df_a10_gmm.loc[:,r'$R^2(D_1)$'],df_a10_grid.loc[:,r'$R^2(D_1)$'], equal_var=False)
t_r2_d2_a10, p_r2_d2_a10 = ttest_ind(df_a10_gmm.loc[:,r'$R^2(D_2)$'],df_a10_grid.loc[:,r'$R^2(D_2)$'], equal_var=False)

t_kendall_d0_a0_1, p_kendall_d0_a0_1 = ttest_ind(df_a0_1_gmm.loc[:,r'$\tau_B(D_0)$'],df_a0_1_grid.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_kendall_d1_a0_1, p_kendall_d1_a0_1 = ttest_ind(df_a0_1_gmm.loc[:,r'$\tau_B(D_1)$'],df_a0_1_grid.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_kendall_d2_a0_1, p_kendall_d2_a0_1 = ttest_ind(df_a0_1_gmm.loc[:,r'$\tau_B(D_2)$'],df_a0_1_grid.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_kendall_d0_a1, p_kendall_d0_a1 = ttest_ind(df_a1_gmm.loc[:,r'$\tau_B(D_0)$'],df_a1_grid.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_kendall_d1_a1, p_kendall_d1_a1 = ttest_ind(df_a1_gmm.loc[:,r'$\tau_B(D_1)$'],df_a1_grid.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_kendall_d2_a1, p_kendall_d2_a1 = ttest_ind(df_a1_gmm.loc[:,r'$\tau_B(D_2)$'],df_a1_grid.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_kendall_d0_a10, p_kendall_d0_a10 = ttest_ind(df_a10_gmm.loc[:,r'$\tau_B(D_0)$'],df_a10_grid.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_kendall_d1_a10, p_kendall_d1_a10 = ttest_ind(df_a10_gmm.loc[:,r'$\tau_B(D_1)$'],df_a10_grid.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_kendall_d2_a10, p_kendall_d2_a10 = ttest_ind(df_a10_gmm.loc[:,r'$\tau_B(D_2)$'],df_a10_grid.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_kendall_d0_a0_1_uns, p_kendall_d0_a0_1_uns = ttest_ind(df_a0_1_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_a0_1_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_kendall_d1_a0_1_uns, p_kendall_d1_a0_1_uns = ttest_ind(df_a0_1_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_a0_1_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_kendall_d2_a0_1_uns, p_kendall_d2_a0_1_uns = ttest_ind(df_a0_1_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_a0_1_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_kendall_d0_a1_uns, p_kendall_d0_a1_uns = ttest_ind(df_a1_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_a1_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_kendall_d1_a1_uns, p_kendall_d1_a1_uns = ttest_ind(df_a1_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_a1_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_kendall_d2_a1_uns, p_kendall_d2_a1_uns = ttest_ind(df_a1_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_a1_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_kendall_d0_a10_uns, p_kendall_d0_a10_uns = ttest_ind(df_a10_gmm_uns.loc[:,r'$\tau_B(D_0)$'],df_a10_grid_uns.loc[:,r'$\tau_B(D_0)$'], equal_var=False)
t_kendall_d1_a10_uns, p_kendall_d1_a10_uns = ttest_ind(df_a10_gmm_uns.loc[:,r'$\tau_B(D_1)$'],df_a10_grid_uns.loc[:,r'$\tau_B(D_1)$'], equal_var=False)
t_kendall_d2_a10_uns, p_kendall_d2_a10_uns = ttest_ind(df_a10_gmm_uns.loc[:,r'$\tau_B(D_2)$'],df_a10_grid_uns.loc[:,r'$\tau_B(D_2)$'], equal_var=False)

t_mantel_a0_1_uns, p_mantel_a0_1_uns = ttest_ind(df_a0_1_gmm_uns.loc[:,r'$\rho(\beta)$'],df_a0_1_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)
t_mantel_a1_uns, p_mantel_a1_uns = ttest_ind(df_a1_gmm_uns.loc[:,r'$\rho(\beta)$'],df_a1_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)
t_mantel_a10_uns, p_mantel_a10_uns = ttest_ind(df_a10_gmm_uns.loc[:,r'$\rho(\beta)$'],df_a10_grid_uns.loc[:,r'$\rho(\beta)$'], equal_var=False)