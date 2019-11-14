#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: prubbens
"""
import itertools
import numpy as np
import pandas as pd

from itertools import permutations
from scipy import spatial, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, RandomizedSearchCV


''' Calculate Hill diversity for q=0 (richness)

Paramaters:
----------

df: dataframe containing abundance data

'''
def calc_D0(df):
    result = df.astype(bool).sum(axis=1)
    return pd.DataFrame(result, columns=[r'$D_0$'])


''' Calculate Hill diversity for q=1 (evenness)

Paramaters:
----------

df: dataframe containing abundance data

'''
def calc_D1(df): 
    return pd.DataFrame(np.exp(stats.entropy(df.T.values)), index=df.index, columns=[r'$D_1$'])


''' Calculate Hill diversity for q=2 (evenness)

Paramaters:
----------

df: dataframe containing abundance data

'''    
def calc_D2(df): 
    df_sum = df.sum(axis=1)
    df_final = (df.div(df_sum, axis=0))**2
    return pd.DataFrame(1./df_final.sum(axis=1), columns=[r'$D_2$'])

'''Concat dataframes from flow cytometry experiment, based on minimum amount of cells in a replicate 

Parameters:
----------

df_meta: metadata file
columns_rep: variable which contains list of filenames
n_cells: number of cells to sample per community
path_data: directory which contains the files
'''
def concat_df(df_meta, n_cells, path_dir, list_filenames): 
    df = pd.DataFrame()
    for index in df_meta.index:
        for file in list_filenames: 
            if file.startswith(index): 
                df_rep = pd.read_csv(path_dir+file, index_col=0, header=0)
                df = pd.concat([df,df_rep.sample(n_cells)], ignore_index=True, axis=0)
    return df


''' Concatenate technical replicates of the same sample into one dataframe

Parameters:
----------
path_data: directory which contains the files
list_filenames: list containing filenames
idx_comm: index of specific sample
frac: fraction of cells to sample
n_cells: number of cells to sample per community
n_rep: number of replicates
'''
def concat_rep(n_cells, path_dir, list_filenames, n_rep): 
    df = pd.DataFrame()
    n_sample = n_cells
    for file in list_filenames: 
        df_rep = pd.read_csv(path_dir + file, index_col=0, header=0)
        if df_rep.shape[0] < n_sample: 
            n_sample = df_rep.shape[0]
    for file in list_filenames: 
        df_rep = pd.read_csv(path_dir + file, index_col=0, header=0)
        df = pd.concat([df,df_rep.sample(n_sample)], ignore_index=True, axis=0)
    return df


''' Concatenate technical replicates of the same sample into one dataframe

Parameters:
----------
path_files: directory which contains the files
list_filenames: list containing filenames
n_rep: number of replicates
n_cells: number of cells to sample per community  
'''
def concat_rep_rw(path_files, list_filenames, n_rep, n_cells): 
    df = pd.DataFrame()
    for i in np.arange(0,n_rep): 
        df_rep = pd.read_csv(path_files + '/' + list_filenames[i], index_col=0, header=0)
        if df_rep.shape[0] < n_cells: 
            n_cells = df_rep.shape[0]
    for j in np.arange(0,n_rep): 
        df_rep = pd.read_csv(path_files + '/' + list_filenames[j], index_col=0, header=0)
        df = pd.concat([df,df_rep.sample(n_cells)], ignore_index=True, axis=0)
    return df

'''Get multiple equally spaced grids, for each bivariate combinations of variables

Parameters:
----------
df: dataframe
features: variables in dataframe
n_bins: number of bins
norm_bool: boolean whether to normalize the data
transform_bool: boolean whether to transform the data according to transform()
'''
def get_binning_grid(df, features, n_bins, norm=False, transform_bool=True): 
    grids = []
    bivariate_combs = list(itertools.combinations(features, 2))
    for feat_comb in bivariate_combs: 
        if transform_bool==True: 
            hist_tot, grid = np.histogramdd(np.arcsinh(df.loc[:,feat_comb].values), bins=n_bins, normed=norm)
        else: 
            hist_tot, grid = np.histogramdd(df.loc[:,feat_comb].values, bins=n_bins, normed=norm)   
        grids.append(grid)
    return grids

''' Return counts per bin for each sample

Parameters: 
----------

df_meta: metadata file, containing sample names 
n_rep: number of replicates
n_cells: number of cells to sample per community
features: variables in dataframe
grids: list of numpy two-dimensional histograms for each pairwise combination of all features
transform_bool: boolean whether to normalize the data
transform_bool: if True, first transform data according to transform()
path_files: directory which contains the files
list_filenames: list containing filenames
'''
def get_fcfp_binning_grid_rw(df_meta, n_rep, n_cells, features, grids, transform_bool, norm, path_files, list_filenames):
    fingerprint = []
    bivariate_combs = list(itertools.combinations(features, 2))    
    for index in df_meta.index:
        list_reps = []
        for file in list_filenames: 
            if file.startswith(index): 
                list_reps.append(file)
        df_sample = concat_rep(n_cells, path_files, list_reps, n_rep)
        if transform_bool==True: 
            df_sample = transform(df_sample, features)
        fingerprint_grid = []
        for grid, comb_feat in zip(grids,bivariate_combs): 
            hist, grid_ = np.histogramdd(df_sample.loc[:,comb_feat].values, bins=grid, normed=norm) 
            hist = np.divide(hist, np.float(df_sample.shape[0]))
            fingerprint_grid.extend(hist.reshape(-1))
        fingerprint.append(fingerprint_grid)
    df_fingerprint = pd.DataFrame(np.array(fingerprint), index=df_meta.index)
    return df_fingerprint


''' Create Gaussian Mixture Model and fit to dataframe df 

Parameters:
----------
df: dataframe
features: variables in dataframe
n_mixtures: number of mixtures to initialize Gaussian Mixture Model
cov_type: 'diag','full' (default in scikit-learn), 'spherical' or tied
transform_bool: if True, first transform data according to transform()
scaler: object to standardize dataframe
'''
def get_gmm_fitted(df, features, n_mixtures=100, cov_type='full', transform_bool=True): 
    gmm = GaussianMixture(n_components=n_mixtures, covariance_type=cov_type, warm_start=False)
    if transform_bool==True: 
        df_trans = transform(df, features)
        gmm.fit(df_trans.loc[:,features].values)
    else: 
        df_trans = df
        gmm.fit(df_trans.loc[:,features].values)
    return gmm

''' Cluster individual samples using a fitted Gaussian Mixture Model and derive cell counts per mixture 

Parameters:
----------
df_meta: metadata file, containing sample names 
n_rep: number of replicates
n_cells: number of cells to sample per community
n_mix: numer of mixtures
features: variables in dataframe
gmm: fitted gaussian mixture model
transform_bool: boolean whether to normalize the data
path_files: directory which contains the files
list_filenames: list containing filenames
'''
def get_fcfp_gmm_rw(df_meta, n_rep, n_cells, n_mix, features, gmm, transform_bool, path_files, list_filenames):
    fingerprint = []
    for index in df_meta.index:
        list_reps = []
        for file in list_filenames: 
            if file.startswith(index): 
                list_reps.append(file)
        df_sample = concat_rep(n_cells, path_files, list_reps, n_rep)
        if transform_bool==True: 
            df_sample = transform(df_sample, features)
        preds = gmm.predict(df_sample.loc[:,features].values)
        fcfp = np.bincount(preds, minlength = n_mix)
        fcfp = np.divide(fcfp, np.float(df_sample.shape[0]))
        fingerprint.append(fcfp.reshape(-1))
    df_fingerprint = pd.DataFrame(fingerprint, index=df_meta.index)
    return df_fingerprint

''' Create K-fold cross-validation object

Parameters: 
----------
n_splits: k, number of splits
'''
def get_KFold(n_splits, shuffle=True): 
    return KFold(n_splits=n_splits, shuffle=shuffle)


''' Return variables for which at least one element is non-zero 

Parameters: 
----------
df: dataframe
'''
def get_nonzero_features(df): 
    return df.columns[(df != 0).any(axis=0)]


''' Get trained Random Forest Regression Model, tuned using a Randomized Grid Search

Parameters: 
----------
df_train: dataframe containing training set
df_target: dataframe containing target variable to predict
features: variables in dataframe
n_trees: number of trees
min_samples_leaf: minimum samples per leaf
n_iter: number of iterations 
cv: cross-validation object
'''
def get_trained_RF(df_train, df_target, features, n_trees, min_samples_leaf, n_iter, cv): 
    len_feat = len(features)
    if np.sqrt(len_feat)*min_samples_leaf < n_iter: 
        rf = RandomizedSearchCV(RandomForestRegressor(n_estimators=n_trees, criterion='mse', oob_score=True), scoring='neg_mean_squared_error', iid = False, param_distributions={'max_features': np.arange(1,len_feat), 'min_samples_leaf': np.arange(1,min_samples_leaf+1)}, cv=cv, n_iter=np.sqrt(len_feat)*min_samples_leaf)
    else: 
        rf = RandomizedSearchCV(RandomForestRegressor(n_estimators=n_trees, criterion='mse', oob_score=True), scoring='neg_mean_squared_error', iid = False, param_distributions={'max_features': np.arange(1,len_feat), 'min_samples_leaf': np.arange(1,min_samples_leaf+1)}, cv=cv, n_iter=n_iter)
    rf.fit(df_train.loc[:,features],df_target)
    return rf

def mantel_test(X, Y, perms=10000, method='pearson', tail='two-tail'):
#Source: https://github.com/jwcarr/MantelTest/blob/master/Mantel.py
  """
  Takes two distance matrices (either redundant matrices or condensed vectors)
  and performs a Mantel test. The Mantel test is a significance test of the
  correlation between two distance matrices.
  Parameters
  ----------
  X : array_like
      First distance matrix (condensed or redundant).
  Y : array_like
      Second distance matrix (condensed or redundant), where the order of
      elements corresponds to the order of elements in the first matrix.
  perms : int, optional
      The number of permutations to perform (default: 10000). A larger number
      gives more reliable results but takes longer to run. If the actual number
      of possible permutations is smaller, the program will enumerate all
      permutations. Enumeration can be forced by setting this argument to 0.
  method : str, optional
      Type of correlation coefficient to use; either 'pearson' or 'spearman'
      (default: 'pearson').
  tail : str, optional
      Which tail to test in the calculation of the empirical p-value; either
      'upper', 'lower', or 'two-tail' (default: 'two-tail').
  Returns
  -------
  r : float
      Veridical correlation
  p : float
      Empirical p-value
  z : float
      Standard score (z-score)
  """

  # Ensure that X and Y are formatted as Numpy arrays.
  X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

  # Check that X and Y are valid distance matrices.
  if spatial.distance.is_valid_dm(X) == False and spatial.distance.is_valid_y(X) == False:
    raise ValueError('X is not a valid condensed or redundant distance matrix')
  if spatial.distance.is_valid_dm(Y) == False and spatial.distance.is_valid_y(Y) == False:
    raise ValueError('Y is not a valid condensed or redundant distance matrix')

  # If X or Y is a redundant distance matrix, reduce it to a condensed distance matrix.
  if len(X.shape) == 2:
    X = spatial.distance.squareform(X, force='tovector', checks=False)
  if len(Y.shape) == 2:
    Y = spatial.distance.squareform(Y, force='tovector', checks=False)

  # Check for size equality.
  if X.shape[0] != Y.shape[0]:
    raise ValueError('X and Y are not of equal size')

  # Check for minimum size.
  if X.shape[0] < 3:
    raise ValueError('X and Y should represent at least 3 objects')

  # If Spearman correlation is requested, convert X and Y to ranks.
  if method == 'spearman':
    X, Y = stats.rankdata(X), stats.rankdata(Y)

  # Check for valid method parameter.
  elif method != 'pearson':
    raise ValueError('The method should be set to "pearson" or "spearman"')

  # Check for valid tail parameter.
  if tail != 'upper' and tail != 'lower' and tail != 'two-tail':
    raise ValueError('The tail should be set to "upper", "lower", or "two-tail"')

  # Now we're ready to start the Mantel test using a number of optimizations:
  #
  # 1. We don't need to recalculate the pairwise distances between the objects
  #    on every permutation. They've already been calculated, so we can use a
  #    simple matrix shuffling technique to avoid recomputing them. This works
  #    like memoization.
  #
  # 2. Rather than compute correlation coefficients, we'll just compute the
  #    covariances. This works because the denominator in the equation for the
  #    correlation coefficient will yield the same result however the objects
  #    are permuted, making it redundant. Removing the denominator leaves us
  #    with the covariance.
  #
  # 3. Rather than permute the Y distances and derive the residuals to calculate
  #    the covariance with the X distances, we'll represent the Y residuals in
  #    the matrix and shuffle those directly.
  #
  # 4. If the number of possible permutations is less than the number of
  #    permutations that were requested, we'll run a deterministic test where
  #    we try all possible permutations rather than sample the permutation
  #    space. This gives a faster, deterministic result.

  # Calculate the X and Y residuals, which will be used to compute the
  # covariance under each permutation.
  X_residuals, Y_residuals = X - X.mean(), Y - Y.mean()

  # Expand the Y residuals to a redundant matrix.
  Y_residuals_as_matrix = spatial.distance.squareform(Y_residuals, force='tomatrix', checks=False)

  # Get the number of objects.
  m = Y_residuals_as_matrix.shape[0]

  # Calculate the number of possible matrix permutations.
  n = np.math.factorial(m)

  # Initialize an empty array to store temporary permutations of Y_residuals.
  Y_residuals_permuted = np.zeros(Y_residuals.shape[0], dtype=float)

  # If the number of requested permutations is greater than the number of
  # possible permutations (m!) or the perms parameter is set to 0, then run a
  # deterministic Mantel test ...
  if perms >= n or perms == 0:

    # Initialize an empty array to store the covariances.
    covariances = np.zeros(n, dtype=float)

    # Enumerate all permutations of row/column orders and iterate over them.
    for i, order in enumerate(permutations(range(m))):

      # Take a permutation of the matrix.
      Y_residuals_as_matrix_permuted = Y_residuals_as_matrix[order, :][:, order]

      # Condense the permuted version of the matrix. Rather than use
      # distance.squareform(), we call directly into the C wrapper for speed.
      spatial.distance._distance_wrap.to_vector_from_squareform_wrap(Y_residuals_as_matrix_permuted, Y_residuals_permuted)

      # Compute and store the covariance.
      covariances[i] = (X_residuals * Y_residuals_permuted).sum()

  # ... otherwise run a stochastic Mantel test.
  else:

    # Initialize an empty array to store the covariances.
    covariances = np.zeros(perms, dtype=float)

    # Initialize an array to store the permutation order.
    order = np.arange(m)

    # Store the veridical covariance in 0th position...
    covariances[0] = (X_residuals * Y_residuals).sum()

    # ...and then run the random permutations.
    for i in range(1, perms):

      # Choose a random order in which to permute the rows and columns.
      np.random.shuffle(order)

      # Take a permutation of the matrix.
      Y_residuals_as_matrix_permuted = Y_residuals_as_matrix[order, :][:, order]

      # Condense the permuted version of the matrix. Rather than use
      # distance.squareform(), we call directly into the C wrapper for speed.
      spatial.distance._distance_wrap.to_vector_from_squareform_wrap(Y_residuals_as_matrix_permuted, Y_residuals_permuted)

      # Compute and store the covariance.
      covariances[i] = (X_residuals * Y_residuals_permuted).sum()

  # Calculate the veridical correlation coefficient from the veridical covariance.
  r = covariances[0] / np.sqrt((X_residuals ** 2).sum() * (Y_residuals ** 2).sum())

  # Calculate the empirical p-value for the upper or lower tail.
  if tail == 'upper':
    p = (covariances >= covariances[0]).sum() / float(covariances.shape[0])
  elif tail == 'lower':
    p = (covariances <= covariances[0]).sum() / float(covariances.shape[0])
  elif tail == 'two-tail':
    p = (abs(covariances) >= abs(covariances[0])).sum() / float(covariances.shape[0])

  # Calculate the standard score.
  #z = (covariances[0] - covariances.mean()) / covariances.std()

  return r, p


''' Transform each variable in the dataframe df by arcsinh(x) 

Parameters:
----------
df: dataframe
features: variables to transform
'''
def transform(df, features): 
    return df.loc[:,features].apply(np.arcsinh)