#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:17:49 2019

@author: prubbens
"""

import numpy as np
import pandas as pd

''' Sample random relative compositions of microbial communities using the Dirichlet distribution, with universal paramater a 

Parameters
----------

n_communities: number of communities to sample
n_populations_min: minimal number of populations that are present
n_populations_max: maximum number of populations that are present
a: universal parameter of the Dirichlet distribution, which determines how the weight among populations will be divided. 
   small a: few populations will be highly abundant
   high a: the abundances are equally spread over the populations

'''

def sample_compositions(n_communities, n_populations_min, n_populations_max, a=1): 
    comp = pd.DataFrame(np.zeros((n_communities,n_populations_max)))
    for comm in np.arange(0,n_communities): 
        S = np.random.randint(n_populations_min,n_populations_max,1)
        mock = np.random.choice(n_populations_max,S,replace=False)
        comp.iloc[comm,mock] = np.random.dirichlet(np.ones(S)*a,size=1)[0] #See: https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
    return comp

np.random.seed(2703)
N_POP_min = 2
N_POP_max = 20
N_TRAIN = 300
N_TEST = 100

a = 0.1
comp_train = sample_compositions(N_TRAIN, N_POP_min, N_POP_max, a)
comp_test = sample_compositions(N_TEST, N_POP_min, N_POP_max, a)
comp_train.to_csv('../Files/Comp_train_a=' + str(a) + '.csv')
comp_test.to_csv('../Files/Comp_test_a=' + str(a) + '.csv')

a = 1
comp_train = sample_compositions(N_TRAIN, N_POP_min, N_POP_max, a)
comp_test = sample_compositions(N_TEST, N_POP_min, N_POP_max, a)
comp_train.to_csv('../Files/Comp_train_a=' + str(a) + '.csv')
comp_test.to_csv('../Files/Comp_test_a=' + str(a) + '.csv')

a = 10
comp_train = sample_compositions(N_TRAIN, N_POP_min, N_POP_max, a)
comp_test = sample_compositions(N_TEST, N_POP_min, N_POP_max, a)
comp_train.to_csv('../Files/Comp_train_a=' + str(a) + '.csv')
comp_test.to_csv('../Files/Comp_test_a=' + str(a) + '.csv')