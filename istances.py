import random

import numpy as np

# define initial parameters
index = random.randint(0,1)
if index == 0:
    n_bins = 5
    n_compulsory_items = 20
    n_non_compulsory_items = 80
    n_items = n_compulsory_items + n_non_compulsory_items
    # generate bin's costs
    cost_bins = np.random.uniform(100,120,n_bins)
    max_budget = 400
    # generate item's profits
    profit_items = np.random.uniform(10,12,n_items)
    weight_items = np.random.uniform(1,5,n_items)
    capacity_bins = np.random.uniform(20,40,n_bins)

if index == 1:
    n_bins = 5
    n_compulsory_items = 20
    n_non_compulsory_items = 80
    n_items = n_compulsory_items + n_non_compulsory_items
    # generate bin's costs
    cost_bins = np.random.uniform(100,120,n_bins)
    max_budget = 400
    # generate item's profits
    profit_items = np.random.poisson(8,n_items)
    weight_items = np.random.gamma(2.5,1,n_items)
    capacity_bins = np.random.uniform(20,40,n_bins)

