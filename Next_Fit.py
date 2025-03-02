import numpy as np


def next_fit(cap_bins, cost_bins, n_b, w_items, p_items, n_it, n_c_i):
    c_bins = cap_bins.copy()
    allocation = [0] * n_it
    j = 0

    # we save in allocation for each item i the number of the bin where the item will be
    # we will have 0 if the item isn't in any bin
    for i in range(n_it):
        t = 0
        while j < n_b:
            if c_bins[j] >= w_items[i]:
                allocation[i] = j + 1
                c_bins[j] = c_bins[j] - w_items[i]
                break
            elif j == n_b - 1:
                if t == 1:
                    break
                else:
                    j = 1
                    t = 1
            else:
                j = j + 1

    bins = np.ones(n_b)
    solution = np.zeros((n_it, n_b))
    residual_capacity_bins = cap_bins.copy()

    # we build a matrix with value 1 if the item i is in the bin j and 0 otherwise
    # we create a vector with the residual capacity for each bin
    for j in range(n_b):
        for i in range(n_it):
            if allocation[i] == j + 1:
                solution[i, j] = 1
                residual_capacity_bins[j] = residual_capacity_bins[j] - w_items[i]

    def tot_cost(b_s):
        cost = 0
        for j in range(n_b):
            cost = cost + cost_bins[j]
            for i in range(n_c_i, n_it):
                cost = cost - p_items[i] * b_s[i, j]
        return cost

    print(tot_cost(solution))

    return solution, bins, residual_capacity_bins, allocation
