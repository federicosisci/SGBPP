import time
from istances import *
from Gurobi import *
from Next_Fit import next_fit
from Tabu_Search import tabu_search
from Genetic_Algorithm import genetic_algorithm

start_gb = time.time()
Gurobi(index, n_bins, n_compulsory_items, n_non_compulsory_items, n_items, cost_bins, max_budget, profit_items,
       weight_items, capacity_bins)
end_gb = time.time()
comp_time_gb = end_gb - start_gb
print(f"Computational time of Gurobi: {comp_time_gb} s.")


start_nf = time.time()
solution_nf, bins_nf, residual_capacity_bins_nf, pop_nf = next_fit(capacity_bins,cost_bins, n_bins, weight_items,
                                                                   profit_items, n_items, n_compulsory_items)
end_nf = time.time()


best_solution = solution_nf.copy()
bins = bins_nf.copy()
residual_capacity = residual_capacity_bins_nf.copy()

start_ts = time.time()
t = 0
max_iteration = 2
while t < max_iteration:
    best_solution, bins, best_solution_cost, residual_capacity = tabu_search(best_solution, bins, residual_capacity,
                                                                             capacity_bins, n_bins, cost_bins, n_items,
                                                                             n_compulsory_items, profit_items,
                                                                             weight_items, max_budget)
    t = t + 1
    print(f"Number of iterations of the Tabu Search: {t}.")
    print(best_solution_cost)
    print(bins)
end_ts = time.time()
comp_time_ts = end_ts - start_ts + end_nf - start_nf
print(f"Computational time of NextFit and Tabu Search: {comp_time_ts} s.")

# to find a solution in a relatively short time, we set the following parameters in this way
size_pop = 200
perc_size_pop = 50
max_iter_while = 100

# if we wanted to move far from the initial solution towards a global minimum
# we have to accept that the algorithm takes longer time and we set the parameters like this
# size_pop = 500
# perc_size_pop = 200
# max_iter_while = 100

start_ga = time.time()
minimum, population = genetic_algorithm(size_pop, perc_size_pop, max_iter_while, pop_nf, capacity_bins, cost_bins,
                                        n_bins, weight_items, profit_items, n_items, n_compulsory_items, max_budget)
end_ga = time.time()
comp_time_ga = end_ga - start_ga + end_nf - start_nf
print(f"Computational time of NextFit and Genetic Algorithm: {comp_time_ga} s.")
