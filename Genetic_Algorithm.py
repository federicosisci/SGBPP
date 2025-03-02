import copy
import random
import numpy as np


def genetic_algorithm(size_population, perc_size_population, max_iter_while, pop_nf, capacity_bins, cost_bins, n_bins,
                      weight_items, profit_items, n_items, n_compulsory_items, max_budget):

    # first we define function used later in the algorithm

    # this function checks if the population found after mutation or crossover is feasible,
    # i.e. if the constraints of the minimization problem are respected
    def is_feasible(solution):
        is_feasible_solution = 0
        c_b = cost_b

        # we check if length of new solution is different from number of item
        if len(solution) != n_i:
            is_feasible_solution = 1

        else:
            # we check in the new solution if the sum of items' weight in a bin is bigger than the capacity of the bin
            # we create a vector as long as the number of bins. The i position in the vector is the i bin and
            # we sum the weight of items collected by i bin
            bins = [0] * n_b
            for i in range(n_i):
                if solution[i] != 0:
                    bins[solution[i] - 1] = bins[solution[i] - 1] + w_i[i]

            j = 0
            # we check this for each bin
            for binWeight in bins:
                if binWeight > cap_b[j]:
                    is_feasible_solution = 1
                    break
                j = j + 1

            # we create a vector as long as the number of bins
            # if the i-th position has value 1, the bin is used, 0 otherwise
            yes_bins = [0] * n_b
            for i in range(len(solution)):
                if solution[i] != 0:
                    yes_bins[solution[i] - 1] = 1

            # we check if the total cost of the bins used by the new solution is bigger than the
            # max budget we have available
            cost_bins = 0
            cost_bins = sum(c_b[j] * yes_bins[j] for j in yes_bins)
            if cost_bins > m_b:
                is_feasible_solution = 1

            # this check is done after my population size is bigger than a certain value to have more elasticity of
            # finding items, we do this check after a fixed dimension of our population.
            # In particular, the first time we do this check, we control that the value of the new solution is smaller
            # than the first value found because the value of the last population can be bigger than the starting point
            if len(population) > perc_size_population:
                if (len(population)-perc_size_population) == 1:
                    val, bins = objective_function(solution, len(yes_bins), c_b, p_i, n_i, n_i_c)
                    if val > value_obj_fun[0]:
                        is_feasible_solution = 1
                else:
                    val, bins = objective_function(solution, len(yes_bins), c_b, p_i, n_i, n_i_c)
                    if val > last_value:
                        is_feasible_solution = 1

        return is_feasible_solution

    # function used in mutate function
    def change_random_gene(all_population, rand_index, not_try, it, bin_el):
        # we create a copy of population passed, then we save the element modified at the end of the list
        pop_mut = copy.deepcopy(all_population)
        # we select the element of the population following the random index chosen in the mutation function
        rand_pop = pop_mut[rand_index]
        # we select a new random number corresponding to an item in rand_pop
        rand_gene = random.randint(0, (len(rand_pop) - 1))

        # the first time the algorithm uses this function we have it=0, so we don't use the while cycle but
        # just save the item's index tried
        not_try.append(rand_gene)

        # from the second time, we have it=1 so we enter in the while cycle
        # we change the item's index as long as it's in the not_try vector, of course if we have already used it
        while it == 1 and (rand_gene in not_try):
            rand_gene = random.randint(0, (len(rand_pop) - 1))

        # we save the old bin that collected the item selected
        old_gene_value = rand_pop[rand_gene]
        # we choose a new bin that collect the item selected
        new_gene_value = random.randint(1, n_b)

        # we check if the old bin and the new bin is the same
        # if it's so, we repeat the choice of the new bin until this is different to the old bin
        while new_gene_value == old_gene_value or new_gene_value in bin_el:
            new_gene_value = random.randint(1, n_b)

        rand_pop[rand_gene] = new_gene_value
        return pop_mut

    # function to mutate the bin selected in a specific population
    def mutate(population, bin_el):
        # vector to save index in the function change_random_gene
        # the aim is to not try item's index that we have already used
        # it=0 for the first time we use the function change_random_gene
        not_try = []
        it = 0

        # we choose a random index to select a random element of the population
        rand_index = random.randint(0, (len(population) - 1))
        # res is result of the core of mutation operation and it's the new population with the mutated element at the
        # end of the list
        res = change_random_gene(population, rand_index, not_try, it, bin_el)
        counter = 0

        # we set a counter parameter because we set a limit to the number of iterations of the while cycle that checks
        # the feasibility until the result is unfeasible, we repeat the function that change the bin selected
        while is_feasible(res[rand_index]) == 1 and counter < max_iter_while:
            # it=1 from the second time we use the function change_random_gene
            it = 1
            res = change_random_gene(population, rand_index, not_try, it, bin_el)
            counter += 1

        return res

    # function used in the crossover function
    def pick_two_parents(parents):
        # in this vector we collect the two positions of the parents
        different_parents = []
        no_of_parents = len(parents)
        p1 = p2 = 0

        # we select the position of first and second parents to crossover later
        # we repeat this until the two position are the same
        while p1 == p2:
            p1 = random.randint(0, (no_of_parents - 1))
            p2 = random.randint(0, (no_of_parents - 1))

        different_parents.append(p1)
        different_parents.append(p2)
        return different_parents

    # function to do the crossover of two elements of the population
    def crossover(parents):
        # we choose a random point to apply the crossover between two element of the population
        point = random.randint(1, (len(parents[0]) - 1))
        parent_length = len(parents[0])
        p1 = p2 = 0
        child = []
        counter = 0

        # the while cycle has the same role of the while cycle in the mutate function, we check that the new solution
        # child is feasible
        while is_feasible(child) == 1 and counter < max_iter_while:
            child = []
            # we collect the positions in the population of the two element to crossover and we save them in p1 and p2
            parents_to_crossover = pick_two_parents(parents)
            p1 = parents[parents_to_crossover[0]]
            p2 = parents[parents_to_crossover[1]]

            # then, we save the two portion in the vector child and then we append this to our population
            for i in range(0, parent_length):
                if i < point:
                    child.append(p1[i])
                else:
                    child.append(p2[i])

            counter += 1
        parents.append(child)

    # function used for calculating the value of objective function of our minimization problem
    def objective_function(solution, n_b, cost_b, p_i, n_i, n_i_c):
        bins = [0] * n_b
        solution_matrix = np.zeros((n_i, n_b))
        costo_bins = 0

        # vector of bins used:
        # 1 if the i-th bins is used or 0 otherwise
        for i in range(n_i):
            if solution[i] != 0:
                bins[solution[i] - 1] = 1

        # matrix of items collected by bins:
        # 1 if the item j is collected by bin i or 0 otherwise
        for i in range(n_b):
            for j in range(n_i):
                if solution[j] == i+1:
                    solution_matrix[j, i] = 1

        # structure of objective function
        for j in range(n_b):
            costo_bins = costo_bins + cost_b[j] * bins[j]
            for i in range(n_i_c, n_i):
                costo_bins = costo_bins - p_i[i] * solution_matrix[i, j]

        return costo_bins, bins

    # this function randomly take the bins we have to empty, save them in the bin_el list and create a list add of
    # compulsory items to be reallocated. At the end we update the solution vector
    def bin_mutation(solution, max_bg):
        bin_el = []
        add = []
        bg =0
        for j in range(len(cost_bins)):
            bg = bg + cost_bins[j]
        while bg > max_bg:
            c = random.randint(0, n_bins - 1)
            if c not in bin_el:
                bin_el.append(c)
                bg = bg - cost_bins[c]
                for i in range(n_i_c):
                    if solution[i] == c+1:
                        solution[i] = 0
                        add.append(i)
                for k in range(n_i_c, n_i):
                    if solution[k] == c+1:
                        solution[k] = 0
        return add, solution, bin_el


    # start of the Genetic Algorithm
    # first we copy some elements from file istances passed by input to not modify elements of this file
    cap_b = capacity_bins.copy()
    n_b = n_bins
    w_i = weight_items.copy()
    n_i = n_items
    n_i_c = n_compulsory_items
    cost_b = cost_bins.copy()
    p_i = profit_items.copy()
    m_b = max_budget

    # the first element of the population is the result "allocation" of Next_Fit.py
    # a row vector long as the number of items in it where the position i is the item and the number in this position
    # is the bin used
    # there's a zero if the item is not collected

    first_pop = pop_nf.copy()
    it = 0

    # now we have to empty some bins to satisfy the budget constraint
    add, solution, bin_el = bin_mutation(first_pop, m_b)
    for j in range(n_b):
        if j not in bin_el:
            for i in add:
                for k in range(n_i_c, n_i):
                    if solution[k] == j+1:
                        solution[k] = 0
                        solution[i] = j + 1
                        break
    population = [solution]

    value_obj_fun = []
    min_bin = []
    # we calculate the value of our objective function
    value, bins = objective_function(solution, n_b, cost_b, p_i, n_i, n_i_c)
    # and append it the vector value_obj_fun to collect them at each iteration
    value_obj_fun.append(value)
    min_bin.append(bins)
    print(bins)
    first_el_pop = population[0]

    # in this while cycle we do the two main operation of the genetic algorithm: mutation and crossover
    # furthermore we calculate the value of the value function after this operation and save the smaller value
    while len(population) < size_population:

        last_value = value_obj_fun[len(value_obj_fun) - 1]
        population = mutate(population, bin_el)

        # at first iteration, we only do mutation because
        # the crossover operation needs more of two elements in population for working good
        if it == 0:

            population.append(first_el_pop)
            value, bins = objective_function(population[len(population)-1], n_b, cost_b, p_i, n_i, n_i_c)
            value_obj_fun.append(value)
            min_bin.append(bins)

        if it > 0:

            value, bins = objective_function(population[len(population)-1], n_b, cost_b, p_i, n_i, n_i_c)
            value_obj_fun[len(value_obj_fun)-1] = value
            crossover(population)
            # it's possible that the result of crossover is the same of an element that already is in the population
            # for this reason we use the following operation to remove the duplicates in the population vector
            # maintaining the order of the elements
            population[:] = [x for i, x in enumerate(population) if i == population.index(x)]
            value, bins = objective_function(population[len(population)-1], n_b, cost_b, p_i, n_i, n_i_c)
            value_obj_fun.append(value)
            min_bin.append(bins)

        # print(len(population))
        minim = min(value_obj_fun)
        i_min = value_obj_fun.index(minim)

        # because the two vectors value_obj_fun and min_bin have the same size (they are updated togheter)
        # we can use the same index i_min to find the vector of bins related to the minimum value
        bin_min = min_bin[i_min]
        it += 1

    # The initial value matches to the objective function's value of the Next_Fit's solution
    print(f"initial value {value_obj_fun[0]}, minimum value {minim}, iterations' number {it}")
    print(bin_min)
    return minim, population