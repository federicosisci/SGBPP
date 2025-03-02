import numpy as np


def tabu_search(solution, bins_nf, residual_capacity_bins, capacity_bins, n_bins, cost_bins, n_items,
                n_compulsory_items, profit_items, weight_items, max_budget):
    bins = bins_nf.copy()
    c_bin = capacity_bins.copy()
    best_solution = solution.copy()

    # function that calculate the cost of every solution
    def tot_cost(b_s, camion):
        cost = 0
        for j in range(n_bins):
            cost = cost + cost_bins[j] * camion[j]
            for i in range(n_items):
                cost = cost - profit_items[i] * b_s[i, j]
        return cost

    cost = tot_cost(best_solution, bins)

    # function that "put out" an item j from the bin s and add the new item i
    def put_out(b_s, i, j, s):
        b_s[i, s] = 1
        b_s[j, s] = 0
        return b_s

    # this for cycle substitute the not compulsory items in the bins with items left out with a higher profit
    # and that have a weight < residual capacity + the element "put out"
    for a in range(n_compulsory_items, n_items):
        for b in range(n_compulsory_items, n_items):
            for c in range(n_bins):
                if bins[c] == 1:
                    if best_solution[a, c] == 1:
                        if all(n == 0 for n in best_solution[b,]):
                            if profit_items[a] < profit_items[b]:
                                if residual_capacity_bins[c] >= weight_items[b] - weight_items[a]:
                                    s_po = put_out(best_solution, b, a, c)
                                    if tot_cost(s_po, bins) < cost:
                                        cost = tot_cost(s_po, bins)
                                        best_solution = s_po.copy()
                                        residual_capacity_bins[c] = residual_capacity_bins[c] + weight_items[
                                            a] - weight_items[b]

    # function that "swap" two items i,j in two bins k,s
    def swap(b_s, i, k, j, s):
        b_s[i, k] = 0
        b_s[j, s] = 0
        b_s[i, s] = 1
        b_s[j, k] = 1
        return b_s

    # this for cycle "swap" two different items in two different bins in order to create more space in one of the two
    # bin. After it try to insert one left out item in the bin
    for a in range(n_items):
        for b in range(n_items):
            for c in range(n_bins):
                for d in range(n_bins):
                    if a != b:
                        if c != d:
                            if best_solution[a, c] == 1:
                                if best_solution[b, d] == 1:
                                    if weight_items[a] > weight_items[b]:
                                        if residual_capacity_bins[d] >= weight_items[a] - weight_items[b]:
                                            s_sw = swap(best_solution, a, c, b, d)
                                            best_solution = s_sw.copy()
                                            residual_capacity_bins[d] = residual_capacity_bins[d] + weight_items[b] - \
                                                                        weight_items[a]
                                            residual_capacity_bins[c] = residual_capacity_bins[c] + weight_items[a] - \
                                                                        weight_items[b]
                                            for e in range(n_items):
                                                if all(n == 0 for n in best_solution[e,]):
                                                    if residual_capacity_bins[c] >= weight_items[e]:
                                                        solution = best_solution.copy()
                                                        solution[e, c] = 1
                                                        if cost > tot_cost(solution, bins):
                                                            best_solution = solution.copy()
                                                            cost = tot_cost(best_solution, bins)
                                                            residual_capacity_bins[c] = residual_capacity_bins[c] - \
                                                                                        weight_items[e]

    # function that count the single contribution of the bin j to the total cost
    def profit_bin(b_s, j):
        prof_bin = cost_bins.copy()
        for i in range(n_items):
            if b_s[i, j] == 1:
                prof_bin[j] = prof_bin[j] - profit_items[i]
        return prof_bin[j]

    # function that create a binary vector with 1 if the compulsory item in position i was in bin k and 0 otherwise
    # and set the solution's column of the bin k =0 for every item
    def erase_bin(b_s, k):
        add = np.zeros(n_compulsory_items)
        for i in range(n_compulsory_items):
            if b_s[i, k] == 1:
                add[i] = 1
        b_s[:, k] = 0
        return b_s, add

    # function that find the not compulsory items of the selected bin that cause a lower "lost" of profit if they
    # would be replaced by the compulsory item j
    # nb: r = residual capacity of the bin
    def best_option(lis_i, lis_p, lis_w, j, r):
        lis_m_1 = []
        t = 0
        l_1 = lis_p.copy()
        r_1 = r
        # create a list with the not compulsory items ordered from the one with lower profit to the higher one
        while any(n != 100 for n in l_1):
            m_1 = min(l_1)
            ind_1 = l_1.index(m_1)
            l_1[ind_1] = 100  # fictitious value
            lis_m_1.append(lis_i[ind_1])
            t = t + 1
        f_1 = []
        length_1 = 0
        best_1 = 0
        # create a list with the items with lower profit that are needed to create enough space on the bin
        # and save the lost profit in the variable best_1
        for s in range(t):
            p = lis_m_1[s]
            if weight_items[j] < r_1 + weight_items[p]:
                f_1.append(p)
                best_1 = best_1 + profit_items[p]
                length_1 = length_1 + 1
                break
            else:
                r_1 = r_1 + weight_items[p]
                f_1.append(p)
                best_1 = best_1 + profit_items[p]
                length_1 = length_1 + 1

        lis_m_2 = []
        l_2 = lis_w.copy()
        r_2 = r
        # create a list with the not compulsory items ordered from the one with higher weight to the lower one
        while any(n != -1 for n in l_2):
            m_2 = max(l_2)
            ind_2 = l_2.index(m_2)
            l_2[ind_2] = -1  # fictitious value
            lis_m_2.append(lis_i[ind_2])
        f_2 = []
        length_2 = 0
        best_2 = 0
        # create a list with the items with higher weight that are needed to create enough space on the bin
        # and save the lost profit in the variable best_2
        for s in range(t):
            p = lis_m_2[s]
            if weight_items[j] < r_2 + weight_items[p]:
                f_2.append(p)
                best_2 = best_2 + profit_items[p]
                length_2 = length_2 + 1
                break
            else:
                r_2 = r_2 + weight_items[p]
                f_2.append(p)
                best_2 = best_2 + profit_items[p]
                length_2 = length_2 + 1

        # take the option with lower loss of profit
        if best_1 <= best_2:
            best = best_1
            f = f_1.copy()
            length = length_1
        else:
            best = best_2
            f = f_2.copy()
            length = length_2

        return best, f, length

    # function that insert all the compulsory item saved in the add vector in the other active bins trying to obtain the
    # lowest loss of profit
    def swap_out(s, add, r_c_b, c, b):
        sol = s.copy()
        error = 0
        for j in range(n_compulsory_items):
            if add[j] == 1:
                lis_b = []
                lis_e = []
                lis_g = []
                lis_l = []
                for k in range(n_bins):
                    if k != c:
                        lis_i = []
                        lis_p = []
                        lis_w = []
                        t = 0
                        w = 0
                        if b[k] != 0:
                            # create the list with the not compulsory items in the bin k and their profit and weight
                            for i in range(n_compulsory_items, n_items):
                                if sol[i, k] == 1:
                                    lis_i.append(i)
                                    lis_p.append(profit_items[i])
                                    lis_w.append(weight_items[i])
                                    t = t + 1
                                    w = w + weight_items[i]
                            # if this bin don't have "not compulsory" items go on with the for cycle and change bin
                            if len(lis_i) == 0:
                                continue
                            # else if there is enough space pulling out all the not compulsory items apply the
                            # Best_Option function and save the output for every bin k in the lists
                            elif r_c_b[k] + w >= weight_items[j]:
                                best, elenco, lg = best_option(lis_i, lis_p, lis_w, j, r_c_b[k])
                                lis_b.append(best)
                                lis_e.append(elenco)
                                lis_g.append(k)
                                lis_l.append(lg)

                # if the list is empty it means that there is no bin with enough space to take the compulsory item j
                # that means that the bin c can't be eliminated
                if len(lis_b) == 0:
                    error = 1
                    break
                # else we take the bin with the lowest loss of profit, then we put the compulsory item j in this bin
                # and take off the items in the list "oggetti" from the bin
                else:
                    minimo = min(lis_b)
                    ind = lis_b.index(minimo)
                    camion = lis_g[ind]
                    oggetti = lis_e[ind]
                    length = lis_l[ind]
                    add[j] = 0
                    s[j, camion] = 1
                    r_c_b[camion] = r_c_b[camion] - weight_items[j]
                    for r in range(length):
                        ogg_f = oggetti[r]
                        ogg = int(ogg_f)
                        s[ogg, camion] = 0
                        r_c_b[camion] = r_c_b[camion] + weight_items[ogg]
        return s, r_c_b, error

    # this function eliminate the bin if there is enough space in the other active bin for the compulsory items that
    # were inside that bin
    def eliminate(sol, residual_capacity_bin, bins_vet, camion):
        s, add = erase_bin(sol, camion)
        b_s_w, r_c_b_so, error = swap_out(s, add, residual_capacity_bin, camion, bins_vet)
        if error == 0:
            bins_vet[camion] = 0
            r_c_b_so[camion] = c_bin[camion]
            b_s = b_s_w.copy()
            r_c_b = r_c_b_so.copy()
        else:
            bins_vet[camion] = -1
            b_s = sol.copy()
            r_c_b = residual_capacity_bin.copy()
        return b_s, r_c_b, bins_vet

    # here we try to eliminate the bins that costs more than the profit that they get from the items inside them
    for c in range(n_bins):
        if bins[c] == 1:
            if profit_bin(best_solution, c) > 0:
                best_solution_el, residual_capacity_bins_el, bins_el = eliminate(best_solution, residual_capacity_bins,
                                                                                 bins, c)
                # if there were no error caused by the reassignment of the compulsory items
                if bins_el[c] != -1:
                    # and if the cost decreased, then this is our new best solution
                    if tot_cost(best_solution_el, bins_el) <= cost:
                        cost = tot_cost(best_solution_el, bins_el)
                        best_solution = best_solution_el.copy()
                        bins = bins_el.copy()
                        residual_capacity_bins = residual_capacity_bins_el.copy()
                    # else is useless eliminate this bin so we keep it
                    else:
                        bins[c] = 1
                # else we have to keep the bin
                else:
                    bins[c] = 1

    # this function find the best combination of left out items able to substitute an item where w is the residual
    # capacity of the bin plus the weight of the item we want to get out
    def all_combination(solution, w):
        # first of all we save all the left out item and their weight in 2 lists, after we order them from the one with
        # lower weight to the higher one's
        lis_lo = []
        lis_lo_w = []
        for i in range(n_compulsory_items, n_items):
            if all(n == 0 for n in solution[i, ]):
                lis_lo.append(i)
                lis_lo_w.append(weight_items[i])
        lis_lo_w2 = lis_lo_w.copy()
        l_lo = []
        while any(n != 100 for n in lis_lo_w2):
            w_min = min(lis_lo_w2)
            ind_min = lis_lo_w2.index(w_min)
            lis_lo_w2[ind_min] = 100  # fictitious value
            l_lo.append(lis_lo[ind_min])
        # l_w is a list with all the left out item with weight lower than w ordered from the lightest to the heaviest
        l_w = []
        for i in l_lo:
            if weight_items[i] <= w:
                l_w.append(i)
        if len(l_w) == 0:
            sol_tot = [0]
            m_tot = [0]
        elif len(l_w) == 1:
            for a in l_w:
                sol_tot = [a]
                m_tot = [profit_items[a]]
        else:
            # here we obtain all the combination with total weight lower than w and we save them in list_k and the
            # respective total profit in list_p
            sol = []
            sol_p = []
            for j in l_w:
                o = l_w.index(j)
                y = o
                w_j = weight_items[j]
                # here we take all the combination with total weight lower than w that start with the item j
                s = np.zeros(len(l_w))
                s[o] = 1
                while y + 1 < len(l_w):
                    k = l_w[y + 1]
                    if w_j + weight_items[k] <= w:
                        w_j = w_j + weight_items[k]
                        e = l_w.index(k)
                        s[e] = 1
                        y = e
                    else:
                        # this mean that we have reached the maximal weight and that we have to try the next combination
                        # taking away the last element saved in the list and try to substitute it with the next one
                        prof = 0
                        it = []
                        list_p = []
                        list_k = []
                        last = [o, j]  # vector where we save the position of the item and the item that was last taken
                        for p in range(len(s)):
                            # here we save the combination
                            if s[p] == 1:
                                k_it = l_w[p]
                                prof = prof + profit_items[k_it]
                                it.append(k_it)
                                last = [p, k_it]  # since our items are ordered from the lightest to the heaviest and
                                # that we go in order the last element is the last with a 1
                        list_p.append(prof)
                        list_k.append(it)
                        y = last[0]
                        s[y] = 0
                        w_j = w_j - weight_items[last[1]]
                        if y == o:
                            # this mean that we have finished the combination that starts with the item j
                            break

                # then for each initial items j we found the combination with higher profit
                m = max(list_p)
                ind = list_p.index(m)
                b = list_k[ind]
                sol.append(b)
                sol_p.append(m)

            # finally we do the same for all the initial items and find the best combination sol_tot and the relative
            # total profit m_tot
            m_tot = max(sol_p)
            ind_tot = sol_p.index(m_tot)
            sol_tot = sol[ind_tot]
        return sol_tot, m_tot

    # in this for cycle we try to exit from the local best solution trying to find new items that combined give
    # a lower cost
    for j in range(n_bins):
        # we save all the not compulsory items inside the bin j ordering them from the one with higher weight to the
        # lower one
        list_items = []
        list_weight = []
        for i in range(n_compulsory_items, n_items):
            if best_solution[i, j] == 1:
                list_items.append(i)
                list_weight.append(weight_items[i])
        if len(list_items) != 0:
            l_w = list_weight.copy()
            l_i = []
            while any(n != -1 for n in l_w):
                w = max(l_w)
                ind = l_w.index(w)
                l_w[ind] = -1  # fictitious value
                l_i.append(list_items[ind])

            # in the for cycle we run all the combination to try to substitute the items in the bin j
            for t in l_i:
                w_i = residual_capacity_bins[j] + weight_items[t]
                l, p = all_combination(best_solution, w_i)
                if p > profit_items[t]:
                    best_solution[t, j] = 0
                    residual_capacity_bins[j] = residual_capacity_bins[j] + weight_items[t]
                    for d in l:
                        best_solution[d, j] = 1
                        residual_capacity_bins[j] = residual_capacity_bins[j] - weight_items[d]
        cost = tot_cost(best_solution, bins)

    # this function try to empty enough bin in order to respect the budget constrain
    def budget_constrain(max_bg, budget, best_sol, rc, bb):
        list_budget = []
        list_b = []
        # we create an ordered list with the profit of all the active bin from the highest to the lowest
        for j in range(n_bins):
            if bins[j] != 0:
                p_b = profit_bin(best_sol, j)
                list_budget.append(p_b)
                list_b.append(j)
        lg = list_budget.copy()
        bg_bins = []
        while any(n != -400 for n in lg):
            bg = max(lg)
            ind_bg = lg.index(bg)
            bg_bins.append(list_b[ind_bg])
            lg[ind_bg] = -400
        tb = 0
        temp_sol = best_sol.copy()  # temporary best solution
        temp_rcb = rc.copy()    # temporary residual capacity vector
        temp_bb = bb.copy()     # temporary vector of the bin
        temp_budget = budget
        err_b = 0
        h = 0
        # here we try to remove the bin with the higher profit in order to increase the objective function with the
        # lowest possible value
        if any(cost_bins[n] >= budget-max_budget for n in bg_bins):
            # we don't want to risk to empty 2 bins instead of only 1
            for k in bg_bins:
                if cost_bins[k] >= budget-max_budget:
                    bs_g, rcb_b, bins_b = eliminate(temp_sol, temp_rcb, temp_bb, k)
                    if bins_b[k] != -1:
                        temp_sol = bs_g.copy()
                        temp_rcb = rcb_b.copy()
                        temp_bb = bins_b.copy()
                        temp_budget = temp_budget - cost_bins[k]
                        break
        # if none of the bins with cost higher than the quantity of budget we have to free can be eliminated
        # or if there is no bin able the free all this budget by its own
        elif temp_budget > max_bg:
            while temp_budget > max_bg:
                if tb < len(bg_bins):
                    in_b = bg_bins[tb]
                    bs_g, rcb_b, bins_b = eliminate(temp_sol, temp_rcb, temp_bb, in_b)
                    if bins_b[in_b] != -1:
                        temp_budget = temp_budget - cost_bins[in_b]
                        temp_sol = bs_g.copy()
                        temp_rcb = rcb_b.copy()
                        temp_bb = bins_b.copy()
                        tb = tb + 1
                    else:
                        tb = tb +1
                elif h < len(bg_bins):
                    temp_sol = best_sol.copy()
                    temp_rcb = rc.copy()
                    temp_bb = bb.copy()
                    temp_budget = budget
                    h = h + 1
                    tb = h
                else:
                    err_b = 1
                    break
        return temp_bb, temp_sol, temp_rcb, err_b


    budget_bins = 0
    for j in range(n_bins):
        budget_bins = budget_bins + cost_bins[j]*bins[j]

    # we control the budget constrain
    if budget_bins > max_budget:
        list_bins, solution_bg, rcb_bg, error_bg = budget_constrain(max_budget, budget_bins, best_solution,
                                                                    residual_capacity_bins, bins)
        if error_bg != 1:
            bins = list_bins.copy()
            best_solution = solution_bg.copy()
            residual_capacity_bins = rcb_bg.copy()
            cost = tot_cost(best_solution, bins)
        else:
            print('Error in the budget constrain')
    else:
        # function that add bins
        for j in range(n_bins):
            if bins[j] == 0:
                if budget_bins + cost_bins[j] <= max_budget:
                    sol_ac, p = all_combination(best_solution, c_bin[j])
                    if p > cost_bins[j]:
                        bins[j] = 1
                        for e in sol_ac:
                            best_solution[e, j] = 1
                            residual_capacity_bins[j] = residual_capacity_bins[j] - weight_items[e]
                        cost = tot_cost(best_solution, bins)

    # in order to be able to compare the results with Gurobi and the Genetic Algorithm we have to calculate the costs
    # without the compulsory item's profit
    best_solution_cost = cost
    for i in range(n_compulsory_items):
        best_solution_cost = best_solution_cost + profit_items[i]

    # security check for the compulsory items
    for a in range(n_compulsory_items):
        if all(n == 0 for n in best_solution[a,]):
            print('Error in the allocation of the compulsory items')

    return best_solution, bins, best_solution_cost, residual_capacity_bins
