import os
import numpy as np
import gurobipy as grb


def Gurobi(index,n_bins, n_compulsory_items, n_non_compulsory_items, n_items, cost_bins, max_budget, profit_items, weight_items, capacity_bins):
    # we create the model
    model = grb.Model('gbpp')

    # then we add the two variables Y and X to the model
    # Y = 1 if bin j is rented
    Y = model.addVars(
        n_bins,
        vtype=grb.GRB.BINARY,
        name='Y'
    )

    # X = 1 if the item i is collected by bin j
    X = model.addVars(
        n_items, n_bins,
        vtype=grb.GRB.BINARY,
        name='X'
    )

    bins = range(n_bins)
    items = range(n_items)

    # we set the objective function of our minimization problem
    expr = sum(
        cost_bins[j] * Y[j] for j in bins
    )
    expr -= grb.quicksum(profit_items[i] * X[i,j] for i in range(n_compulsory_items, n_items) for j in bins)

    # we add the objective function to the model with the goal to minimize
    model.setObjective(expr, grb.GRB.MINIMIZE)
    model.update()

    # then we add all the constraints of our minimization problem
    # here we have the capacity constraints
    model.addConstrs(
        ( grb.quicksum(weight_items[i]*X[i,j] for i in items) <= capacity_bins[j]*Y[j] for j in bins),
        name="capacity_constraint"
    )

    # here we have the compulsory item constraints
    # a compulsory item have to be collect by one bin
    model.addConstrs(
        ( grb.quicksum(X[i,j] for j in bins) == 1 for i in range(n_compulsory_items)),
        name="compulsory_item"
    )

    # here we have the noncompulsory item constraints
    # a noncompulsory item can be collect by one bin or not
    model.addConstrs(
        ( grb.quicksum(X[i,j] for j in bins) <= 1 for i in range(n_compulsory_items, n_items)),
        name="non_compulsory_item"
    )

    # here we have the budget constraints
    model.addConstr(
        ( grb.quicksum(cost_bins[j]*Y[j] for j in bins) <= max_budget),
        name="budget_constraint"
    )

    # we set some limit:
    # after a % of gap, the process stops
    model.setParam('MIPgap', 0.1)
    # after a time of 360 s, the process stops
    model.setParam(grb.GRB.Param.TimeLimit, 360)
    model.setParam('OutputFlag', 1)

    # Gurobi saves result in a log file
    model.setParam(
        'LogFile',
        os.path.join('.', 'logs', 'gurobi.log')
    )
    model.write(
        os.path.join('.', 'logs', "model.lp")
    )

    model.optimize()

    # we print what set of instances we use and for each bin, the bin used and the compulsory and noncompulsory items in this bin
    if model.status == grb.GRB.Status.OPTIMAL:
        print(f"from initial istances number: {index}")
        for j in bins:
            if Y[j].X > 0.5:

                print(f"Bin {j+1}")
                print("Compulsory items:")
                for i in range(n_compulsory_items):
                    if X[i, j].X > 0.5:

                        print(f"{i+1}")
                print("Non Compulsory items:")
                for i in range(n_compulsory_items, n_items):
                     if X[i, j].X > 0.5:

                        print(f"{i+1}")

    return model
