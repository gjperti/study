from pyomo.environ import *
from utils import *
from copy import deepcopy

def outer_approx(model, nlp_solver_name='gorubi', milp_solver_name='gorubi'):

    # Initializing solvers
    nlp_solver = SolverFactory(nlp_solver_name)
    milp_solver = SolverFactory(milp_solver_name)

    # Get variable information
    var_info ={'integer':[], 'real':[]}
    for v in model.component_objects(Var, active=True):
        if v.domain == Integers:
            var_info['integer'].append(v)
        else:
            var_info['real'].append(v)

    solve_nlp(model, nlp_solver, var_info)
    return var_info

    


def solve_nlp(model, solver, var_info):

    nlp = model

    for var in var_info['integer']:
        var.domain = Reals

    result = solver.solve(nlp)
    

    