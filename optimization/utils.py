import numpy as np
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
from pyomo.environ import *

warnings.filterwarnings('ignore')

def plot_model(model, include_obj=True):

    N = 100
    xs = np.linspace(0, 4, N)
    ys = np.linspace(0, 4, N)

    model = deepcopy(model)

    X, Y = np.meshgrid(xs, ys)

    if include_obj:
        # Evaluate the Pyomo function dynamically for 2D plotting
        obj = np.zeros_like(X)
        for i in range(N):
            for j in range(N):
                model.x.set_value(xs[i])
                model.y.set_value(ys[j], skip_validation=True)  # Skip domain validation
                obj[i,j] = model.obj()

        plt.contourf(X, Y, obj, levels=30, cmap="viridis")
        plt.colorbar(label="Objective Value")

    for c in model.component_objects(Constraint, active=True):
        con = np.zeros_like(X)
        for i in range(N):
            for j in range(N):
                model.x.set_value(xs[i])
                model.y.set_value(ys[j], skip_validation=True)  # Skip domain validation
                con[i,j] = c() > 0

    plt.scatter(X,Y, c=con, alpha=0.05, cmap='Grays')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Optimization with Pyomo")
    plt.yticks([0,1,2,3,4])
    plt.grid()
