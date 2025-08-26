import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.integrate import solve_ivp

def beta_func(t, beta0, rate):
    return beta0 - rate*t

def F_saddlenode(t, x, beta_func):
    return -(x[0] + 1.0)*((x[0] - 1.0)**2 - beta_func(t)) + 0.01*np.random.normal(0, 1)

def saddlenode_solver(t_eval, x0, beta_func, rtol, atol):
    solver = solve_ivp(lambda t, x: F_saddlenode(t, x, beta_func), (t_eval[0], t_eval[-1]), [x0], t_eval = t_eval, rtol = rtol, atol = atol)

    return solver.t, solver.y[0]

def saddlenode_simulation():
    x0 = 1.8
    beta0 = 1.0
    dt = 0.01
    step_size = 2000
    T = step_size*dt

    # for bifurcation simulation
    list_rate = [
        0.001,
        0.0015,
        0.002,
        0.003,
        0.004,
        0.006,
        0.01,
        0.015,
        0.02,
        0.03
    ]

    # for non-bifurcation simulation
    list_init = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.08,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5
    ]

    rtol = 1e-6
    atol = 1e-6

    X_dataset = []
    time_dataset = []
    label_dataset = []

    for r in list_rate:
        bifurcation_point = beta0/r 
        t_eval = np.arange(0.0, bifurcation_point, dt)

        t, x = saddlenode_solver(t_eval, x0, beta_func = lambda tt, r = r: beta_func(tt, beta0, r), rtol = rtol, atol = atol)

        X_dataset.append(x[-step_size:])
        time_dataset.append(t[-step_size:])
        label_dataset.append(1)

    for beta_init in list_init:
        T_end = T + 200.0
        t_eval = np.arange(0.0, T_end, dt)

        t, x = saddlenode_solver(t_eval, x0, beta_func = lambda tt, c = beta_init: c, rtol = rtol, atol = atol)

        X_dataset.append(x[-step_size:])
        time_dataset.append(t[-step_size:])
        label_dataset.append(0)

    X_dataset = np.stack(X_dataset, axis = 0)
    time_dataset = np.stack(time_dataset, axis = 0)
    label_dataset = np.array(label_dataset, dtype = int)

    rows = []
    for i in range(20):
        for j in range(step_size):
            rows.append({
                'id': i,
                'label': int(label_dataset[i]),
                't': float(time_dataset[i, j]),
                'x': float(X_dataset[i, j])
            })

    df_saddlenode = pd.DataFrame(rows)
    csv_path = '../alldata_saddlenode.csv'
    df_saddlenode.to_csv(csv_path, index = False)

if __name__ == "__main__": 
    saddlenode_simulation()