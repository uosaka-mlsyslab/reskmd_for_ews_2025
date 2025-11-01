import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.integrate import solve_ivp

def beta_func(t, beta0, rate):
    return beta0 + rate*t

def F_hopf(t, x, beta_func):
    x1, x2 = x
    r = x1**2 + x2**2
    dx1 = beta_func(t)*x1 - 2*np.pi*x2 - x1*r*(r - 1.0) + 0.001*np.random.normal(0, 1)
    dx2 = 2*np.pi*x1 + beta_func(t)*x2 - x2*r*(r - 1.0) + 0.001*np.random.normal(0, 1)
    return [dx1, dx2]

def hopf_solver(t_eval, x0, beta_func, rtol, atol):
    solver = solve_ivp(lambda t, x: F_hopf(t, x, beta_func), (t_eval[0], t_eval[-1]), x0, t_eval = t_eval, rtol = rtol, atol = atol)

    return solver.t, solver.y[0], solver.y[1]

def hopf_simulation():
    x0 = [0.1, 0.0]
    beta0 = -1.0
    dt = 0.01
    step_size = 2000
    T = step_size*dt

    # for bifurcation simulation
    list_rate = [
        0.001,
        0.0015,
        0.002,
        0.0025,
        0.003,
        0.0035,
        0.004,
        0.0045,
        0.005,
        0.0055
    ]

    # for non-bifurcation simulation
    list_init = [
        -0.9,
        -0.8,
        -0.7,
        -0.6,
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        -0.05
    ]

    rtol = 1e-6
    atol = 1e-6

    X1_dataset = []
    X2_dataset = []
    time_dataset = []
    label_dataset = []

    for r in list_rate:
        bifurcation_point = (0.12 - beta0)/r 
        t_eval = np.arange(0.0, bifurcation_point, dt)

        t, x1, x2 = hopf_solver(t_eval, x0, beta_func = lambda tt, r = r: beta_func(tt, beta0, r), rtol = rtol, atol = atol)

        X1_dataset.append(x1[-step_size:])
        X2_dataset.append(x2[-step_size:])
        time_dataset.append(t[-step_size:])
        label_dataset.append(1)

    for beta_init in list_init:
        T_end = T + 200.0
        t_eval = np.arange(0.0, T_end, dt)

        t, x1, x2 = hopf_solver(t_eval, x0, beta_func = lambda tt, c = beta_init: c, rtol = rtol, atol = atol)

        X1_dataset.append(x1[-step_size:])
        X2_dataset.append(x2[-step_size:])
        time_dataset.append(t[-step_size:])
        label_dataset.append(0)

    X1_dataset = np.stack(X1_dataset, axis = 0)
    X2_dataset = np.stack(X2_dataset, axis = 0)
    time_dataset = np.stack(time_dataset, axis = 0)
    label_dataset = np.array(label_dataset, dtype = int)

    rows = []
    for i in range(20):
        for j in range(step_size):
            rows.append({
                'id': i,
                'label': int(label_dataset[i]),
                't': float(time_dataset[i, j]),
                'x1': float(X1_dataset[i, j]),
                'x2': float(X2_dataset[i, j])
            })

    df_hopf = pd.DataFrame(rows)
    csv_path = '../alldata_hopf.csv'
    df_hopf.to_csv(csv_path, index = False)

    fig, axes = plt.subplots(4, 5, figsize=(18, 10), sharex=False, sharey=False)
    axes = axes.ravel()
    for i in range(20):
        ax = axes[i]
        ax.plot(time_dataset[i], X1_dataset[i], lw=1.0, label='x(t)')
        ax.plot(time_dataset[i], X2_dataset[i], lw=1.0, linestyle='--', label='y(t)')
        ax.set_xlabel("time"); ax.set_ylabel("state")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle("All cases: x(t) & y(t) vs time (last 2000 points per case)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.92)
    plot_path = "all_hopf_cases_xy_vs_time.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"saved figure: {plot_path}")

if __name__ == "__main__": 
    hopf_simulation()