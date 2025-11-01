import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.stats import kendalltau
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from ews_module import Stochastic_Resilience
from ews_module import MaxEigenvalue_DMD
from ews_module import EWS_DeepLearning
from ews_module import Koopman_Resilience

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
    step_size = 20000
    T = step_size*dt

    r = 0.001

    rtol = 1e-6
    atol = 1e-6

    bifurcation_point = beta0/r
    t_eval = np.arange(0.0, bifurcation_point, dt)

    t, x = saddlenode_solver(t_eval, x0, beta_func = lambda tt, r = r: beta_func(tt, beta0, r), rtol = rtol, atol = atol)

    t = t[-step_size:]
    x = x[-step_size:].reshape([1, -1])

    return t, x

def select_parameter_rbf_and_laplacian(X, type_dmd = 'rbf', dim_delay = 400, low_rank = 0.9):
    candidates_parameter = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    best_loss = float('inf')
    best_param = 0.0
    for param in candidates_parameter:
        koopman_estimater = Koopman_Resilience(X, type_dmd = type_dmd, dim_delay = dim_delay, low_rank = low_rank, kernel_params = {'gamma': param})
        loss = np.abs(koopman_estimater.resKMD())
        if loss < best_loss:
            best_loss = loss
            best_param = param

    return param

def calculate_ews(time, X, skip = 500):
    window_length = int(X.shape[1]/2)
    # window_length = 5000
    # window_length = 1500
    dim_delay = 800

    time_select = []
    ews_var = []
    ews_ac = []
    ews_eig = []
    ews_dl = []
    ews_res = []

    dl_model = EWS_DeepLearning()
    for i in range(int((window_length + 1)/skip)):
        print('NOW ON CALCULATING: {}/{}'.format(i + 1, int((window_length + 1)/skip)))

        time_select.append(time[(i*skip + window_length)])
        Xwindow = X[:, i*skip:(i*skip + window_length)]

        var_model = Stochastic_Resilience(Xwindow, type_ews = 'variance')
        ews_var.append(var_model())

        if Xwindow.shape[0] == 1:
            ac_model = Stochastic_Resilience(Xwindow, type_ews = 'lag1-ac')
            ews_ac.append(ac_model())
        else:
            ac_model = Stochastic_Resilience(Xwindow[0, :].reshape([1, -1]), type_ews = 'lag1-ac')
            ews_ac.append(ac_model())

        ews_eig.append(np.abs(MaxEigenvalue_DMD(Xwindow, dim_delay = dim_delay)))

        if Xwindow.shape[1] > 1500:
            dl_model(Xwindow[0, -1500:].reshape([1, -1]))
        else:
            dl_model(Xwindow[0].reshape([1, -1]))
        ews_dl.append(dl_model.predict())

        if i == 0:
            gamma_rbf = select_parameter_rbf_and_laplacian(Xwindow, type_dmd = 'rbf', dim_delay = dim_delay)
        koopman_byrbf = Koopman_Resilience(Xwindow, type_dmd = 'rbf', dim_delay = dim_delay, kernel_params = {'gamma': gamma_rbf})
        ews_res.append(np.abs(koopman_byrbf.resKMD()))

    time_select = np.array(time_select)
    ews_var = np.array(ews_var)
    ews_ac = np.array(ews_ac)
    ews_eig = np.array(ews_eig)
    ews_dl = np.array(ews_dl)
    ews_res = np.array(ews_res)

    kendall_var, p_var = kendalltau(time_select, ews_var)
    kendall_ac, p_ac = kendalltau(time_select, ews_ac)
    kendall_eig, p_eig = kendalltau(time_select, ews_eig)
    last_dl = np.max(ews_dl)
    kendall_res, p_res = kendalltau(time_select, ews_res)

    plt.figure()
    plt.plot(time_select, ews_var, color = 'red', marker = 'o')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_var) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_var.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_ac, color = 'orange', marker = 'x')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_ac) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_ac.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_eig, color = 'green', marker = 's')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_eig) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_eig.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_dl, color = 'black', marker = '*')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(last_dl) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_dl.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_res, color = 'cyan', marker = 'v')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_res))
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_res.pdf', format = 'pdf')
    plt.close()

if __name__ == "__main__": 
    t, X = saddlenode_simulation()
    X = X + 0.01*np.random.normal(0, 1, size = X.shape)
    calculate_ews(t, X)