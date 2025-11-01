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

    r = 0.002

    rtol = 1e-6
    atol = 1e-6

    bifurcation_point = np.abs(0.12 - beta0)/r
    t_eval = np.arange(0.0, bifurcation_point, dt)

    t, x1, x2 = hopf_solver(t_eval, x0, beta_func = lambda tt, r = r: beta_func(tt, beta0, r), rtol = rtol, atol = atol)

    t = t[-step_size:]
    x1 = x1[-step_size:].reshape([1, -1])
    x2 = x2[-step_size:].reshape([1, -1])
    X = np.concatenate([x1, x2], axis = 0)

    return t, X 

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
    
def select_parameter_poly(X, type_dmd = 'poly', dim_delay = 400, low_rank = 0.9):
    candidates_gamma = [1.0, 0.1, 0.01]
    candidates_degree = [2.0, 3.0, 4.0]
    
    best_loss = float('inf')
    best_gamma = 0.0
    best_degree = 0
    for gamma in candidates_gamma:
        for degree in candidates_degree:
            koopman_estimater = Koopman_Resilience(X, type_dmd = type_dmd, dim_delay = dim_delay, low_rank = low_rank, kernel_params = {'gamma': gamma, 'coef0': 1, 'degree': degree})
            loss = np.abs(koopman_estimater.resKMD())
            if loss < best_loss:
                best_loss = loss
                best_gamma = gamma
                best_degree = degree

    return best_gamma, best_degree

def calculate_ews(time, X, skip = 50):
    window_length = int(X.shape[1]/2)
    dim_delay = 400

    time_select = []
    ews_var = []
    ews_ac = []
    ews_eig = []
    ews_dl = []
    ews_res1 = []
    ews_res2 = []
    ews_res3 = []
    ews_res4 = []

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

        koopman_byvanilla = Koopman_Resilience(Xwindow, type_dmd = 'vanilla', dim_delay = dim_delay)
        ews_res1.append(np.abs(koopman_byvanilla.resKMD()))

        if i == 0:
            gamma_rbf = select_parameter_rbf_and_laplacian(Xwindow, type_dmd = 'rbf', dim_delay = dim_delay)
        koopman_byrbf = Koopman_Resilience(Xwindow, type_dmd = 'rbf', dim_delay = dim_delay, kernel_params = {'gamma': gamma_rbf})
        ews_res2.append(np.abs(koopman_byrbf.resKMD()))

        if i == 0:
            gamma_laplacian = select_parameter_rbf_and_laplacian(Xwindow, type_dmd = 'laplacian', dim_delay = dim_delay)
        koopman_bylaplacian = Koopman_Resilience(Xwindow, type_dmd = 'laplacian', dim_delay = dim_delay, kernel_params = {'gamma': gamma_laplacian})
        ews_res3.append(np.abs(koopman_bylaplacian.resKMD()))

        if i == 0:
            gamma_poly, degree_poly = select_parameter_poly(Xwindow, type_dmd = 'poly', dim_delay = dim_delay)
        koopman_bypoly = Koopman_Resilience(Xwindow, type_dmd = 'poly', dim_delay = dim_delay, kernel_params = {'gamma': gamma_poly, 'coef0': 1, 'degree': degree_poly})
        ews_res4.append(np.abs(koopman_bypoly.resKMD()))

    time_select = np.array(time_select)
    ews_var = np.array(ews_var)
    ews_ac = np.array(ews_ac)
    ews_eig = np.array(ews_eig)
    ews_dl = np.array(ews_dl)
    ews_res1 = np.array(ews_res1)
    ews_res2 = np.array(ews_res2)
    ews_res3 = np.array(ews_res3)
    ews_res4 = np.array(ews_res4)

    kendall_var, p_var = kendalltau(time_select, ews_var)
    kendall_ac, p_ac = kendalltau(time_select, ews_ac)
    kendall_eig, p_eig = kendalltau(time_select, ews_eig)
    last_dl = np.max(ews_dl)
    kendall_res1, p_res1 = kendalltau(time_select, ews_res1)
    kendall_res2, p_res2 = kendalltau(time_select, ews_res2)
    kendall_res3, p_res3 = kendalltau(time_select, ews_res3)
    kendall_res4, p_res4 = kendalltau(time_select, ews_res4)

    plt.figure()
    plt.plot(time, X[0, :].flatten(), lw = 1.0)
    plt.plot(time, X[1, :].flatten(), lw = 1.0, linestyle = '--')
    plt.xlabel('time')
    plt.ylabel('state')
    plt.grid()
    plt.tight_layout()
    plt.savefig('time_series_hopf.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(X[0, :].flatten(), X[1, :].flatten())
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()
    plt.tight_layout()
    plt.savefig('phase_portrait_hopf.pdf', format = 'pdf')
    plt.close()

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
    plt.title('maximum probability {}'.format(last_dl) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_dl.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_res1, color = 'pink', marker = '^')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_res1) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_res1.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_res2, color = 'cyan', marker = 'v')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_res2) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_res2.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_res3, color = 'blue', marker = '>')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_res3) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_res3.pdf', format = 'pdf')
    plt.close()

    plt.figure()
    plt.plot(time_select, ews_res4, color = 'magenta', marker = '<')
    plt.xlabel('time')
    plt.ylabel('ews')
    plt.title('kendall tau {}'.format(kendall_res4) )
    plt.grid()
    plt.tight_layout()
    plt.savefig('ews_res4.pdf', format = 'pdf')
    plt.close()

if __name__ == "__main__": 
    t, X = hopf_simulation()
    calculate_ews(t, X)