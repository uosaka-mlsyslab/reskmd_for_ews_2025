import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import kendalltau
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from ews_module import Stochastic_Resilience
from ews_module import MaxEigenvalue_DMD
from ews_module import EWS_DeepLearning
from ews_module import Koopman_Resilience

def calculate_ews(time, X, skip = 20):
    window_length = int(X.shape[1]/2)

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
        time_select.append(time[:, (i*skip + window_length)])
        Xwindow = X[:, i*skip:(i*skip + window_length)]

        var_model = Stochastic_Resilience(Xwindow, type_ews = 'variance')
        ews_var.append(var_model())

        if Xwindow.shape[0] == 1:
            ac_model = Stochastic_Resilience(Xwindow, type_ews = 'lag1-ac')
            ews_ac.append(ac_model())
        else:
            ac_model = Stochastic_Resilience(Xwindow[0, :].reshape([1, -1]), type_ews = 'lag1-ac')
            ews_ac.append(ac_model())

        ews_eig.append(MaxEigenvalue_DMD(Xwindow, dim_delay = 300))

        if Xwindow.shape[0] == 1:
            dl_model(Xwindow)
            ews_dl.append(dl_model.predict())
        else:
            dl_model(Xwindow[0].reshape([1, -1]))
            ews_dl.append(dl_model.predict())

        koopman_byvanilla = Koopman_Resilience(Xwindow, type_dmd = 'vanilla', dim_delay = 300)
        ews_res1.append(koopman_byvanilla.resKMD())

        koopman_byrbf = Koopman_Resilience(Xwindow, type_dmd = 'rbf', dim_delay = 300, kernel_params = {'gamma': 0.001})
        ews_res2.append(koopman_byrbf.resKMD())

        koopman_bylaplacian = Koopman_Resilience(Xwindow, type_dmd = 'laplacian', dim_delay = 300, kernel_params = {'gamma': 0.001})
        ews_res3.append(koopman_bylaplacian.resKMD())

        koopman_bypoly = Koopman_Resilience(Xwindow, type_dmd = 'poly', dim_delay = 300, kernel_params = {'gamma': 1, 'coef0': 1, 'degree': 4})
        ews_res4.append(koopman_bypoly.resKMD())

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

    return kendall_var, kendall_ac, kendall_eig, last_dl, kendall_res1, kendall_res2, kendall_res3, kendall_res4

def calculate_roc(name):
    kendalls_var = []
    kendalls_ac = []
    kendalls_eig = []
    lasts_dl = []
    kendalls_res1 = []
    kendalls_res2 = []
    kendalls_res3 = []
    kendalls_res4 = []
    labels = []

    df = pd.read_csv('datas/alldata_' + name + '.csv')
    list_id = df['id'].unique()
    for id in list_id:
        print('{}: {}/{}'.format(name, id, len(list_id)))
        df_select = df[(df['id'] == id)].reset_index(drop = True)
        labels.append(df_select['label'].iloc[0])
        time = df_select['t'].to_numpy().reshape([1, -1])
        if name == 'hopf':
            X1 = df_select['x1'].to_numpy().reshape([1, -1])
            X2 = df_select['x2'].to_numpy().reshape([1, -1])
            X = np.concatenate([X1, X2], axis = 0)
        else:
            X = df_select['x'].to_numpy().reshape([1, -1])
        
        var, ac, eig, dl, res1, res2, res3, res4 = calculate_ews(time, X)
        kendalls_var.append(var)
        kendalls_ac.append(ac)
        kendalls_eig.append(eig)
        lasts_dl.append(dl)
        kendalls_res1.append(res1)
        kendalls_res2.append(res2)
        kendalls_res3.append(res3)
        kendalls_res4.append(res4)

    fpr_var, tpr_var, threshold_var = roc_curve(labels, kendalls_var)
    auc_var = roc_auc_score(labels, kendalls_var)
    fpr_ac, tpr_ac, threshold_ac = roc_curve(labels, kendalls_ac)
    auc_ac = roc_auc_score(labels, kendalls_ac)
    fpr_eig, tpr_eig, threshold_eig = roc_curve(labels, kendalls_eig)
    auc_eig = roc_auc_score(labels, kendalls_eig)
    fpr_dl, tpr_dl, threshold_dl = roc_curve(labels, lasts_dl)
    auc_dl = roc_auc_score(labels, lasts_dl)
    fpr_res1, tpr_res1, threshold_res1 = roc_curve(labels, kendalls_res1)
    auc_res1 = roc_auc_score(labels, kendalls_res1)
    fpr_res2, tpr_res2, threshold_res2 = roc_curve(labels, kendalls_res2)
    auc_res2 = roc_auc_score(labels, kendalls_res2)
    fpr_res3, tpr_res3, threshold_res3 = roc_curve(labels, kendalls_res3)
    auc_res3 = roc_auc_score(labels, kendalls_res3)
    fpr_res4, tpr_res4, threshold_res4 = roc_curve(labels, kendalls_res4)
    auc_res4 = roc_auc_score(labels, kendalls_res4)

    plt.figure()
    plt.plot(fpr_var, tpr_var, color = 'red', marker = 'o')
    plt.plot(fpr_ac, tpr_ac, color = 'orange', marker = 'x')
    plt.plot(fpr_eig, tpr_eig, color = 'green', marker = 's')
    plt.plot(fpr_dl, tpr_dl, color = 'black', marker = '*')
    plt.plot(fpr_res1, tpr_res1, color = 'pink', marker = '^')
    plt.plot(fpr_res2, tpr_res2, color = 'cyan', marker = 'v')
    plt.plot(fpr_res3, tpr_res3, color = 'blue', marker = '>')
    plt.plot(fpr_res4, tpr_res4, color = 'magenta', marker = '<')
    plt.xlabel('FPR: False Positive Rate')
    plt.ylabel('TPR: True Positive Rate')
    plt.grid()
    plt.tight_layout()
    plt.savefig('roc_{}.pdf'.format(name), format = 'pdf')
    plt.close()

    return {'variance': auc_var, 'lag1-ac': auc_ac, 'max eigenvalue': auc_eig, 'deep learning': auc_dl, 'residual(vanilla)': auc_res1, 'residual(rbf)': auc_res2, 'residual(laplacian)': auc_res3, 'residual(polynomial)': auc_res4}

def main():
    auc_saddlenode = calculate_roc('saddlenode')
    auc_hopf = calculate_roc('hopf')
    auc_thermoacoustic = calculate_roc('thermoacoustic')
    auc_paleoclimate = calculate_roc('paleoclimate')
    auc_anoxia = calculate_roc('anoxia')

    rows = []
    rows.append({
        'data': 'saddlenode',
        'variance': auc_saddlenode['variance'],
        'lag1-ac': auc_saddlenode['lag1-ac'],
        'max eigenvalue': auc_saddlenode['max eigenvalue'],
        'deep learning': auc_saddlenode['deep learning'],
        'residual(vanilla)': auc_saddlenode['residual(vanilla)'],
        'residual(rbf)': auc_saddlenode['residual(rbf)'],
        'residual(laplacian)': auc_saddlenode['residual(laplacian)'],
        'residual(polynomial)': auc_saddlenode['residual(polynomial)']
    })
    rows.append({
        'data': 'hopf',
        'variance': auc_hopf['variance'],
        'lag1-ac': auc_hopf['lag1-ac'],
        'max eigenvalue': auc_hopf['max eigenvalue'],
        'deep learning': auc_hopf['deep learning'],
        'residual(vanilla)': auc_hopf['residual(vanilla)'],
        'residual(rbf)': auc_hopf['residual(rbf)'],
        'residual(laplacian)': auc_hopf['residual(laplacian)'],
        'residual(polynomial)': auc_hopf['residual(polynomial)']
    })
    rows.append({
        'data': 'thermoacoustic',
        'variance': auc_thermoacoustic['variance'],
        'lag1-ac': auc_thermoacoustic['lag1-ac'],
        'max eigenvalue': auc_thermoacoustic['max eigenvalue'],
        'deep learning': auc_thermoacoustic['deep learning'],
        'residual(vanilla)': auc_thermoacoustic['residual(vanilla)'],
        'residual(rbf)': auc_thermoacoustic['residual(rbf)'],
        'residual(laplacian)': auc_thermoacoustic['residual(laplacian)'],
        'residual(polynomial)': auc_thermoacoustic['residual(polynomial)']
    })
    rows.append({
        'data': 'paleoclimate',
        'variance': auc_paleoclimate['variance'],
        'lag1-ac': auc_paleoclimate['lag1-ac'],
        'max eigenvalue': auc_paleoclimate['max eigenvalue'],
        'deep learning': auc_paleoclimate['deep learning'],
        'residual(vanilla)': auc_paleoclimate['residual(vanilla)'],
        'residual(rbf)': auc_paleoclimate['residual(rbf)'],
        'residual(laplacian)': auc_paleoclimate['residual(laplacian)'],
        'residual(polynomial)': auc_paleoclimate['residual(polynomial)']
    })
    rows.append({
        'data': 'anoxia',
        'variance': auc_anoxia['variance'],
        'lag1-ac': auc_anoxia['lag1-ac'],
        'max eigenvalue': auc_anoxia['max eigenvalue'],
        'deep learning': auc_anoxia['deep learning'],
        'residual(vanilla)': auc_anoxia['residual(vanilla)'],
        'residual(rbf)': auc_anoxia['residual(rbf)'],
        'residual(laplacian)': auc_anoxia['residual(laplacian)'],
        'residual(polynomial)': auc_anoxia['residual(polynomial)']
    })
    df_auc = pd.DataFrame(rows)
    csv_path = "auc_all.csv"
    df_auc.to_csv(csv_path, index = False)

if __name__ == "__main__": 
    main()