import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.signal import detrend

def anoxia_simulation():
    X_dataset = []
    time_dataset = []
    label_dataset = []

    df = pd.read_csv('data_anoxia.csv')
    list_tsid = df['tsid'].unique()
    for tsid in list_tsid:
        df_temp = df[(df['tsid'] == tsid)]
        df_select = df_temp[df_temp['Age [ka BP]'] >= df_temp['t_transition_start'].iloc[0]].copy()
        df_select['Age [ka BP]'] = -df_select['Age [ka BP]']
        df_select = df_select[::-1]

        # for Mo
        df_Mo = df_select.set_index('Age [ka BP]')['Mo [ppm]']
        t_Mo = df_select['Age [ka BP]'].to_numpy().flatten()
        x_Mo = df_Mo.to_numpy().flatten()

        cs_Mo = CubicSpline(t_Mo, x_Mo, bc_type = 'natural')
        time_dataset.append(np.linspace(t_Mo[0], t_Mo[-1], 2000))
        X_dataset.append(cs_Mo(np.linspace(t_Mo[0], t_Mo[-1], 2000)))
        label_dataset.append(1)

        # for U
        df_U = df_select.set_index('Age [ka BP]')['U [ppm]']
        t_U = df_select['Age [ka BP]'].to_numpy().flatten()
        x_U = df_U.to_numpy().flatten()

        cs_U = CubicSpline(t_U, x_U, bc_type = 'natural')
        time_dataset.append(np.linspace(t_U[0], t_U[-1], 2000))
        X_dataset.append(cs_U(np.linspace(t_U[0], t_U[-1], 2000)))
        label_dataset.append(1)

    X_dataset = np.stack(X_dataset, axis = 0)
    time_dataset = np.stack(time_dataset, axis = 0)
    label_dataset = np.array(label_dataset, dtype = int)

    Xnull_dataset = []
    timenull_dataset = []
    labelnull_dataset = []

    for i in range(X_dataset.shape[0]):
        x = X_dataset[i]
        xsub = x[:int(x.shape[0]/5)]
        xsub_detrend = detrend(xsub, type = 'linear')
        var = np.var(xsub_detrend)
        xsub_detrend_mean = np.mean(xsub_detrend)
        xsub_forac = xsub_detrend - xsub_detrend_mean
        ac = np.corrcoef(xsub_forac[:-1], xsub_forac[1:])[0, 1]

        alpha = ac
        sigma = np.sqrt(var*(1 - alpha**2))

        xtemp = x[0]
        list_x = [xtemp]
        for j in range(1, X_dataset.shape[1]):
            epsilon = np.random.normal(loc = 0, scale = 1)
            xtemp = alpha*xtemp + sigma*epsilon
            list_x.append(xtemp)

        Xnull_dataset.append(np.array(list_x))
        timenull_dataset.append(time_dataset[i])
        labelnull_dataset.append(0)

    Xnull_dataset = np.stack(Xnull_dataset, axis = 0)
    timenull_dataset = np.stack(timenull_dataset, axis = 0)
    labelnull_dataset = np.array(labelnull_dataset, dtype = int)

    X_dataset = np.concatenate([X_dataset, Xnull_dataset], axis = 0)
    time_dataset = np.concatenate([time_dataset, timenull_dataset], axis = 0)
    label_dataset = np.concatenate([label_dataset, labelnull_dataset], axis = 0)

    rows = []
    for i in range(X_dataset.shape[0]):
        for j in range(2000):
            rows.append({
                'id': i,
                'label': int(label_dataset[i]),
                't': float(time_dataset[i, j]),
                'x': float(X_dataset[i, j])
            })

    df_anoxia = pd.DataFrame(rows)
    csv_path = '../alldata_anoxia.csv'
    df_anoxia.to_csv(csv_path, index = False)

if __name__ == "__main__": 
    anoxia_simulation()