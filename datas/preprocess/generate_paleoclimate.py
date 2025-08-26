import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.signal import detrend

def interpolate(df):
    df_prior = df[df["Age"] >= df["Transition"].iloc[0]].copy()

    t_inter_vals = np.linspace(df_prior["Age"].iloc[0], df_prior["Age"].iloc[-1], len(df_prior))
    df_inter = pd.DataFrame({"Age": t_inter_vals, "Inter": True})
    df2 = pd.concat([df_prior, df_inter]).set_index("Age")
    df2 = df2.interpolate(method = "index")

    df_inter = df2[df2["Inter"] == True][["Proxy", "Transition"]].reset_index()

    return df_inter

def paleoclimate_simulation():
    X_dataset = []
    time_dataset = []
    label_dataset = []

    df = pd.read_csv('data_paleoclimate.csv')
    list_records = df['Record'].unique()
    for record in list_records:
        df_select = df[df['Record'] == record]
        df_inter = interpolate(df_select[['Age', 'Proxy', 'Transition']])
        df_inter['Age'] = -df_inter['Age']
        df_proxy = df_inter.set_index('Age')['Proxy']
        t = df_inter['Age'].to_numpy().flatten()
        x = df_proxy.to_numpy().flatten()

        cs = CubicSpline(t, x, bc_type = 'natural')
        time_dataset.append(np.linspace(t[0], t[-1], 2000))
        X_dataset.append(cs(np.linspace(t[0], t[-1], 2000)))
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

    df_paleoclimate = pd.DataFrame(rows)
    csv_path = '../alldata_paleoclimate.csv'
    df_paleoclimate.to_csv(csv_path, index = False)

if __name__ == "__main__": 
    paleoclimate_simulation()