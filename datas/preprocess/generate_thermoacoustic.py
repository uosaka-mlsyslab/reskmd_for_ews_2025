import numpy as np
import pandas as pd

from scipy.interpolate import CubicSpline
from scipy.signal import detrend

def thermoacoustic_simulation():
    X_dataset = []
    time_dataset = []
    label_dataset = []

    df = pd.read_csv('data_thermoacoustic.csv')
    list_tsid = df['tsid'].unique()
    for tsid in list_tsid:
        df_select = df[(df['tsid'] == tsid)].reset_index(drop = True)
        time_trans = df_select['transition time (s)'].iloc[0]
        idx_trans = df_select[df_select['Time (s)'] <= time_trans].index[-1]
        df_prior = df_select.iloc[idx_trans - 2000 : idx_trans]

        time_dataset.append(df_prior['Time (s)'])
        X_dataset.append(df_prior.set_index('Time (s)')['Pressure (kPa)'])
        label_dataset.append(1)

    X_dataset = np.stack(X_dataset, axis = 0)
    time_dataset = np.stack(time_dataset, axis = 0)
    label_dataset = np.array(label_dataset, dtype = int)

    Xnull_dataset = []
    timenull_dataset = []
    labelnull_dataset = []

    df_nulls = pd.read_csv('nulldata_thermoacoustic.csv')
    list_tsid = np.arange(1, 11)
    for tsid in list_tsid:
        df_select = df_nulls[(df_nulls['tsid'] == tsid)].reset_index(drop = True)
        for i in [1, 2]:
            idx_start = np.random.choice(np.arange(len(df_select) - 2000))
            df_select1500 = df_select.iloc[idx_start : idx_start + 2000]
            
            timenull_dataset.append(df_select1500['Time (s)'])
            Xnull_dataset.append(df_select1500.set_index('Time (s)')['Pressure (kPa)'])
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

    df_thermoacoustic = pd.DataFrame(rows)
    csv_path = '../alldata_thermoacoustic.csv'
    df_thermoacoustic.to_csv(csv_path, index = False)

if __name__ == "__main__": 
    thermoacoustic_simulation()