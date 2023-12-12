import pandas as pd
import plotly.express as px
import numpy as np

def clip_value(series, levels):
    bins = np.zeros(len(levels)+1)
    diff = np.diff(levels)/2
    bins[0] = levels[0] - diff[0]
    bins[1:-1] = levels[:-1] + diff
    bins[-1] = levels[-1] + diff[-1]
    return pd.to_numeric(pd.cut(series, bins=bins, labels=levels))

def process_calibration_data(
        df: pd.DataFrame,
        names : list, 
        levels=None,
        reference='mean',
        der_limit = 0.0003,
        rolling_window_size=240,
        T_min=20,
        T_max=1000,
        T_step=5):
    
    names = list(set(df.columns) & set(names)) #intersection

    if levels is None:
        levels = np.arange(T_min, T_max, T_step)

    df = df.set_index('Time')
    df.sort_index(inplace=True)
    df = df[names]
    df['mean'] = df.mean(axis=1)
    df['set'] = clip_value(df['mean'], levels)
    mask = df['mean'].diff().rolling(rolling_window_size).mean().abs() < der_limit
    df_all_steadies = df[mask].sort_index()

    df[names] = df[names].sub(df[reference], axis=0)
    df_groups = df[mask].groupby('set').agg('mean')
    

    return df_groups, df_all_steadies
