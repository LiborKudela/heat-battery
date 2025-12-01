from dash import Patch
import numpy as np
import pandas as pd
import datetime
import tsdownsample
import re

DEFAULT_TSDOWNSAMPLE_N = 1000
DEFAULT_TSDOWNSAMPLE_METHOD = tsdownsample.MinMaxLTTBDownsampler()

def regex_list_selector(list: list[str], regexes: list[str]) -> list[str]:
    "Compare a str items in a list and if it matches the regex add it to the result list"
    filtered_list = []
    for regex in regexes:
        for item in list:
            if re.match(f"^{regex}$", item):
                filtered_list.append(item)
            elif regex == item:
                filtered_list.append(item)
    return filtered_list

def limit_to_number(value):
    if isinstance(value, datetime.datetime):
        return value.timestamp()
    elif isinstance(value, str):
        return datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=datetime.timezone.utc).timestamp()
    elif isinstance(value, float):
        return value
    else:
        raise ValueError(f"Unknown value type: {type(value)}")
    
def downsample_1d(
        x:np.ndarray, 
        y:np.ndarray, 
        x_min:float, 
        x_max:float, 
        n_out:int=DEFAULT_TSDOWNSAMPLE_N):
    assert len(x.shape) == 1, "x must be 1D"
    assert len(y.shape) == 1, "y must be 1D"
    assert x.shape[0] > 0, "x and y must have at least one element"
    assert x.shape[0] == y.shape[0], "x and y must have the same length"
    if x.shape[0] < n_out:
        return x, y
    x_min = limit_to_number(x_min)
    x_max = limit_to_number(x_max)
    x_min = max(x_min, x[0])
    x_max = min(x_max, x[-1])
    x_min_index = np.searchsorted(x, x_min, side='left')
    x_max_index = np.searchsorted(x, x_max, side='right')
    x = x[x_min_index:x_max_index].copy()
    y = y[x_min_index:x_max_index].copy()
    if x_max_index - x_min_index < n_out:
        return x, y
    
    # downsample when needed
    ds_idxs = DEFAULT_TSDOWNSAMPLE_METHOD.downsample(x, y, n_out=n_out, parallel=False)
    x_out = np.empty(n_out)
    x_out[:] = x[ds_idxs]
    y_out = np.empty(n_out)
    y_out[:] = y[ds_idxs]
    return x_out, y_out

def downsample_from_dataframe(df, x_name, y_names, x_min, x_max, n_out=DEFAULT_TSDOWNSAMPLE_N):
    x = df[x_name].values
    res = []
    for y_name in y_names:
        y = df[y_name].values
        x_out, y_out = downsample_1d(x, y, x_min, x_max, n_out)
        res.append({'x': x_out, 'y': y_out})
    return res

def downsample_to_dict(df, x_name, y_names, x_min, x_max, n_out=DEFAULT_TSDOWNSAMPLE_N):
    d = {'data': []}
    res = downsample_from_dataframe(df, x_name, y_names, x_min, x_max, n_out)
    for i, trace in enumerate(res):
        d['data'].append({'x': trace['x'], 'y': trace['y']})
    return d

def minify_for_chart_editor(df: pd.DataFrame, n_out=DEFAULT_TSDOWNSAMPLE_N):
    n_in = len(df.index)
    n_step = n_in // n_out
    mini_df = df.iloc[::n_step] 
    return mini_df

def convert_to_patch(data):
    patch = Patch()
    for i, item in enumerate(data['data']):
        patch['data'][i]['x'] = item['x']
        patch['data'][i]['y'] = item['y']
        patch['data'][i]['name'] = item['name']
    return patch

def get_timeseries_figure_update(df, trace_names, x_name, relayout_data, as_patch=True):
    print(f"Trace names: {trace_names}")
    if 'xaxis.range[0]' not in relayout_data:
        x_min = df[x_name].iloc[0]
    else:
        x_min = relayout_data['xaxis.range[0]']

    if 'xaxis.range[1]' not in relayout_data:
        x_max = df[x_name].iloc[-1]
    else:
        x_max = relayout_data['xaxis.range[1]']
    d = downsample_to_dict(df, x_name, trace_names, x_min, x_max)
    for i, t_name in enumerate(trace_names):
        x = d['data'][i]['x']
        x = pd.to_datetime(x, unit='s', origin='unix', utc=True)
        d['data'][i]['x'] = x
        d['data'][i]['name'] = t_name
    if as_patch:
        return convert_to_patch(d)
    else:
        return d

def fill_timeseries_figure_initial(df, fig_data):
    fig_data['y_names'] = regex_list_selector(df.columns, fig_data['y_names'])
    data = get_timeseries_figure_update(df, fig_data['y_names'], fig_data['x_name'], {}, as_patch=False)
    for i, item in enumerate(data['data']):
        item['x'] = item['x'].strftime('%Y-%m-%d %H:%M:%S')
        item['xsrc'] = fig_data['x_name']
        item['ysrc'] = item['name']
    fig_data['data'] = data['data']

def no_update_patch():
    return Patch()

def get_aggregated_figure_update(df, trace_names, x_name, relayout_data):
    return Patch()