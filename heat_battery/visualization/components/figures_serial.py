from .base import ContentItem, dash_enrich
from dash import Patch
from copy import deepcopy
import numpy as np
import pandas as pd
import datetime

import tsdownsample #rust
DEFAULT_TSDOWNSAMPLE_N = 1000
DEFAULT_TSDOWNSAMPLE_METHOD = tsdownsample.MinMaxLTTBDownsampler()
DEFAULT_REFRESH_PERIOD = 5000
LOAD_OUT_OF_RELAYOUT_DATA = False

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

def downsample_to_patch(df, x_name, y_names, x_min, x_max, n_out=DEFAULT_TSDOWNSAMPLE_N):
    fig_patch = Patch()
    fig_patch['data'] = downsample_from_dataframe(df, x_name, y_names, x_min, x_max, n_out)
    return fig_patch

def downsample_to_dict(df, x_name, y_names, x_min, x_max, n_out=DEFAULT_TSDOWNSAMPLE_N):
    d = {'data': []}
    res = downsample_from_dataframe(df, x_name, y_names, x_min, x_max, n_out)
    for i, trace in enumerate(res):
        d['data'].append({'x': trace['x'], 'y': trace['y']})
    return d

class BaseFigureItem(ContentItem):
    """This represents the basic extendable layout of a plotly figure"""
    def __init__(self, f_template, id_int, x_name='datetime', animate=False, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.GRAPH_ID = {'type': 'graph', 'index': f"{id_int}"}
        self.f_template = f_template
        self.x_name = x_name
        self.first_data_updata_call = True
        self.animate = animate
        self.controls = []
        self.invisibles = []
        self.figure_template = self.f_template(self.parent.dummy_data)
        self.figure_template['layout']['showlegend'] = False
    
    def get_new_id(self):
        ContentItem.last_id += 1
        return f'resampling-figure-item-{ContentItem.last_id}'

    def get_fresh_figure(self, df, qs_data:dict|None=None):
        return self.f_template(df)

    def get_layout(self, df, qs_data:dict|None=None):
        return dash_enrich.dcc.Graph(
                figure=self.get_fresh_figure(df, qs_data),
                id=self.GRAPH_ID,
                config={
                    'responsive':True, 
                    'scrollZoom':False,
                    'displaylogo': False,
                    'displayModeBar': 'hover',
                },
                style={
                    "margin":"2px", 
                    'borderRadius': '10px', 
                    'overflow':'hidden', 
                    'height':'100%',
                    #'boxShadow': '0px 0px 5px 2px #0ff',
                },
                animate=self.animate,
            )

class TimeSeriesResamplingFigure(BaseFigureItem):
    def __init__(self, f_template, id_int, x_name='datetime', x_to_date=True, parent=None, **kwargs):
        super().__init__(f_template, id_int, x_name, parent=parent, **kwargs)
        self.x_to_date = x_to_date

    def get_fresh_figure(self, df, qs_data:dict|None=None):
        y_names = [trace['name'] for trace in self.figure_template['data']]
        x_min = df[self.x_name].iloc[0]
        x_max = df[self.x_name].iloc[-1]
        new_figure = deepcopy(self.figure_template)
        d = downsample_to_dict(df, self.x_name, y_names, x_min, x_max)
        for i, trace in enumerate(new_figure['data']):
            x = d['data'][i]['x']
            if self.x_to_date:
                x = pd.to_datetime(x, unit='s', origin='unix', utc=True)
            trace['x'] = x
            trace['y'] = d['data'][i]['y']
        return new_figure

