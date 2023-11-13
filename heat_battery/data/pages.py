import dash_extensions.enrich as dash_enrich
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


from trace_updater import TraceUpdater
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

class VisualizerPage():
    def __init__(self, name):
        self.name = name.capitalize()
        self.href = '/' + name.lower()
        self.data = None # initial data
        self.disable_client_interval = False
        self.disable_server_updates = False
        self.data_updated = False

    def get_href(self):
        return self.href
    
    def get_link(self):
        return dbc.NavLink(self.name, href=self.href, active="exact")

    def _update_data(self):
        '''static page switch handling'''
        if self.disable_server_updates and self.data_updated:
            pass
        else:
            self.data_updated = False
            self.update_data()
            self.data_updated = True

    def update_data(self):
        '''defined in subclass'''
        pass

class HomePage(VisualizerPage):
    def __init__(self, name="Home"):
        super().__init__(name=name)
        self.disable_client_interval = True

    def get_layout(self):
        return dash_enrich.html.Div(
            [],
        )

class GridLayoutPage(VisualizerPage):
    def __init__(self, name, f, *args, **kwargs):
        super().__init__(name=name)
        self.args = args
        self.kwargs = kwargs
        self.figure_constructor = f

    def subplot_grid_size(self, n):
        rows, cols = 1, 1
        for i in range(n):
            if rows*cols >= n:
                break
            else:
                if cols + 1 > rows:
                    rows += 1
                else:
                    cols += 1
        return rows, cols
    
    def get_grid_items(self) -> list:
        '''defined in subclass'''
        pass

    def get_hidden_children(self) -> list:
        '''defined in subclass'''
        pass
    
    def get_layout(self):
        rows, cols = self.subplot_grid_size(len(self.data))
        div = dash_enrich.html.Div(children=[
            dash_enrich.html.Div(
                self.get_grid_items(),
                style={
                    'display':'grid',
                    'grid-template-columns': f'repeat({rows}, 1fr)',
                    'grid-template-rows': f'repeat({cols}, 1fr)',
                    'position':'relative', 'top':0, 'left':0, 'bottom':0, 'right':0, 'width':'100%', 'height':'95vh', 'margin':0, 'padding':0, 'overflow':'hidden',
                },
            ),
            dash_enrich.html.Div(children=self.get_hidden_children())],
        )
        return div

class FigurePage(GridLayoutPage):
    '''Auto-updating page with plotly figures organised in a grid (for small datasets)'''
    def __init__(self, name, f, *args, **kwargs):
        super().__init__(name, f, *args, **kwargs)
    
    def get_grid_items(self):
        return [dash_enrich.dcc.Graph(
            figure=fig, 
            config={'responsive':True}, 
            style={'className':'grid-item'}) for fig in self.data]
    
    def update_data(self):
        data = self.figure_constructor(*self.args, **self.kwargs)
        if isinstance(data, list):
            self.data = data
        else:
            self.data = [data]

class ResampingFigurePage(GridLayoutPage):
    '''Static page with plotly figures organised in a grid (for big datasets)'''
    def __init__(self, name, f, *args, **kwargs):
        super().__init__(name, f, *args, **kwargs)
        self.disable_client_interval = True # do not update data at client
        self.disable_server_updates = True  # do not update data at server
    
    def get_grid_items(self):
        grid_items = []
        for i, fig in enumerate(self.data):
            grid_items.append(
                dash_enrich.dcc.Graph(
                    id={"type": "dynamic-graph", "index": f"{self.href}-{i}"}, 
                    figure=fig, 
                    config={'responsive':True}, 
                    style={'className':'grid-item'},
                ),
            )
        return grid_items
    
    def get_hidden_children(self) -> list:
        hidden_updaters = []
        for i, fig in enumerate(self.data):
            hidden_updaters.append(
                TraceUpdater(
                    id={"type": "dynamic-updater", "index": f"{self.href}-{i}"},
                    gdID=f"{self.href}-{i}",
                ),
            )
        return hidden_updaters
    
    def resampler_wrapper(self, fig):
        resampler = FigureResampler(
            fig,
            default_n_shown_samples=2000,
            default_downsampler=MinMaxLTTB(parallel=True),
        )
        return resampler
    
    def update_data(self):
        data = self.figure_constructor(*self.args, **self.kwargs)
        if isinstance(data, list):
            self.data = []
            for data_item in data:
                self.data.append(self.resampler_wrapper(data_item))
        else:
            self.data = [self.resampler_wrapper(data)]
