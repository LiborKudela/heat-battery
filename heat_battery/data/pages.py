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

    def get_href(self):
        return self.href
    
    def get_link(self):
        return dbc.NavLink(self.name, href=self.href, active="exact")

    def update_data(self):
        pass

class HomePage(VisualizerPage):
    def __init__(self, name="Home"):
        super().__init__(name=name)
        self.disable_interval = True

    def get_layout(self):
        return dash_enrich.html.Div(
            [],
        )
    
class FigurePage(VisualizerPage):
    def __init__(self, name, f, *args, **kwargs):
        super().__init__(name=name)
        self.args = args
        self.kwargs = kwargs
        self.figure_constructor = f
        self.disable_interval = False

    def get_layout(self):
        div = dash_enrich.dcc.Graph(
            figure=self.data,
            config={'responsive':True},
            style={'position':'relative', 'top':0, 'left':0, 'bottom':0, 'right':0, 'width':'100%', 'height':'95vh', 'margin':0, 'padding':0, 'overflow':'hidden'},
            )
        return div
    
    def update_data(self):
        self.data = self.figure_constructor(*self.args, **self.kwargs)

class ResampingFigurePage(VisualizerPage):
    def __init__(self, name, f, *args, **kwargs):
        super().__init__(name=name)
        self.args = args
        self.kwargs = kwargs
        self.figure_constructor = f
        self.disable_interval = True
        self.static = False
        self.data_ready = False

    def get_layout(self):
        div = dash_enrich.html.Div(
            children=[
                dash_enrich.dcc.Graph(id={"type": "dynamic-graph", "index": self.href}, figure=go.Figure(), config={'responsive':True}, style={'height':'95vh'}),
                dash_enrich.dcc.Loading(dash_enrich.dcc.Store(id={"type": "store", "index": self.href})),
                TraceUpdater(id={"type": "dynamic-updater", "index": self.href}, gdID=f"{self.href}"),
                dash_enrich.dcc.Interval(id={"type": "interval", "index": self.href}, interval=1, max_intervals=1),
            ],
            style={'position':'relative', 'top':0, 'left':0, 'bottom':0, 'right':0, 'width':'100%', 'height':'95vh', 'margin':0, 'padding':0, 'overflow':'hidden'},
        )
        return div
    
    def update_data(self):
        self.data = self.figure_constructor(*self.args, **self.kwargs)
    
def subplot_grid_size(n):
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
    
class SubPlotsPage(VisualizerPage):
    def __init__(self, name, f, *args, **kwargs):
        super().__init__(name=name)
        self.args = args
        self.kwargs = kwargs
        self.figure_constructor = f
        self.disable_interval = False

    def get_layout(self):
        figs = self.data
        rows, cols = subplot_grid_size(len(figs))
        grid_items = [dash_enrich.dcc.Graph(figure=fig, config={'responsive':True}, style={'className':'grid-item'}) for fig in figs]
        div = dash_enrich.html.Div(
            grid_items, 
            style={
                'display':'grid',
                'grid-template-columns': f'repeat({rows}, 1fr)',
                'grid-template-rows': f'repeat({cols}, 1fr)',
            },
        )
        return div
    
    def update_data(self):
        self.data = self.figure_constructor(*self.args, **self.kwargs)
