import dash
import dash_bootstrap_components as dbc
from dash_extensions import Lottie

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

    def get_layout(self):
        return dash.html.Div(
            [
                Lottie(options=dict(loop=True, autoplay=True), width="30%", url=dash.get_asset_url('fire_lottie.json')),
            ],
        )
    
class FigurePage(VisualizerPage):
    def __init__(self, name, f, *args, **kwargs):
        super().__init__(name=name)
        self.args = args
        self.kwargs = kwargs
        self.figure_constructor = f

    def get_layout(self):
        div = dash.dcc.Graph(
            figure=self.data,
            config={'responsive':True},
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

    def get_layout(self):
        figs = self.data
        rows, cols = subplot_grid_size(len(figs))
        grid_items = [dash.dcc.Graph(figure=fig, config={'responsive':True}, style={'className':'grid-item'}) for fig in figs]
        div = dash.html.Div(
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
