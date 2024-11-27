from heat_battery.visualization.pages.base import VisualizerApp, dash_enrich
from dash_extensions import Lottie
from dash import get_asset_url

class HomePage(VisualizerApp):
    def __init__(self, name="Home"):
        super().__init__(name=name)

    def get_layout(self):
        return dash_enrich.html.Div(
            children=[
                Lottie(
                    id='welcome-lottie',
                    options=dict(loop=True, autoplay=True), 
                    height="60vh",
                    url=get_asset_url('lotties/solar_battery.json'),
                ),
            ],
        )
    
class SingleItemPage(VisualizerApp):
    def __init__(self, name, item):
        super().__init__(name=name)
        self.item = item

    def get_children(self):
        return [self.item]

    def update_data(self):
        self.item.update_data()

    def get_layout(self):
        div = dash_enrich.html.Div(
            children=[self.item.get_layout()],
            style={'height':'100%'},
            )
        return div
    
    def set_callbacks(self, server):
        self.item.set_callbacks(server)