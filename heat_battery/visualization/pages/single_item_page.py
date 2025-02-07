from heat_battery.visualization.pages.base import VisualizerApp, dash_enrich
from dash_extensions import Lottie
from dash import get_asset_url

class SingleItemPage(VisualizerApp):
    def __init__(self, name, item, parent=None):
        super().__init__(name=name, parent=parent)
        self.item = item

    def get_children(self):
        return [self.item]

    def preload_cache_data(self):
        self.item.preload_cache_data()

    def get_layout(self, qs_data:dict|None=None):
        item_layout = self.item.get_layout(qs_data)
        div = dash_enrich.html.Div(
            children=[item_layout],
            style={'height':'100%'},
            )
        return div
    
    def set_callbacks(self, server):
        self.item.set_callbacks(server)