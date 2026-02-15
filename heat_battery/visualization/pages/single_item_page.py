from heat_battery.visualization.pages.base import VisualizerApp, dash_enrich
from dash_extensions import Lottie
from dash import get_asset_url

class SingleItemPage(VisualizerApp):
    def __init__(self, name, item, parent=None, icon=None, tooltip_text=None):
        super().__init__(name=name, parent=parent, icon=icon, tooltip_text=tooltip_text)
        self.item = item
        # Set this page as the parent of the item so it can access parent methods
        if hasattr(self.item, 'parent'):
            self.item.parent = self

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