from heat_battery.visualization.pages.base import VisualizerApp, dash_enrich
from dash_extensions import Lottie
from dash import get_asset_url

class HomePage(VisualizerApp):
    def __init__(self, name="Home"):
        super().__init__(name=name)

    def get_layout(self, qs_data:dict|None=None):
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