from mpi4py import MPI
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash_extensions import Lottie
import logging
import threading
import time
from .pages import HomePage

def on_master(f):
    def decorated_f(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            return f(*args, **kwargs)
    return decorated_f

def interval_trigger(triggers, interval):
    trigger=dcc.Interval(
        id={'type':'refresh-trigger', 'trigers':triggers},
        interval=interval,
        n_intervals=1)
    return trigger

class Visualizer():
    def __init__(self, name=""):

        self.name = name

        # visual components
        self.pages = {}
        homepage = HomePage()
        self.register_page(homepage)

        # trigers
        self.figure_interval = 1000
        self.display_interval = 50

    @on_master
    def build_app(self):
        self.app = Dash(__name__, assets_folder='./assets', external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.set_layout()
        self.set_callbacks()

    @on_master
    def start_app(self):
        if MPI.COMM_WORLD.rank == 0:
            logging.getLogger('werkzeug').setLevel(logging.ERROR)
            thread = threading.Thread(target = self.app.run_server, daemon=True)
            thread.start()

    @on_master
    def debug_mode(self):
        self.app.run(debug=True)

    def register_page(self, page):
        self.pages[page.get_href()] = page

    def update_data(self):
        for href, page in self.pages.items():
            page.update_data()

    @on_master
    def set_layout(self):
        SIDEBAR_STYLE = {
            "position": "fixed",
            "top": 0,   
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "2rem 1rem",
            "background-color": "#f8f9fa",
        }

        sidebar = dbc.Offcanvas(
            [
                dcc.Location(id="url", refresh=False, pathname='/home'),
                html.H2("Menu", className="display-4"),
                html.Hr(),
                html.P(
                    "Select graph", className="lead"
                ),
                dbc.Nav(
                    [page.get_link() for key, page in self.pages.items()],
                    vertical=True,
                    pills=True,
                ),
            ],
            id="offcanvas-scrollable",
            scrollable=True,
            title=Lottie(options=dict(loop=True, autoplay=True), width="35%", url=dash.get_asset_url('fire_lottie.json')),
            is_open=False,
            style=SIDEBAR_STYLE,
        )

        self.app.layout = html.Div(
            [
                sidebar,    
                dbc.Button(
                    "MENU",
                    id="open-offcanvas-scrollable",
                    n_clicks=0,
                ),
                dbc.Button(
                    "REFRESFH",
                    id="resfresh-button",
                    n_clicks=0,
                ),
                html.Div(id="page-content"),
                #interval_trigger('displays', self.display_interval),
                interval_trigger('figures', self.figure_interval),
            ],
        )

    @on_master
    def set_callbacks(self):

        @self.app.callback(
            dash.Output("offcanvas-scrollable", "is_open"),
            dash.Input("open-offcanvas-scrollable", "n_clicks"),
            dash.State("offcanvas-scrollable", "is_open"),
        )
        def toggle_offcanvas_scrollable(n1, is_open):
            if n1:
                return not is_open
            return is_open
        
        @self.app.callback(
            dash.Output("page-content", "children"),
            [
            dash.Input({'type': 'refresh-trigger', 'trigers': 'figures'},'n_intervals'),
            dash.Input('resfresh-button','n_clicks'),
            dash.Input("url", "pathname")
            ]
        )
        def update_content(n_intervals, n_clicks, pathname):
            return self.pages[pathname].get_layout()