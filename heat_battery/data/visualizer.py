from mpi4py import MPI
from dash import get_asset_url
import dash_extensions.enrich as dash_enrich
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
    trigger=dash_enrich.dcc.Interval(
        id={'type':'refresh-trigger', 'trigers':triggers},
        interval=interval,
        n_intervals=1)
    return trigger

class Visualizer():
    def __init__(self, name=""):

        self.name = name

        # added pages
        self.pages = {}
        homepage = HomePage()
        self.register_page(homepage)
     
        # trigers at client side
        self.figure_interval = 1000    # checks for data updates
        self.breathing_interval = 5000 # checks for server connection

        #server
        self.data_time_stamp = self.get_current_time_stamp()

        #TODO: add client counter
        
        #TODO: add reference counter of opened client pages 
        # so we can avoid unnecessary server data updates

    @on_master
    def build_app(self):
        '''Dash app needs to exist only on rank 0'''
        self.app = dash_enrich.DashProxy(
            __name__, 
            assets_folder='./assets', 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            transforms=[],
            prevent_initial_callbacks=True)
        
        self.breathing_icon = Lottie(
            options=dict(loop=False, autoplay=True), 
            width="30px", 
            url=get_asset_url('fire_lottie.json')
        )

        self.set_layout()
        self.set_callbacks()

    @on_master
    def start_app(self, host='127.0.0.1', port=8050):
        '''The dash server need to run in separate thread so it does not block other stuff'''
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        thread = threading.Thread(target = self.app.run_server, daemon=True, kwargs={'host': host, 'port':port})
        thread.start()

    def debug_mode(self):
        '''The dash server runs in serial so it blocks other evaluations'''
        if MPI.COMM_WORLD.rank == 0:
            self.app.run(debug=True)
        MPI.COMM_WORLD.Barrier()

    def stay_alive(self, timeout=365*3600):
        MPI.COMM_WORLD.Barrier()
        time.sleep(timeout)

    def register_page(self, page):
        '''All ranks need acces to page constructors'''
        self.pages[page.get_href()] = page

    def update_data(self):
        '''Some data need all mpi ranks to evaluate sucesfully'''
        for href, page in self.pages.items():
            page._update_data()
        self.data_time_stamp = self.get_current_time_stamp()

    def get_current_time_stamp(self):
        return time.strftime("%m/%d/%Y, %H:%M:%S.%f")

    @on_master
    def set_layout(self):
        '''The frontend layout constuctor (only on rank = 0)'''
        SIDEBAR_STYLE = {
            "position": "fixed",
            "top": 1,   
            "left": 0,
            "bottom": 0,
            "width": "16rem",
            "padding": "1rem 1rem",
            "background-color": "#f8f9fa",
        }

        sidebar = dbc.Offcanvas(
            [
                dash_enrich.dcc.Location(id="url", refresh=False, pathname='/home'),
                dash_enrich.html.H2("Menu", className="display-4"),
                dash_enrich.html.Hr(),
                dash_enrich.html.P(
                    "Select page", className="lead"
                ),
                dbc.Nav(
                    [page.get_link() for key, page in self.pages.items()],
                    vertical=True,
                    pills=True,
                ),
            ],
            id="offcanvas-scrollable",
            scrollable=True,
            close_button=False,
            title=Lottie(options=dict(loop=True, autoplay=True), width="35%", url=get_asset_url('fire_lottie.json')),
            is_open=False,
            style=SIDEBAR_STYLE,
        )

        self.app.layout = dash_enrich.html.Div(
            [
                sidebar,
                dash_enrich.dcc.Store(id="update-data-status", data=1),
                dash_enrich.html.Div(
                    [    
                        dbc.Button(
                            "MENU",
                            id="open-offcanvas-scrollable",
                            n_clicks=0,
                            size="sm",
                            className="me-2",
                            style={'display': 'inline-block'}
                        ),
                        dash_enrich.html.Div(
                            self.breathing_icon,
                            id="breathing-server",
                            style={'display': 'inline-block', 'vertical-align': 'top', 'text-align': 'center', 'flex-grow': '1'}
                        ),
                    ],
                style={'display':'flex'},
                ),
                dash_enrich.html.Div(id="page-content"),
                interval_trigger('breathing', self.breathing_interval),
                interval_trigger('figures', self.figure_interval),
            ],
        )

    @on_master
    def set_callbacks(self):
        '''Sets all callbacks (callbacks are handled only by rank = 0)'''

        # opens and closes the side menu
        @self.app.callback(
            dash_enrich.Output("offcanvas-scrollable", "is_open"),
            dash_enrich.Input("open-offcanvas-scrollable", "n_clicks"),
            dash_enrich.State("offcanvas-scrollable", "is_open"),
        )
        def toggle_offcanvas_scrollable(n1, is_open):
            if n1:
                return not is_open
            return is_open
        
        # makes the fire lottie at top screen breathe if server is alive and comunicating with the UI
        @self.app.callback(
            dash_enrich.Output("breathing-server", "children"),
            dash_enrich.Input({'type': 'refresh-trigger', 'trigers': 'breathing'},'n_intervals'),  
        )
        def send_icon(n_intervals):
            # Periodicaly resets the fire animation at the top. If it stops 
            # blinking at the client side, it means that the connection to
            # the server is lost.
            return self.breathing_icon
        
        # updates figures with new data (only when server updated, or url requested)
        @self.app.callback(
            dash_enrich.Output("update-data-status", 'data'),
            dash_enrich.Output("page-content", "children"),
            dash_enrich.Output({'type': 'refresh-trigger', 'trigers': 'figures'}, "disabled"),
            dash_enrich.State('update-data-status', 'data'),
            dash_enrich.Input({'type': 'refresh-trigger', 'trigers': 'figures'},'n_intervals'),
            dash_enrich.Input("url", "pathname"),
            prevent_initial_call=True,
        )
        def update_content(data, n_intervals, pathname):
            # Periodic trigger at client side checks whether server has new data,
            # if so we send the whole page to the client.
            # If the client changed the URL (page) that he is watching we also
            # send the new page.
            # If the client has the newest data we do not update.
            if (data != self.data_time_stamp) or dash_enrich.ctx.triggered_id == 'url':
                data = self.data_time_stamp # update the data timestamp of data send from server to client
                return data, self.pages[pathname].get_layout(), self.pages[pathname].disable_client_interval
            else:
                return dash_enrich.no_update
        
        # updates a resampling figure of any url (page) that the client is currently watching
        @self.app.callback(
            dash_enrich.Output({"type": "dynamic-updater", "index": dash_enrich.MATCH}, "updateData"),
            dash_enrich.Input({"type": "dynamic-graph", "index": dash_enrich.MATCH}, "relayoutData"),
            prevent_initial_call=True,
            memoize=False,
        )
        def update_fig(relayoutdata):
            # Variable relayoutdata contains zoom (area) of a figure sellected
            # by a user. We check which page (href) and which graph (i) on
            # that page is sending the relayoutdata. Then we compute the data.
            # The attribute data[i] in the page[href] is a FigureResample 
            # (see https://github.com/predict-idlab/plotly-resampler)
            index = dash_enrich.ctx.triggered_id['index']
            href, i = index.split('-') #TODO: can tubple be used instead of this string?
            return self.pages[href].data[int(i)].construct_update_data(relayoutdata)

        # callback for class Table
        @self.app.callback(
            dash_enrich.Output({'type':'datatable-interactivity', "index": dash_enrich.MATCH}, 'style_data_conditional'),
            dash_enrich.Input({'type':'datatable-interactivity', "index": dash_enrich.MATCH}, 'selected_columns'),
            dash_enrich.Input({'type':'datatable-interactivity', "index": dash_enrich.MATCH}, 'selected_rows'),
        )
        def update_styles(selected_columns, selected_rows):
            cs = [{'if': { 'column_id': i }, 'border': '1px solid black'} for i in selected_columns]
            rs = [{'if': { 'row_id': i }, 'background_color': '#D2F3FF'} for i in selected_rows]
            return cs #+ rs
        
        # update stuff under table
        @self.app.callback(
            #dash_enrich.Output({'type':'datatable-interactivity-container', "index": dash_enrich.MATCH}, 'children'),
            dash_enrich.Output({'type':'datatable-loading-div', "index": dash_enrich.MATCH}, 'children'),
            dash_enrich.Input({'type':'datatable-interactivity', "index": dash_enrich.MATCH}, "derived_virtual_data"),
            dash_enrich.Input({'type':'datatable-interactivity', "index": dash_enrich.MATCH}, "derived_virtual_selected_rows"),
            prevent_initial_call=True,
            )
        def update_view(rows, derived_virtual_selected_rows):
            href = dash_enrich.ctx.triggered_id['index']
            if not derived_virtual_selected_rows:
                return dash_enrich.no_update
            else:
                row_id = derived_virtual_selected_rows[0]
                return self.pages[href].get_content(rows[row_id])
            