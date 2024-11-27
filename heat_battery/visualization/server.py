from mpi4py import MPI
from dash import get_asset_url
import dash_extensions.enrich as dash_enrich
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash_extensions import EventListener, Lottie
import logging
import threading
import time
from .pages import HomePage
import datetime
from..utilities import hash_data
import random
import json
from dash_iconify import DashIconify

def on_master(f):
    def decorated_f(*args, **kwargs):
        if MPI.COMM_WORLD.rank == 0:
            return f(*args, **kwargs)
    return decorated_f

class Dasboard():
    def __init__(self, name=""):

        self.name = name
        self.server_id = hash_data((time.time(), random.random()))

        # added pages
        self.pages = {}
        self.updaters = {}
        self.server_data = {}
        homepage = HomePage()
        self.register_page(homepage)
        self.server_ready = False
        self.server_load_progress = 0.0
        self.server_load_progress_info = ""
     
        # trigers at client side
        self.figure_interval = 1000    # checks for data updates

        #server
        self.data_time_stamp = self.get_current_time_stamp()

        #TODO: add client counter
        self.active_session_ids = dict() #id : last ping time

    @on_master
    def build_app(self):
        '''Dash app needs to exist only on rank 0'''
        self.app = dash_enrich.DashProxy(
            __name__, 
            external_scripts=["https://cdn.jsdelivr.net/npm/lodash/lodash.min.js", "https://cdn.plot.ly/plotly-2.18.2.min.js"],
            assets_folder='./assets', 
            external_stylesheets=[dbc.themes.QUARTZ],
            transforms=[],
            prevent_initial_callbacks=True)

        self.pass_server_data_to_pages()
        self.set_layout()
        self.set_callbacks()

    @on_master
    def start_app(self, daemon=False, host='127.0.0.1', port=8050):
        '''The dash server need to run in separate thread so it does not block other stuff'''
        #logging.getLogger('werkzeug').setLevel(logging.ERROR)
        thread = threading.Thread(target = self.app.run_server, daemon=daemon, kwargs={'host': host, 'port':port})
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
        '''All ranks need access to page constructors'''
        self.pages[page.get_href()] = page
        
    def register_data_updater(self, name, updater):
        self.updaters[name] = updater
        self.server_data[name] = None

    def update_data(self):
        '''Some data need all mpi ranks to evaluate sucesfully'''
        now = time.time()
        active_ids = list(self.active_session_ids.keys())
        for session_id in active_ids:
            if (now - self.active_session_ids[session_id]) > 30:
                self.active_session_ids.pop(session_id)
                print(f'Client disconected with id: {session_id}')

        active_client = len(self.active_session_ids) > 0
        active_client = MPI.COMM_WORLD.bcast(active_client)
        total = len(self.updaters) + len(self.pages)
        finished = 0
        self.server_load_progress = 0.0
        self.server_load_progress_info = "initialisation"
        if active_client or not self.server_ready:
            u_start = time.time()
            for name, updater in self.updaters.items():
                self.server_load_progress_info = f"Running updater: {name}"
                self.server_data[name] = updater.reload()
                finished += 1
                self.server_load_progress = finished/total
            print(f"Updaters: {time.time() - u_start}.")
            u_start = time.time()
            for href, page in self.pages.items():
                self.server_load_progress_info = f"Loading page: {href}"
                page._update_data()
                finished += 1
                self.server_load_progress = finished/total
                self.server_load_progress_info = f"Finishing"
            print(f"Apps: {time.time() - u_start}.")
            self.data_time_stamp = self.get_current_time_stamp()
            self.server_ready = True

    def auto_update(self, freq=10):
        while True:
            u_start = time.time()
            self.update_data()
            print(f"Update time: {time.time() - u_start} -> {len(self.active_session_ids)} clients.")
            time.sleep(freq)

    def get_current_time_stamp(self):
        return datetime.datetime.now().strftime(f'%m/%d/%Y, %H:%M:%S.%f')

    @on_master
    def pass_server_data_to_pages(self):
        for path, page in self.pages.items():
            page.set_server_data(self.server_data)

    @on_master
    def set_layout(self):
        '''The frontend layout constuctor (only on rank = 0)'''

        search_bar = dash_enrich.html.Div(
            children=[
                dbc.Input(
                    type="search",
                    placeholder="Search...",
                    style={
                        'height':40,
                        'margin-left': 'auto',
                        'margin-right': 4,
                        'display':'inline-block',
                    },
                ),
                dbc.Button(
                    DashIconify(icon="bi:search", width=30, flip="horizontal"),
                    size="sm",   
                    className="btn btn-primary",
                    n_clicks=0,
                    style={
                        'margin-right': 'auto',
                    },
                ),
            ],
            style={'display':'flex'}  
        )

        side_bar = dbc.Offcanvas(
            id="side-menu",
            children=[  
                Lottie(
                    options=dict(
                        loop=True, 
                        autoplay=True, 
                        src=get_asset_url('lotties/robot_hi.json'),
                    ), 
                    width="50%", 
                    url=get_asset_url('lotties/robot_hi.json'),
                ),
                dash_enrich.html.H2("Main menu", style={'text-align':'center'}),
                #dash_enrich.html.Hr(),
                search_bar,
                dash_enrich.html.Hr(),
                dash_enrich.html.P(
                    "Primary apps", className="lead"
                ),
                dbc.Nav(
                    [page.get_link() for key, page in self.pages.items()],
                    vertical=True,
                    pills=True,
                ),
                dash_enrich.html.Hr(),
                dash_enrich.html.P(
                    "Secondary selection", className="lead"
                ),
                dbc.Button(
                    "Contact info",
                    id="open-contacts",
                    n_clicks=0,
                    size="sm",
                    className="btn btn-primary",
                ),
            ],
            scrollable=True,
            close_button=True,
            is_open=False,
        )

        nav_bar = dash_enrich.html.Div(
            id='top-menu-bar',
                children=[
                dbc.Button(
                    DashIconify(icon="majesticons:menu-line", width=30),
                    id="open-side-menu",
                    n_clicks=0,
                    size="sm",
                    className="btn btn-primary",
                    style={'margin-left': '0'}
                ),
                dbc.Button(
                        DashIconify(icon="carbon:apps", width=30),
                        id="open-apps",
                        n_clicks=0,
                        size="sm",
                        className="btn btn-primary",
                        style={
                            'margin-right': 4,
                            'margin-left': 'auto',
                            }
                    ),    
                dbc.Button(
                        DashIconify(icon="radix-icons:avatar", width=30),
                        id="open-login",
                        n_clicks=0,
                        size="sm",
                        className="btn btn-primary",
                        style={
                            'margin-right': 4,
                        }
                    ),
                dbc.Button(
                        DashIconify(icon="carbon:help", width=30),
                        id="open-help",
                        n_clicks=0,
                        size="sm",
                        className="btn btn-primary",
                        style={'margin-left': '4'}
                    ),
                ],
        style={
            'padding':4, 
            'display':'flex',
            'background-color':'rgb(17,17,17)',
            #'margin-bottom':4,
            }
        )

        contact_modal = dbc.Modal(
            id="contact-modal",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Contact")),
                dbc.ModalBody(
                    "Contact information",
                    ),
                dbc.ModalFooter(
                ),
            ],
            is_open=False,
            centered=True,
            )
        
        loggin_modal = dbc.Modal(
            id="login-modal",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Login")),
                dbc.ModalBody(
                    children=[
                        dbc.InputGroup(
                            children=[
                                dbc.InputGroupText(
                                    DashIconify(icon="radix-icons:person"),
                                ), 
                                dbc.Input(placeholder="Username")],
                            className="mb-3",
                        ),
                        dbc.InputGroup(
                            children=[
                                dbc.InputGroupText(
                                    DashIconify(icon="radix-icons:lock-closed"),
                                ), 
                                dbc.Input(placeholder="Password")],
                            className="mb-3",
                        ),
                        dbc.Checklist(
                            id=f'login-remember-me',
                            value=[1],
                            options=[
                                {'label': dash_enrich.html.Div(
                                    'Remember me', 
                                    style={
                                        'color':'white',
                                        'display':'inline', 
                                        'padding-right': 5,
                                        },
                                    ), 
                                'value': 1,
                                },
                            ],
                        ),
                        dbc.Button(
                            "Login",
                            outline=True,
                            id="send-login",
                            n_clicks=0,
                            size="sm-2",
                            className="btn btn-primary",
                            style={'margin-left': '4', 'width':100}
                    ),
                    ],
                ),
                dbc.ModalFooter(
                ),
            ],
            is_open=False,
            centered=True,
            )
        
        loading_overlay = dash_enrich.html.Div(
            id="page-loading-overlay",
            children=dbc.Card(
                dbc.CardBody(
                    children=dash_enrich.html.H4(
                        "Loading app, please wait..", className="card-text"
                    ),
                ),
                color='primary',
                style={'margin':'auto', 'width':400, 'top':'45%'},
            ),
            style={
                'zIndex':10,
                'background-color':'rgb(255, 255, 255, 0.3)',
                'position':'absolute',
                'top':0,
                'left':0,
                'height':'100%',
                'width':'100%',
                },
                hidden=True,
        )

        fs_event = {"event": "fullscreenchange", "props": None}
        self.app.layout = dash_enrich.html.Div(
            [   
                dash_enrich.dcc.Location(id="url"),
                EventListener(id='fullscreen-listener', events=[fs_event]),
                dash_enrich.dcc.Store(id="update-data-status", data=1),
                dash_enrich.dcc.Store(id="client-session-id", data="initial"),
                side_bar,
                nav_bar,
                contact_modal,
                loggin_modal,
                dash_enrich.html.Div(
                    children=[
                        dbc.Fade(
                            is_in=True,
                            id="page-content", 
                            style={
                                'height': '100%', 
                                'padding-top':4, 
                                "transition": "opacity 1000ms ease",
                            },
                        ),
                        loading_overlay,
                    ],
                    style={'height': '100%', 'position':'relative'},
                ),
                dash_enrich.dcc.Store(
                    id='page-load-state', 
                    data='initial',
                ),
                dash_enrich.dcc.Interval(
                    id='reload-page-content-trigger',
                    interval=2000,
                    n_intervals=1,
                ),
                dash_enrich.dcc.Interval(
                    id='server-ping-trigger',
                    interval=5000,
                    n_intervals=1,
                ),
            ],
            style={'height': '100vh', 'display': 'flex', 'flex-direction': 'column'},
        )

    @on_master
    def set_callbacks(self):
        '''Sets all callbacks (callbacks are handled only by rank = 0)'''

        js_open_close = f"""
            function(n1, is_open) {{
                if (n1>0) {{
                    return !is_open
                }} else {{
                    return is_open
                }};
            }}
            """
        
        # open and close side menu
        self.app.clientside_callback(
            js_open_close,
            dash_enrich.Output("side-menu", "is_open"),
            dash_enrich.Input("open-side-menu", "n_clicks"),
            dash_enrich.State("side-menu", "is_open"),
        )

        # add callback for toggling the collapse on small screens
        self.app.clientside_callback(
            js_open_close,
            dash_enrich.Output("navbar-collapse", "is_open"),
            dash_enrich.Input("navbar-toggler", "n_clicks"),
            dash_enrich.State("navbar-collapse", "is_open"),
        )
        
        # open and close contacts
        self.app.clientside_callback(
            js_open_close,
            dash_enrich.Output("contact-modal", "is_open"),
            dash_enrich.Input("open-contacts", "n_clicks"), 
            dash_enrich.State("contact-modal", "is_open"),
        )

        self.app.clientside_callback(
            js_open_close,
            dash_enrich.Output("help-modal", "is_open"),
            dash_enrich.Input("open-help", "n_clicks"), 
            dash_enrich.State("help-modal", "is_open"),
        )

        self.app.clientside_callback(
            js_open_close,
            dash_enrich.Output("login-modal", "is_open"),
            dash_enrich.Input("open-login", "n_clicks"), 
            dash_enrich.State("login-modal", "is_open"),
        )

        self.app.clientside_callback(
            """function(pathname) {{
                if (pathname === "/home") {{
                    return [true, true]
                }} else {{
                    return [false, false]
                }};
            }}
            """,
            dash_enrich.Input("url", "pathname"),
            dash_enrich.Output("page-loading-overlay", "hidden"), 
            dash_enrich.Output("page-content", "is_in"),
            prevent_initial_call=True,
        )

        # client id handling
        @self.app.callback(
            dash_enrich.Output("client-session-id", "data"),
            dash_enrich.Input("server-ping-trigger", "n_intervals"),
            dash_enrich.State("client-session-id", "data"),
        )
        def update_client_counter(n_intervals, data):
            if data in self.active_session_ids.keys():
                self.active_session_ids[data] = time.time()
                return dash_enrich.no_update
            else:
                new_client_id = hash_data((time.time(), random.random()))
                print(f"new client connected: {new_client_id}")
                self.active_session_ids[new_client_id] = time.time()
                return new_client_id
            
        waiting_content = dash_enrich.html.Div(
                        id="please-wait-page",
                        children=[Lottie(
                            options=dict(loop=True, autoplay=True), 
                            height="60vh", 
                            url=get_asset_url('lotties/robot_table.json'),
                        ),
                        dbc.Progress(
                            id='server-load-progress',
                            value=0,
                            striped=True,
                            animated=True,
                            style={
                                'width':'50%',
                                'margin': 'auto',
                            },
                        ),
                        dash_enrich.html.H2(
                            children='The server is getting ready - please wait few seconds.',
                            style={'text-align': 'center', 'margin-top': 5}
                        ),
                        dash_enrich.html.H6(
                            children='It will automaticaly load when ready.',
                            style={'text-align': 'center'},
                        ),
                        ],
                        style={'text-align':'center'},
                    )

        missing_content_404 = dash_enrich.html.Div(
                    id='missing-page-404',
                    children=[
                        Lottie(
                            options=dict(loop=True, autoplay=True), 
                            height="60vh", 
                            url=get_asset_url('lotties/robot_404.json'),
                        ),
                        dash_enrich.html.H1(
                            children='Oops! - something went wrong.',
                            style={'text-align': 'center', 'margin-top': 5}
                        ),
                        dash_enrich.html.H5(
                            children='It seems like this content does not exist.',
                            style={'text-align': 'center'},
                        ),
                   ],
                   style={'text-align':'center', 'margin': 'auto'},
                )
        
        # inserts requested page to content div
        @self.app.callback(
            dash_enrich.Output("page-content", "children"),
            dash_enrich.Output("reload-page-content-trigger", "disabled"),
            dash_enrich.Output('page-load-state', "data"),
            dash_enrich.Output("page-loading-overlay", "hidden", allow_duplicate=True), 
            dash_enrich.Output("page-content", "is_in", allow_duplicate=True),
            dash_enrich.Input("url", "pathname"),
            dash_enrich.Input("reload-page-content-trigger", "n_intervals"),
            dash_enrich.State('page-load-state', "data"),
            #prevent_initial_call=True,
        )
        def update_content(pathname, n_intervals, load_state):
            # Client changed the url -> load new content
            # If page not ready on serverside wait and resend
            if pathname == '/':
                pathname = '/home'

            if not self.server_ready:
                # page is not ready
                if load_state == 'waiting':
                    # waiting content was send before so keep waiting for page
                    return dash_enrich.no_update
                
                elif load_state == 'initial':
                    # send waiting content on first try
                    return waiting_content, False, 'waiting', True, True

            # wrong URL request
            if self.pages.get(pathname) is None:
                return missing_content_404, True, 'loaded', True, True

            # if page is ready send page and disable the reload trigger
            return self.pages[pathname].get_page_content(), True, 'loaded', True, True
            
        # update loading bar
        @self.app.callback(
            dash_enrich.Output('server-load-progress', "value", allow_duplicate=True),
            dash_enrich.Input("reload-page-content-trigger", "n_intervals"),
            prevent_initial_call=True,
        )
        def update_progress(n_intervals):
            return int(self.server_load_progress*100)
        
        # set individual Content item callbacks
        for href, page in self.pages.items():
            page.set_callbacks(self)
            