from dash import get_asset_url, _dash_renderer, ClientsideFunction
import dash_extensions.enrich as dash_enrich
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
import dash_chart_editor as dce
from dash_extensions import EventListener, Lottie
import time
from .pages import HomePage
import datetime
from..utilities import hash_data
import random
from dash_iconify import DashIconify
from urllib.parse import urlparse, parse_qs
from flask import send_file, session
import os
import secrets
_dash_renderer._set_react_version("18.2.0")

class Dasboard():
    def __init__(self, name="", bootstrap_style='bootstrap', credentials=None, secret_key=None, user_permissions=None):

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
        self.bootstrap_style = bootstrap_style
        # trigers at client side
        self.figure_interval = 1000    # checks for data updates
        #server
        self.data_time_stamp = self.get_current_time_stamp()

        #TODO: add client counter
        self.active_session_ids = dict() #id : last ping time
        
        # Authentication
        self.credentials = credentials if credentials else {'admin': 'password'}  # Default credentials
        # Flask session secret key (generate new one if not provided)
        self.secret_key = secret_key if secret_key else secrets.token_hex(32)
        # User permissions: maps username to list of permissions
        # If None, all authenticated users have full access (backwards compatible)
        self.user_permissions = user_permissions

    def build_app(self):
        '''Dash app needs to exist only on rank 0'''
        self.app = dash_enrich.DashProxy(
            __name__, 
            external_scripts=["https://cdn.jsdelivr.net/npm/lodash/lodash.min.js", "https://cdn.plot.ly/plotly-3.0.0.min.js"],
            assets_folder='./assets', 
            external_stylesheets=[
                eval(f"dbc.themes.{self.bootstrap_style.upper()}"), 
                dmc.styles.CODE_HIGHLIGHT, 
                "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.css",
            ],
            transforms=[],
            suppress_callback_exceptions=True,
            prevent_initial_callbacks=True,
        )
        
        # Set Flask secret key for server-side sessions
        self.app.server.secret_key = self.secret_key
        # Configure session to be permanent (stays until browser closes)
        self.app.server.config['SESSION_TYPE'] = 'filesystem'
        self.app.server.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(hours=24)

        #self.pass_server_data_to_pages()
        self.set_layout()
        self.set_callbacks()
        self.add_download_routes()

    def start_app(self, host='127.0.0.1', port=8050):
        '''Run the dash server'''
        #logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self.app.run_server(host=host, port=port)

    def debug_mode(self, host='127.0.0.1', port=8050):
        '''The dash server runs in serial so it blocks other evaluations'''
        self.app.run(host=host, port=port, debug=True)

    # def stay_alive(self, timeout=365*3600):
    #     MPI.COMM_WORLD.Barrier()
    #     time.sleep(timeout)

    def register_page(self, page):
        '''All ranks need access to page constructors'''
        self.pages[page.get_href()] = page
    
    def get_user_permissions(self, username):
        '''Get permissions for a user'''
        if self.user_permissions is None:
            # No RBAC configured - all authenticated users have full access
            return ['*']  # Wildcard = all permissions
        return self.user_permissions.get(username, [])
    
    def user_has_permission(self, username, required_permission):
        '''Check if user has required permission'''
        if required_permission is None or required_permission == 'public':
            return True  # No permission required
        
        user_perms = self.get_user_permissions(username)
        
        # Wildcard permission grants all access
        if '*' in user_perms:
            return True
        
        # Check if user has the specific permission
        return required_permission in user_perms
    
    def add_download_routes(self):
        '''Add Flask routes for file downloads'''
        @self.app.server.route('/download/install_ubuntu.sh')
        def download_install_script():
            # Get the path to the install script relative to the project root
            # The server.py is in heat_battery/visualization/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            script_path = os.path.join(project_root, 'install_scripts', 'install_ubuntu.sh')
            
            if os.path.exists(script_path):
                return send_file(
                    script_path,
                    as_attachment=True,
                    download_name='install_ubuntu.sh',
                    mimetype='text/x-shellscript'
                )
            else:
                return "Install script not found", 404
        
        @self.app.server.route('/download/worker.py')
        def download_worker_script():
            # Get the path to the worker_test.py file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            worker_path = os.path.join(project_root, 'worker_test.py')
            
            if os.path.exists(worker_path):
                return send_file(
                    worker_path,
                    as_attachment=True,
                    download_name='worker.py',
                    mimetype='text/x-python'
                )
            else:
                return "Worker script not found", 404
        
        @self.app.server.route('/download/config_template.yaml')
        def download_config_template():
            # Get the path to the config_template.yaml file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            template_path = os.path.join(project_root, 'config_template.yaml')
            
            if os.path.exists(template_path):
                return send_file(
                    template_path,
                    as_attachment=True,
                    download_name='config.yaml',
                    mimetype='text/yaml'
                )
            else:
                return "Config template not found", 404
        
    # def register_data_updater(self, name, updater):
    #     self.updaters[name] = updater
    #     self.server_data[name] = None

    # def update_data(self):
    #     '''Some data need all mpi ranks to evaluate sucesfully'''
    #     now = time.time()
    #     active_ids = list(self.active_session_ids.keys())
    #     for session_id in active_ids:
    #         if (now - self.active_session_ids[session_id]) > 30:
    #             self.active_session_ids.pop(session_id)
    #             print(f'Client disconected with id: {session_id}')

    #     active_client = len(self.active_session_ids) > 0
    #     active_client = MPI.COMM_WORLD.bcast(active_client)
    #     total = len(self.updaters) + len(self.pages)
    #     finished = 0
    #     self.server_load_progress = 0.0
    #     self.server_load_progress_info = "initialisation"
    #     if active_client or not self.server_ready:
    #         u_start = time.time()
    #         for name, updater in self.updaters.items():
    #             self.server_load_progress_info = f"Running updater: {name}"
    #             self.server_data[name] = updater.reload()
    #             finished += 1
    #             self.server_load_progress = finished/total
    #         print(f"Updaters: {time.time() - u_start}.")
    #         u_start = time.time()
    #         for href, page in self.pages.items():
    #             self.server_load_progress_info = f"Loading page: {href}"
    #             page._update_data()
    #             finished += 1
    #             self.server_load_progress = finished/total
    #             self.server_load_progress_info = f"Finishing"
    #         print(f"Apps: {time.time() - u_start}.")
    #         self.data_time_stamp = self.get_current_time_stamp()
    #         self.server_ready = True

    def preload_cache_data(self):
        for page in self.pages.values():
            page.preload_cache_data()

    # def auto_update(self, freq=10):
    #     while True:
    #         u_start = time.time()
    #         self.update_data()
    #         print(f"Update time: {time.time() - u_start} -> {len(self.active_session_ids)} clients.")
    #         time.sleep(freq)

    def get_current_time_stamp(self):
        return datetime.datetime.now().strftime(f'%m/%d/%Y, %H:%M:%S.%f')

    def set_layout(self):
        '''The frontend layout constuctor (only on rank = 0)'''

        search_bar = dash_enrich.html.Div(
            children=[
                dbc.Input(
                    type="search",
                    placeholder="Search...",
                    style={
                        'height':40,
                        'marginLeft': 'auto',
                        'marginRight': 4,
                        'display':'inline-block',
                    },
                ),
                dbc.Button(
                    DashIconify(icon="bi:search", width=30, flip="horizontal"),
                    size="sm",   
                    className="btn btn-primary",
                    n_clicks=0,
                    style={
                        'marginRight': 'auto',
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
                    id="sidebar-nav",
                    children=[page.get_link() for key, page in self.pages.items() if key == '/home'],
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
                    style={'marginLeft': '0'}
                ),
                dbc.Button(
                        DashIconify(icon="carbon:apps", width=30),
                        id="open-apps",
                        n_clicks=0,
                        size="sm",
                        className="btn btn-primary",
                        style={
                            'marginRight': 4,
                            'marginLeft': 'auto',
                            }
                    ),    
                dash_enrich.html.Div(
                        id="auth-status-display",
                        children=[
                            dbc.Button(
                                DashIconify(icon="radix-icons:avatar", width=30),
                                id="open-login",
                                n_clicks=0,
                                size="sm",
                                className="btn btn-primary",
                                style={
                                    'marginRight': 4,
                                }
                            ),
                        ],
                        style={'display': 'flex', 'alignItems': 'center', 'gap': '4px'}
                    ),
                dbc.Button(
                        DashIconify(icon="carbon:help", width=30),
                        id="open-help",
                        n_clicks=0,
                        size="sm",
                        className="btn btn-primary",
                        style={'marginLeft': '4'}
                    ),
                ],
        style={
            'padding':4, 
            'display':'flex',
            'backgroundColor':'rgb(17,17,17)',
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

        code_modal = dbc.Modal(
            id="code-modal",
            size='xl',
            fullscreen=True,
            fade=True,
            children=[
                dbc.ModalHeader(
                    dbc.ModalTitle(id="code-modal-title"),
                ),
                dbc.ModalBody(
                    dmc.CodeHighlightTabs(
                        id="code-modal-content",
                        code="",
                        copyLabel="Copy to clipboard",
                        copiedLabel="Copied!",
                        style={'minHeight':'100%'},
                        
                    ),
                ),
            ],
        )
        
        loggin_modal = dbc.Modal(
            id="login-modal",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Login")),
                dbc.ModalBody(
                    children=[
                        dbc.Alert(
                            id="login-alert",
                            children="",
                            color="danger",
                            is_open=False,
                            dismissable=True,
                            className="mb-3",
                        ),
                        dbc.InputGroup(
                            children=[
                                dbc.InputGroupText(
                                    DashIconify(icon="radix-icons:person"),
                                ), 
                                dbc.Input(
                                    id="login-username",
                                    placeholder="Username",
                                    type="text",
                                )],
                            className="mb-3",
                        ),
                        dbc.InputGroup(
                            children=[
                                dbc.InputGroupText(
                                    DashIconify(icon="radix-icons:lock-closed"),
                                ), 
                                dbc.Input(
                                    id="login-password",
                                    placeholder="Password",
                                    type="password",
                                )],
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
                        dash_enrich.html.Div(
                            children=[
                                dbc.Button(
                                    "Login",
                                    outline=True,
                                    id="send-login",
                                    n_clicks=0,
                                    size="sm-2",
                                    className="btn btn-primary",
                                    style={'marginLeft': '4', 'width':100}
                                ),
                            ],
                        ),
                    ],
                ),
                dbc.ModalFooter(
                ),
            ],
            is_open=False,
            centered=True,
            )
        
        # chart_editor_modal = dbc.Modal(
        #     id="chart-editor-modal",
        #     fullscreen=True,
        #     is_open=False,
        #     children=[
        #         dbc.ModalHeader(
        #             close_button=False,
        #             style={'gap': 4},
        #             children=[
        #                 dbc.ModalTitle("Chart editor"),
        #                 dbc.Button(
        #                     "Close without saving", 
        #                     id="close-chart-editor-no-save", 
        #                     color="danger",
        #                     size="sm",
        #                     style={'marginLeft': 'auto'},
        #                 ),
        #                 dbc.Button(
        #                     "Save", 
        #                     id="save-chart-editor", 
        #                     color="primary",
        #                     size="sm",
        #                 ),
        #                 dbc.Button(
        #                     "Save & Close", 
        #                     id="save-and-close-chart-editor", 
        #                     color="success",
        #                     size="sm",
        #                 ),
        #             ],
        #         ),
        #         dash_enrich.dcc.Store(
        #             id="chart-editor-store",
        #             data=None,
        #         ),
        #         dce.DashChartEditor(
        #             id="chart-editor-editor",
        #             # saveState=True,
        #             dataSources={
        #                 'x': [1, 2, 3, 4, 5],
        #                 'y': [10, 20, 30, 40, 50],
        #             },
        #             style={'width': '100%', 'height': '100%'},
        #         ),
        #     ],
        #     style={'width': '100%', 'height': '100%'},
        # )
        
        loading_overlay = dash_enrich.html.Div(
            id="page-loading-overlay",
            children=dbc.Card(
                dbc.CardBody(
                    children=dash_enrich.html.H4(
                        "Loading app, please wait..", className="card-text"
                    ),
                ),
                color='primary',
                style={'margin':'auto', 'width':'min(400px, 80vw)', 'top':'45%'},
            ),
            style={
                'zIndex':10,
                'backgroundColor':'rgb(255, 255, 255, 0.3)',
                'position':'absolute',
                'top':0,
                'left':0,
                'height':'100%',
                'width':'100%',
                },
                hidden=True,
        )

        fs_event = {"event": "fullscreenchange", "props": None}
        self.app.layout = dmc.MantineProvider(dash_enrich.html.Div(
            [   
                dash_enrich.dcc.Location(id="url"),
                EventListener(id='fullscreen-listener', events=[fs_event]),
                dash_enrich.dcc.Store(id="update-data-status", data=1),
                dash_enrich.dcc.Store(id="client-session-id", data="initial"),
                dash_enrich.dcc.Store(id="auth-state", storage_type='session', data={'authenticated': False, 'username': None}),
                side_bar,
                nav_bar,
                contact_modal,
                loggin_modal,
                # chart_editor_modal,
                code_modal,
                dash_enrich.html.Div(
                    children=[
                        dbc.Fade(
                            is_in=True,
                            id="page-content", 
                            style={
                                'height': '100%', 
                                'paddingTop':4, 
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
            style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'},
        ))
    
    def get_waiting_content(self):
        waiting_content = dash_enrich.html.Div(
            id="please-wait-page",
            children=[Lottie(
                options=dict(loop=True, autoplay=True), 
                height="60vh", 
                url=get_asset_url('lotties/robot_table.json'),
            ),
            dash_enrich.dcc.Interval(
                id='waiting-page-polling-interval',
                interval=2000,
                n_intervals=1,
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
        return waiting_content

    def get_404_content(self):
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
        return missing_content_404

    def get_content_by_url(self, url, auth_state=None):
        parsed = urlparse(url)
        qs_data = parse_qs(parsed.query)
        
        # Homepage is always accessible
        if parsed.path == '/' or parsed.path == '/home':
            return self.pages['/home'].get_page_content(qs_data)

        # Find the target page
        url_paths = parsed.path.split('/')[1:]
        subpage = self
        for path in url_paths:
            subpage = subpage.pages.get(f"/{path}")
        if subpage is None:
            return self.get_404_content()
        
        # Check page permission requirements
        required_permission = getattr(subpage, 'required_permission', 'authenticated')
        
        # Public pages don't require authentication
        if required_permission == 'public':
            return subpage.get_page_content(qs_data)
        
        # SECURITY: Check server-side session (not client-side auth_state)
        # This prevents attackers from bypassing auth by modifying browser storage
        server_authenticated = session.get('authenticated', False)
        username = session.get('username', None)
        
        # If not authenticated and page requires authentication
        if not server_authenticated:
            return dash_enrich.html.Div(
                children=[
                    Lottie(
                        options=dict(loop=True, autoplay=True), 
                        height="60vh", 
                        url=get_asset_url('lotties/robot_404.json'),
                    ),
                    dash_enrich.html.H1(
                        children='Authentication Required',
                        style={'text-align': 'center', 'margin-top': 5}
                    ),
                    dash_enrich.html.H5(
                        children='Please log in to access this page.',
                        style={'text-align': 'center'},
                    ),
                ],
                style={'text-align':'center', 'margin': 'auto'},
            )
        
        # Check if user has required permission
        if required_permission != 'authenticated' and not self.user_has_permission(username, required_permission):
            return dash_enrich.html.Div(
                children=[
                    Lottie(
                        options=dict(loop=True, autoplay=True), 
                        height="60vh", 
                        url=get_asset_url('lotties/robot_404.json'),
                    ),
                    dash_enrich.html.H1(
                        children='Access Denied',
                        style={'text-align': 'center', 'margin-top': 5}
                    ),
                    dash_enrich.html.H5(
                        children=f'You do not have permission to access this page.',
                        style={'text-align': 'center'},
                    ),
                    dash_enrich.html.P(
                        children=f'Required permission: {required_permission}',
                        style={'text-align': 'center', 'color': '#999'},
                    ),
                ],
                style={'text-align':'center', 'margin': 'auto'},
            )
        
        # User is authenticated and has required permission
        return subpage.get_page_content(qs_data)

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

        # open and close help
        self.app.clientside_callback(
            js_open_close,
            dash_enrich.Output("help-modal", "is_open"),
            dash_enrich.Input("open-help", "n_clicks"), 
            dash_enrich.State("help-modal", "is_open"),
        )

        # open and close login
        self.app.clientside_callback(
            js_open_close,
            dash_enrich.Output("login-modal", "is_open"),
            dash_enrich.Input("open-login", "n_clicks"), 
            dash_enrich.State("login-modal", "is_open"),
        )

        # insert dataSources to chart editor
        # @self.app.callback(
        #     dash_enrich.Input("chart-editor-modal", "is_open"),
        #     dash_enrich.Output("chart-editor-editor", "dataSources"),
        #     prevent_initial_call=True,
        # )
        # def insert_dataSources_to_chart_editor(is_open):
        #     if is_open:
        #         return dash_enrich.no_update
        #     else:
        #         return dash_enrich.no_update

        # save and close chart editor
        # self.app.clientside_callback(
        #     ClientsideFunction(
        #         namespace='clientside',
        #         function_name='save_and_close_chart_editor',
        #     ),
        #     dash_enrich.Input("save-and-close-chart-editor", "n_clicks"), 
        #     dash_enrich.Output("chart-editor-modal", "is_open", allow_duplicate=True),
        #     dash_enrich.Output("chart-editor-editor", "saveState", allow_duplicate=True),
        #     prevent_initial_call=True,
        # )
        
        # # save chart editor without closing
        # self.app.clientside_callback(
        #     ClientsideFunction(
        #         namespace='clientside',
        #         function_name='save_chart_editor',
        #     ),
        #     dash_enrich.Input("save-chart-editor", "n_clicks"), 
        #     dash_enrich.Output("chart-editor-editor", "saveState", allow_duplicate=True),
        #     prevent_initial_call=True,
        # )

        # # close chart editor without saving
        # self.app.clientside_callback(
        #     ClientsideFunction(
        #         namespace='clientside',
        #         function_name='close_no_save_chart_editor',
        #     ),
        #     dash_enrich.Input("close-chart-editor-no-save", "n_clicks"), 
        #     dash_enrich.Output("chart-editor-modal", "is_open", allow_duplicate=True),
        #     prevent_initial_call=True,
        # )

        # # replace figure on save in editor
        # self.app.clientside_callback(
        #     ClientsideFunction(
        #         namespace='clientside',
        #         function_name='replace_figure_on_save',
        #     ),
        #     dash_enrich.Input("chart-editor-editor", "figure"),
        #     dash_enrich.State("chart-editor-store", "data"),
        #     prevent_initial_call=True,
        # ) 

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
        
        # Handle login authentication
        @self.app.callback(
            dash_enrich.Output("auth-state", "data"),
            dash_enrich.Output("login-modal", "is_open", allow_duplicate=True),
            dash_enrich.Output("login-alert", "children"),
            dash_enrich.Output("login-alert", "is_open"),
            dash_enrich.Output("login-username", "value"),
            dash_enrich.Output("login-password", "value"),
            dash_enrich.Input("send-login", "n_clicks"),
            dash_enrich.Input("login-username", "n_submit"),
            dash_enrich.Input("login-password", "n_submit"),
            dash_enrich.State("login-username", "value"),
            dash_enrich.State("login-password", "value"),
            dash_enrich.State("auth-state", "data"),
            prevent_initial_call=True,
        )
        def handle_login(n_clicks, n_submit_username, n_submit_password, username, password, auth_state):
            if not n_clicks and not n_submit_username and not n_submit_password:
                return dash_enrich.no_update, dash_enrich.no_update, "", False, dash_enrich.no_update, dash_enrich.no_update
            
            if username and password and username in self.credentials and self.credentials[username] == password:
                # Successful login - set server-side session
                session['authenticated'] = True
                session['username'] = username
                session['permissions'] = self.get_user_permissions(username)  # Store user permissions
                session.permanent = True  # Keep session until browser closes (or 24h as configured)
                
                # Also update client-side state for UI updates
                return {'authenticated': True, 'username': username}, False, "", False, "", ""
            else:
                # Failed login - ensure session is cleared
                session.clear()
                return auth_state, True, "Invalid username or password", True, dash_enrich.no_update, ""
        
        # Verify server-side session on page load
        @self.app.callback(
            dash_enrich.Output("auth-state", "data", allow_duplicate=True),
            dash_enrich.Input("url", "href"),
            prevent_initial_call=False,  # Run on initial load
        )
        def verify_session_on_load(href):
            """Sync client-side auth-state with server-side session"""
            # Check server-side session
            if session.get('authenticated', False):
                return {
                    'authenticated': True,
                    'username': session.get('username', 'User')
                }
            else:
                # No valid server session - ensure client is logged out
                return {'authenticated': False, 'username': None}
        
        # Update sidebar navigation based on authentication and permissions
        @self.app.callback(
            dash_enrich.Output("sidebar-nav", "children"),
            dash_enrich.Input("auth-state", "data"),
        )
        def update_sidebar_nav(auth_state):
            is_authenticated = auth_state and auth_state.get('authenticated', False)
            username = session.get('username', None) if is_authenticated else None
            
            visible_pages = []
            for key, page in self.pages.items():
                # Get page permission requirement
                required_permission = getattr(page, 'required_permission', 'authenticated')
                
                # Public pages are always visible
                if required_permission == 'public' or key == '/home':
                    visible_pages.append(page)
                    continue
                
                # Skip if not authenticated and page requires authentication
                if not is_authenticated:
                    continue
                
                # If page only requires authentication (no specific permission)
                if required_permission == 'authenticated':
                    visible_pages.append(page)
                    continue
                
                # Check if user has specific permission
                if self.user_has_permission(username, required_permission):
                    visible_pages.append(page)
            
            return [page.get_link() for page in visible_pages]
        
        # Update auth status display in navbar
        @self.app.callback(
            dash_enrich.Output("auth-status-display", "children"),
            dash_enrich.Input("auth-state", "data"),
        )
        def update_auth_display(auth_state):
            if auth_state and auth_state.get('authenticated', False):
                username = auth_state.get('username', 'User')
                return [
                    dash_enrich.html.Span(
                        f"Welcome, {username}",
                        style={'color': 'white', 'marginRight': '8px', 'fontSize': '14px'}
                    ),
                    dbc.Button(
                        children=[
                            DashIconify(icon="mdi:logout", width=20),
                        ],
                        id="logout-button",
                        color="danger",
                        size="sm",
                        outline=True,
                        style={'marginRight': 4}
                    ),
                ]
            else:
                return [
                    dbc.Button(
                        DashIconify(icon="radix-icons:avatar", width=30),
                        id="open-login",
                        n_clicks=0,
                        size="sm",
                        className="btn btn-primary",
                        style={'marginRight': 4}
                    ),
                ]
        
        # Handle logout
        @self.app.callback(
            dash_enrich.Output("auth-state", "data", allow_duplicate=True),
            dash_enrich.Input("logout-button", "n_clicks"),
            prevent_initial_call=True,
        )
        def handle_logout(n_clicks):
            if n_clicks:
                # Clear server-side session
                session.clear()
                # Update client-side state
                return {'authenticated': False, 'username': None}
            return dash_enrich.no_update

        # inserts requested page to content div
        @self.app.callback(
            dash_enrich.Output("page-content", "children"),
            dash_enrich.Input("url", "href"),
            dash_enrich.Input("auth-state", "data"),
            running=[
                (dash_enrich.Output("page-loading-overlay", "hidden"), False, True),
                (dash_enrich.Output("page-content", "is_in"), False, True),
                ],
            
            #prevent_initial_call=True,
        )
        def update_content(pathname, auth_state):
            # Client changed the url -> load new content
            # If page not ready on serverside wait and resend
            return self.get_content_by_url(pathname, auth_state)
            
        # update loading bar
        @self.app.callback(
            dash_enrich.Output('server-load-progress', "value", allow_duplicate=True),
            dash_enrich.Input("waiting-page-polling-interval", "n_intervals"),
            prevent_initial_call=True,
        )
        def update_progress(n_intervals):
            return int(self.server_load_progress*100)
        
        # set individual Content item callbacks
        for href, page in self.pages.items():
            page.set_callbacks(self)
            