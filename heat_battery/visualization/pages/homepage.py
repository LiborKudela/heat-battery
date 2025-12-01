from heat_battery.visualization.pages.base import VisualizerApp, dash_enrich
from dash_extensions import Lottie
from dash import get_asset_url
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import time

class HomePage(VisualizerApp):
    def __init__(self, name="Home"):
        super().__init__(name=name, required_permission='public')

    def get_layout(self, qs_data:dict|None=None):
        # Modal for installation instructions
        install_instructions_modal = dbc.Modal(
            id="install-instructions-modal",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Installation Instructions")),
                dbc.ModalBody(
                    children=[
                        dash_enrich.html.H6(
                            "Select your operating system:", 
                            className="mb-2",
                            style={'color': '#ffffff'}
                        ),
                        dbc.RadioItems(
                            id="os-selector",
                            options=[
                                {"label": "Ubuntu / Linux", "value": "ubuntu"},
                                {"label": "Windows (WSL)", "value": "windows"},
                            ],
                            value="ubuntu",
                            inline=True,
                            className="mb-3",
                            style={'color': '#ffffff'}
                        ),
                        dash_enrich.html.Hr(style={'borderColor': '#ffffff'}),
                        
                        # Windows WSL Instructions
                        dash_enrich.html.Div(
                            id="windows-instructions",
                            children=[
                                dash_enrich.html.H6(
                                    "Windows Installation (WSL Ubuntu 22.04):", 
                                    className="mb-3",
                                    style={'color': '#ffffff'}
                                ),
                                dbc.Card(
                                    dbc.CardBody([
                                        dash_enrich.html.H6(
                                            "Step 1: Enable Virtualization in BIOS", 
                                            className="mb-2",
                                            style={'color': '#198754', 'fontWeight': 'bold'}
                                        ),
                                        dash_enrich.html.P([
                                            "Before installing WSL, ensure virtualization is enabled in your BIOS/UEFI:",
                                        ], style={'color': '#ffffff', 'fontSize': '0.95em', 'marginBottom': '10px'}),
                                        dash_enrich.html.Ul([
                                            dash_enrich.html.Li("Restart your computer and enter BIOS/UEFI (usually F2, F12, Del, or Esc during boot)", style={'color': '#ffffff', 'fontSize': '0.9em'}),
                                            dash_enrich.html.Li("Look for 'Virtualization Technology', 'Intel VT-x', or 'AMD-V'", style={'color': '#ffffff', 'fontSize': '0.9em'}),
                                            dash_enrich.html.Li("Enable the virtualization option", style={'color': '#ffffff', 'fontSize': '0.9em'}),
                                            dash_enrich.html.Li("Save and exit BIOS", style={'color': '#ffffff', 'fontSize': '0.9em'}),
                                        ]),
                                    ]),
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    dbc.CardBody([
                                        dash_enrich.html.H6(
                                            "Step 2: Install WSL Ubuntu 22.04", 
                                            className="mb-2",
                                            style={'color': '#198754', 'fontWeight': 'bold'}
                                        ),
                                        dash_enrich.html.P(
                                            "Open PowerShell as Administrator and run:",
                                            style={'color': '#ffffff', 'fontSize': '0.95em', 'marginBottom': '10px'}
                                        ),
                                        dash_enrich.html.Pre(
                                            dash_enrich.html.Code(
                                                "wsl --install -d Ubuntu-22.04",
                                                style={'color': '#212529'}
                                            ),
                                            style={
                                                'backgroundColor': '#f8f9fa', 
                                                'padding': '10px', 
                                                'borderRadius': '5px',
                                                'border': '1px solid #dee2e6'
                                            }
                                        ),
                                        dash_enrich.html.Small(
                                            "You may need to restart your computer after installation",
                                            style={'color': '#ffffff', 'fontWeight': '500'}
                                        ),
                                    ]),
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    dbc.CardBody([
                                        dash_enrich.html.H6(
                                            "Step 3: Set up Ubuntu user", 
                                            className="mb-2",
                                            style={'color': '#198754', 'fontWeight': 'bold'}
                                        ),
                                        dash_enrich.html.P(
                                            "After installation, Ubuntu will prompt you to create a username and password. Follow the on-screen instructions.",
                                            style={'color': '#ffffff', 'fontSize': '0.95em'}
                                        ),
                                    ]),
                                    className="mb-3",
                                ),
                                dbc.Card(
                                    dbc.CardBody([
                                        dash_enrich.html.H6(
                                            "Step 4: Continue with Ubuntu installation", 
                                            className="mb-2",
                                            style={'color': '#198754', 'fontWeight': 'bold'}
                                        ),
                                        dash_enrich.html.P(
                                            "Once WSL Ubuntu is set up, switch to 'Ubuntu / Linux' tab above and follow those instructions inside your WSL terminal.",
                                            style={'color': '#ffffff', 'fontSize': '0.95em'}
                                        ),
                                    ]),
                                    className="mb-3",
                                ),
                            ],
                            style={'display': 'none'}
                        ),
                        
                        # Ubuntu Instructions
                        dash_enrich.html.Div(
                            id="ubuntu-instructions",
                            children=[
                                dash_enrich.html.Div(
                                    dbc.Button(
                                        children=[
                                            DashIconify(icon="mdi:download", width=20, style={'marginRight': '8px'}),
                                            "Download Ubuntu Install Script"
                                        ],
                                        href="/download/install_ubuntu.sh",
                                        external_link=True,
                                        color="primary",
                                        size="lg",
                                    ),
                                    style={'textAlign': 'center', 'marginBottom': '20px'}
                                ),
                                dash_enrich.html.H6(
                                    "Follow these steps to install Heat Battery:", 
                                    className="mb-3",
                                    style={'color': '#ffffff'}
                                ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Step 1: Make the script executable", 
                                    className="mb-2",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.Pre(
                                    dash_enrich.html.Code(
                                        "chmod +x install_ubuntu.sh",
                                        style={'color': '#212529'}
                                    ),
                                    style={
                                        'backgroundColor': '#f8f9fa', 
                                        'padding': '10px', 
                                        'borderRadius': '5px',
                                        'border': '1px solid #dee2e6'
                                    }
                                ),
                            ]),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Step 2: Run the basic installation", 
                                    className="mb-2",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.Pre(
                                    dash_enrich.html.Code(
                                        "./install_ubuntu.sh -y",
                                        style={'color': '#212529'}
                                    ),
                                    style={
                                        'backgroundColor': '#f8f9fa', 
                                        'padding': '10px', 
                                        'borderRadius': '5px',
                                        'border': '1px solid #dee2e6'
                                    }
                                ),
                            ]),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Step 3: (Optional) Install with PostgreSQL", 
                                    className="mb-2",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.P(
                                    "Install with PostgreSQL if you intend to host a project database yourself.",
                                    style={'color': '#ffffff', 'fontSize': '0.95em', 'marginBottom': '10px'}
                                ),
                                dash_enrich.html.Pre(
                                    dash_enrich.html.Code(
                                        "./install_ubuntu.sh -y -p --ppass yourpassword --ppassc yourpassword",
                                        style={'color': '#212529'}
                                    ),
                                    style={
                                        'backgroundColor': '#f8f9fa', 
                                        'padding': '10px', 
                                        'borderRadius': '5px',
                                        'border': '1px solid #dee2e6'
                                    }
                                ),
                                dash_enrich.html.Small(
                                    "Replace 'yourpassword' with your desired PostgreSQL password",
                                    style={'color': '#ffffff', 'fontWeight': '500'}
                                ),
                            ]),
                            className="mb-3",
                        ),
                                dbc.Alert(
                                    [
                                        DashIconify(icon="mdi:information", width=20, style={'marginRight': '8px'}),
                                        dash_enrich.html.Span("For more options, run: ", style={'color': '#ffffff'}),
                                        dash_enrich.html.Code(
                                            "./install_ubuntu.sh --help",
                                            style={'color': '#212529', 'backgroundColor': '#e7f3ff', 'padding': '2px 6px', 'borderRadius': '3px'}
                                        )
                                    ],
                                    color="info",
                                ),
                            ],
                        ),
                    ],
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-install-instructions-modal", color="secondary", size="sm")
                ),
            ],
            size="lg",
            is_open=False,
            centered=True,
            scrollable=True,
        )
        
        # Modal for worker instructions
        worker_instructions_modal = dbc.Modal(
            id="worker-instructions-modal", 
            children=[
                dbc.ModalHeader(dbc.ModalTitle("How to Run Worker Script")),
                dbc.ModalBody(
                    children=[
                        dash_enrich.html.Div(
                            dbc.Button(
                                children=[
                                    DashIconify(icon="mdi:download", width=20, style={'marginRight': '8px'}),
                                    "Download Basic Worker Script"
                                ],
                                href="/download/worker.py",
                                external_link=True,
                                color="success",
                                size="lg",
                            ),
                            style={'textAlign': 'center', 'marginBottom': '20px'}
                        ),
                        dash_enrich.html.H6(
                            "Follow these steps to run the worker script:", 
                            className="mb-3",
                            style={'color': '#ffffff'}
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Step 1: Download the required files", 
                                    className="mb-2",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.P([
                                    "You need both the ",
                                    dash_enrich.html.Strong("worker.py"),
                                    " script and the ",
                                    dash_enrich.html.Strong("config.yaml"),
                                    " file to run the worker.",
                                    " The config file contains the database connection details ",
                                    "including the password for the host database.",
                                    "This will allow the worker to pull jobs from the project,",
                                    "Run them and send results to the database."
                                ], style={'color': '#ffffff', 'fontSize': '0.95em'}),
                            ]),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Step 2: Configure the config.yaml", 
                                    className="mb-2",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.P(
                                    "Download and edit the config.yaml file to set your database connection details (host, port, username, password, etc.).",
                                    style={'color': '#ffffff', 'fontSize': '0.95em', 'marginBottom': '10px'}
                                ),
                                dash_enrich.html.Div(
                                    dbc.Button(
                                        children=[
                                            DashIconify(icon="mdi:download", width=18, style={'marginRight': '6px'}),
                                            "Download Config Template"
                                        ],
                                        href="/download/config_template.yaml",
                                        external_link=True,
                                        color="info",
                                        size="sm",
                                        outline=True,
                                    ),
                                    style={'marginTop': '10px'}
                                ),
                            ]),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Step 3: Place the config file", 
                                    className="mb-2",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.P(
                                    "Place the config.yaml file in the same directory as your worker.py script, or in your project root directory.",
                                    style={'color': '#ffffff', 'fontSize': '0.95em', 'marginBottom': '10px'}
                                ),
                                dash_enrich.html.Pre(
                                    dash_enrich.html.Code(
                                        "your_project/\n├── config.yaml\n└── worker.py",
                                        style={'color': '#212529'}
                                    ),
                                    style={
                                        'backgroundColor': '#f8f9fa', 
                                        'padding': '10px', 
                                        'borderRadius': '5px',
                                        'border': '1px solid #dee2e6'
                                    }
                                ),
                            ]),
                            className="mb-3",
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Step 4: Run the worker script", 
                                    className="mb-2",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.P([
                                    "Execute the worker script using MPI and Python.",
                                    "Replace the number after -n with the number of cores you want to use.",
                                    "The rule of thumb is to use few less cores than the number of physical cores on the machine.",
                                ],
                                    style={'color': '#ffffff', 'fontSize': '0.95em', 'marginBottom': '10px'}
                                ),
                                dash_enrich.html.Pre(
                                    dash_enrich.html.Code(
                                        "mpirun -n 4 python3 worker.py",
                                        style={'color': '#212529'}
                                    ),
                                    style={
                                        'backgroundColor': '#f8f9fa', 
                                        'padding': '10px', 
                                        'borderRadius': '5px',
                                        'border': '1px solid #dee2e6'
                                    }
                                ),
                            ]),
                            className="mb-3",
                        ),
                        dbc.Alert(
                            [
                                DashIconify(icon="mdi:information", width=20, style={'marginRight': '8px'}),
                                dash_enrich.html.Span(
                                    ["The worker will connect to the project database and process scheduled ",
                                    "jobs automatically with checkpoint policy specified in the host project."],
                                    style={'color': '#ffffff'}),
                            ],
                            color="info",
                        ),
                    ],
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-worker-instructions-modal", color="secondary", size="sm")
                ),
            ],
            size="lg",
            is_open=False,
            centered=True,
            scrollable=True,
        )
        
        # Modal for request access instructions
        request_access_modal = dbc.Modal(
            id="request-access-modal",
            children=[
                dbc.ModalHeader(dbc.ModalTitle("Request Access")),
                dbc.ModalBody(
                    children=[
                        dash_enrich.html.Div(
                            children=[
                                DashIconify(icon="mdi:email-outline", width=60, style={'color': '#667eea'}),
                            ],
                            style={'textAlign': 'center', 'marginBottom': '20px'}
                        ),
                        dash_enrich.html.H5(
                            "How to Request an Account",
                            className="mb-3",
                            style={'color': '#ffffff', 'textAlign': 'center'}
                        ),
                        dash_enrich.html.P(
                            "To request access to the HiTepOptim Project Viewer, please send an email with the following information:",
                            style={'color': '#ffffff', 'fontSize': '0.95em', 'marginBottom': '20px'}
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                dash_enrich.html.H6(
                                    "Email Template",
                                    className="mb-3",
                                    style={'color': '#198754', 'fontWeight': 'bold'}
                                ),
                                dash_enrich.html.P([
                                    dash_enrich.html.Strong("To: ", style={'color': '#ffffff'}),
                                    dash_enrich.html.A(
                                        "admin@hitepoptim.org",
                                        href="mailto:admin@hitepoptim.org?subject=Account%20Request%20-%20HiTepOptim%20Viewer&body=Hello%2C%0A%0AI%20would%20like%20to%20request%20access%20to%20the%20HiTepOptim%20Project%20Viewer.%0A%0AName%3A%20%0AOrganization%3A%20%0AEmail%3A%20%0AReason%20for%20access%3A%20%0A%0AThank%20you.",
                                        style={'color': '#667eea', 'textDecoration': 'underline'}
                                    ),
                                ], style={'marginBottom': '15px'}),
                                dash_enrich.html.P([
                                    dash_enrich.html.Strong("Subject: ", style={'color': '#ffffff'}),
                                    dash_enrich.html.Span("Account Request - HiTepOptim Viewer", style={'color': '#a0aec0'}),
                                ], style={'marginBottom': '15px'}),
                                dash_enrich.html.P(
                                    dash_enrich.html.Strong("Body:", style={'color': '#ffffff'}),
                                    style={'marginBottom': '10px'}
                                ),
                                dash_enrich.html.Pre(
                                    dash_enrich.html.Code(
                                        "Hello,\n\n"
                                        "I would like to request access to the HiTepOptim Project Viewer.\n\n"
                                        "Name: [Your Full Name]\n"
                                        "Organization: [Your Organization]\n"
                                        "Email: [Your Email]\n"
                                        "Reason for access: [Brief description]\n\n"
                                        "Thank you.",
                                        style={'color': '#212529', 'fontSize': '0.85em'}
                                    ),
                                    style={
                                        'backgroundColor': '#f8f9fa',
                                        'padding': '15px',
                                        'borderRadius': '5px',
                                        'border': '1px solid #dee2e6',
                                        'overflowX': 'auto'
                                    }
                                ),
                            ]),
                            className="mb-3",
                        ),
                        dbc.Alert(
                            [
                                DashIconify(icon="mdi:information", width=20, style={'marginRight': '8px'}),
                                dash_enrich.html.Span(
                                    "You will receive your login credentials via email within 24-48 hours.",
                                    style={'color': '#ffffff'}
                                ),
                            ],
                            color="info",
                        ),
                        dash_enrich.html.Div(
                            dbc.Button(
                                children=[
                                    DashIconify(icon="mdi:email-fast", width=20, style={'marginRight': '8px'}),
                                    "Send Email Now"
                                ],
                                href="mailto:admin@hitepoptim.org?subject=Account%20Request%20-%20HiTepOptim%20Viewer&body=Hello%2C%0A%0AI%20would%20like%20to%20request%20access%20to%20the%20HiTepOptim%20Project%20Viewer.%0A%0AName%3A%20%0AOrganization%3A%20%0AEmail%3A%20%0AReason%20for%20access%3A%20%0A%0AThank%20you.",
                                color="primary",
                                size="lg",
                                external_link=True,
                                className="mt-3",
                            ),
                            style={'textAlign': 'center'}
                        ),
                    ],
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-request-access-modal", color="secondary", size="sm")
                ),
            ],
            size="lg",
            is_open=False,
            centered=True,
            scrollable=True,
        )
        
        return dash_enrich.html.Div(
            children=[
                # Animated title
                dash_enrich.html.Div(
                    children=[
                        dash_enrich.html.H1(
                            "HiTepOptim Project Viewer",
                            style={
                                'fontSize': 'clamp(2rem, 6vw, 4rem)',
                                'fontWeight': '800',
                                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
                                'WebkitBackgroundClip': 'text',
                                'WebkitTextFillColor': 'transparent',
                                'backgroundClip': 'text',
                                'textAlign': 'center',
                                'margin': '0',
                                'padding': '20px 10px',
                                'letterSpacing': '2px',
                                'animation': 'slideDown 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55), shimmer 3s ease-in-out infinite',
                                'position': 'relative',
                            }
                        ),
                        dash_enrich.html.Div(
                            style={
                                'width': '200px',
                                'height': '4px',
                                'background': 'linear-gradient(90deg, transparent, #667eea, #764ba2, transparent)',
                                'margin': '0 auto 20px',
                                'borderRadius': '2px',
                                'animation': 'expandWidth 1s ease-out 0.5s both',
                            }
                        ),
                    ],
                    style={
                        'marginBottom': '20px',
                    }
                ),
                Lottie(
                    id='welcome-lottie',
                    options=dict(loop=True, autoplay=True), 
                    height="42vh",
                    url=get_asset_url('lotties/solar_battery.json'),
                ),
                dash_enrich.html.Div(
                    children=[
                        dash_enrich.html.H4(
                            "Get Started",
                            style={'textAlign': 'center', 'marginBottom': '20px'}
                        ),
                        dbc.Button(
                            children=[
                                DashIconify(icon="mdi:information", width=20, style={'marginRight': '8px'}),
                                "Installation Instructions"
                            ],
                            id="open-install-instructions-modal",
                            color="primary",
                            size="lg",
                            style={'width': '350px'},
                        ),
                        dbc.Button(
                            children=[
                                DashIconify(icon="mdi:information", width=20, style={'marginRight': '8px'}),
                                "How to Run Worker Script"
                            ],
                            id="open-worker-instructions-modal",
                            color="success",
                            size="lg",
                            style={'width': '350px'},
                        ),
                        dbc.Button(
                            children=[
                                DashIconify(icon="mdi:book-open-page-variant", width=20, style={'marginRight': '8px'}),
                                "Read Full Documentation"
                            ],
                            href="https://github.com/yourusername/HeatBattery/wiki",
                            target="_blank",
                            color="warning",
                            size="lg",
                            external_link=True,
                            style={'width': '350px'},
                        ),
                        # Login and Request Access buttons side by side
                        dash_enrich.html.Div(
                            children=[
                                dbc.Button(
                                    children=[
                                        DashIconify(icon="mdi:account-plus", width=20, style={'marginRight': '8px'}),
                                        "Request Access"
                                    ],
                                    id="open-request-access-modal",
                                    color="info",
                                    size="lg",
                                    style={'width': '170px'},
                                ),
                                dbc.Button(
                                    children=[
                                        DashIconify(icon="mdi:login", width=20, style={'marginRight': '8px'}),
                                        "Log In"
                                    ],
                                    id="open-login-homepage",
                                    color="secondary",
                                    size="lg",
                                    style={'width': '170px'},
                                ),
                            ],
                            style={
                                'display': 'flex',
                                'gap': '10px',
                                'justifyContent': 'center',
                            }
                        ),
                    ],
                    style={
                        'textAlign': 'center', 
                        'marginTop': '20px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'gap': '5px'
                    }
                ),
                install_instructions_modal,
                worker_instructions_modal,
                request_access_modal,
            ],
        )
    
    def set_callbacks(self, dashboard):
        '''Set up callbacks for the homepage'''
        
        # Open/close installation instructions modal
        @dashboard.app.callback(
            dash_enrich.Output("install-instructions-modal", "is_open"),
            dash_enrich.Input("open-install-instructions-modal", "n_clicks"),
            dash_enrich.Input("close-install-instructions-modal", "n_clicks"),
            dash_enrich.State("install-instructions-modal", "is_open"),
            prevent_initial_call=True,
        )
        def toggle_install_instructions_modal(n_open, n_close, is_open):
            return not is_open
        
        # Toggle between Windows and Ubuntu installation instructions
        @dashboard.app.callback(
            dash_enrich.Output("windows-instructions", "style"),
            dash_enrich.Output("ubuntu-instructions", "style"),
            dash_enrich.Input("os-selector", "value"),
        )
        def toggle_os_instructions(selected_os):
            if selected_os == "windows":
                return {'display': 'block'}, {'display': 'none'}
            else:
                return {'display': 'none'}, {'display': 'block'}
        
        # Open/close worker instructions modal
        @dashboard.app.callback(
            dash_enrich.Output("worker-instructions-modal", "is_open"),
            dash_enrich.Input("open-worker-instructions-modal", "n_clicks"),
            dash_enrich.Input("close-worker-instructions-modal", "n_clicks"),
            dash_enrich.State("worker-instructions-modal", "is_open"),
            prevent_initial_call=True,
        )
        def toggle_worker_instructions_modal(n_open, n_close, is_open):
            return not is_open
        
        # Open login modal from homepage
        @dashboard.app.callback(
            dash_enrich.Output("login-modal", "is_open", allow_duplicate=True),
            dash_enrich.Input("open-login-homepage", "n_clicks"),
            dash_enrich.State("login-modal", "is_open"),
            prevent_initial_call=True,
        )
        def open_login_from_homepage(n_clicks, is_open):
            if n_clicks:
                return True
            return is_open
        
        # Open/close request access modal
        @dashboard.app.callback(
            dash_enrich.Output("request-access-modal", "is_open"),
            dash_enrich.Input("open-request-access-modal", "n_clicks"),
            dash_enrich.Input("close-request-access-modal", "n_clicks"),
            dash_enrich.State("request-access-modal", "is_open"),
            prevent_initial_call=True,
        )
        def toggle_request_access_modal(n_open, n_close, is_open):
            return not is_open