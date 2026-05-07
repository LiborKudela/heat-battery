
from heat_battery.visualization import Dasboard, pages
from heat_battery.simulations.postgresql_project import Project
import os

# Configure authentication
credentials = {
    'demo_user': 'demo123',
}
user_permissions = {
    'demo_user': ['*'],  # Can see project overview and results
}

# For production, set: export FLASK_SECRET_KEY='your-secure-random-key'
secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-example-05-secret-key')

# ============================================================================
# PROJECT AND DASHBOARD SETUP
# ============================================================================
project_legacy = Project('project_example_05')
project_copied = Project('project_example_05_copy')

# Create dashboard with secure authentication and RBAC
dashboard = Dasboard(
    'Testing server', 
    bootstrap_style='darkly',           # Bootstrap theme name
    credentials=credentials,            # Enable authentication
    secret_key=secret_key,              # Secure Flask session management
    user_permissions=user_permissions,  # Enable role-based access control
)

# Define initial figure templates for visualization of result data
initial_figures_templates = [
    {   'active': True,
        'title': 'Temperature',
        'plotly_type': 'scattergl',
        'updater_type': 'timeseries',
        'y_names': ['T[0]', 'T_.*'], # column regex or name
        'x_name': 't_timestamp',
        'layout': {
            'yaxis': {
                'title': 'Temperature (°C)',
                #'range': [0, 600]  # Set y-axis limits [min, max]
            }
        }
    },
    {   'active': True,
        'title': 'Heat Flow',
        'plotly_type': 'scattergl',
        'updater_type': 'timeseries',
        'y_names': ['Q_.*'],
        'x_name': 't_timestamp',
        'layout': {
            'yaxis': {
                'title': 'Heat Flow (W)'
                # 'range': [0, 1000]  # Uncomment and adjust as needed
            }
        }
    },
    {   'active': True,
        'title': 'Total Energy',
        'plotly_type': 'scattergl',
        'updater_type': 'timeseries',
        'y_names': ['H_.*'],
        'x_name': 't_timestamp',
        'layout': {
            'yaxis': {
                'title': 'Total Energy (J)'
                # 'range': [0, 1e6]  # Uncomment and adjust as needed
            }
        }
    },
    {   'active': True,
        'title': 'Fraction/Efficiency',
        'plotly_type': 'scattergl',
        'updater_type': 'timeseries',
        'y_names': ['Eff_.*', 'power_toggle'],
        'x_name': 't_timestamp',
        'layout': {
            'yaxis': {
                'title': 'Fraction/Efficiency (-)',
                'range': [0, 1]  # Example: limit to 0-1 range for efficiency
            }
        }
    },
    {   'active': True,
        'title': 'Solver Iterations per Step',
        'plotly_type': 'scattergl',
        'updater_type': 'timeseries',
        'y_names': ['NLS_iter_step', 'KSP_iter_step'],
        'x_name': 't_timestamp',
        'layout': {
            'yaxis': {
                'title': 'Iterations per accepted step (-)',
                #'type': 'log',  # uncomment if KSP_iter_step dwarfs NLS_iter_step
            }
        }
    },
]

# Define variable descriptions for the right-click info modal
# Keys should match the trace names (column names) in your result data
variable_descriptions = {
    'T[0]': 'Temperature at middle of the heating cartridge',
    'T_amb': 'Ambient temperature',
    'T_avg_m': 'Average temperature of the through hole pipes (those that give het to the room)',
    'T_avg_room': 'Average temperature of the room',
    'T_avg_s': 'Average temperature of the outer surface of the storage (usually passing heat to the room)',
    'Q_amb': 'Actual heat loss (heatflow) of the building',
    'Q_bivalent': 'Actual heating power of the backup source',
    'Q_c': 'Actual heating power of the heating cartridges',
    'Q_pv': 'Actual power of the PV panels',
    'Q_m': 'Heat flow rate to the mem pipes',
    'Q_s': 'Heat flow rate to the outer surface of the storage',
    'Q_s2r_total': 'Total heat flow from the storage to the room (via through hole pipes and outer surface)',
    'H_bivalent': 'Total energy passed to the room when heating season active (time integral of Q_bivalent)',
    'H_c': 'Total energy generated in the heating cartridges (time integral of Q_c)',
    'H_pv': 'Total energy generated in the PV panels (time integral of Q_pv)',
    'H_demand': 'Total energy demand of the building (time integral of Q_amb + Q_bivalent)',
    'H_s2r_loss': 'Total energy lost (passed to the room) from the storage when not needed for heating (time integral of Q_loss)',
    'H_s2r_used': 'Total energy passed to the room from the storage when heating season active (time integral of Q_s2r_total)',
    'H_storage': 'Total energy stored in the storage (against reference temperature) (time integral of H_storage)',
    'NLS_iter_step': 'Total SNES (Newton) iterations across all solve attempts in one accepted time step (includes failed retries / dt halving)',
    'KSP_iter_step': 'Total KSP (linear solver) iterations across all solve attempts in one accepted time step',
}

# Register project page (will be protected by authentication)
project_page = pages.ProjectViewerSuperApp(
    project_legacy, 
    fig_theme='plotly_dark',
    initial_figures=initial_figures_templates,
    variable_descriptions=variable_descriptions,
)
dashboard.register_page(project_page)

backup_page = pages.ProjectViewerSuperApp(
    project_copied, 
    fig_theme='plotly_dark',
    initial_figures=initial_figures_templates,
    variable_descriptions=variable_descriptions,
)
dashboard.register_page(backup_page)

# Build the application
dashboard.build_app()
server = dashboard.app.server

if __name__ == '__main__':
    # Start the server in debug mode
    dashboard.debug_mode(host='147.229.141.75', port=8050)
