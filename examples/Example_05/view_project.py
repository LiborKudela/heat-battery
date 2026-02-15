
from heat_battery.visualization import Dasboard, pages
from heat_battery.simulations.postgresql_project import Project
import os

# ============================================================================
# AUTHENTICATION AND ROLE-BASED ACCESS CONTROL CONFIGURATION
# ============================================================================
# Configure authentication credentials
# WARNING: Change these credentials for production use!
credentials = {
    'admin': 'heatbattery2024',
    'provyko': 'secret',
    'demo': 'demo123',
    'viewer': 'viewonly',
}

# Configure user permissions (Role-Based Access Control)
# Each user is mapped to a list of permissions they have
user_permissions = {
    'admin': ['*'],  # '*' = all permissions (wildcard)
    'provyko': ['*'],
    'demo': ['view_project', 'view_results'],  # Can see project overview and results
    'viewer': ['view_project'],  # Can only see project main page
}

# Get Flask secret key from environment variable (recommended)
# If not set, uses development key (users logged out on server restart)
# For production, set: export FLASK_SECRET_KEY='your-secure-random-key'
secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-example-05-secret-key')

# ============================================================================
# PROJECT AND DASHBOARD SETUP
# ============================================================================
project_legacy = Project('project_example_05')

# Create dashboard with secure authentication and RBAC
dashboard = Dasboard(
    'Testing server', 
    bootstrap_style='darkly',
    credentials=credentials,        # Enable authentication
    secret_key=secret_key,          # Secure Flask session management
    user_permissions=user_permissions,  # Enable role-based access control
)

# Define initial figure templates for visualization
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
}

# Define initial comparator chart templates
initial_comparator_templates = [
    {
        'active': True,
        'title': 'Job Priority vs Mesh Size',
        'chart_type': 'scatter',
        'x_column': 'priority',
        'y_column': 'input_mesh_p_size',
        'layout': {
            'xaxis': {
                'title': 'Job Priority'
            },
            'yaxis': {
                'title': 'Mesh Size (P)',
                #'range': [0, 10]
            }
        }
    },
    {
        'active': True,
        'title': 'Storage Size vs Prize',
        'chart_type': 'scatter',
        'x_column': 'input_mesh_p_size',
        'y_column': 'output_total_price_value',
        'layout': {
            'xaxis': {
                'title': 'Size of the storage (Diameter, Height)'
            },
            'yaxis': {
                'title': 'Prize (€)',
                #'range': [0, 10]
            }
        }
    },
    {
        'active': True,
        'title': 'Storage Size vs Prize',
        'chart_type': 'scatter',
        'x_column': 'input_mesh_p_size',
        'y_column': 'output_total_price_value',
        'layout': {
            'xaxis': {
                'title': 'Size of the storage (Diameter, Height)'
            },
            'yaxis': {
                'title': 'Prize (€)',
                #'range': [0, 10]
            }
        }
    },
    {
        'active': True,
        'title': 'Storage Size vs Prize',
        'chart_type': 'scatter',
        'x_column': 'input_mesh_p_size',
        'y_column': 'output_total_price_value',
        'layout': {
            'xaxis': {
                'title': 'Size of the storage (Diameter, Height)'
            },
            'yaxis': {
                'title': 'Prize (€)',
                #'range': [0, 10]
            }
        }
    }
]

# Define default data transform presets
# These will be automatically loaded into localStorage when the page loads
# Users can add, edit, or delete these transforms, but the defaults will be available
# Each transform is composed of multiple steps that are executed in sequence
initial_transforms_presets = [   
    {
        'id': 'default_monthly_change',
        'name': 'Monthly Energy Change',
        'description': 'Calculate monthly change values for all energy columns',
        'steps': [
            {
                'type': 'column_selection',
                'description': 'Select allenergy columns starting with H_',
                'column_pattern': 'H_.*',
                'operation': 'select'
            },
            {
                'type': 'time_aggregation',
                'description': 'Resample to values at the end of each month',
                'operation': 'resample',
                'frequency': '1M',
                'method': 'last'
            },
            {
                'type': 'column_transform',
                'description': 'Calculate difference between current and previous month',
                'operation': 'diff',
                'new_column_suffix': '_diff'
            }
        ]
    }
]

# Register project page (will be protected by authentication)
project_page = pages.ProjectViewerSuperApp(
    project_legacy, 
    fig_theme='plotly_dark',
    initial_figures=initial_figures_templates,
    initial_comparator_figures=initial_comparator_templates,
    variable_descriptions=variable_descriptions,
    initial_transforms=initial_transforms_presets,
)

# Set permissions for project and subpages
# Main project page requires 'view_project' permission
project_page.set_permission('public')

# Set specific permissions for each subpage
# Jobs table requires 'view_jobs' permission
project_page.pages['/jobs-overview'].set_permission('view_jobs')

# Result data requires 'view_results' permission
project_page.pages['/result-data'].set_permission('view_results')

# Result comparator requires 'view_comparator' permission
project_page.pages['/result-comparator'].set_permission('view_comparator')

dashboard.register_page(project_page)

# Build the application
dashboard.build_app()
server = dashboard.app.server

if __name__ == '__main__':
    # Print authentication info
    # print("=" * 80)
    # print("Heat Battery Dashboard - Example 05 with Role-Based Access Control")
    # print("=" * 80)
    # print("\n🔒 AUTHENTICATION ENABLED (Flask Sessions + RBAC)")
    # print("\nAvailable login credentials:")
    # print("  ┌─────────────┬──────────────────┬────────────────────────────────────────┐")
    # print("  │ Username    │ Password         │ Permissions                            │")
    # print("  ├─────────────┼──────────────────┼────────────────────────────────────────┤")
    # print("  │ provyko     │ secret           │ All access (*)                         │")
    # print("  │ admin       │ heatbattery2024  │ All access (*)                         │")
    # print("  │ researcher  │ lab_access       │ Project, Jobs, Results, Comparator     │")
    # print("  │ demo        │ demo123          │ Project, Results only                  │")
    # print("  │ viewer      │ viewonly         │ Project overview only                  │")
    # print("  └─────────────┴──────────────────┴────────────────────────────────────────┘")
    # print("=" * 80)
    # print()
    
    # Start the server in debug mode
    dashboard.debug_mode(host='147.229.141.75', port=8050)
