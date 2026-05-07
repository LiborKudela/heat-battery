from heat_battery.visualization.pages.base import dash_enrich, ClientsideFunction, dbc      
from heat_battery.visualization.components.grids import GridLayout
from heat_battery.simulations.postgresql_project import Project
import pandas as pd
import time
import numpy as np
import dash_chart_editor as dce
from dash import no_update, Patch

DEFAULT_COMPARATOR_REFRESH_INTERVAL = 10000  # milliseconds

DROP_COLUMNS = [
    'checkpoint_data',
    'required_source_files',
    'error_log', 'probe_columns',
    'input_PRIORITY','input_SIGNATURE', 'input_SIM_CLASS_NAME', 'input_GROUP_SIGNATURE',
    'input_MESH_BUILDER_NAME', 'input_SIM_CLASS_FILENAME',
    'input_SIM_CLASS_SIGNATURE', 'input_MESH_BUILDER_FILENAME',
    'input_MESH_BUILDER_SIGNATURE',
]

class ResultComparatorComponent(GridLayout):
    """Component for comparing results across multiple jobs"""
    
    def __init__(
            self, 
            project: Project, 
            parent=None,
            fig_theme: str = 'bootstrap',
            initial_figures: list | None = None,
        ):
        self.fig_theme = fig_theme
        self.initial_figures = initial_figures or []
        self.project = project
        self.comparator_data = None
        super().__init__([], parent=parent)

    def _build_comparator_dataframe(self, exclude_pending=True):
        """Build a dataframe combining job metadata and input parameters for comparison
        
        Args:
            exclude_pending: If True, exclude jobs with status SCHEDULED or FAILED
        """
        total_start_time = time.time()
        
        # Get jobs data from the database
        print('[Comparator] Fetching jobs from database...')
        jobs_start = time.time()
        from heat_battery.simulations.jobs import Job
        if exclude_pending:
            jobs_rows = self.project._get_non_scheduled_jobs_query()
        else:
            jobs_rows = self.project._get_jobs_query()
        jobs_df = pd.DataFrame(jobs_rows, columns=Job.COLUMNS.keys())
        print(f'[Comparator] Jobs query completed in {time.time() - jobs_start:.3f}s ({len(jobs_df)} jobs)')
        
        if jobs_df.empty:
            print('[Comparator] No job data found')
            return pd.DataFrame()
        
        # Flatten p_inputs column into separate columns
        print('[Comparator] Flattening p_inputs...')
        flatten_start = time.time()
        
        def flatten_dict(inputs_dict, prefix='input'):
            """Flatten nested input dictionary into flat dict with prefixed keys"""
            if not inputs_dict or not isinstance(inputs_dict, dict):
                return {}
            flat = {}
            for key, value in inputs_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat[f'{prefix}_{key}_{subkey}'] = subvalue
                else:
                    flat[f'{prefix}_{key}'] = value
            return flat

        def flatten_inputs(inputs_dict):
            return flatten_dict(inputs_dict, 'input')

        def flatten_output(output_dict):
            return flatten_dict(output_dict, 'output')

        
        # Apply flattening to each row's p_inputs and expand into columns
        flattened_inputs = jobs_df['p_inputs'].apply(flatten_inputs)
        flattened_outputs = jobs_df['output_build'].apply(flatten_output)
        inputs_df = pd.DataFrame(flattened_inputs.tolist())
        outputs_df = pd.DataFrame(flattened_outputs.tolist())
        
        # Combine original dataframe with flattened inputs
        # Drop the original p_inputs column as it's now flattened
        comparator_df = pd.concat([jobs_df, inputs_df, outputs_df], axis=1)

        # drop all columns ending with _unit
        comparator_df.drop(columns=DROP_COLUMNS, inplace=True)
        comparator_df.drop(columns=comparator_df.columns[comparator_df.columns.str.endswith('_unit')], inplace=True)
        print(comparator_df['output_total_price_value'])

        # add types to columns in brackets
        
        print(f'[Comparator] Flattening completed in {time.time() - flatten_start:.3f}s')
        
        total_time = time.time() - total_start_time
        print(f'[Comparator] Total build time: {total_time:.3f}s ({len(comparator_df)} rows, {len(comparator_df.columns)} columns)')
        return comparator_df

    def get_layout(self, qs_data: dict | None = None):
        start_time = time.time()
        
        # Preload dataset on page load (with exclude_pending=True by default)
        df = self._build_comparator_dataframe(exclude_pending=True)
        if df.empty:
            initial_dataset = None
            initial_status = 'No data available'
        else:
            initial_dataset = df.to_dict('records')
            initial_status = f'Dataset loaded: {len(df)} jobs, {len(df.columns)} columns'
        
        # Top bar with controls
        top_bar = dash_enrich.html.Div(
            id=f'{self.id}-top-bar',
            children=[
                dbc.DropdownMenu(
                    label='Add chart',
                    id=f'{self.id}-add-chart-dropdown', 
                    color='success',
                    menu_variant='dark',
                    size='sm',
                    style={
                        'margin': '0px', 
                        'fontWeight': 'bold',   
                    },
                    children=[
                        dbc.DropdownMenuItem("Chart types", header=True),
                        dbc.DropdownMenuItem(
                            children='Scatter plot',
                            id=f'{self.id}-add-scatter-button',
                            n_clicks=0,
                        ),
                        dbc.DropdownMenuItem(
                            children='Bar chart',
                            id=f'{self.id}-add-bar-button',
                            n_clicks=0,
                        ),
                        dbc.DropdownMenuItem(
                            children='Box plot',
                            id=f'{self.id}-add-box-button',
                            n_clicks=0,
                        ),
                    ],
                ),
                dbc.Button(
                    dbc.Label(
                        'Refresh dataset', 
                        style={
                            'fontWeight': 'bold',
                        }
                    ), 
                    id=f'{self.id}-refresh-button', 
                    color='primary',
                    size='sm',
                    style={
                        'margin': '0px', 
                        'fontWeight': 'bold',
                    },
                    n_clicks=0,
                ),
                dbc.Checklist(
                    id=f'{self.id}-exclude-pending-checkbox',
                    options=[{'label': 'Started jobs only', 'value': 'exclude_pending'}],
                    value=['exclude_pending'],  # Enabled by default
                    inline=True,
                    style={
                        'marginLeft': '10px',
                        'lineHeight': '32px',
                        'fontSize': '14px',
                    },
                    input_style={
                        'marginRight': '5px',
                    },
                ),
                dash_enrich.html.Div(
                    id=f'{self.id}-status-text',
                    children=initial_status,
                    style={
                        'marginLeft': '10px',
                        'lineHeight': '32px',
                        'fontStyle': 'italic',
                        'color': '#888',
                    }
                ),
                self.parent.parent.get_icons_menu(icon_width=20, dashboard=True),
            ],
            style={
                'display': 'flex', 
                'paddingBottom': '2px', 
                'height': '32px',
                'marginLeft': '10px',
                'gap': '4px',
            },
        )

        # Grid div for charts - same styling as ResultViewerComponent
        grid_div = dash_enrich.html.Div(
            id=f'grid-div-{self.id}',
            children=[],
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(min(900px, 100vw), 1fr))',
                'gridAutoRows': 'minmax(250px, auto)',
                'position': 'relative', 
                'top': 0, 
                'left': 0, 
                'bottom': 0, 
                'right': 0, 
                'width': '100%', 
                'height': '100%', 
                'margin': 0, 
                'padding': 0, 
                'overflowY': 'auto',
            }
        )

        # Per-instance viewer id store so clientside JS can read self.id and
        # build correctly-namespaced target IDs for dash_clientside.set_props.
        viewer_id_store = dash_enrich.dcc.Store(
            id=f'{self.id}-viewer-id-store',
            data=self.id,
        )

        # Stores for managing state
        comparator_dataset_store = dash_enrich.dcc.Store(
            id=f'{self.id}-comparator-dataset-store',
            data=initial_dataset,
        )
        
        comparator_updaters_store = dash_enrich.dcc.Store(
            id=f'{self.id}-comparator-updaters-store',
            data=[],
        )
        
        remove_comparator_chart_store = dash_enrich.dcc.Store(
            id=f'{self.id}-remove-comparator-chart-store',
            data=[],
        )

        chart_editor_store = dash_enrich.dcc.Store(
            id=f'{self.id}-comparator-chart-editor-store',
            data=None,
        )

        # Chart editor modal
        chart_editor_modal = dbc.Modal(
            id=f'{self.id}-comparator-chart-editor-modal',
            fullscreen=True,
            is_open=False,
            children=[
                dbc.ModalHeader(
                    close_button=False,
                    style={'gap': 4},
                    children=[
                        dbc.ModalTitle(
                            id=f'{self.id}-comparator-chart-editor-modal-title', 
                            children="Chart editor"),
                        dbc.Button(
                            "Close without saving", 
                            id=f'{self.id}-comparator-close-chart-editor-no-save', 
                            color="danger",
                            size="sm",
                            style={'marginLeft': 'auto'},
                        ),
                        dbc.Button(
                            "Save", 
                            id=f'{self.id}-comparator-save-chart-editor', 
                            color="primary",
                            size="sm",
                        ),
                        dbc.Button(
                            "Save & Close", 
                            id=f'{self.id}-comparator-save-and-close-chart-editor', 
                            color="success",
                            size="sm",
                        ),
                    ],
                ),
                dce.DashChartEditor(
                    id=f'{self.id}-comparator-chart-editor-editor',
                    dataSources=df.to_dict('list'),
                    style={'width': '100%', 'height': '100%'},
                ),
            ],
            style={'width': '100%', 'height': '100%'},
        )

        initial_figures_store = dash_enrich.dcc.Store(
            id=f'{self.id}-comparator-initial-figures-store',
            data=self.initial_figures,
        )

        div = dash_enrich.html.Div(
            children=[
                top_bar,
                dash_enrich.html.Div(
                    id=f'{self.id}-grid-container',   
                    children=[
                        grid_div,
                        viewer_id_store,
                        comparator_dataset_store,
                        comparator_updaters_store,
                        remove_comparator_chart_store,
                        chart_editor_modal,
                        chart_editor_store,
                        initial_figures_store,
                    ],  
                    style={
                        "height": "calc(100% - 32px)",
                        "paddingTop": 0,
                    },
                ),
            ],
            style={'height': '100%'},
        ) 
        print(f'Time taken to get comparator layout: {time.time() - start_time:.5f}s') 
        return div

    def set_callbacks(self, server):

        viewer_id = self.id

        # Refresh dataset callback
        @server.app.callback(
            dash_enrich.Input(f'{self.id}-refresh-button', 'n_clicks'),
            dash_enrich.Output(f'{self.id}-comparator-dataset-store', 'data'),
            dash_enrich.Output(f'{self.id}-status-text', 'children', allow_duplicate=True),
            dash_enrich.State(f'{self.id}-exclude-pending-checkbox', 'value'),
            prevent_initial_call=True,
        )
        def refresh_comparator_dataset(n_clicks, exclude_pending_value):
            if not n_clicks:
                return dash_enrich.no_update, dash_enrich.no_update
            
            df = self._build_comparator_dataframe(exclude_pending='exclude_pending' in (exclude_pending_value or []))
            if df.empty:
                return None, 'No data available'
            
            # Convert to dict for storage
            data_dict = df.to_dict('records')
            status_text = f'Dataset loaded: {len(df)} jobs, {len(df.columns)} columns'
            return data_dict, status_text

        # Update charts when dataset changes
        @server.app.callback(
            dash_enrich.Input(f'{self.id}-comparator-dataset-store', 'data'),
            dash_enrich.State(f'{self.id}-comparator-updaters-store', 'data'),
            dash_enrich.Output({'type': 'comparator-graph', 'viewer_id': viewer_id, 'index': dash_enrich.ALL}, 'figure'),
            prevent_initial_call=True,
        )
        def update_charts_on_dataset_change(dataset, updaters_data):
            if not dataset or not updaters_data:
                return [dash_enrich.no_update] * len(dash_enrich.ctx.outputs_list)
            
            df = pd.DataFrame(dataset)
            patches = []
            
            for i, ud in enumerate(updaters_data):
                x_col = ud.get('x_column')
                y_col = ud.get('y_column')
                
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    # Create a patch to update the figure data
                    patch = Patch()
                    patch['data'][0]['x'] = df[x_col].tolist()
                    patch['data'][0]['y'] = df[y_col].tolist()
                    patch['data'][0]['mode'] = 'markers'
                    patches.append(patch)
                else:
                    patches.append(dash_enrich.no_update)
            
            return patches

        # Add scatter chart
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_comparator_scatter_chart'
            ),
            dash_enrich.Input(f'{self.id}-add-scatter-button', 'n_clicks'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State(f'{self.id}-comparator-updaters-store', 'data'),
            dash_enrich.State(f'{self.id}-viewer-id-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output(f'{self.id}-comparator-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # Add bar chart
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_comparator_bar_chart'
            ),
            dash_enrich.Input(f'{self.id}-add-bar-button', 'n_clicks'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State(f'{self.id}-comparator-updaters-store', 'data'),
            dash_enrich.State(f'{self.id}-viewer-id-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output(f'{self.id}-comparator-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # Add box chart
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_comparator_box_chart'
            ),
            dash_enrich.Input(f'{self.id}-add-box-button', 'n_clicks'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State(f'{self.id}-comparator-updaters-store', 'data'),
            dash_enrich.State(f'{self.id}-viewer-id-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output(f'{self.id}-comparator-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # Remove chart
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='remove_comparator_chart'
            ),
            dash_enrich.Input(f'{self.id}-remove-comparator-chart-store', 'data'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State(f'{self.id}-comparator-updaters-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output(f'{self.id}-remove-comparator-chart-store', 'data', allow_duplicate=True),
            dash_enrich.Output(f'{self.id}-comparator-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # Chart editor callbacks
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='save_and_close_comparator_chart_editor',
            ),
            dash_enrich.Input(f'{self.id}-comparator-save-and-close-chart-editor', "n_clicks"), 
            dash_enrich.Output(f'{self.id}-comparator-chart-editor-modal', "is_open", allow_duplicate=True),
            dash_enrich.Output(f'{self.id}-comparator-chart-editor-editor', "saveState", allow_duplicate=True),
            prevent_initial_call=True,
        )

        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='save_comparator_chart_editor',
            ),
            dash_enrich.Input(f'{self.id}-comparator-save-chart-editor', "n_clicks"), 
            dash_enrich.Output(f'{self.id}-comparator-chart-editor-editor', "saveState", allow_duplicate=True),
            prevent_initial_call=True,
        )

        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='close_no_save_comparator_chart_editor',
            ),
            dash_enrich.Input(f'{self.id}-comparator-close-chart-editor-no-save', "n_clicks"), 
            dash_enrich.Output(f'{self.id}-comparator-chart-editor-modal', "is_open", allow_duplicate=True),
            prevent_initial_call=True,
        )

        # Replace figure on save
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='replace_comparator_figure_on_save',
            ),
            dash_enrich.Input(f'{self.id}-comparator-chart-editor-editor', "saveState"),
            dash_enrich.State(f'{self.id}-comparator-chart-editor-editor', "figure"),
            dash_enrich.State(f'{self.id}-comparator-chart-editor-store', "data"),
            dash_enrich.State(f'{self.id}-comparator-updaters-store', "data"),
            dash_enrich.State(f'{self.id}-viewer-id-store', 'data'),
            prevent_initial_call=True,
        )

        # Update chart editor data sources when editing
        @server.app.callback(
            dash_enrich.Input(f'{self.id}-comparator-chart-editor-store', 'data'),
            dash_enrich.State(f'{self.id}-comparator-dataset-store', 'data'),
            dash_enrich.Output(f'{self.id}-comparator-chart-editor-editor', 'dataSources'),   
            prevent_initial_call=True,
        )
        def update_chart_editor_datasources(chart_editor_store_data, dataset):
            if chart_editor_store_data is None or dataset is None:
                return dash_enrich.no_update
            
            # Convert dataset back to dataframe
            df = pd.DataFrame(dataset)
            
            # Return as dict for chart editor
            return df.to_dict('list')

        # Initialize figures on load
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='initialize_comparator_charts'
            ),
            dash_enrich.Input(f'{self.id}-comparator-initial-figures-store', 'data'),
            dash_enrich.State(f'{self.id}-comparator-dataset-store', 'data'),
            dash_enrich.State(f'{self.id}-viewer-id-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children'),
            dash_enrich.Output(f'{self.id}-comparator-updaters-store', 'data'),
            dash_enrich.Output(f'{self.id}-status-text', 'children', allow_duplicate=True),
            prevent_initial_call=False
        )

