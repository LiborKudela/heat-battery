from heat_battery.visualization.pages.base import dash_enrich, ClientsideFunction, dbc      
from heat_battery.visualization.components.grids import GridLayout
from heat_battery.simulations.postgresql_project import Project
#from heat_battery.visualization.components.figures_serial import TimeSeriesResamplingFigure, BaseFigureItem
from heat_battery.visualization.components.figure_updaters import *
import pandas as pd
import time
import numpy as np
import plotly.express as px
from dash_bootstrap_templates import load_figure_template, dbc_templates
from heat_battery.visualization.linux_shm_cache import SHMCache
import dash_chart_editor as dce
import dash_table
import datetime
import dateutil

DEFAULT_RESULT_VIEWER_REFRESH_INTERVAL = 10000 # miliseconds

class ResultViewerComponent(GridLayout):
    HBRES_CACHE_KEY_ROUTE = ("hbres", "data")
    HBRES_LAST_UPDATE_KEY_ROUTE = ("hbres", "last_update")
    def __init__(
            self, 
            project: Project, 
            parent=None,
            fig_theme:str='bootstrap',
            initial_figures: list|None=None,
            variable_descriptions: dict|None=None,
            initial_transforms: list|None=None,
        ):

        self.fig_theme = fig_theme
        self.initial_figures = initial_figures or []
        self.variable_descriptions = variable_descriptions or {}
        self.initial_transforms = initial_transforms or []


        self.project = project
        self.n_rows_loaded = 0
        self.init_result_table_signatures = self.project._get_result_table_signatures_query()
        self.first_data_updata_call = True
        self.client_signature_map = {}
        self.cache = SHMCache(f'ResultViewer-{self.project.project_name}')
        super().__init__([], parent=parent)
    
    # def _parse_timestamp_column(self, df):
    #     """Hidden first step: Parse t_timestamp column using the same logic as figure_updaters"""
    #     if 't_timestamp' not in df.columns:
    #         print("t_timestamp column not found in dataframe")
    #         return df

    #     df = df.copy()
    #     # Apply parsing to t_timestamp column
    #     df['t_timestamp'] = pd.to_datetime(df['t_timestamp'], unit='s', origin='unix', utc=True)
    #     df = df.set_index('t_timestamp')
        
    #     return df

  
    def get_layout(self, qs_data:dict|None=None):
        start_time = time.time()
        available_result_signatures = self.project._get_result_table_signatures_query()

        if qs_data:
            if qs_data.get('signature'):
                signature = qs_data['signature'][0]
            else:
                signature = None
        else:
            if len(available_result_signatures) > 0:
                signature = available_result_signatures[0]
            else:
                signature = None


        top_bar = dash_enrich.html.Div(
            id=f'{self.id}-top-bar',
            children=[
                dbc.Button(
                    dbc.Label(
                        'Data Transforms', 
                        style={
                            'fontWeight': 'bold',
                        }
                    ), 
                    id=f'data-transforms-button', 
                    color='secondary',
                    size='sm',
                    style={
                        'margin': '0px', 
                        'fontWeight': 'bold',
                        'whiteSpace':'nowrap',
                    },
                ),
                dbc.DropdownMenu(
                    label='Add graph',
                    id=f'{self.id}-add-figure-dropdown', 
                    color='success',
                    menu_variant='dark',
                    size='sm',
                    style={
                        'margin': '0px', 
                        #'fontSize': '12px', 
                        'fontWeight': 'bold',  
                        'whiteSpace':'nowrap', 
                    },
                    children=[
                        dbc.DropdownMenuItem("Graph templates", header=True),
                        *[],
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Timeseries graphs", header=True),
                        dbc.DropdownMenuItem(
                            children='Timeseries',
                            id=f'{self.id}-add-timeseries-figure-button',
                            n_clicks=0,
                        ),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Transform graphs", header=True, id=f'{self.id}-transform-graphs-header'),
                        dash_enrich.html.Div(id=f'{self.id}-transform-graphs-menu-items', children=[]),
                    ],
                ),
                dbc.Button(
                    dbc.Label(
                        'Refresh Data', 
                        style={
                            #   'fontSize': '12px', 
                            'fontWeight': 'bold',
                        }
                    ), 
                    id=f'{self.id}-refresh-button', 
                    color='primary',
                    size='sm',
                    style={
                        'margin': '0px', 
                        'whiteSpace':'nowrap',
                        #'fontSize': '12px', 
                        'fontWeight': 'bold',
                    },
                ),
                # dbc.DropdownMenu(   
                #     label='Timeseries rolling span',
                #     id=f'{self.id}-rolling-span-dropdown', 
                #     color='info',
                #     size='sm',
                #     style={
                #         'margin': '0px', 
                #         #'fontSize': '12px', 
                #         'fontWeight': 'bold',   
                #     },
                #     children=[
                #         dbc.DropdownMenuItem('1 hour'),
                #         dbc.DropdownMenuItem('1 day'),
                #         dbc.DropdownMenuItem('7 days'),
                #         dbc.DropdownMenuItem('30 days'),
                #         dbc.DropdownMenuItem('1 year'),
                #         dbc.DropdownMenuItem('Max'),
                #     ],
                # ),
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

        job_status = self.project._get_status_query(signature)
        disable_autorefresh = not ("RUNNING" in job_status)
        cache_interval = dash_enrich.dcc.Interval(
            id=f'{self.id}-cache-interval', 
            interval=DEFAULT_RESULT_VIEWER_REFRESH_INTERVAL,
            disabled=disable_autorefresh, #TODO: this need to be
            n_intervals=0
        )
        if signature is None:
            grid_div = dash_enrich.html.Div(
                children=[
                    dash_enrich.html.H5('No data tables are available yet'),
                    dash_enrich.dcc.Interval(id=f'{self.id}-interval-wait', interval=1000, n_intervals=0),
                ],
                style={'height':'100%'},
            )
        elif signature not in available_result_signatures:
            grid_div = dash_enrich.html.Div(
                children=[
                    dash_enrich.html.H5(f'There is no result table for requested signature: {signature}'),
                ],
                style={'height':'100%'},
            )
        else:
            # signature is valid if this is called
            qs_data.update({'auto-refresh': [0] if disable_autorefresh else [1]})
            qs_data.update({'bar-title': [job_status]}) 
            qs_data.update({'signature': [signature]})
            last_update = self.project._get_last_updated_query(signature) 
            if self.cache.read_object(("hbres", "last_update", signature)) != last_update:
                self.cache.write_object(("hbres", "last_update", signature), last_update)
                df = self.project._get_result_dataframe(signature)
                self.cache.write_hbres_dataframe(("hbres", "data", signature), df, expiration_time=600)
            else:
                df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
            grid_div = self.get_grid_div(df, qs_data)

        signature_store = dash_enrich.dcc.Store(
            id='signature-store',
            data=signature,
        )
        reriodic_relayout_data_store = dash_enrich.dcc.Store(
            id='reriodic-relayout-data-store',
            data={},
        )
        figure_updaters_store = dash_enrich.dcc.Store(
            id='figure-updaters-store',
            data=[],
        )
        remove_figure_store = dash_enrich.dcc.Store(
            id='remove-figure-store',
            data=[],
        )

        chart_editor_store = dash_enrich.dcc.Store(
            id="chart-editor-store",
            data=None,
        )

        data_transforms_store = dash_enrich.dcc.Store(
            id='data-transforms-store',
            data=[],
        )
        initial_transforms_store = dash_enrich.dcc.Store(
            id='initial-transforms-store',
            data=self.initial_transforms,
        )
        expanded_transform_store = dash_enrich.dcc.Store(
            id='expanded-transform-store',
            data=None,  # Stores the index of the expanded transform
        )

        data_transforms_modal = dbc.Modal(
            id="data-transforms-modal",
            size='xl',
            centered=True,
            is_open=False,
            fullscreen=True,
            children=[
                dbc.ModalHeader(children="Data Transforms", close_button=True),
                dbc.ModalBody(
                    children=[
                        dash_enrich.html.Div(
                            children=[
                                dbc.Button(
                                    "Add Transform",
                                    id="add-transform-button",
                                    color="success",
                                    size="sm",
                                ),
                            ],
                            style={
                                'marginBottom': '15px',
                                'display': 'flex',
                                'gap': '8px',
                            }
                        ),
                        dash_enrich.html.Hr(),
                        dash_enrich.html.H5("Current Transforms:", style={'marginBottom': '10px'}),
                        dbc.ListGroup(
                            id="transforms-list",
                            children=[],
                            style={'maxHeight': '600px', 'overflowY': 'auto'},
                        ),
                    ],
                    className='bg-body',
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-data-transforms-button", color="secondary", size="sm")
                ),
            ],
        )

        step_preview_modal = dbc.Modal(
            id="step-preview-modal",
            size='xl',
            centered=True,
            is_open=False,
            children=[
                dbc.ModalHeader(
                    dbc.ModalTitle(id="step-preview-modal-title", children="Step Preview"),
                    close_button=True,
                ),
                dbc.ModalBody(
                    id="step-preview-modal-body",
                    children="Preview will appear here",
                    style={'fontSize': '14px'},
                    className='bg-body',
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-step-preview-button", color="secondary", size="sm"),
                ),
            ],
        )

        chart_editor_modal = dbc.Modal(
            id="chart-editor-modal",
            fullscreen=True,
            is_open=False,
            children=[
                dbc.ModalHeader(
                    close_button=False,
                    style={'gap': 4},
                    children=[
                        dbc.ModalTitle(
                            id="chart-editor-modal-title", 
                            children="Chart editor"),
                        dbc.Button(
                            "Close without saving", 
                            id="close-chart-editor-no-save", 
                            color="danger",
                            size="sm",
                            style={'marginLeft': 'auto'},
                        ),
                        dbc.Button(
                            "Save", 
                            id="save-chart-editor", 
                            color="primary",
                            size="sm",
                        ),
                        dbc.Button(
                            "Save & Close", 
                            id="save-and-close-chart-editor", 
                            color="success",
                            size="sm",
                        ),
                    ],
                ),
                dce.DashChartEditor(
                    id="chart-editor-editor",
                    dataSources={'dummy_data': []},
                    # saveState=True,
                    style={'width': '100%', 'height': '100%'},
                ),
            ],
            style={'width': '100%', 'height': '100%'},
        )

        # Modal for displaying clicked line info (right-click context)
        line_click_modal = dbc.Modal(
            id="line-click-modal",
            is_open=False,
            centered=True,
            children=[
                dbc.ModalHeader(
                    dbc.ModalTitle(id="line-click-modal-title", children="Line Info"),
                    close_button=True,
                ),
                dbc.ModalBody(
                    id="line-click-modal-body",
                    children="Click on a line in any chart to see details here.",
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="line-click-modal-close", color="secondary", size="sm"),
                ),
            ],
        )

        line_click_store = dash_enrich.dcc.Store(
            id='line-click-store',
            data=None,
        )

        variable_descriptions_store = dash_enrich.dcc.Store(
            id='variable-descriptions-store',
            data=self.variable_descriptions,
        )

        for fig in self.initial_figures:
            if fig['active']:
                if fig['updater_type'] == 'timeseries':
                    fill_timeseries_figure_initial(df, fig)

        initial_figures_store = dash_enrich.dcc.Store(
            id='initial-figures-store',
            data=self.initial_figures,
        )

        div = dash_enrich.html.Div(
            children=[
                top_bar,
                dash_enrich.html.Div(
                    id=f'{self.id}-grid-container',   
                    children=[
                        grid_div,
                        cache_interval,
                        signature_store,
                        data_transforms_modal,
                        step_preview_modal,
                        data_transforms_store,
                        initial_transforms_store,
                        expanded_transform_store,
                        chart_editor_modal,
                        chart_editor_store,
                        figure_updaters_store,
                        reriodic_relayout_data_store,
                        remove_figure_store,
                        initial_figures_store,
                        line_click_modal,
                        line_click_store,
                        variable_descriptions_store,
                    ],  
                    style={
                        "height":"calc(100% - 32px)",
                        "paddingTop":0,
                    },
                ),
            ],
            style={'height':'100%'},
        ) 
        print(f'Time to result viewer layout: {time.time() - start_time:.5f}s') 
        return div

    def set_callbacks(self, server):

        # add timeseries graph to grid
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_timeseries_graph_to_grid'
            ),
            dash_enrich.Input(f'{self.id}-add-timeseries-figure-button', 'n_clicks'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output('figure-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # Update transform graphs menu items based on transforms
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='update_transform_graphs_menu'
            ),
            dash_enrich.Input('data-transforms-store', 'data'),
            dash_enrich.Output(f'{self.id}-transform-graphs-menu-items', 'children'),
        )
        
        # Add transform graph to grid (generic callback for all transform buttons)
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_transform_graph_to_grid'
            ),
            dash_enrich.Input({'type': 'add-transform-graph-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output('figure-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )

        # remove figure from grid
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='remove_figure_from_grid'
            ),
            dash_enrich.Input('remove-figure-store', 'data'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output('remove-figure-store', 'data', allow_duplicate=True),
            dash_enrich.Output('figure-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # save chart editor and close
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='save_and_close_chart_editor',
            ),
            dash_enrich.Input("save-and-close-chart-editor", "n_clicks"), 
            dash_enrich.Output("chart-editor-modal", "is_open", allow_duplicate=True),
            dash_enrich.Output("chart-editor-editor", "saveState", allow_duplicate=True),
            prevent_initial_call=True,
        )

        # save chart editor without closing
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='save_chart_editor',
            ),
            dash_enrich.Input("save-chart-editor", "n_clicks"), 
            dash_enrich.Output("chart-editor-editor", "saveState", allow_duplicate=True),
            prevent_initial_call=True,
        )

        # close chart editor without saving
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='close_no_save_chart_editor',
            ),
            dash_enrich.Input("close-chart-editor-no-save", "n_clicks"), 
            dash_enrich.Output("chart-editor-modal", "is_open", allow_duplicate=True),
            prevent_initial_call=True,
        )

        # replace figure on save in editor
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='replace_figure_on_save',
            ),
            dash_enrich.Input("chart-editor-editor", "saveState"),
            dash_enrich.State("chart-editor-editor", "figure"),
            dash_enrich.State("chart-editor-store", "data"),
            dash_enrich.State("figure-updaters-store", "data"),
            prevent_initial_call=True,
        ) 

        # periodicaly update cache for the selected signature
        @server.app.callback(
            dash_enrich.Input(f'{self.id}-cache-interval', 'n_intervals'),
            dash_enrich.State('signature-store', 'data'),
            dash_enrich.State('chart-editor-modal', 'is_open'),
            dash_enrich.State({'type': 'graph', 'index': dash_enrich.ALL}, 'relayoutData'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.Output({'type': 'graph', 'index': dash_enrich.ALL}, 'figure', allow_duplicate=True),
            prevent_initial_call=True,
        )
        def update_figures(n_intervals, signature, is_open, relayout_data, updater_data, transforms):
            #print(f'Updating figures: {updater_data}')
            if is_open:
                return [no_update_patch()] * len(dash_enrich.ctx.outputs_list)
            start_time = time.time()
            last_update = self.project._get_last_updated_query(signature)
            if self.cache.read_object(("hbres", "last_update", signature)) != last_update:
                df = self.project._get_result_dataframe(signature)
                self.cache.write_hbres_dataframe(("hbres", "data", signature), df, expiration_time=600)
                self.cache.write_object(("hbres", "last_update", signature), last_update)
                # perform all transforms on the dataframe
                # for transform in transforms:
                #     steps = transform.get('steps', [])
                #     for step in steps:
                #         df = self._apply_transform_step(df, step)
                #     transform_signature = hashlib.sha256(str(transform).encode('utf-8')).hexdigest()
                #     self.cache.write_small_dataframe(("hbres", f"transform_data", signature, transform_signature), df, expiration_time=600)
            else:
                print(f'Time to update result viewer figures (cache hit): {time.time() - start_time:.5f}s')
                return [dash_enrich.no_update] * len(dash_enrich.ctx.outputs_list)
            patches = []
            # for ud in updater_data:
            #     print(ud)
            # for f in dash_enrich.ctx.outputs_list:
            #     print(f)
            for output, rd in zip(dash_enrich.ctx.outputs_list, relayout_data):
                i = int(output['id']['index'])
                ud = updater_data[i]
                if ud.get('y_names') is None:
                    patches.append(dash_enrich.no_update)
                elif ud.get('updater_type') == 'transform' and transforms and 'transform_index' in ud:
                    # Apply transforms to dataframe
                    transform_index = ud['transform_index']
                    if transform_index < len(transforms):
                        transform = transforms[transform_index]
                        result_df = apply_transform(df, transform)
                        if result_df is None:
                            patches.append(dash_enrich.no_update)
                            break
                        patches.append(get_timeseries_figure_update(result_df, ud['y_names'], ud['x_name'], rd))
                    else:
                        patches.append(dash_enrich.no_update)
                else:
                    patches.append(get_timeseries_figure_update(df, ud['y_names'], ud['x_name'], rd))

            print(f'Time to update result viewer figures (cache miss): {time.time() - start_time:.5f}s')
            return patches
        
        #update figure on relayout event
        @server.app.callback(
            dash_enrich.Input({'type': 'graph', 'index': dash_enrich.MATCH}, 'relayoutData'),
            dash_enrich.State('signature-store', 'data'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.Output({'type': 'graph', 'index': dash_enrich.MATCH}, 'figure'),
            #dash_enrich.Output('reriodic-relayout-data-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )
        def update_figure(relayout_data, signature, updater_data, transforms):
            #{'xaxis.autorange': True, 'xaxis.showspikes': False, 'yaxis.autorange': True, 'yaxis.showspikes': False}
            start_time = time.time()
            if not any(key.startswith('xaxis.') for key in relayout_data.keys()):
                print(f"No xaxis.autorange in relayout data {time.time() - start_time:.5f}s")
                return dash_enrich.no_update
            if relayout_data.get('autosize') is not None:
                print(f"No autosize in relayout data {time.time() - start_time:.5f}s")
                return dash_enrich.no_update
            if relayout_data is None:
                relayout_data = {}
            if {'xaxis.autorange': True, 'xaxis.showspikes': False, 'yaxis.autorange': True, 'yaxis.showspikes': False} == relayout_data:
                relayout_data = {}
            if relayout_data =={'xaxis.autorange': True, 'yaxis.autorange': True}:
                print(f"No xaxis.autorange and yaxis.autorange in relayout data {time.time() - start_time:.5f}s")
                return dash_enrich.no_update
            df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
            index = int(dash_enrich.ctx.inputs_list[0]['id']['index'])
            ud = updater_data[index]
            if ud['y_names'] is None:
                print(f"No y_names in updater data {time.time() - start_time:.5f}s")
                return dash_enrich.no_update
            if ud['updater_type'] == 'timeseries':
                patch = get_timeseries_figure_update(df, ud['y_names'], ud['x_name'], relayout_data)
            elif ud['updater_type'] == 'transform':
                # Get transforms from store
                if transforms and 'transform_index' in ud:
                    transform_index = ud['transform_index']
                    if transform_index < len(transforms):
                        transform = transforms[transform_index]
                        print(f"updater: {ud}")
                        result_df = apply_transform(df, transform)
                        #patch = get_timeseries_figure_update(result_df, ud['y_names'], ud['x_name'], relayout_data)
                        patch = get_transform_figure_update(result_df, ud['y_names'], ud['x_name'], relayout_data)
                    else:
                        print(f"Invalid transform index {transform_index} {time.time() - start_time:.5f}s")
                        return dash_enrich.no_update
                else:
                    print(f"No transform index in updater data {time.time() - start_time:.5f}s")
                    return dash_enrich.no_update
            else:
                print(f"No xaxis.autorange and yaxis.autorange in relayout data {time.time() - start_time:.5f}s")
                return dash_enrich.no_update
            print(f'Time taken to relayout figure: {time.time() - start_time:.5f}s')
            return patch#, relayout_data
        
        @server.app.callback(
            dash_enrich.Input('chart-editor-store', 'data'),
            dash_enrich.State('signature-store', 'data'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.Output('chart-editor-editor', 'dataSources'),   
            prevent_initial_call=True,
        )
        def replace_chart_editor_dataSources_on_chart_edit(chart_editor_store_data, signature, transforms):
            if chart_editor_store_data is None:
                return dash_enrich.no_update    
            id = chart_editor_store_data['id']
            chart_type = chart_editor_store_data['type']
            if chart_type == 'timeseries':
                df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
                df_minified = minify_for_chart_editor(df)
                df_minified['t_timestamp'] = pd.to_datetime(df_minified['t_timestamp'], unit='s', origin='unix', utc=True)
                print("sending data to editor - timeseries")
                return df_minified.to_dict('list')
            elif chart_type == 'transform':
                # Get transform_index from chart_editor_store_data
                transform_index = chart_editor_store_data.get('transform_index')
                if transform_index is None or not transforms or transform_index >= len(transforms):
                    print("sending data to editor - transform (error: invalid transform_index)")
                    return dash_enrich.no_update
                
                # Get the dataframe
                df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
                if df is None:
                    return dash_enrich.no_update
                
                # Apply transform
                transform = transforms[transform_index]
                result_df = apply_transform(df, transform)
                
                # Ensure t_timestamp is a column (not index) for chart editor
                if result_df.index.name == 't_timestamp' or (hasattr(result_df.index, 'name') and result_df.index.name == 't_timestamp'):
                    result_df = result_df.reset_index()
                
                # Minify for chart editor
                #df_minified = minify_for_chart_editor(result_df)
                
                # Convert t_timestamp to datetime if it exists and is not already datetime
                if 't_timestamp' in result_df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(result_df['t_timestamp']):
                        result_df['t_timestamp'] = pd.to_datetime(result_df['t_timestamp'], unit='s', origin='unix', utc=True)
                
                print(f"sending data to editor - transform (transform_index: {transform_index})")
                return result_df.to_dict('list')
            
        # Add the initialization callback
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='initialize_figures'
            ),
            dash_enrich.Input('initial-figures-store', 'data'),
            dash_enrich.State('signature-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children'),
            dash_enrich.Output('figure-updaters-store', 'data'),
            prevent_initial_call=False  # This is important - it runs on page load
        )

        # Handle data transforms modal - open modal when data-transforms-button is clicked
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='open_data_transforms_modal'
            ),
            dash_enrich.Input('data-transforms-button', 'n_clicks'),
            dash_enrich.Output('data-transforms-modal', 'is_open'),
        )

        # Handle data transforms modal - close modal when close-data-transforms-button is clicked
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='close_data_transforms_modal'
            ),
            dash_enrich.Input('close-data-transforms-button', 'n_clicks'),
            dash_enrich.Output('data-transforms-modal', 'is_open', allow_duplicate=True),
        )

        # Initialize default transforms on page load
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='initialize_default_transforms'
            ),
            dash_enrich.Input('initial-transforms-store', 'data'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
            prevent_initial_call=False,
        )

        # Load transforms from localStorage when modal opens
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='load_transforms_from_storage'
            ),
            dash_enrich.Input('data-transforms-modal', 'is_open'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
        )

        # Display transforms list
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='display_transforms_list'
            ),
            dash_enrich.Input('data-transforms-store', 'data'),
            dash_enrich.Input('expanded-transform-store', 'data'),
            dash_enrich.Output('transforms-list', 'children'),
        )

        # Handle add transform button
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_transform'
            ),
            dash_enrich.Input('add-transform-button', 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )

        # Handle edit transform button - toggle expansion
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='toggle_transform_expansion'
            ),
            dash_enrich.Input({'type': 'edit-transform-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('expanded-transform-store', 'data'),
            dash_enrich.Output('expanded-transform-store', 'data'),
            prevent_initial_call=True,
        )

        # Handle add step button
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_step_to_transform'
            ),
            dash_enrich.Input({'type': 'add-step-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.State('expanded-transform-store', 'data'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )

        # Handle remove step button
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='remove_step_from_transform'
            ),
            dash_enrich.Input({'type': 'remove-step-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.State('expanded-transform-store', 'data'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )

        # Handle move step up button
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='move_step_up'
            ),
            dash_enrich.Input({'type': 'move-step-up-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.State('expanded-transform-store', 'data'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )

        # Handle move step down button
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='move_step_down'
            ),
            dash_enrich.Input({'type': 'move-step-down-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.State('expanded-transform-store', 'data'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )

        # Handle step and transform preview modal - merged callback (computes preview and displays modal)
        @server.app.callback(
            dash_enrich.Input({'type': 'preview-step-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.Input({'type': 'preview-transform-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.State('expanded-transform-store', 'data'),
            dash_enrich.State('signature-store', 'data'),
            dash_enrich.Output('step-preview-modal', 'is_open', allow_duplicate=True),
            dash_enrich.Output('step-preview-modal-title', 'children', allow_duplicate=True),
            dash_enrich.Output('step-preview-modal-body', 'children', allow_duplicate=True),
            prevent_initial_call=True,
        )
        def preview_data_trasform(step_n_clicks_array, transform_n_clicks_array, transforms, expanded_index, signature):
            # Check if a preview button was actually clicked
            triggered_id = dash_enrich.ctx.triggered_id
            if not triggered_id:
                return dash_enrich.no_update
            
            button_type = triggered_id.get('type')
            clicked_index = triggered_id.get('index')

            # prevent preview if no button was clicked
            if (transform_n_clicks_array and 
                transform_n_clicks_array[clicked_index] and 
                transform_n_clicks_array[clicked_index] > 0) or \
                (step_n_clicks_array and
                step_n_clicks_array[clicked_index] and 
                step_n_clicks_array[clicked_index] > 0):
                pass
            else:
                return dash_enrich.no_update


            # Handle step preview button
            if button_type == 'preview-transform-button':
                # full preview
                transform = transforms[clicked_index]
                n_steps = None
            elif button_type == 'preview-step-button':
                # partial preview
                if expanded_index is None:
                    return dash_enrich.no_update
                transform = transforms[expanded_index]
                n_steps = clicked_index + 1
           
            # Get the dataframe
            print(f"Reading preview dataframe for signature: {signature}")
            df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
            if df is None:
                return True, "Step Preview - Error", "No data available"
            
            result_df = apply_transform(df, transform, n_steps=n_steps)
            if result_df is None:
                return True, "Step Preview - Error", "Error applying transform"
            
            # Get preview data (head 10 rows)
            preview_df = result_df.head(10).round(2)
            preview_str = str(preview_df)
            
            shape_info = f"Shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns (showing first 10 rows)"
            
            return True, f"Step {clicked_index + 1} Preview", [
                dash_enrich.html.P(shape_info, style={'marginBottom': '10px', 'fontWeight': 'bold'}),
                dash_enrich.html.Pre(preview_str, style={'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace'})
            ]
        
        # Close step preview modal
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='close_step_preview_modal'
            ),
            dash_enrich.Input('close-step-preview-button', 'n_clicks'),
            dash_enrich.Output('step-preview-modal', 'is_open', allow_duplicate=True),
            prevent_initial_call=True,
        )

        # Handle delete transform button (from list items)
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='delete_transform'
            ),
            dash_enrich.Input({'type': 'delete-transform-button', 'index': dash_enrich.ALL}, 'n_clicks'),
            dash_enrich.State('data-transforms-store', 'data'),
            dash_enrich.Output('data-transforms-store', 'data', allow_duplicate=True),
            prevent_initial_call=True,
        )
        # Handle line click modal - open modal when line-click-store is updated
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='open_line_click_modal'
            ),
            dash_enrich.Input('line-click-store', 'data'),
            dash_enrich.State('variable-descriptions-store', 'data'),
            dash_enrich.Output('line-click-modal', 'is_open'),
            dash_enrich.Output('line-click-modal-title', 'children'),
            dash_enrich.Output('line-click-modal-body', 'children'),
            prevent_initial_call=True
        )

        # Close line click modal
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='close_line_click_modal'
            ),
            dash_enrich.Input('line-click-modal-close', 'n_clicks'),
            dash_enrich.Output('line-click-modal', 'is_open', allow_duplicate=True),
            prevent_initial_call=True
        )