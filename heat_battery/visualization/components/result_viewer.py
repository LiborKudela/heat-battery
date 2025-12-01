from heat_battery.visualization.pages.base import dash_enrich, ClientsideFunction, dbc      
from heat_battery.visualization.components.grids import GridLayout
from heat_battery.simulations.postgresql_project import Project
from heat_battery.visualization.components.figures_serial import TimeSeriesResamplingFigure, BaseFigureItem
from heat_battery.visualization.components.figure_updaters import *
import pandas as pd
import time
import numpy as np
import plotly.express as px
from dash_bootstrap_templates import load_figure_template, dbc_templates
from heat_battery.visualization.linux_shm_cache import SHMCache
import dash_chart_editor as dce

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
        ):

        self.fig_theme = fig_theme
        self.initial_figures = initial_figures or []


        self.project = project
        self.n_rows_loaded = 0
        self.init_result_table_signatures = self.project._get_result_table_signatures_query()
        self.first_data_updata_call = True
        self.client_signature_map = {}
        self.cache = SHMCache(f'ResultViewer-{self.project.project_name}')
        super().__init__([], parent=parent)

  
    def get_layout(self, qs_data:dict|None=None):
        start_time = time.time()
        available_result_signatures = self.project._get_result_table_signatures_query()

        if qs_data:
            signature = qs_data['signature'][0]
        else:
            if len(available_result_signatures) > 0:
                signature = available_result_signatures[0]
            else:
                signature = None

        top_bar = dash_enrich.html.Div(
            id=f'{self.id}-top-bar',
            children=[
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
                        dbc.DropdownMenuItem("Aggregated graphs", header=True),
                        dbc.DropdownMenuItem(
                            children='Last value',
                            id=f'{self.id}-add-last-value-aggregated-figure-button',
                            n_clicks=0,
                        ),
                        dbc.DropdownMenuItem(
                            children='Monthly',
                            id=f'{self.id}-add-monthly-aggregated-figure-button',
                            n_clicks=0,
                        ),
                        dbc.DropdownMenuItem(
                            children='Yearly',
                            id=f'{self.id}-add-yearly-aggregated-figure-button',
                            n_clicks=0,
                        ),
                    ],
                ),
                dbc.Button(
                    dbc.Label(
                        'Refresh', 
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
                        #'fontSize': '12px', 
                        'fontWeight': 'bold',
                    },
                ),
                dbc.DropdownMenu(   
                    label='Timeseries rolling span',
                    id=f'{self.id}-rolling-span-dropdown', 
                    color='info',
                    size='sm',
                    style={
                        'margin': '0px', 
                        #'fontSize': '12px', 
                        'fontWeight': 'bold',   
                    },
                    children=[
                        dbc.DropdownMenuItem('1 hour'),
                        dbc.DropdownMenuItem('1 day'),
                        dbc.DropdownMenuItem('7 days'),
                        dbc.DropdownMenuItem('30 days'),
                        dbc.DropdownMenuItem('1 year'),
                        dbc.DropdownMenuItem('Max'),
                    ],
                ),
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
                        chart_editor_modal,
                        chart_editor_store,
                        figure_updaters_store,
                        remove_figure_store,
                        initial_figures_store,
                    ],  
                    style={
                        "height":"calc(100% - 32px)",
                        "paddingTop":0,
                    },
                ),
            ],
            style={'height':'100%'},
        ) 
        print(f'Time taken to get layout: {time.time() - start_time:.5f}s') 
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

        # add last value aggregated graph to grid
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_last_value_aggregated_graph_to_grid'
            ),
            dash_enrich.Input(f'{self.id}-add-last-value-aggregated-figure-button', 'n_clicks'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output('figure-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # add monthly aggregated graph to grid
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_monthly_aggregated_graph_to_grid'
            ),
            dash_enrich.Input(f'{self.id}-add-monthly-aggregated-figure-button', 'n_clicks'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output('figure-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
        )

        # add yearly aggregated graph to grid
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='add_yearly_aggregated_graph_to_grid'
            ),
            dash_enrich.Input(f'{self.id}-add-yearly-aggregated-figure-button', 'n_clicks'),
            dash_enrich.State(f'grid-div-{self.id}', 'children'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.Output(f'grid-div-{self.id}', 'children', allow_duplicate=True),
            dash_enrich.Output('figure-updaters-store', 'data', allow_duplicate=True),
            prevent_initial_call=True
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
            dash_enrich.Output({'type': 'graph', 'index': dash_enrich.ALL}, 'figure', allow_duplicate=True),
            prevent_initial_call=True,
        )
        def update_figures(n_intervals, signature, is_open, relayout_data, updater_data):
            print(f'Updating figures: {updater_data}')
            if is_open:
                return [no_update_patch()] * len(dash_enrich.ctx.outputs_list)
            start_time = time.time()
            last_update = self.project._get_last_updated_query(signature)
            if self.cache.read_object(("hbres", "last_update", signature)) != last_update:
                df = self.project._get_result_dataframe(signature)
                self.cache.write_hbres_dataframe(("hbres", "data", signature), df, expiration_time=600)
                self.cache.write_object(("hbres", "last_update", signature), last_update)
            else:
                print(f'Time taken to update figures: {time.time() - start_time:.5f}s')
                return [dash_enrich.no_update] * len(dash_enrich.ctx.outputs_list)
            patches = []
            for ud in updater_data:
                print(ud)
            for f in dash_enrich.ctx.outputs_list:
                print(f)
            for output, rd in zip(dash_enrich.ctx.outputs_list, relayout_data):
                i = int(output['id']['index'])
                ud = updater_data[i]
                if ud.get('y_names') is None:
                    patches.append(dash_enrich.no_update)
                else:
                    patches.append(get_timeseries_figure_update(df, ud['y_names'], ud['x_name'], rd))

            print(f'Time taken to update figures: {time.time() - start_time:.5f}s')
            return patches
        
        #update figure on relayout event
        @server.app.callback(
            dash_enrich.Input({'type': 'graph', 'index': dash_enrich.MATCH}, 'relayoutData'),
            dash_enrich.State('signature-store', 'data'),
            dash_enrich.State('figure-updaters-store', 'data'),
            dash_enrich.Output({'type': 'graph', 'index': dash_enrich.MATCH}, 'figure'),
            prevent_initial_call=True,
        )
        def update_figure(relayout_data, signature, updater_data):
            #{'xaxis.autorange': True, 'xaxis.showspikes': False, 'yaxis.autorange': True, 'yaxis.showspikes': False}
            start_time = time.time()
            if relayout_data.get('autosize') is not None:
                return dash_enrich.no_update
            if relayout_data is None:
                relayout_data = {}
            if {'xaxis.autorange': True, 'xaxis.showspikes': False, 'yaxis.autorange': True, 'yaxis.showspikes': False} == relayout_data:
                relayout_data = {}
            if relayout_data =={'xaxis.autorange': True, 'yaxis.autorange': True}:
                return dash_enrich.no_update
            df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
            index = int(dash_enrich.ctx.inputs_list[0]['id']['index'])
            ud = updater_data[index]
            if ud['y_names'] is None:
                return dash_enrich.no_update
            if ud['updater_type'] == 'timeseries':
                patch = get_timeseries_figure_update(df, ud['y_names'], ud['x_name'], relayout_data)
            elif ud['updater_type'] == 'aggregated':
                patch = get_aggregated_figure_update(df, ud['y_names'], ud['x_name'], relayout_data)
            print(f'Time taken to relayout figure: {time.time() - start_time:.5f}s')
            return patch
        
        @server.app.callback(
            dash_enrich.Input('chart-editor-store', 'data'),
            dash_enrich.State('signature-store', 'data'),
            dash_enrich.Output('chart-editor-editor', 'dataSources'),   
            prevent_initial_call=True,
        )
        def replace_chart_editor_dataSources_on_chart_edit(chart_editor_store_data, signature):
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
            elif chart_type == 'last_value_aggregated':
                df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
                values = df.iloc[-1].values.tolist()
                labels = df.columns.tolist()
                print("sending data to editor - last value aggregated")
                return {'values': values, 'labels': labels}
            elif chart_type == 'monthly_aggregated':
                print("sending data to editor - monthly aggregated")
                dummy_df = pd.DataFrame(columns=['t_sim_days', 'value'])
                dummy_df['t_sim_days'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                dummy_df['value'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                return dummy_df.to_dict('list')
            elif chart_type == 'yearly_aggregated':
                print("sending data to editor - yearly aggregated")
                dummy_df = pd.DataFrame(columns=['t_sim_days', 'value'])
                dummy_df['t_sim_days'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                dummy_df['value'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                return dummy_df.to_dict('list')
            
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