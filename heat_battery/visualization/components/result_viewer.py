from heat_battery.visualization.pages.base import dash_enrich, ClientsideFunction, dbc      
from heat_battery.visualization.components.grids import GridLayout
from heat_battery.simulations.postgresql_project import Project
from heat_battery.visualization.components.figures_serial import TimeSeriesResamplingFigure
import pandas as pd
import time
import numpy as np
import plotly.express as px
from dash_bootstrap_templates import load_figure_template, dbc_templates
from heat_battery.visualization.linux_shm_cache import SHMCache


DEFAULT_RESULT_VIEWER_REFRESH_INTERVAL = 10000 # miliseconds

class ResultViewerComponent(GridLayout):
    HBRES_CACHE_KEY_ROUTE = ("hbres", "data")
    HBRES_LAST_UPDATE_KEY_ROUTE = ("hbres", "last_update")
    def __init__(
            self, 
            project: Project, 
            result_chart_data: dict, 
            figure_creators_getters: dict|None=None,
            parent=None,
            fig_theme:str='bootstrap',
        ):
        if fig_theme in dbc_templates:
            load_figure_template(fig_theme)
        else:
            print(f'Warning: {fig_theme} is not in dbc_templates, skipping the loading')
        self.fig_theme = fig_theme
        figure_creators = []
        self.dataset_columns = ['t_sim_days']
        for fig_name, data in result_chart_data.items():
            self.dataset_columns += data[1]
            if fig_name in figure_creators_getters:
                figure_creators.append(figure_creators_getters[fig_name](data[0], data[1]))
            else:
                figure_creators.append(self.get_figure_creator(data[0], data[1]))

        dummy_array = np.ones((2, len(self.dataset_columns)))
        self.dummy_data = pd.DataFrame(columns=self.dataset_columns, data=dummy_array)
            
        overview_items = [TimeSeriesResamplingFigure(
            fig_i, 
            id_int=i,
            x_name='t_timestamp', 
            x_to_date=True,
            rolling_span=24*7,
            parent=self,
            animate=False,
            fig_theme=fig_theme,
        ) for i, fig_i in enumerate(figure_creators)]

        self.project = project
        self.n_rows_loaded = 0
        self.init_result_table_signatures = self.project._get_result_table_signatures_query()
        self.first_data_updata_call = True
        self.client_signature_map = {}
        self.cache = SHMCache(f'ResultViewer-{self.project.project_name}')
        #self.cache = LocalCache(f'ResultViewer-{self.project.project_name}')
        super().__init__(overview_items, parent=parent)

    # def get_fresh_data(self, signature:str):
    #     last_update = self.project._get_last_updated_query(signature)
    #     df = self.project._get_result_dataframe(signature)
    #     return df
  
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
                dbc.Button(
                    'Add graph', 
                    id=f'{self.id}-add-figure-button', 
                    color='success',
                    size='sm',
                    style={
                        'margin': '0px', 
                        #'fontSize': '12px', 
                        'fontWeight': 'bold',
                    },
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
                    label='Rolling span',
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

        ready_store = dash_enrich.dcc.Store(
            id='ready-store',
            data=signature,
        )

        # def make_expandable_file(file_name):
        #     return dbc.Text(
        #         [], 
        #         style={"paddingTop": 10}
        #     )

        # modal_selector = dbc.Modal(
        #     children=[
        #         dbc.ModalHeader(dbc.ModalTitle("Select case from project")),
        #         dbc.ModalBody(),
        #         dbc.ModalFooter(),
        #     ],
        # )

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
            #self.update_source_data_cache(signature)
            last_update = self.project._get_last_updated_query(signature) 
            if self.cache.read_object(("hbres", "last_update", signature)) != last_update:
                self.cache.write_object(("hbres", "last_update", signature), last_update)
                df = self.project._get_result_dataframe(signature)
                self.cache.write_hbres_dataframe(("hbres", "data", signature), df, expiration_time=600)
                #grid_div = self.get_grid_div(df, qs_data)  
                #self.cache.write_object(("hbres", "init_grid_div", signature), grid_div, expiration_time=600)
            else:
                df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
                #grid_div = self.cache.read_object(("hbres", "init_grid_div", signature)) 
            grid_div = self.get_grid_div(df, qs_data)

        div = dash_enrich.html.Div(
            children=[
                top_bar,
                dash_enrich.html.Div(
                    id=f'{self.id}-grid-container',   
                    children=[
                        grid_div,
                        cache_interval,
                        ready_store,
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
        for item in self.get_children():
            item.set_clientside_callbacks(server)

        # periodicaly update cache for the selected signature
        @server.app.callback(
            dash_enrich.Input(f'{self.id}-cache-interval', 'n_intervals'),
            dash_enrich.State('ready-store', 'data'),
            dash_enrich.State({'type': 'graph', 'index': dash_enrich.ALL}, 'relayoutData'),
            dash_enrich.Output({'type': 'graph', 'index': dash_enrich.ALL}, 'figure', allow_duplicate=True),
            prevent_initial_call=True,
        )
        def update_figures(n_intervals, signature, relayout_data):
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
            for output, rd in zip(dash_enrich.ctx.outputs_list, relayout_data):
                i = int(output['id']['index'])
                patches.append(self.items[i].get_update_data(df, rd))
            print(f'Time taken to update figures: {time.time() - start_time:.5f}s')
            return patches
        
        # update figure on relayout event
        @server.app.callback(
            dash_enrich.State('ready-store', 'data'),
            dash_enrich.Input({'type': 'graph', 'index': dash_enrich.MATCH}, 'relayoutData'),
            dash_enrich.Output({'type': 'graph', 'index': dash_enrich.MATCH}, 'figure'),
            prevent_initial_call=True,
        )
        def update_figure(signature, relayout_data):
            #{'xaxis.autorange': True, 'xaxis.showspikes': False, 'yaxis.autorange': True, 'yaxis.showspikes': False}
            start_time = time.time()
            if (
                relayout_data.get('yaxis.autorange', False)
                or relayout_data.get('autosize', False)
                and not relayout_data.get('xaxis.autorange', False)
            ):
                return dash_enrich.no_update
            df = self.cache.read_hbres_dataframe(("hbres", "data", signature))
            index = int(dash_enrich.ctx.triggered_id['index'])
            patch = self.items[index].get_update_data(df, relayout_data)
            print(f'Time taken to relayout figure: {time.time() - start_time:.5f}s')
            return patch

        # sets up new buttons in figures bacause it canot be done in python
        # FIXME: this is a hack, it should be done in the figure item itself
        #        maybe we can overload some chunk of plotly codebase for this?
        # FIXME: this is not working as expected, buttons are not visible on first load
        #        the page needs refresh, we have to use diferent trigger for this that
        #        runs after the fugures are fully loaded
        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='update_graph_config'
            ),
            dash_enrich.Input({'type': 'graph', 'index': dash_enrich.MATCH}, 'config'),
            dash_enrich.State({'type': 'graph', 'index': dash_enrich.MATCH}, 'id'),
            dash_enrich.Output({'type': 'graph', 'index': dash_enrich.MATCH}, 'config'),
            prevent_initial_call=False
        )

    def get_figure_creator(self, unit_title, y_names, y_range=[None, None]):
        layout_args = dict(
            uirevision="None",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Time",
            yaxis_title=unit_title,
            yaxis_range=y_range,
        )
        def create_figure(df):
            fig = px.line(
                df,
                template=self.fig_theme,
                x='t_sim_days', 
                y=y_names,
            ).update_layout(**layout_args,
                title=dict(
                    text="Live data plot",
                    x=0.5,
                    y=0.98,
                    xanchor='center',
                    yanchor='top',
                    font=dict(
                        size=10,
                    ),
                ),
                xaxis_tickfont=dict(
                    size=10,
                ),
                yaxis_tickfont=dict(
                    size=10,
                ),
            )

            return fig
        return create_figure