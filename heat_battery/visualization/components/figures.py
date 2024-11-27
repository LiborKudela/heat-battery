from .base import ContentItem, dash_enrich, dbc
from dash import Patch
from plotly_resampler import FigureResampler
import dash_chart_editor as dce
import time

DEFAULT_REFRESH_PERIOD = 5000

class BaseFigureItem(ContentItem):
    """This represents the basic extendable layout of a plotly figure"""
    def __init__(self, f_template, data_name, x_name='datetime', animate=False, **kwargs):
        super().__init__()
        self.GRAPH_ID = f'{self.id}-graph'
        self.FIGURE_TYPE = 'Figure Item'
        self.f_template = f_template
        self.data_name = data_name
        self.x_name = x_name
        self.first_data_updata_call = True
        self.animate = animate
        self.controls = []
        self.invisibles = []

    def add_control(self, control):
        self.controls.append(control)

    def add_invisible(self, invisible):
        self.invisibles.append(invisible)

    def get_new_id(self):
        ContentItem.last_id += 1
        return f'resampling-figure-item-{ContentItem.last_id}'

    def update_data(self):
        if self.first_data_updata_call:
            self.figure_template = self.f_template(self.server_data[self.data_name])
            self.data = self.figure_template
            self.update_current_time_stamp()
            self.first_data_updata_call = False

    def get_fresh_figure(self):
        return self.figure_template
    
    def get_layout(self):
        init_relayout = {
            'xaxis.autorange': True, 
            'xaxis.showspikes': False, 
            'yaxis.autorange': True, 
            'yaxis.showspikes': False,
            }

        return dash_enrich.html.Div(
            id=self.id,
            children=[
            dash_enrich.dcc.Store(
                id=f'figure-timestampStore-{self.id}',
                data=init_relayout),
            *self.invisibles,
            dash_enrich.html.Div(
                children=[
                    dash_enrich.html.Div(
                        children=self.FIGURE_TYPE, 
                        style = {
                            'padding-left': 2, 
                            'display':'inline',
                            'padding-right': 15, 
                        },
                    ),
                    dbc.Checklist(
                        id=f'show-legend-{self.id}',
                        value=[],
                        inline=True,
                        switch=True,
                        options=[
                            {'label': dash_enrich.html.Div(
                                'Show legend', 
                                style={
                                    'color':'white',
                                    'display':'inline', 
                                    'padding-right': 5,
                                    },
                                ), 
                            'value': 1,
                            },
                            ],
                        style = {'padding-left': 0, 'display':'inline'},
                    ),
                    dbc.Checklist(
                        id=f'fullscreen-{self.id}',
                        value=[],
                        inline=True,
                        switch=True,
                        options=[
                            {'label': dash_enrich.html.Div(
                                'Fullscreen', 
                                style={
                                    'color':'white',
                                    'display':'inline', 
                                    'padding-right': 0
                                    },
                                ), 
                            'value': 1,
                            },
                        ],
                        style = {'display':'inline'},
                    ),
                    *self.controls, 
                ],
                style={'display':'flex'}
            ),
            dash_enrich.dcc.Graph(
                figure=self.get_fresh_figure(),
                id=self.GRAPH_ID,
                config={
                    'responsive':True, 
                    'displaylogo': False,
                },
                style={
                    "margin":"4px", 
                    'border-radius': '10px', 
                    'overflow':'hidden', 
                    'height':'100%',
                },
                animate=self.animate,
            ),
            ],
            style={
                'display':'flex', 
                'flex-direction':'column', 
                "height":"100%",
            },
        )
    
    def set_callbacks(self, server):
        # show and hide legend
        # TODO: make this client side
        @server.app.callback(
            dash_enrich.Output(self.GRAPH_ID, "figure", allow_duplicate=True),
            dash_enrich.Input(f'show-legend-{self.id}', "value"),
            prevent_initial_call=False,
        )
        def update_fig(value):
            patch = Patch()
            patch['layout']['showlegend'] = 1 in value
            return patch
        
        # toggle fullscreen
        server.app.clientside_callback(
            f"""
            function(value) {{
                var target = document.getElementById('{self.id}');
                if (value.includes(1) && document.fullscreenElement === null){{
                    target.requestFullscreen();
                }} else if (value.length === 0) {{
                    if (document.fullscreenElement?.id === '{self.id}') {{
                        document.exitFullscreen();
                    }}
                }};
            }}
            """,
            dash_enrich.Input(f'fullscreen-{self.id}', 'value'),
        )

        # set fullscreen switch to off if fullcreen exited by ESC
        server.app.clientside_callback(
            f"""
            function(n_events) {{
            console.log("trigger fullscreen change");
                if (document.fullscreenElement?.id === '{self.id}') {{
                    return [1]
                }} else {{
                    return []
                }};
                
            }}
            """,
            dash_enrich.Input(f'fullscreen-listener', 'n_events'),
            dash_enrich.Output(f'fullscreen-{self.id}', 'value'),
        )

class LiveFigureItem(BaseFigureItem):
    def __init__(self, f_template, data_name, x_name='datetime', refresh_period=DEFAULT_REFRESH_PERIOD, rolling_span=None, **kwargs):
        super().__init__(f_template, data_name, x_name, **kwargs)
        self.REFRESH_PERIOD = refresh_period
        self.ROLLING_SPAN = rolling_span

        self.add_control(
            dbc.Checklist(
                id=f'auto-refresh-{self.id}',
                value=[1] if self.REFRESH_PERIOD is not None else [],
                inline=True,
                switch=True,
                options=[
                    {'label': dash_enrich.html.Div(
                        'Auto-refresh', 
                        style={
                            'color':'white',
                            'display':'inline', 
                            'padding-right': 0
                            },
                        ), 
                    'value': 1,
                    },
                ],
                style = {
                    'display':'none' if self.REFRESH_PERIOD is None else 'inline'},
            ),
        )

        self.add_invisible(
            dash_enrich.dcc.Interval(
                id=f'trigger-{self.id}',
                disabled=self.REFRESH_PERIOD is None,
                interval=self.REFRESH_PERIOD,
                n_intervals=1),
        )
    
    def update_data(self):
        if self.first_data_updata_call:
            self.figure_template = self.f_template(self.server_data[self.data_name])
            self.first_data_updata_call = False
            self.data = self.figure_template
        self.update_current_time_stamp()

    def get_fresh_figure(self):
        return self.f_template(self.server_data[self.data_name])

    def set_callbacks(self, server):
        super().set_callbacks(server)

        # enable/disable automatic refresh        
        server.app.clientside_callback(
            f"""
            function(value) {{
                return !value.includes(1);
            }}
            """,
            dash_enrich.Input(f'auto-refresh-{self.id}', 'value'),
            dash_enrich.Output(f'trigger-{self.id}', 'disabled'),
        )

        @server.app.callback(
            dash_enrich.Output(self.GRAPH_ID, "figure"),
            dash_enrich.Output(f'figure-timestampStore-{self.id}', "data"),
            dash_enrich.Input(f'trigger-{self.id}', 'n_intervals'),
            dash_enrich.State(f'figure-timestampStore-{self.id}', "data"),
            prevent_initial_call=True,
            )
        def update_figure(n_intervals, data_time_stamp):
            if n_intervals < 1 or self.data is None or data_time_stamp==self.data_time_stamp:
                return dash_enrich.no_update
            else:
                patch = Patch()
                patch['data'] = self.data['data']
                
                return patch, self.data_time_stamp

class FigureResamplerItem(BaseFigureItem):
    """This class makes it convenient to use plotly-resampler. It will be 
        automaticaly refreshing data from the server to the clien side. It is 
        perfect for showing big timeseries data stored in CSV files of pandas
        Dataframes.
    """
    def __init__(self, 
            f_template, data_name, x_name='datetime', refresh_period=DEFAULT_REFRESH_PERIOD, 
            rolling_span=100, **kwargs):
        
        super().__init__(f_template, data_name, x_name, **kwargs)
        self.REFRESH_PERIOD = refresh_period
        self.ROLLING_SPAN = rolling_span
        
        self.FIGURE_TYPE = 'Live Figure'
        self.init_relayout = {
            'xaxis.autorange': True, 
            'xaxis.showspikes': False, 
            'yaxis.autorange': True, 
            'yaxis.showspikes': False,
            }
        
        self.add_control(
            dbc.Checklist(
                id=f'show-rolling-{self.id}',
                value=[],
                inline=True,
                switch=True,
                options=[
                    {'label': dash_enrich.html.Div(
                        'Rolling', 
                        style={
                            'color':'white',
                            'display':'inline', 
                            'padding-right': 5
                            },
                        ), 
                    'value': 1,
                    },
                ],
                style = {
                    'display':'none' if self.ROLLING_SPAN is None else 'inline'},
            ),
        )

        self.add_control(
            dbc.Checklist(
                id=f'auto-refresh-{self.id}',
                value=[1] if self.REFRESH_PERIOD is not None else [],
                inline=True,
                switch=True,
                options=[
                    {'label': dash_enrich.html.Div(
                        'Auto-refresh', 
                        style={
                            'color':'white',
                            'display':'inline', 
                            'padding-right': 0
                            },
                        ), 
                    'value': 1,
                    },
                ],
                style = {
                    'display':'none' if self.REFRESH_PERIOD is None else 'inline'},
            ),
        )

        self.add_invisible(
            dash_enrich.dcc.Interval(
                id=f'trigger-{self.id}',
                disabled=self.REFRESH_PERIOD is None,
                interval=self.REFRESH_PERIOD,
                n_intervals=1),
        )

        self.add_invisible(
            dash_enrich.dcc.Store(
                id=f'figure-relayoutDataStore-{self.id}', 
                data=self.init_relayout),
        )

    def update_data(self):
        if self.first_data_updata_call:
            self.figure_template = self.f_template(self.server_data[self.data_name])
            self.first_data_updata_call = False
            self.data = FigureResampler(self.figure_template)
        self.update_current_time_stamp()

    def get_fresh_figure(self):
        for trace_data in self.figure_template['data']:
            trace_data['x'] = self.server_data[self.data_name][self.x_name]
            trace_data['y'] = self.server_data[self.data_name][trace_data['name']]
        return FigureResampler(self.figure_template)

    def get_rolling_relayoutData(self, data):
        end = data['data'][0]['x'][-1]
        start = end - self.ROLLING_SPAN
        return {'xaxis.range[0]': start, 'xaxis.range[1]': end}
    
    def set_callbacks(self, server):
        super().set_callbacks(server)

        # enable/disable automatic refresh        
        server.app.clientside_callback(
            f"""
            function(value) {{
                return !value.includes(1);
            }}
            """,
            dash_enrich.Input(f'auto-refresh-{self.id}', 'value'),
            dash_enrich.Output(f'trigger-{self.id}', 'disabled'),
        )

        @server.app.callback(
            dash_enrich.Output(self.GRAPH_ID, "figure"),
            dash_enrich.Output(f'figure-timestampStore-{self.id}', "data"),
            dash_enrich.Input(f'trigger-{self.id}', 'n_intervals'),
            dash_enrich.State(f'figure-relayoutDataStore-{self.id}', 'data'),
            dash_enrich.State(f'show-rolling-{self.id}', "value"),
            dash_enrich.State(f'figure-timestampStore-{self.id}', "data"),
            prevent_initial_call=True,
            )
        def update_figure(n_intervals, relayoutdata, value, data_time_stamp):
            if n_intervals < 1 or self.data is None or data_time_stamp==self.data_time_stamp:
                return dash_enrich.no_update
            else:
                data = self.get_fresh_figure()
                if 1 in value:
                    return data.construct_update_data_patch(self.get_rolling_relayoutData(data)), self.data_time_stamp
                else:
                    return data.construct_update_data_patch(relayoutdata), self.data_time_stamp
        
        # relayout resample data callback
        @server.app.callback(
            dash_enrich.Output(self.GRAPH_ID, "figure", allow_duplicate=True),
            dash_enrich.Output(f'figure-relayoutDataStore-{self.id}', 'data'),
            dash_enrich.Input(self.GRAPH_ID, "relayoutData"),
            dash_enrich.Input(f'show-rolling-{self.id}', "value"),
            prevent_initial_call=True,
        )
        def update_fig(relayoutdata, value):
            if dash_enrich.ctx.triggered_id == f'show-rolling-{self.id}':
                relayoutdata = {
                    'xaxis.autorange': True, 
                    'xaxis.showspikes': False, 
                    'yaxis.autorange': True, 
                    'yaxis.showspikes': False,
                }
            if self.data is None or relayoutdata == {'autosize': True}:
                return dash_enrich.no_update
            else:
                data = self.get_fresh_figure()
                if 1 in value:
                    return data.construct_update_data_patch(self.get_rolling_relayoutData(data)), relayoutdata
                else:
                    return data.construct_update_data_patch(relayoutdata), relayoutdata