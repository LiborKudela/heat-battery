from heat_battery.visualization.pages.base import VisualizerApp, dash_enrich
from heat_battery.simulations.postgresql_project import Project
import dash_bootstrap_components as dbc
from dash import get_asset_url
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# import website components
from heat_battery.visualization.pages.single_item_page import SingleItemPage
from heat_battery.visualization.components.tables import JobsTable
from heat_battery.visualization.components.result_viewer import ResultViewerComponent
from heat_battery.visualization.components.chart_studio import ChartStudio

df_test = pd.DataFrame({
    'progress': 100*np.random.rand(10),
    'dim_1': np.random.rand(10),
    'dim_2': 2*np.random.rand(10),
    'dim_3': 3*np.random.rand(10),
    'dim_4': 4*np.random.rand(10),
    'dim_5': 5*np.random.rand(10),
})

def get_overview_chart(data, dimensions=['dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5']):
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(
                color = data['progress'],
                colorscale = 'Electric',
                showscale = False,
                cmin = 0,
                cmax = 100,
            ),
            dimensions = [
                dict(range = [data[d_name].min(), data[d_name].max()],
                    #constraintrange = [data[d_name].min(), data[d_name].max()],
                    label = d_name, values = data[d_name])
                for d_name in dimensions
            ]
        ),
    )
    return fig

class ProjectViewerSuperApp(VisualizerApp):
    def __init__(
            self, 
            project: Project, 
            #result_chart_data: dict,
            #figure_creators_getters: dict|None=None,
            parent=None,
            fig_theme:str='bootstrap',
            initial_figures: list|None=None,
        ):
        self.project = project
        super().__init__(name=project.project_name, parent=parent)

        self.pages = {
            '/jobs-overview': SingleItemPage(
                "Jobs table", 
                JobsTable(project), 
                parent=self),

            '/result-data': SingleItemPage(
                "Result viewer", 
                ResultViewerComponent(
                    project, 
                    #result_chart_data, 
                    #figure_creators_getters,
                    fig_theme=fig_theme,
                    initial_figures=initial_figures,
                ), 
                parent=self
            ),

            '/result-comparator': SingleItemPage(
                "Result comparator", 
                ChartStudio(project), 
                parent=self,
            ),
        }
        self.set_subpages_hrefs()

    def get_children(self):
        return list(self.pages.values())

    def preload_cache_data(self):
        for page in self.pages.values():
            page.preload_cache_data()

    def get_layout(self, qs_data:dict|None=None):

        title = dash_enrich.html.Div(
            children=[
                dash_enrich.html.H3(
                children=f"Project dashboard", 
                style = {
                    'paddingLeft': 2,  
                    'padding-right': 15,
                    'margin':'auto',
                    'font-weight':'bold',
                },
            ),
            dash_enrich.html.H5(
                children=f"{self.project.project_name}", 
                style = {
                    'paddingLeft': 2, 
                    'padding-right': 15,
                    'margin':'auto',
                    'font-style':'italic',
                },
            ),
            ],
        )

        overview_chart = dash_enrich.html.Div(
            children=[
                dash_enrich.dcc.Graph(
                    id='overview-chart',
                    figure=get_overview_chart(df_test),
                    style={
                        'border-radius': '10px', 
                        'overflow':'hidden',
                        'height':'100%',
                    },
            )
            ],
            style={
                "width": "calc(54rem + 30px)", 
                'margin-bottom':'15px',
                'height':'40%',
            }
        )

        overview_card = dbc.Card(
            children=[
                dbc.CardImg(src=get_asset_url('images/table.png'), top=True),
                dbc.CardBody(
                    [
                        dash_enrich.html.H4("Jobs Overview", className="card-title"),
                        dash_enrich.html.P(
                            "Live data about the running jobs within the project in "
                            "tabular form",
                            className="card-text",
                        ),
                        dbc.Nav(
                            dbc.NavItem(
                                dbc.NavLink(
                                    dbc.Button(
                                        "Open overview", 
                                        style={'width':'100%'}
                                    ),
                                    href=self.pages['/jobs-overview'].get_href(), 
                                    active="exact"
                                ),
                                style={'width':'100%'}
                                
                            ),
                        ),
                    ]
                ),
            ],
            style={"width": "18rem"},
        )

        result_data_card = dbc.Card(
            children=[
                dbc.CardImg(src=get_asset_url('images/chart.png'), top=True),
                dbc.CardBody(
                    [
                        dash_enrich.html.H4("Result Viewer", className="card-title"),
                        dash_enrich.html.P(
                            "Live charts viewer for the simulation data of "
                            "individual jobs within the project",
                            className="card-text",
                        ),
                        dbc.Nav(
                            dbc.NavItem(
                                dbc.NavLink(
                                    dbc.Button(
                                        "Open result viewer", 
                                        style={'width':'100%'}
                                    ),
                                    href=self.pages['/result-data'].get_href(), 
                                    active="exact"
                                ),
                                style={'width':'100%'}
                            ),
                        ),
                    ]
                ),
            ],
            style={"width": "18rem"},
        )

        compare_card = dbc.Card(
            children=[
                dbc.CardImg(src=get_asset_url('images/table.png'), top=True),
                dbc.CardBody(
                    [
                        dash_enrich.html.H4("Result Comparator", className="card-title"),
                        dash_enrich.html.P(
                            "Compare results of finished jobs from their "
                            "input/output perspective",
                            className="card-text",
                        ),
                        dbc.Nav(
                            dbc.NavItem(
                                dbc.NavLink(
                                    dbc.Button(
                                        "Open comparator", 
                                        style={'width':'100%'}
                                    ),
                                    href=self.pages['/result-comparator'].get_href(), 
                                    active="exact"
                                ),
                                style={'width':'100%'}
                            ),
                        ),
                    ]
                ),
            ],
            style={"width": "18rem"},
        )

        cards = dash_enrich.html.Div(
            children=dash_enrich.html.Div(
                children=[
                    overview_card,
                    result_data_card,
                    compare_card,
                    
                ],
                style={
                'display':'flex', 
                'flex-wrap':'wrap',
                'align-items':'stretch', 
                'height':'100%', 
                'gap':'15px'
                },
                className='card-deck',
            )
        )

        div = dash_enrich.html.Div(
            id=f'project-dashboard-{self.project.project_name}',
            children=[
                title,
                overview_chart,
                cards,
            ],
            style={
                'height':'90%', 
                'width':'90%', 
                'margin':'auto',
                'text-align':'center'
            },
        )
        return div
    
    def set_callbacks(self, server):
        for page in self.pages.values():
            page.set_callbacks(server)