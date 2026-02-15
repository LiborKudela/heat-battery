from heat_battery.visualization.pages.base import VisualizerApp, dash_enrich
from heat_battery.simulations.postgresql_project import Project
import dash_bootstrap_components as dbc
from dash import get_asset_url
from dash_iconify import DashIconify
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# import website components
from heat_battery.visualization.pages.single_item_page import SingleItemPage
from heat_battery.visualization.components.tables import JobsTable
from heat_battery.visualization.components.result_viewer import ResultViewerComponent
from heat_battery.visualization.components.result_comparator import ResultComparatorComponent

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
            initial_comparator_figures: list|None=None,
            variable_descriptions: dict|None=None,
            initial_transforms: list|None=None,
        ):
        self.project = project
        super().__init__(name=project.project_name, parent=parent)

        self.pages = {
            '/jobs-overview': SingleItemPage(
                "Jobs table", 
                JobsTable(project), 
                parent=self,
                icon="mdi:table",
                tooltip_text="Jobs table",
            ),

            '/result-data': SingleItemPage(
                "Result viewer", 
                ResultViewerComponent(
                    project, 
                    #result_chart_data, 
                    #figure_creators_getters,
                    fig_theme=fig_theme,
                    initial_figures=initial_figures,
                    variable_descriptions=variable_descriptions,
                    initial_transforms=initial_transforms,
                ), 
                parent=self,
                icon="mdi:chart-line",
                tooltip_text="Result viewer",
            ),

            '/result-comparator': SingleItemPage(
                "Result comparator", 
                ResultComparatorComponent(
                    project,
                    fig_theme=fig_theme,
                    initial_figures=initial_comparator_figures,
                ), 
                parent=self,
                icon="mdi:compare",
                tooltip_text="Result comparator",
            ),
        }
        self.set_subpages_hrefs()

    def get_children(self):
        return list(self.pages.values())

    # def get_menu(self, dashboard=True, item_style=None, icon_width=30, menu_style=None):
    #     menu = []
    #     if dashboard:
    #         menu.append(dbc.NavLink(
    #             DashIconify(icon="mdi:home", width=30),
    #             href=self.href,
    #             active="exact",
    #         ))
    #     for page in self.pages.values():
    #         btn = dbc.NavLink(
    #             DashIconify(icon=page.icon, width=30),
    #             href=page.href,
    #             active="exact",
    #         )
    #         if item_style:
    #             btn.style = item_style
    #         menu.append(btn)
    #     return menu

    def preload_cache_data(self):
        for page in self.pages.values():
            page.preload_cache_data()

    def get_link(self):
        # Override to return a simple NavLink instead of DropdownMenu
        # This removes the submenu from the side menu
        return dbc.NavLink(self.name, href=self.href, active="exact")

    def get_icons_menu(self, icon_width=30, dashboard=True, vertical=False):
        btns = []
        if dashboard:
            btns.append(self.get_menu_icon(icon_width=icon_width, tooltip_text="Dashboard", icon="mdi:monitor-dashboard"))
        for page in self.pages.values():
            btns.append(page.get_menu_icon(icon_width=icon_width))

        return dbc.Nav(
            children=btns,
            pills=False,
            vertical=vertical,
            className="bg-dark rounded justify-content-center project-component-nav-icon",
            style={
                'width': '100%' if not vertical else 'height: 100%',
                'padding': '3px',
            }
        )


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

        # Navigation icons in horizontal navbar at bottom
        # nav_icons = dbc.Nav(
        #     children=[page.get_menu_icon(icon_width=30) for page in self.pages.values()],
        #     pills=False,
        #     className="bg-dark justify-content-center project-component-nav-icon",
        #     style={
        #         'width': '100%',
        #         'padding': '10px',
        #     }
        # )

        # Main content area
        main_content = dash_enrich.html.Div(
            children=[
                title,
                overview_chart,
            ],
            style={
                'flex': '1',
                'text-align': 'center',
                'overflow-y': 'auto',
            }
        )

        # Container with main content and bottom navigation
        div = dash_enrich.html.Div(
            id=f'project-dashboard-{self.project.project_name}',
            children=[
                self.get_icons_menu(icon_width=20, vertical=False, dashboard=False),
                main_content,
            ],
            style={
                'height': '100%',
                'width': '100%',
                'margin': 'auto',
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'stretch',
            },
        )
        return div
    
    def set_callbacks(self, server):
        for page in self.pages.values():
            page.set_callbacks(server)