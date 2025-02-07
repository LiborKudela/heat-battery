import dash_chart_editor as dce
from .base import ContentItem, dash_enrich, dbc
from heat_battery.simulations.postgresql_project import Project
import pandas as pd


class ChartStudio(ContentItem):
    def __init__(self, project:Project, parent=None):
        super().__init__(parent=parent)
        self.project = project
        
    def get_new_id(self):
        ContentItem.last_id += 1
        return f'chart-studio-{ContentItem.last_id}'

    def get_layout(self, qs_data:dict|None=None):
        #group_names = qs_data['group_names']
        #print(group_names)
        #df = self.project.get_parameter_space(group_names)
        df_test = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50],
        })

        return dash_enrich.html.Div(
            [
                #dash_enrich.html.H4("Comparator studio"),
                dce.DashChartEditor(
                    dataSources=df_test.to_dict("list"),
                    style={'width': '100%', 'height': '100%'},
                )
            ],
            style={'width': '100%', 'height': '100%'},
        )