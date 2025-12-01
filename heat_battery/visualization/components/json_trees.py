import dash_bootstrap_components as dbc
from heat_battery.visualization.pages.base import dash_enrich, dbc, ClientsideFunction

from heat_battery.visualization.components.base import ContentItem
class JSONTreeView(ContentItem):
    def __init__(self):
        super().__init__()
        self.id = 'json-tree-view'
    
    def get_new_id(self):
        ContentItem.last_id += 1
        return f'json-tree-view-{ContentItem.last_id}'
    
    def recursive_walk_node(self, node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key, value
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield f'[{i}]', value
    
    def get_layout(self, json_data):
        return dbc.Accordion(
            [
                dbc.AccordionItem(
                    children=[dash_enrich.html.Div(id=self.id)]
                )
            ]
        )
    