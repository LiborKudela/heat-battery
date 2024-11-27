import dash_extensions.enrich as dash_enrich
import dash_bootstrap_components as dbc

class VisualizerApp():
    def __init__(self, name):
        self.page_ready = False
        self.name = name.capitalize()
        self.href = '/' + name.lower()
        self.data = None # initial data
        self.data_updated = False
        self.in_menu = True

    def get_children(self):
        return []

    def set_server_data(self, server_data):
        self.server_data = server_data
        self.pass_server_data_to_children()

    def pass_server_data_to_children(self):
        children = self.get_children()
        assert isinstance(children, list), "get_children must return list"
        for child in children:
            print(f'passing server_data page: {self.name} -> child: {child.id}')
            child.set_server_data(self.server_data)

    def get_href(self):
        return self.href
    
    def get_link(self):
        return dbc.NavLink(self.name, href=self.href, active="exact")
    
    def get_page_content(self):
        return dash_enrich.html.Div(
            id=f'{self.href}-pagecontainer',
            children=[
                self.get_layout(),
            ],
            style={"height":"100%"},
        )
    
    def get_layout(self):
        return dash_enrich.html.Div(
            [],
        )

    def _update_data(self):
        '''static page switch handling'''
        self.update_data()
        self.page_ready = True

    def update_data(self):
        '''defined in subclass'''
        pass

    def set_callbacks(self, server):
        pass