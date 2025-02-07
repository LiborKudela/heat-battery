import dash_extensions.enrich as dash_enrich
import dash_bootstrap_components as dbc
from dash import ClientsideFunction
from dash_iconify import DashIconify

class VisualizerApp():
    def __init__(self, name, parent=None, bootstrap_style:str='bootstrap'):
        self.page_ready = False
        self.name = name.capitalize()
        self.href = '/' + name.lower()
        self.data = None # initial data
        self.data_updated = False
        self.in_menu = True
        self.parent = parent
        self.pages = {}
        self.bootstrap_style = bootstrap_style

    def set_subpages_hrefs(self):
        for href, page in self.pages.items():
            page.href = self.href + href

    def preload_cache_data(self):
        for page in self.pages.values():
            page.preload_cache_data()

    def pass_server_data_to_children(self):
        children = self.get_children()
        assert isinstance(children, list), "get_children must return list"
        for child in children:
            print(f'passing server_data page: {self.name} -> child: {child.id}')
            child.set_server_data(self.server_data)

    def preload_cache_data(self):
        for page in self.pages.values():
            page.preload_cache_data()

    def get_href(self):
        return self.href
    
    def get_link(self): 
        if not self.pages:
            return dbc.NavLink(self.name, href=self.href, active="exact")
        else:
            return dbc.DropdownMenu(
                label=self.name,
                nav=True,
                group=True,
                direction="bottom",
                children=[
                    dbc.DropdownMenuItem(page.get_link())
                    for page in self.pages.values()
                ],
            )
    
    def get_page_content(self, qs_data:dict|None=None):
        return dash_enrich.html.Div(
            id=f'{self.href}-pagecontainer',
            children=[
                self.get_layout(qs_data),
            ],
            style={"height":"100%"},
        )
    
    def get_layout(self, qs_data:dict|None=None):
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