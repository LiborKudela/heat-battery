import dash_extensions.enrich as dash_enrich
import dash_bootstrap_components as dbc
from dash import ClientsideFunction
from dash_iconify import DashIconify

class VisualizerApp():
    def __init__(self, 
        name, parent=None, bootstrap_style:str='bootstrap', 
        required_permission='authenticated', icon=None, tooltip_text=None):

        self.page_ready = False
        self.name = name.capitalize()
        self.tooltip_text = tooltip_text
        self.href = '/' + name.lower()
        self.data = None # initial data
        self.data_updated = False
        self.in_menu = True
        self.parent = parent
        self.pages = {}
        self.bootstrap_style = bootstrap_style
        # Permission required to access this page
        # Options: 'public', 'authenticated', or any custom permission string
        self.required_permission = required_permission
        self.icon = icon

    def set_subpages_hrefs(self):
        for href, page in self.pages.items():
            page.href = self.href + href
    
    def set_permission(self, permission):
        '''Set the required permission for this page'''
        self.required_permission = permission
        return self  # Allow chaining
    
    def set_subpages_permissions(self, permission):
        '''Set permissions for all subpages'''
        for page in self.pages.values():
            page.set_permission(permission)
        return self  # Allow chaining

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
            # Include a "Dashboard" link to the main page when there are subpages
            dropdown_items = [
                dbc.DropdownMenuItem(
                    dbc.NavLink("Dashboard", href=self.href, active="exact"),
                )
            ]
            # Add all subpage links
            dropdown_items.extend([
                dbc.DropdownMenuItem(page.get_link())
                for page in self.pages.values()
            ])
            
            return dbc.DropdownMenu(
                label=self.name,
                nav=True,
                group=True,
                direction="bottom",
                children=dropdown_items,
            )
    
    def get_menu_icon(self, icon_width=30, tooltip_text=None, icon=None):
        if icon is None:
            if self.icon is None:
                raise ValueError(f'This page has no default icon and no icon was provided: {self.name}')
            icon = self.icon
        if tooltip_text is None:
            if self.tooltip_text is None:
                raise ValueError(f'This page has no tooltip text and no tooltip text was provided: {self.name}')
            tooltip_text = self.tooltip_text
        tooltip = dbc.Tooltip(
            tooltip_text,
            target=f'{self.name}-menu-icon',
            placement="top",
            style={'position':'fixed'}
        ) if tooltip_text is not None else None
        return dbc.NavItem(
            [
                tooltip,
                dbc.NavLink(
                        DashIconify(icon=icon, width=icon_width),
                        id=f'{self.name}-menu-icon',
                        href=self.get_href(),
                        active="exact",
                        className="rounded d-flex justify-content-center align-items-center me-2",
                        style={
                            'padding': '3px',
                        }
                    )
            ]
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