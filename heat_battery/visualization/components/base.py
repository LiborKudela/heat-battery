import dash_extensions.enrich as dash_enrich
import dash_bootstrap_components as dbc
import datetime

class ContentItem:

    last_id = 0

    def __init__(self):
        self.id = self.get_new_id()
        self.data = None
        self.update_current_time_stamp()

    def get_children(self):
        return []

    def set_server_data(self, server_data):
        self.server_data = server_data
        self.pass_server_data_to_children()

    def pass_server_data_to_children(self):
        children = self.get_children()
        assert isinstance(children, list), "get_children must return list"
        for child in children:
            print(f'passing server_data item: {self.id} -> child: {child.id}')
            child.set_server_data(self.server_data)

    def get_new_id(self):
        ContentItem.last_id += 1
        return f'content-item-{ContentItem.last_id}'
    
    def update_current_time_stamp(self):
        self.data_time_stamp = datetime.datetime.now().strftime(f'%m/%d/%Y, %H:%M:%S.%f')

    def get_layout(self):
        return dash_enrich.html.Div(
            [],
        )
    
    def update_data(self):
        self.update_current_time_stamp()
    
    def set_callbacks(self, server):
        pass