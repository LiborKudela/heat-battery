from .base import ContentItem, dash_enrich, dbc


# modal_help = dbc.Modal(
#     [
#         dbc.ModalHeader(dbc.ModalTitle("Help")),
#         dbc.ModalBody(
#             children=dash_enrich.dcc.Markdown(
#                 '''
#                 ## Help content:
#                 Example of equations:  
#                 $\\LARGE{H_{c}} [J] - Generated heat$ 

#                 $\\LARGE{\\eta_{u}=\\frac{H_{sr,u}}{H_{c}}}$

#                 $\\LARGE{\\eta_{pv,u}=\\frac{H_{sr,u}}{H_{c}}}$
                
#                 \\eta_{used}=\\frac{H_{s2r, used}}{H_{c}}
#                 ''', mathjax=True),
#             ),
#         dbc.ModalFooter(
#         ),
#     ],
#     id="help-modal",
#     is_open=False,
#     centered=True,
#     )

class GridLayout(ContentItem):

    def __init__(self, items, parent=None):
        super().__init__(parent=parent)
        self.items = items
        self.items_id_map = {self.items[i].id: i for i in range(len(self.items))}

    def get_new_id(self):
        ContentItem.last_id += 1
        return f'grid-item-{ContentItem.last_id}'
    
    def preload_cache_data(self):
        for item in self.items:
            item.preload_cache_data()

    def get_children(self):
        return self.items

    # def subplot_grid_size(self, n):
    #     rows, cols = 1, 1
    #     for i in range(n):
    #         if rows*cols >= n:
    #             break
    #         else:
    #             if rows + 1 > cols:
    #                 cols += 1
    #             else:
    #                 rows += 1
    #     return rows, cols
    
    def get_grid_items(self, df=None, qs_data:dict|None=None) -> list:
        '''defined in subclass'''
        divs = []
        for item in self.items:
            div = dash_enrich.html.Div(
            id={'type':'grid-item', 'index':f'container-{item.id}'},
            children=[
                item.get_layout(df, qs_data),
            ],
            style={
                'className': 'grid-item', 
                'display': 'flex', 
                'flexDirection': 'column', 
                }
            )
            divs.append(div)
        return divs

    def get_grid_div(self, df=None, qs_data:dict|None=None):
        #rows, cols = self.subplot_grid_size(len(self.items))
        return dash_enrich.html.Div(
            id=f'grid-div-{self.id}',
            children=self.get_grid_items(df, qs_data),  
            style={
                'display':'grid',
                #'gridTemplateColumns': f'repeat({cols}, 1fr)',
                'gridTemplateColumns': f'repeat(auto-fit, minmax(min(900px, 100vw), 1fr))',
                #'grid-template-rows': f'repeat({rows}, 1fr)',
                'gridAutoRows': 'minmax(250px, auto)',
                'position':'relative', 
                'top':0, 
                'left':0, 
                'bottom':0, 
                'right':0, 
                'width':'100%', 
                'height':'100%', 
                'margin':0, 
                'padding':0, 
                "overflowY": "auto",
            },
        )
    
    def get_layout(self):
        div = dash_enrich.html.Div(
            id=f'{self.id}-grid-container',
            children=self.get_grid_div(),
            style={"height":"100%"},
            )
        return div

    def set_callbacks(self, server):
        for i, item in enumerate(self.items):
            item.set_callbacks(server)