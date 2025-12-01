import datetime
import pandas as pd
from heat_battery.simulations.jobs import Job
import dash_ag_grid as dag
import pprint
import time
import json
from heat_battery.visualization.pages.base import dash_enrich, dbc, ClientsideFunction, DashIconify
from heat_battery.simulations.postgresql_project import Project
from heat_battery.visualization.components.base import ContentItem
from heat_battery.visualization.linux_shm_cache import SHMCache

DEFAULT_REFRESH_PERIOD = 5000
DEFAULT_MAX_NUM_FILTER_CONDITIONS = 5

def format_python_data(data, header_comment):
    return f"{header_comment}\n{json.dumps(data, indent=4).replace('true', 'True').replace('false', 'False')}"

class JobsTable(ContentItem):
    CLIENT_SIDE_COLUMNS = [
        'group_priority',
        'priority', 
        'status',   
        'progress',
        'remaining_time',
        'elapsed_time',
        'signature', 
        'group_name', 
        'created_by', 
        'insert_datetime',
        'last_updated',
        'last_checkpoint_progress',
        'last_checkpoint_date',
        'active_node_address',
    ]
    
    def __init__(self, project: Project, refresh_period: int|None=DEFAULT_REFRESH_PERIOD, parent=None):
        super().__init__(parent=parent)
        self.project = project
        self.AG_GRID_ID = f'{self.id}-ag-grid'
        self.LATEST_DATA_TIMESTAMP_ID = f'{self.id}-latest-data-timestamp'
        self.FILTERS_TABLE_BODY_ID = f'{self.id}-filters-table-body'
        self.DATA_TABLE_BODY_ID = f'{self.id}-data-table-body'
        self.REFRESH_TRIGGER_ID = f'{self.id}-refresh-trigger'
        self.REFRESH_PERIOD = refresh_period
        self.INPUTS_MODAL_TEXT_ID = f'{self.id}-inputs-modal-text'
        self.INPUTS_STORE_ID = f'{self.id}-inputs-store'
        self.FILTER_INPUT_ID_TYPE = f'{self.id}-filter-input'
        self.cache = SHMCache(f'JobsTable-{self.project.project_name}')

    def get_new_id(self):
        ContentItem.last_id += 1
        return f'jobs-table-item-{ContentItem.last_id}'

    def get_fresh_table(self):
        jobs_rows = self.project._get_jobs_query()
        jobs = pd.DataFrame(jobs_rows, columns=Job.COLUMNS.keys())
        return jobs

    def get_layout(self, qs_data:dict|None=None):
        table_columnDefs = [
            {
                # select column
                "headerName": "Selection",
                "field": "signature", 
                "filter": 'agTextColumnFilter', 
                'sortable': True, 
                'headerTooltip': 'Select jobs',
                'cellRenderer': 'signatureCell',    
                #'tooltipValueGetter':{'function': "'Select the job to open its results: ' + params.data.signature"},
                'checkboxSelection': True,
                "headerCheckboxSelection": {"function": 'params.column ? params.column == params.api.getAllDisplayedColumns()[0] : false'},
                "headerCheckboxSelectionFilteredOnly": {"function": 'params.column == params.api.getAllDisplayedColumns()[0]'},
                'headerTooltip': 'Unique identifier of the job',
                'tooltipValueGetter':{'function': " params.data && params.data.signature ? 'The signature is: ' + params.data.signature : ''"},
                'minWidth': 120,
                'hide': False,
            },
            {
                # job created by
                "field": "created_by", 
                'filter':False,
                'headerName': 'Owner', 
                'cellRenderer': 'createdByAvatar',
                'headerTooltip': 'User who inserted the job into the database queue',
                'tooltipValueGetter':{'function': " params.data && params.data.created_by ? 'Name: ' + params.data.created_by.first_name + ' ' + params.data.created_by.last_name : ''"},
                'minWidth': 90,
                'maxWidth': 100,
                'hide': False,
            },
            {
                # open actions menu button
                "headerName": "Actions",
                "field": "signature", 
                "cellRenderer": "actionsMenu", 
                'cellRendererParams': {'project_name': self.project.project_name},
                "filter": False, 
                'sortable': False, 
                'headerTooltip': 'Open action menu for given job',
                'tooltipValueGetter':{'function': " params.data && params.data.signature ? 'Open action menu for job with signature: ' + params.data.signature : ''"},
                'minWidth': 100,
                'maxWidth': 100,
                "cellStyle": {"overflow": "visible"},
                'hide': False,
            },
            {
                # group priority
                "field": "group_priority", 
                'filter': 'agNumberColumnFilter',
                'headerName': 'GP', 
                'headerTooltip':'Priority of the simulation group (aka Group Priority)', 
                'tooltipValueGetter':{'function': " params.data && params.data.group_priority ? 'The priority of the group is: ' + params.data.group_priority : ''"},
                'minWidth': 75,
                'maxWidth': 100,
                'hide': False,
            },
            {
                # job priority
                "field": "priority",
                'filter': 'agNumberColumnFilter',
                'headerName': 'P', 
                'headerTooltip':'Priority of job within its simulation group', 
                'tooltipValueGetter':{'function': " params.data && params.data.priority ? 'The priority of the job is: ' + params.data.priority : ''"},
                'minWidth': 75,
                'maxWidth': 100,
                'hide': False,
            },
            {
                # job status
                "field": "status", 
                'filter': 'agTextColumnFilter',
                'headerName': 'Status',
                "cellRenderer": "statusBadge",
                'headerTooltip': 'Status of the job',
                'tooltipValueGetter':{'function': " params.data && params.data.status ? 'This job is: ' + params.data.status : ''"},
                'minWidth': 135,
                'hide': False,
            },
            {
                # job progress
                "field": "progress", 
                'filter': 'agNumberColumnFilter',
                'headerName': 'Progress',
                "cellRenderer": "progressBar", 
                "flex": 1,
                'headerTooltip': 'Progress of simulation job',
                'tooltipValueGetter':{'function': " params.data && params.data.progress ? 'The progress of the job is: ' + params.data.progress.toFixed(3) + '%' : ''"},
                'minWidth': 125,
                'hide': False,
            },
            {
                # job remaining time
                "field": "remaining_time",
                'filter': 'agNumberColumnFilter', 
                'headerName': 'Remaining',
                "cellRenderer": "remainingTime",
                'headerTooltip': 'Estimated remaining time to finish the job (calculated from the performance of last active node)',
                'tooltipValueGetter':{'function': " params.data && params.data.remaining_time ? 'The remaining time in seconds: ' + params.data.remaining_time + 's' : ''"},
                'minWidth': 130,
                'hide': False,
            },
            {
                # job elapsed time
                "field": "elapsed_time",
                'filter': 'agNumberColumnFilter',  
                'headerName': 'Elapsed',
                "cellRenderer": "elapsedTime",
                'headerTooltip': 'Elapsed CPU time of the job (calculated from all nodes that participated on this job)',
                'tooltipValueGetter':{'function': " params.data && params.data.elapsed_time ? 'The elapsed time is: ' + params.data.elapsed_time + ' s' : ''"},
                'minWidth': 130,
                'hide': False,
            },
            {
                # job last updated
                "field": "last_updated", 
                'filter': 'agDateColumnFilter',
                'headerName': 'Last update', 
                'cellRenderer': 'lastUpdated',
                'headerTooltip': 'Time when the job data was last updated',
                'tooltipValueGetter':{'function': " params.data && params.data.last_updated ? 'Last interaction was at: ' + params.data.last_updated : ''"},
                'minWidth': 130,
                'hide': False,
            },
            {
                # job last checkpoint date
                "field": "last_checkpoint_date", 
                'filter': 'agDateColumnFilter',
                'headerName': 'Last checkpoint', 
                'cellRenderer': 'lastCheckpoint',
                'headerTooltip': 'Time when the job was last checkpointed',
                'tooltipValueGetter':{'function': " params.data && params.data.last_checkpoint_date ? 'Last checkpoint was at: ' + params.data.last_checkpoint_date + ' with progress: ' + params.data.last_checkpoint_progress : ''"},
                'minWidth': 130,
                'hide': False,
            },
            {
                # group name
                "field": "group_name", 
                'filter': 'agTextColumnFilter',
                'headerName': 'Group', 
                'headerTooltip': 'User provided name of the simulation group (aka Group Name)',
                'tooltipValueGetter':{'function': " params.data && params.data.group_name ? 'The group name is: ' + params.data.group_name : ''"},
                'minWidth': 130,
                'hide': False,
            },
            {
                # job insert datetime
                "field": "insert_datetime", 
                'filter': 'agDateColumnFilter',
                'headerName': 'Inserted', 
                'cellRenderer': 'inserted',
                'headerTooltip': 'Datetime when the job was inserted into the database queue',
                'tooltipValueGetter':{'function': " params.data && params.data.insert_datetime ? 'The job was inserted at: ' + params.data.insert_datetime : ''"},
                'minWidth': 130,
                'hide': False,
            },
            {
                # job active node address
                "field": "active_node_address", 
                'filter': 'agTextColumnFilter',
                'headerName': 'Active node', 
                'headerTooltip': 'Address/nickname of the node where the job is running',
                'tooltipValueGetter':{'function': "params.data && params.data.active_node_address ? 'The job is active at: ' + params.data.active_node_address : ''"},
                'minWidth': 130,
                'hide': False,
            },
        ]

        tools = dash_enrich.html.Div(
            children=[
                dbc.Button('Show/hide columns', color='primary', size='sm', style={'font-weight':'bold'}, id='select-columns-button'),
                dbc.Button('Add job', color='success', size='sm', style={'font-weight':'bold'}),
            ],
            style={
                'display':'flex',
                'height':'32px',
                #'padding':'4px',
                'margin-bottom':'4px',
                'margin-left':'4px',
                'gap':'4px',
            },
        )

        select_columns_modal = dbc.Modal(
            id='select-columns-modal',
            is_open=False,
            size='lg',
            children=[
                dbc.ModalHeader('Select columns'),
                dbc.ModalBody(
                    children=[
                        dbc.Checklist(
                            id='select-all-columns-checklist',
                            options=[
                                {'label': 'Select all', 'value': 1} 
                            ],
                            value=[1],
                        ),
                        dbc.Checklist(
                            id='select-columns-checklist',
                            options=[
                                {'label': " - ".join([cd['headerName'], cd['headerTooltip']]), 'value': i} for i, cd in enumerate(table_columnDefs)
                            ],
                            value=[i for i, cd in enumerate(table_columnDefs)],
                        ),
                    ],
                ),
            ],
        )

        grid = dag.AgGrid(
            id=self.AG_GRID_ID,
            columnDefs=table_columnDefs,
            #rowData=jobs_table.to_dict(orient='records'),
            rowModelType="infinite",
            defaultColDef={
                #"flex": 1, 
                "sortable": True, 
                "resizable": False, 
                "filter": True,
                'filterParams': {
                    'maxNumConditions': DEFAULT_MAX_NUM_FILTER_CONDITIONS,
                },
            },
            columnSize='responsiveSizeToFit',
            style={
                'height': '100%', 
                'width': '100%',
                'margin': 'auto',
            },
            dashGridOptions={
                "pagination": True, 
                "paginationAutoPageSize": True,
                'tooltipShowDelay': 10,
                'tooltipMouseTrack': True,
                # 'autoSizeStrategy': {
                #     'type': 'fitGridWidth',
                # },
                "rowSelection": "multiple",
                "suppressRowClickSelection": True,
                "suppressCellFocus": True,
                "rowBuffer": 0,
                "maxBlocksInCache": 1,
                "rowSelection": "multiple",
            },
        )

        table = dbc.Container(
            children=[
                grid,
            ],
            className="dbc-ag-grid",
            style={
                'height': 'calc(100% - 36px)', 
                'maxWidth': '100%', 
                'width': '100%', 
                'padding': '0px', # around the AGGRID table
                'margin': '0px',
            },
        )
        
        refresh_trigger = dash_enrich.dcc.Interval(
            id=self.REFRESH_TRIGGER_ID,
            disabled=self.REFRESH_PERIOD is None,
            interval=self.REFRESH_PERIOD,
            n_intervals=1
            )
        
        latest_data_timestamp = dash_enrich.dcc.Store(
            id=self.LATEST_DATA_TIMESTAMP_ID,
            data=datetime.datetime(1,1,1,0,0,tzinfo=datetime.timezone.utc),
        )   

        return dash_enrich.html.Div(
            id=self.id,
            children=[
                tools,
                table, 
                select_columns_modal,
                refresh_trigger,
                latest_data_timestamp,
            ],
            style={'height': '100%'},
        )


    def set_callbacks(self, server):
        super().set_callbacks(server)

        @server.app.callback(
            dash_enrich.Output(self.AG_GRID_ID, "getRowsResponse"),
            dash_enrich.Input(self.AG_GRID_ID, "getRowsRequest"),
            #dash_enrich.Input(self.REFRESH_TRIGGER_ID, 'n_intervals'),
        )
        def infinite_scroll(
            request, 
            #n_intervals
            ):

            if request:
                
                dff, total_count = self.project._get_ag_grid_row_request_query(
                    JobsTable.CLIENT_SIDE_COLUMNS,
                    request,
                )

                lines = max(total_count, 1)
                return {"rowData": dff.to_dict("records"), "rowCount": lines}

        # updata data in table every refresh period
        # @server.app.callback(
        #     dash_enrich.Output(self.AG_GRID_ID, 'rowData'),
        #     dash_enrich.Output(self.LATEST_DATA_TIMESTAMP_ID, 'data'),
        #     dash_enrich.Input(self.REFRESH_TRIGGER_ID, 'n_intervals'),
        #     dash_enrich.State(self.LATEST_DATA_TIMESTAMP_ID, 'data'),
        #     prevent_initial_call=True,
        # )
        # def update_table(n_intervals, client_timestamp):
        #     df = self.get_fresh_table()
        #     server_timestamp = df['last_updated'].max()
        #     client_timestamp = datetime.datetime.fromisoformat(client_timestamp)
        #     print(server_timestamp, client_timestamp)

        #     if server_timestamp > client_timestamp:
        #         jobs_table = df[JobsTable.CLIENT_SIDE_COLUMNS].sort_values(by='priority')
        #         return jobs_table.to_dict('records'), server_timestamp
        #     else:
        #         return dash_enrich.no_update
            

        @server.app.callback(
            dash_enrich.Output('code-modal', 'is_open'),
            dash_enrich.Output('code-modal-title', 'children'),
            dash_enrich.Output('code-modal-content', 'code'),
            dash_enrich.Input(self.AG_GRID_ID, 'cellRendererData'),
            prevent_initial_call=True,
        )
        def show_sim_inputs(cellRendererData):
            if cellRendererData is None:
                return dash_enrich.no_update
            res = self.project._get_input_outputs_query(signature=cellRendererData['value']['signature'])
            sim_p_formatted = format_python_data(
                res['inputs']['sim_p'], 
                f'# inputs for simulation runner'
            )
            mesh_p_formatted = format_python_data(
                res['inputs']['mesh_p'], 
                '# inputs for mesh generator'
            )
            outputs_postprocess_formatted = format_python_data(
                res['output_postprocess'], 
                f'# outputs of the simulation result postprocessing {"(not available yet)" if not res["output_postprocess"] else ""}'
            )
            outputs_build_formatted = format_python_data(
                res['output_build'], 
                f'# outputs of the simulation result build {"(not available yet)" if not res["output_build"] else ""}'
            )
            tabs = [
                {
                    'fileName': "sim_p.py", 
                    'code': sim_p_formatted.strip(), 
                    'language': "python",
                    "icon": DashIconify(icon="devicon:python"),
                },
                {
                    'fileName': "mesh_p.py", 
                    'code': mesh_p_formatted, 
                    'language': "python",
                    "icon": DashIconify(icon="devicon:python"),
                },
                {
                    'fileName': "outputs_build.py", 
                    'code': outputs_build_formatted, 
                    'language': "python",
                    "icon": DashIconify(icon="devicon:python"),
                },
                {
                    'fileName': "outputs_postprocess.py", 
                    'code': outputs_postprocess_formatted, 
                    'language': "python",
                    "icon": DashIconify(icon="devicon:python"),
                },
                {
                    'fileName': "error_log.txt", 
                    'code': res['error_log'], 
                    'language': "text",
                    "icon": DashIconify(icon="devicon:python"),
                },
            ]
            return True, "Inputs for job " + cellRendererData['value']['signature'], tabs

        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='showhide_columns'
            ),
            dash_enrich.Output(self.AG_GRID_ID, "columnDefs"),
            dash_enrich.Input("select-columns-checklist", "value"),
            dash_enrich.State(self.AG_GRID_ID, "columnDefs"),
            prevent_initial_call=True,
        )

        server.app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='show_select_columns_modal'
            ),
            dash_enrich.Input('select-columns-button', 'n_clicks'),
            dash_enrich.Output('select-columns-modal', 'is_open'),
            prevent_initial_call=True,
        )