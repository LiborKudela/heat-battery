import plotly.express as px
from heat_battery.visualization import Dasboard, pages, components
from heat_battery.visualization.io import ProjectResultTableReloader
from heat_battery.simulations.postgresql_project import Project
import pandas as pd
from dash_bootstrap_templates import load_figure_template
import datetime

fig_theme = 'vapor'
load_figure_template([fig_theme])

project = Project('project_0')
df_jobs = project.get_jobs(as_dataframe=True)
df_jobs = df_jobs[df_jobs['status']!='SCHEDULED']
signatures = df_jobs['signature'].tolist()

dashboard = Dasboard('Testing server')

def add_view(server: Dasboard, signature: str):
    print(f'Adding view for {signature}')

    def get_figure_creator(unit_title, y_names, y_range=[None, None]):
        layout_args = dict(
            uirevision="None",
            legend=dict(

                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                ),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Time [h]",
            yaxis_title=unit_title,
            yaxis_range=y_range,
        )
        def create_figure(df):
            fig = px.line(
                df,
                template=fig_theme,
                x='t_sim_days', 
                y=y_names,
            ).update_layout(**layout_args)
            return fig
        return create_figure

    fig_Q = get_figure_creator("Heat Flow [W]", ['Q_amb', 'Q_bivalent', 'Q_c', 'Q_pv', 'Q_m', 'Q_s', 'Q_s2r_total'])
    fig_T = get_figure_creator("Temperature [°C]", ['T[0]', 'T_amb', 'T_avg_m', 'T_avg_room', 'T_avg_s', 'T_avg_sand'])
    fig_H = get_figure_creator("Total Energy [J]", ['H_bivalent', 'H_c', 'H_pv', 'H_demand', 'H_s2r_loss', 'H_s2r_used', 'H_storage'])
    fig_Eff = get_figure_creator("Fraction/Efficiency [-]", ['Eff_storage', 'Eff_used', 'Eff_used_pv', 'Eff_pv_used','power_toggle'], y_range=[-0.01, 1.01])
    overview_figs = [fig_Q, fig_T, fig_H, fig_Eff]
    overview_items = [components.FigureResamplerItem(fig_i, data_name=signature, x_name='t_sim_days', rolling_span=24*7) for fig_i in overview_figs]

    server.register_data_updater(signature, ProjectResultTableReloader(project, signature))
    server.register_page(pages.SingleItemPage(signature, components.GridLayout(overview_items)))

for signature in signatures:
    add_view(dashboard, signature)

dashboard.build_app()
server = dashboard.app.server

if __name__ == '__main__':
    # editor = components.DataViewer(pd.read_csv(path_dict['size_5_v3wp30m02r05']))
    # server.register_page(pages.SingleItemPage('test', editor))
    # server.update_data()
    # server.debug_mode()
    dashboard.start_app(host='147.229.141.75', port=8050)
    dashboard.auto_update(5)