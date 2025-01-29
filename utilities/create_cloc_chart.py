import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_csv('assets/cloc/line_count_cache.csv', sep=';')
df['date'] = pd.to_datetime(df['date'])
fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_trace(
    go.Scatter(
        x=df['date'], 
        y=df['cloc_total_lines'], 
        mode='markers+lines', 
        name='All lines',
    )
)
fig.add_trace(
    go.Scatter(
        x=df['date'], 
        y=df['cloc_code_lines'], 
        mode='markers+lines', 
        name='Only code',
    ),
)
fig.add_trace(
    go.Scatter(
        x=df['date'],
        y=df['norm_pages'], 
        mode='markers+lines', 
        name='All text in normalized pages',
    ), 
    secondary_y=True,
)
fig.update_layout(
    title='Lines of code over time', 
    xaxis_title='Date', 
    yaxis_title='Lines of Code', 
    yaxis2=dict(
        title='Normalized Pages', 
        overlaying='y', 
        side='right',
    ),
    legend=dict(
        orientation='v',
        yanchor='bottom',
        y=1.02,
        xanchor='center',
        x=0.5,
    ),
)
fig.write_image('assets/cloc/line_count_cache.svg', scale=2, width=1024, height=512)
