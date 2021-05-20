from dash import Dash
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
import plotly.express as px
from collections import deque
import pandas as pd
import pickle




#cACCx	cACCy	cACCz	cECG	cEMG	cEDA	cTemp	cResp	wACCx	wACCy	wACCz	wBVP	wEDA	wTEMP
colors = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921','#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df_quest = pd.read_csv('data/quest.csv')

score_map = {1:'Not at all', 2 : 'A little bit', 3: 'Somewhat', 4 :'Very much', 5:'Extremely'}


available_feelings = list(df_quest.columns)[:-1]
available_stage = list(df_quest['stage'].unique())

def init_dashboard(server):
    dash_app = dash.Dash(
    server = server,
    routes_pathname_prefix = '/dashapp/',
    #external_stylesheets = external_stylesheets
    )

    dash_app.layout = html.Div([
    html.Div([
    dcc.Dropdown(
        id='crossfilter-feeling',
        options=[{'label':i, 'value':i} for i in available_feelings],
        value='Scared'
    ),
    dcc.Dropdown(
        id='crossfilter-stage',
        options=[{'label':i, 'value':i} for i in available_stage],
        value='TSST'
    ),
    ]),
    html.Div([
    dcc.Graph(id='graph_bar-feel')
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
    ])
    init_callbacks(dash_app)
    return dash_app.server

def init_callbacks(dash_app):
    @dash_app.callback(
    Output('graph_bar-feel','figure'),
    [Input('crossfilter-feeling','value'),
    Input('crossfilter-stage','value')])
    def update_graph0(feeling,stage):
        tt = df_quest[df_quest['stage'] == stage][feeling].value_counts()
        x = list(tt.index)
        x = [score_map[i] for i in x]
        data = go.Bar(x=x ,y=list(tt.values))
        return {'data':[data],
        'layout':go.Layout(
        #xaxis = dict(range=[min(X), max(X)]),
        #yaxis=dict(range=[0, 5]),
        title='Change in feelings according to stress levels',
                   xaxis_title='Score',
                   yaxis_title='Count'
        )}
