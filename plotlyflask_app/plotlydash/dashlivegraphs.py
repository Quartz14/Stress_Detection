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
import datetime

X= deque(maxlen = 180)
Ypred = deque(maxlen = 180)
Ypred_emoji = deque(maxlen = 180)

Yx= deque(maxlen = 180)
Yy= deque(maxlen = 180)
Yz= deque(maxlen = 180)
Y_cecg= deque(maxlen = 180)
Y_ceda= deque(maxlen = 180)
Y_cemg= deque(maxlen = 180)
Y_ctemp= deque(maxlen = 180)
Y_cresp= deque(maxlen = 180)

#cEMG cTemp	cResp


Ywx= deque(maxlen = 180)
Ywy= deque(maxlen = 180)
Ywz= deque(maxlen = 180)
Y_wbvp= deque(maxlen = 180)

X.append(0)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('data/S7.csv')
loaded_model = pickle.load(open("models/s7_dt.sav", 'rb'))

available_sensors = list(df.columns)[:-1]
X_data = df[available_sensors]
preds = loaded_model.predict(X_data)
pred_dict = {1:'baseline', 2:'stress', 3:'amusement', 4:'meditation'}
pred_emogi_dict = {1:'😐', 2:'😣', 3:'😃', 4:'😇'}
preds_text = [pred_dict[i] for i in preds]
preds_emoji = [pred_emogi_dict[i] for i in preds]





def init_dashlive(server):
    dash_app = dash.Dash(
    server = server,
    routes_pathname_prefix = '/dashlive/',
    external_stylesheets = external_stylesheets
    )

    dash_app.layout = html.Div([
    html.Div([
        html.H4('Sensor Data Feed'),
        html.Div(id = 'live-update-text'),
        dcc.Graph(id = 'live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval = 1*10000, #millisec
            n_intervals=0
        )
    ],style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
    dcc.Graph(id='sensor_pred', animate=True),
    dcc.Interval(id='graph-update_pred', interval=1000, n_intervals=0)
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),

    ])

    init_callbacks(dash_app)
    return dash_app.server

def init_callbacks(dash_app):

    @dash_app.callback(Output('live-update-graph', 'figure'),
    Input('interval-component', 'n_intervals'))
    def update_graph_live(n):
        print("N: ",n)
        for i in range(4):
            X.append(X[-1]+0.25)

            Y_cecg.append(df['cECG'][4*n+i])
            Y_ceda.append(df['cEDA'][4*n+i])
            Y_cemg.append(df['cEMG'][4*n+i])
            Y_ctemp.append(df['cTemp'][4*n+i])
            Y_cresp.append(df['cResp'][4*n+i])

            Yx.append(df['cACCx'][4*n+i])
            Yy.append(df['cACCy'][4*n+i])
            Yz.append(df['cACCz'][4*n+i])



        fig = plotly.tools.make_subplots(rows=3, cols=2, vertical_spacing=0.2)
        fig['layout']['margin'] = {'l':30, 'r':10, 'b':30, 't':10}
        fig['layout']['legend'] = {'x':1, 'y':0, 'xanchor':'right'}
        fig.append_trace({
            'x': list(X),
            'y':list(Y_cecg),
            'name':'Chest ECG',
            'mode':'lines',
            'type': 'scatter'
        }, 1,1)
        fig.append_trace({
            'x': list(X),
            'y':[-0.081879]*len(X),
            'name':'mode during stress',
            'mode':'lines',
            'type': 'scatter'
        }, 1,1)
        fig.append_trace({
            'x': list(X),
            'y':list(Y_ceda),
            'text': list(Y_ceda),
            'name':'Chest EDA',
            'mode':'lines',
            'type': 'scatter'
        }, 1,2)
        fig.append_trace({
            'x': list(X),
            'y':list(Y_cemg),
            'text': list(Y_cemg),
            'name':'Chest EMG',
            'mode':'lines',
            'type': 'scatter'
        }, 2,1)
        fig.append_trace({
            'x': list(X),
            'y':list(Y_ctemp),
            'text': list(Y_ctemp),
            'name':'Chest Temperature',
            'mode':'lines+markers',
            'type': 'scatter'
        }, 2,2)
        fig.append_trace({
            'x': list(X),
            'y':list(Y_cresp),
            'text': list(Y_cresp),
            'name':'Respiration',
            'mode':'lines+markers',
            'type': 'scatter'
        }, 3,1)
        fig.append_trace({
            'x': list(X),
            'y':list(Yx),
            'text': list(Yx),
            'name':'Accelerometer x',
            'mode':'lines',
            'type': 'scatter',
            'line_color':'rgb(57,105,172)',
            'line_shape':'spline'
        }, 3,2)
        fig.append_trace({
            'x': list(X),
            'y':list(Yy),
            'text': list(Yy),
            'name':'Accelerometer y',
            'mode':'lines',
            'type': 'scatter',
            'line_color':'rgb(242,183,1)',
            'line_shape':'spline'
        }, 3,2)
        fig.append_trace({
            'x': list(X),
            'y':list(Yz),
            'text': list(Yz),
            'name':'Accelerometer z',
            'mode':'lines',
            'type': 'scatter',
            'line_color':'rgb(242,183,1)',
            'line_shape':'spline'
        }, 3,2)

        fig.update_xaxes(title_text='Seconds passed')
        # Update yaxis properties
        fig.update_yaxes(title_text="Chest ECG (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Chest EDA (mu S)", row=1, col=2)
        fig.update_yaxes(title_text="Chest EMG (mV)", showgrid=False, row=2, col=1)
        fig.update_yaxes(title_text="Chest Temperature (C)", row=2, col=2)
        fig.update_yaxes(title_text="Chest Respiration (%)", row=3, col=1)
        fig.update_yaxes(title_text="Chest Accelerometer readings (g)", row=3, col=2)
        return fig



    @dash_app.callback(
    Output('sensor_pred','figure'),
    [Input('graph-update_pred','n_intervals')])
    def update_graph3(n):
        for i in range(4):
            X.append(X[-1]+0.25)

            Ypred.append(preds[4*n+i])
            Ypred_emoji.append(preds_emoji[4*n+i])
        print(Ypred)

        datax = go.Bar(x=list(X), y=list(Ypred), text=list(Ypred_emoji),textposition='outside')
        return {'data':[datax],
        'layout':go.Layout(
        xaxis = dict(range=[min(X), max(X)]),
        yaxis=dict(range=[0, 5]),
        title='Predictions of State',
                   xaxis_title='Seconds passed',
                   yaxis_title='Mental State'
                   )}
