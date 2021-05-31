import dash
from dash.dependencies import Input, Output, State
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

from definition import Data

import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_table
import pandas as pd
import base64
import datetime as dt
import io
import math
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
sample_rate_chest = [1,4,700]
sample_rate_wrist = [1,4]

obj_Data = Data()

def downsample_values(sensor_data,original_sr,stype='avg',target_sr=1):
    under_sampled = []
    step_size = int(original_sr/target_sr)
    for i in range(0,len(sensor_data),step_size):
        sample = sensor_data[i:i+step_size]
        if(stype=='freq'):
            sample = Counter(sample)
            under_sampled.append(sample.most_common(1)[0][0])
        elif(stype=='avg'):
            under_sampled.append(np.mean(sample))
    return under_sampled



def init_dashprocess(server):
    dash_app = dash.Dash(
    server = server,
    routes_pathname_prefix = '/dashprocess/',
    external_stylesheets = external_stylesheets
    )

    upload_layout =html.Div([dcc.Upload(
        id='upload-data',
        children = html.Div(['Drag and Drop or ',html.A('Select Files')],
        ),style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),html.Div(id='output-data-upload'),
    ])



    dash_app.layout = html.Div([
        html.Div([
            html.P("Respiban File Upload:", className="control_label",),
            upload_layout,
            html.Div(id='slider-output-container'),
            html.Br(),
            dcc.RadioItems(id='resp_sampling_rate',
            options=[{'label': i, 'value': i} for i in sample_rate_chest],
            value='4'),

    ])
    ])

    init_callbacks(dash_app)
    return dash_app.server

def parse_contents(samp_rate, contents, filename, date):
    vcc=3
    chan_bit=2**16

    Cmin = 28000
    Cmax = 38000

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'txt' in filename:
            # Assume that the user uploaded an excel file
            decoded = str(decoded)
            #print(decoded)
            ll = decoded.split('\\r')
            #print("******************************************")
            #print(ll)

            resp_list = [line.split('\\t')[2:-1] for line in ll[3:]]
            #print("******************************************")
            #print(resp_list)
            #for l in resp_list:
            #    print(l, len(l))

            #print(pd.DataFrame(resp_list))

            raw_resp = pd.DataFrame(resp_list)

            raw_resp.columns = ["cECG", "cEDA", "cEMG", "cTemp", "cACCx", "cACCy", "cACCz", "cResp"]
            for col in raw_resp.columns:
                print(col, raw_resp[raw_resp[col].isnull()])
            print(raw_resp.isna().sum())
            #print(raw_resp.to_dict('records'))
            print(raw_resp.head())


            raw_resp['cECG'] = [((float(signal)/chan_bit-0.5)*vcc) for signal in raw_resp['cECG']]
            raw_resp['cEDA'] = [(((float(signal)/chan_bit)*vcc)/0.12) for signal in raw_resp['cEDA']]
            raw_resp['cEMG'] = [((float(signal)/chan_bit-0.5)*vcc) for signal in raw_resp['cEMG']]
            si_temp = []
            for signal in list(raw_resp['cTemp']):
                vout = (float(signal)*vcc)/(chan_bit-1.)
                rntc = ((10**4)*vout)/(vcc-vout)
                si_temp.append(- 273.15 + 1./(1.12764514*(10**(-3)) + 2.34282709*(10**(-4))*math.log(rntc) + 8.77303013*(10**(-8))*(math.log(rntc)**3)))

            raw_resp['cTemp'] = si_temp
            raw_resp['cACCx'] = [(float(signal)-Cmin)/(Cmax-Cmin)*2-1 for signal in raw_resp['cACCx']]
            raw_resp['cACCy'] = [(float(signal)-Cmin)/(Cmax-Cmin)*2-1 for signal in raw_resp['cACCy']]
            raw_resp['cACCz'] = [(float(signal)-Cmin)/(Cmax-Cmin)*2-1 for signal in raw_resp['cACCz']]
            raw_resp['cResp'] = [(float(signal) / chan_bit - 0.5) * 100 for signal in raw_resp['cResp']]

            #def get_chest_df(df_s7):
            df7_chest = pd.DataFrame()
            for col in raw_resp.columns:
                    #if(key == 'ACC'):
                    #    for i,axis in enumerate(['x','y','z']):
                    #        df7_chest['c'+key+axis] = downsample_values(df_s7['signal']['chest'][key][:,i],700,stype='avg',target_sr=4)
                    #else:
                df7_chest[col]=downsample_values(raw_resp[col],700,stype='avg',target_sr=int(samp_rate))


        obj_Data.df = pd.DataFrame(df7_chest)


    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),
        html.Button("Download csv", id="btn"), Download(id="download"),


        dash_table.DataTable(
            data=raw_resp.head().to_dict('records'),
            columns=[{'name': i, 'id': i} for i in raw_resp.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browse
    ])



def downsample_values(sensor_data,original_sr,stype='avg',target_sr=1):
    under_sampled = []
    step_size = int(original_sr/target_sr)
    print("step size: ",step_size)
    for i in range(0,len(sensor_data),step_size):
        sample = sensor_data[i:i+step_size]
        if(stype=='freq'):
            sample = Counter(sample)
            under_sampled.append(sample.most_common(1)[0][0])
        elif(stype=='avg'):
            under_sampled.append(np.mean(sample))
    return under_sampled


def init_callbacks(dash_app):
    @dash_app.callback(
    Output('output-data-upload', 'children'),
    Input('resp_sampling_rate', "value"),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
    def update_output(samp_rate,list_of_contents, list_of_names, list_of_dates):
        print('sampling rate selected: ',samp_rate)
        if list_of_contents is not None:
            children = [
            parse_contents(samp_rate,c,n,d) for c,n,d in zip(list_of_contents, list_of_names,list_of_dates)]
            return children

    @dash_app.callback(Output("download", "data"), [Input("btn", "n_clicks"),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')])
    def generate_csv(n_nlicks,list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            print(obj_Data.df.head())
            return send_data_frame(obj_Data.df.head().to_csv, filename="resp_processed.csv")
