import base64
import datetime as dt
import io
import xml
from xml.etree import ElementTree

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_table
from definition import Data

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydot

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score ,confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report


df = pd.DataFrame()
obj_Data = Data()

target_dict = {1:'baseline', 2:'stress', 3:'amusement', 4:'meditation'}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
models = ['Random Forest', 'DT', 'ADABoost']

def init_dashmodel(server):
    dash_app = dash.Dash(
    server = server,
    routes_pathname_prefix = '/dashmodel/',
    external_stylesheets = external_stylesheets
    )

    upload_layout =html.Div([dcc.Upload(
        id='upload-data',
        children = html.Div(['Drag and Drop or ',html.A('Select Files')],
        ),style={
            'width': '100%',
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
            html.P("File Upload:", className="control_label",),
            upload_layout,
            html.Div(id='slider-output-container'),
            html.Br(),
            daq.Slider(id = 'slider',
                            min=0,
                            max=100,
                            value=70,
                            handleLabel={"showCurrentValue": True,"label": "Train test split"},
                            step=10),


            html.P("Models", className="control_label"),
            dcc.Dropdown(
                id="select_models",
                options = [{'label':x, 'value':x} for x in models],
                value = models,
                multi=True,
                clearable=False,
                className="dcc_control",
                ),


        ]),
                html.Div([
                daq.PowerButton(
                id='my-power-button',
                on=False,
                color='#33FF9E'
                ),
                html.Div(id='power-button-output')
                ])
            ])


    init_callbacks(dash_app)
    return dash_app.server

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
        'Error processing file'
        ])

    #print(df.head())
    obj_Data.df = pd.DataFrame(df)
    corr_matrix = obj_Data.df.corr()
    fig = px.imshow(corr_matrix, title= "Corelation Matrix")
    print(obj_Data.df.head())
    return html.Div([
        html.Div([
        html.P("Select Target", className="control_label"),
        dcc.Dropdown(
        id = "select_target",
        options=[{'label':x, 'value':x} for x in obj_Data.df.columns],
        multi=False,
        value = 'label',
        clearable=False,
        className = "dcc_control"
        ),
        html.P("Select Variable", className="control_label"),
        dcc.Dropdown(
            id="select_independent",
            options=[{'label':x, 'value':x} for x in obj_Data.df.columns],
            value=list(obj_Data.df.columns),
            multi=True,
            className="dcc_control",
            ),
        ]),
        html.Div([dcc.Graph(id="pie_graph", figure =fig)],
        className="pretty_container six columns",),
        dash_table.DataTable(
            data=df.to_dict('records')[:5],
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),
        html.Div(id='selected_values'),


    ])

def create_clfmat(y_test, predictions, target_names):
    cr = classification_report(y_test,predictions, target_names=target_names,output_dict=True)
    cr = pd.DataFrame(cr).transpose()
    cr.reset_index(level=0, inplace=True)
    for col in list(cr.columns)[1:]:
        cr[col] = cr[col].map("{:,.3f}".format)
    return cr

def conf_table(cr):
    return (dash_table.DataTable(
        data=cr.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in cr.columns],
        style_data_conditional=[
        {
        'if': {
        'column_id':str(x),
        'filter_query': '{{{0}}} > 0.9'.format(x)
        },'backgroundColor': 'rgb(0, 147, 146)',
        'color':'white'
        } for x in list(cr.columns)[1:] ] + [
        {'if': {
        'column_id':str(x),
        'filter_query': '{{{0}}}>0.8 && {{{0}}} <= 0.9'.format(x)
        },'backgroundColor': 'rgb(156, 203, 134)'
        } for x in list(cr.columns)[1:]
        ] +  [
        {'if': {
        'column_id':str(x),
        'filter_query': '{{{0}}}>0.7 && {{{0}}} <= 0.8'.format(x)
        },'backgroundColor': 'rgb(238, 180, 121)'
        } for x in list(cr.columns)[1:] ]+ [
        {'if': {
        'column_id':str(x),
        'filter_query': '{{{0}}}>0.6 && {{{0}}} <= 0.7'.format(x)
        },'backgroundColor': 'rgb(232, 132, 113)'
        } for x in list(cr.columns)[1:]] + [
        {'if': {
        'column_id':str(x),
        'filter_query': '{{{0}}} <= 0.6'.format(x)
        },'backgroundColor': 'rgb(207, 89, 126)'
        } for x in list(cr.columns)[1:]],
        style_header={
        'backgroundColor': '#61caff',
        'fontWeight': 'bold'
        }
    ))

#['rgb(0, 147, 146)', 'rgb(57, 177, 133)', 'rgb(156, 203, 134)', 'rgb(233, 226, 156)', 'rgb(238, 180, 121)', 'rgb(232, 132, 113)', 'rgb(207, 89, 126)']
print(px.colors.diverging.Temps)
def svg_to_fig(svg_bytes, title=None, plot_bgcolor="white", x_lock=False, y_lock=True):
    svg_enc = base64.b64encode(svg_bytes)
    svg = f"data:image/svg+xml;base64, {svg_enc.decode()}"

    # Get the width and height
    xml_tree = ElementTree.fromstring(svg_bytes.decode())
    img_width = int(xml_tree.attrib["width"].strip("pt"))
    img_height = int(xml_tree.attrib["height"].strip("pt"))

    fig = go.Figure()
    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width],
            y=[img_height, 0],
            mode="markers",
            marker_opacity=0,
            hoverinfo="none",
        )
    )
    fig.add_layout_image(
        dict(
            source=svg,
            x=0,
            y=0,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
            opacity=1,
            layer="below",
        )
    )

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(showgrid=False, visible=False, range=[0, img_width])
    fig.update_yaxes(showgrid=False, visible=False, range=[img_height, 0])

    if x_lock is True:
        fig.update_xaxes(constrain="domain")
    if y_lock is True:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(plot_bgcolor=plot_bgcolor, margin=dict(r=5, l=5, b=5))

    if title:
        fig.update_layout(title=title)

    return fig

def init_callbacks(dash_app):
    @dash_app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
            parse_contents(c,n,d) for c,n,d in zip(list_of_contents, list_of_names,list_of_dates)]
            print(obj_Data.df.tail())
            return children

    @dash_app.callback(Output('selected_values','children'),
    [Input("select_target","value"),
    Input("select_independent", "value"),
    Input("slider", "value"),
    Input("select_models", "value"),],prevent_initial_call=True)
    def model_features(target,independent, slider, selected_models):
        obj_Data.train_test_split = float(slider)/100
        obj_Data.target = str(target)
        obj_Data.features = list(independent)
        obj_Data.models = list(selected_models)
        print(f'You have selected target = {obj_Data.features}, features = {obj_Data.target}, split = {obj_Data.train_test_split} and models = {obj_Data.models}')
        return f'You have selected target = {target}, features = {independent}, split = {slider} and models = {selected_models}'

    @dash_app.callback(Output('power-button-output', 'children'),
    [Input('my-power-button', 'on'),prevent_initial_call=True])
    def disp_model_res(button_on):
        print('inside button')
        print(f'You have selected target = {obj_Data.features}, features = {obj_Data.target}, split = {obj_Data.train_test_split} and models = {obj_Data.models}')
        print(obj_Data.df)
        print(obj_Data.features)
        X = obj_Data.df[obj_Data.features]
        y = obj_Data.df[obj_Data.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(obj_Data.train_test_split), random_state=42)
        res = []
        target_names = list(obj_Data.df[obj_Data.target].unique())
        target_names = [target_dict[i] for i in target_names]
        for m in obj_Data.models:
            if(m == 'DT'):
                clf_dt = tree.DecisionTreeClassifier()
                clf_dt.fit(X_train,y_train)
                predictions = clf_dt.predict(X_test)
                cr = create_clfmat(y_test, predictions, target_names)


                dot_data = export_graphviz(
                            clf_dt,
                            out_file=None,
                            filled=True,
                            rounded=True,
                            feature_names=obj_Data.features,
                            class_names=target_names,
                            proportion=True,
                            rotate=True,
                            precision=2,)

                pydot_graph = pydot.graph_from_dot_data(dot_data)[0]
                #output_graphviz_svg = pydot_graph.create_svg()
                svg_bytes = pydot_graph.create_svg()
                fig = svg_to_fig(svg_bytes, title="Decision Tree Explanation")
                dash_table_st = conf_table(cr)

                res.append(html.Div([html.H3("Decision Tree Classifier", className="control_label"),
                html.Div(dash_table_st),
                html.Div(
                        dcc.Graph(id="aggregate_graph3", figure = fig),
                                    className="pretty_container six columns",
                                )
                ]))

            elif(m == 'Random Forest'):
                clf_rf = RandomForestClassifier()
                clf_rf.fit(X_train,y_train)
                predictions = clf_rf.predict(X_test)
                cr = create_clfmat(y_test, predictions, target_names)

                feature_imp = clf_rf.feature_importances_.argsort()
                print(feature_imp)
                print(X_train.columns[feature_imp])
                print(clf_rf.feature_importances_[feature_imp])
                temp=pd.DataFrame()
                temp['features'] = X_train.columns[feature_imp]
                temp['importance'] = clf_rf.feature_importances_[feature_imp]
                fig_featureImp = px.bar(temp,y='features', x='importance',
                            title= 'Random Forest- Variable Importance')

                dash_table_st = conf_table(cr)
                res.append(html.Div([html.H3("Random Forest Classifier", className="control_label"),
                html.Div(dash_table_st),
                html.Div(
                        dcc.Graph(id="aggregate_graph1", figure = fig_featureImp),
                                    className="pretty_container six columns",
                                )
                ]))

            elif(m == 'ADABoost'):
                clf_ad = AdaBoostClassifier()
                clf_ad.fit(X_train,y_train)
                predictions = clf_ad.predict(X_test)
                cr = create_clfmat(y_test, predictions, target_names)

                feature_imp = clf_ad.feature_importances_.argsort()
                print(feature_imp)
                print(X_train.columns[feature_imp])
                print(clf_ad.feature_importances_[feature_imp])
                temp=pd.DataFrame()
                temp['features'] = X_train.columns[feature_imp]
                temp['importance'] = clf_ad.feature_importances_[feature_imp]
                fig_featureImp = px.bar(temp,y='features', x='importance',
                            title= 'AdaBoost- Variable Importance')
                dash_table_st = conf_table(cr)
                res.append(html.Div([html.H3("AdaBoost Classifier", className="control_label"),
                html.Div(dash_table_st),
                html.Div(
                        dcc.Graph(id="aggregate_graph2", figure = fig_featureImp),
                                    className="pretty_container six columns",
                                )
                ]))
        return res
