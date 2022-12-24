import dash
from dash import Dash, dcc, html, dash_table
from dash import dcc as dcc
from dash import html as html
#import dash_table
import dash_bootstrap_components as dbc
#from dash_core_components import set_theme
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import datetime
import numpy as np
import io

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])


def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Text"),
                ], style={'textAlign': 'center'})
            ])
        ),
    ])

def kpi_one():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H4("Percentage of Null Values:"),
                    html.Div(id = 'output_missing_vals'),
                ], style={'textAlign': 'center'})
            ])
        ),
    ])

def kpi_two():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H4("Count of Distinct Values:"),
                    html.Div(id = 'output_distinct_vals'),
                ], style={'textAlign': 'center'})
            ])
        ),
    ])

def kpi_three():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H4("Datatype:"),
                    html.Div(id = 'output_column_type'),
                ], style={'textAlign': 'center'})
            ])
        ),
    ])


def kpi_four():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H4("Mode/Median Value:"),
                    html.Div(id = 'output_stats_props'),
                ], style={'textAlign': 'center'})
            ])
        ),
    ])


app.layout = html.Div([
    dbc.Tabs(
        [
            dbc.Tab(label="Tab 1", tab_id="tab-1", active_label_style={
                "backgroundColor": "white",
                "color": "black",
            }),
            dbc.Tab(label="Tab 2", tab_id="tab-2", active_label_style={
                "backgroundColor": "white",
                "color": "black",
            }),
            dbc.Tab(label="Tab 3", tab_id="tab-3", active_label_style={
                "backgroundColor": "white",
                "color": "black",
            }),
        ],
        id="tabs",
        active_tab="tab-1",
        className="justify-content-center",
    ),
    html.Div(id="tab-content"),
])

@app.callback(Output('tabs', 'active_tab'),
              [Input('tab-1', 'tab_id'),
               Input('tab-2', 'tab_id'),
               Input('tab-3', 'tab_id')])
def update_active_tab(tab_1, tab_2, tab_3):
    if tab_1:
        return 'tab-1'
    elif tab_2:
        return 'tab-2'
    elif tab_3:
        return 'tab-3'

@app.callback(Output('tab-content', 'children'),
              [Input('tabs', 'active_tab')])
def render_content(active_tab):
    if active_tab == 'tab-1':
        return html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select a File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                }
            )
        ])
    elif active_tab == 'tab-2':
        return html.Div([
            html.H3('Tab 2 content')
        ])
    elif active_tab == 'tab-3':
        return html.Div([
            html.H3('Tab 3 content')
        ])

if __name__ == "__main__":
    app.run_server(debug=True)