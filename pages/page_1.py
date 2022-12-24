import dash
import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash import Input, Output, html, dcc
from dash import dcc as dcc
from dash import html as html
#import dash_table
import dash_bootstrap_components as dbc



dash.register_page(__name__,
                   path ='/',
                   name = 'Home',
                   title = 'Upload',
                   description = 'Home Page to upload data.')
                   #image = use this to add an image when app is linked)

# Text field
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



layout = html.Div([



    html.Div([
            html.Center(
                html.H2('Welcome to The Data Science Application! Please upload your data to get started.')
            )
        ], className="row"),


    html.Div([
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        kpi_one()
                    ], width=3),
                    dbc.Col([
                        kpi_two()
                    ], width=3),
                    dbc.Col([
                        kpi_three()
                    ], width=3),
                    dbc.Col([
                        kpi_four()
                    ], width=3),
                ], align='center')
            ])
        ], className='mb-2', style={
                    'backgroundColor': 'rgba(0,0,0,0)',
                    'color': 'white',
                    'text-align': 'center'
                })
            ], className = "row"),

    dcc.Dropdown(
        id='dropdown',
        options=[],
        style={'color': 'black',
               'textAlign': 'center'},
        value=None
        ),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files'),

        ],
        ),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    html.Div([
        html.Div(id='output-data-upload'),
        dcc.Store(id='store')
    ], className="row"),
])

