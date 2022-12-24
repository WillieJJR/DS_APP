from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

import dash_bootstrap_components as dbc
#from dash_core_components import set_theme
from dash.dependencies import Input, Output, State
import dash_table
import dash_table
import pandas as pd
import base64
import datetime
import numpy as np
import io

# Iris bar figure
def drawFigure():
    return  html.Div([
        dbc.Card(
            dbc.CardBody([
                dcc.Graph(
                    figure=px.bar(
                        df, x="sepal_width", y="sepal_length", color="species"
                    ).update_layout(
                        template='plotly_dark',
                        plot_bgcolor= 'rgba(0, 0, 0, 0)',
                        paper_bgcolor= 'rgba(0, 0, 0, 0)',
                    ),
                    config={
                        'displayModeBar': False
                    }
                )
            ])
        ),
    ])

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

# Data


# Build App
app = Dash(external_stylesheets=[dbc.themes.SUPERHERO])

app.theme = {
    'plotly': True,
    'primary_color': '#1E77B4',
    'secondary_color': '#F9A908'
}

app.layout = html.Div([

html.Div([
        html.Center(
            html.H2('Welcome to The Data Science Application! Please upload your data to get started.')
        )
    ], className="row"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Dropdown(id='dropdown', options=[])
                ]),
            ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0.4)',
                'textAlign': 'center'
            })
        ])
    ]),

    html.Div([
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
                dbc.Col([
                    drawText()
                ], width=3),
            ], align='center')
        ])
    ], className='mb-2', style={
                'backgroundColor': 'rgba(0,0,0,0)',
                'color': 'white',
                'text-align': 'center'
            })
        ], className = "row"),

html.Div([
        # First column

            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.Button('Drag and Drop or Select Files (CSV/XLSX Format ONLY)', style = {'backgroundColor': 'rgba(0,0,0,0.4)',
                'color': 'white',
                                                                                                 'width': '100%',
                    'height': '40px',
                    'lineHeight': '40px',
                    'borderWidth': '1px',
                    'borderStyle': 'solid',
                    'borderRadius': '20px',
                    'textAlign': 'center'}),
                ]),
                style={'backgroundColor': 'rgba(0,0,0,0)',
                'color': 'white',
                    'width': '100%',
                    'height': '40px',
                    'lineHeight': '40px',
                    'borderWidth': '1px',
                    'borderStyle': 'solid',
                    'borderRadius': '20px',
                    'textAlign': 'center',
                       'border':'none'
                },
                multiple=True
            ),

        ]),
    html.Div([(
        html.Div(id='output-data-upload')
    )], className="row")

])



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            style_header={'backgroundColor': 'rgba(0,0,0,0)',
                          'color':'white',
                          'fontWeight': 'bold',
                          'textAlign': 'center', },
            style_table={'overflowX': 'scroll'},
            style_cell={'minWidth': '180px', 'width': '180px',
                        'maxWidth': '180px', 'whiteSpace': 'normal',
                        'backgroundColor': 'rgba(0,0,0,0)',
                        'color': 'white'},
            style_data_conditional=[
                {
                    # Set the font color for all cells to black
                    'if': {'column_id': 'all'},
                    'color': 'white'
                },
                {
                    # Set the font color for cells in the 'Name' column to white
                    # when the row is highlighted
                    'if': {'column_id': 'Name', 'row_index': 'odd'},
                    'color': 'black'
                }
            ],
            row_selectable="multi",
            editable=False,
            page_size=10,
            sort_mode='multi',
            sort_action='multi',
            filter_action='native'
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter = r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df

@app.callback(
    Output('dropdown', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])

def update_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []




@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))


def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

# Run app and display result inline in the notebook
app.run_server()

