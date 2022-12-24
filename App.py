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
#import dash_bootstrap_components as dbc


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
        value=None,
        ),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
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
        },
    ),
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
            sort_action='native',
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

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))


def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [parse_contents(list_of_contents, list_of_names, list_of_dates)]
        return children




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


#callback for the dash table for uploaded data
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

#callback for the null value kpi
@app.callback(
    Output('output_missing_vals', 'children'),
    [Input('dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
     ]
)

def kpi_one(value, contents, filename):
    if contents:
        new_line = '\n'
        df = parse_data(contents, filename)
        if value is not None:
            null_vals = df[value].isnull().sum() / len(df[value])
            null_vals = round(null_vals * 100, 2)
            return f'''{null_vals} %'''
        else:
            null_vals = 'Please select a column to see the percentage of Null Values.'
            return f'''{null_vals}'''
    else:
        no_contents = f'''Please upload data to generate column selections.'''
        return f'''{no_contents}'''

#callback for the distinct value kpi
@app.callback(
    Output('output_distinct_vals', 'children'),
    [Input('dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
     ]
)

def kpi_two(value, contents, filename):
    if contents:
        new_line = '\n'
        df = parse_data(contents, filename)
        if value is not None:
            distinct_vals = df[value].nunique()
            return f'''{distinct_vals} '''
        else:
            distinct_vals = 'Please select a column to see the Number of unique Values.'
            return f'''{distinct_vals}'''
    else:
        no_contents = f'''Please upload data to generate column selections.'''
        return f'''{no_contents}'''


#callback for the column type kpi
@app.callback(
    Output('output_column_type', 'children'),
    [Input('dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
     ]
)

def kpi_three(value, contents, filename):
    if contents:
        new_line = '\n'
        df = parse_data(contents, filename)
        if value is not None:
            datatype_vals = df[value].dtypes
            return f'''{datatype_vals} '''
        else:
            datatype_vals = 'Please select a column to see the datatype of the selected column.'
            return f'''{datatype_vals}'''
    else:
        no_contents = f'''Please upload data to generate column selections.'''
        return f'''{no_contents}'''

#callback for the mode/median kpi
@app.callback(
    Output('output_stats_props', 'children'),
    [Input('dropdown', 'value'),
     Input('upload-data', 'contents'),
     Input('upload-data', 'filename')
     ]
)

def kpi_four(value, contents, filename):
    if contents:
        new_line = '\n'
        df = parse_data(contents, filename)
        if value is not None:
            numbs = ['int', 'float', 'int64', 'float64']
            cat = ['object']
            if df[value].dtypes in (numbs):
                data_describe = df[value].median()
                return f'''Median:    {data_describe} '''
            elif df[value].dtypes == 'object':
                data_describe = df[value].mode()[0]
                data_describe_mode = round((df[value].value_counts().loc[data_describe]/len(df[value]))*100,2)
                return f'''Mode:    {data_describe} ({data_describe_mode} % of column values) '''
            else:
                return f'''Neither a Mode or Median can be applied to this datatype.'''
        else:
            data_describe = 'Please select a column to see the Mode/Median Value of the selected column.'
            return f'''{data_describe}'''
    else:
        no_contents = f'''Please upload data to generate column selections.'''
        return f'''{no_contents}'''



if __name__ == '__main__':
    app.run_server(debug=True)