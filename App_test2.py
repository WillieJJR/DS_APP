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

#Buttons for Page 2 - Feature Exploration
button1 = html.Button(
    'Scatter Plot',
    id='button-1',
    style={
        'backgroundColor': 'rgba(0,0,0,0.3)',
        'color': 'white',
        'text-align': 'center',
        'margin-right': '10px',  # add a right margin to create a space between the buttons
        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
        'height': '60px',  # set the height of the buttons
        'width': '150px',  # set the width of the buttons
        'font-size': '16px'  # set the font size of the button labels
    }
)
button2 = html.Button(
    'Correlation Matrix',
    id='button-2',
    style={
        'backgroundColor': 'rgba(0,0,0,0.3)',
        'color': 'white',
        'text-align': 'center',
        'margin-right': '10px',  # add a right margin to create a space between the buttons
        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
        'height': '60px',  # set the height of the buttons
        'width': '150px',  # set the width of the buttons
        'font-size': '16px'  # set the font size of the button labels
    }
)
button3 = html.Button(
    'Pearson Correlation',
    id='button-3',
    style={
        'backgroundColor': 'rgba(0,0,0,0.3)',
        'color': 'white',
        'text-align': 'center',
        'margin-right': '10px',  # add a right margin to create a space between the buttons
        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
        'height': '60px',  # set the height of the buttons
        'width': '150px',  # set the width of the buttons
        'font-size': '16px'  # set the font size of the button labels
    }
)

# Use the html.Div component to create a container for the buttons
button_container = html.Div(
    children=[button1, button2, button3],
    style={'display': 'flex', 'margin': 'auto', 'justify-content': 'center'}  # set the display property to flex to arrange the buttons horizontally
)


app.layout = html.Div([

    dcc.Store(id='intermediate-value'),
    dbc.Tabs([
        dbc.Tab(label='Home', children=[
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
            ], className="row"),

            dcc.Dropdown(
                id='dropdown',
                options=[],
                placeholder = 'Please select a Column',
                style={'color': 'black',
                       'textAlign': 'center',
                       #'backgroundColor': 'rgba(0,0,0,0)',
                       'width': '800px',
                       'display': 'block',
                       'margin': '0 auto',
                         '.css-1wa3eu0-placeholder': {'color': 'red'}
                       },
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
        ]),
        dbc.Tab(label='Feature Exploration', children=[
            html.H1('Understand Your variables! Please select a technique to better explore your data.', style={'text-align': 'center'}),

            html.Div([
                html.Div([
                    html.Center(
                        html.H2('Understand Your variables! Please select a technique to better explore your data.')
                    )
                ], className="row"),

                dbc.Row([button_container]),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown_x',
                            options=[],
                            placeholder = 'Please select an X value',
                            style={'display': 'none',
                                   'color': 'black',
                                    'textAlign': 'center',
                                   # 'backgroundColor': 'rgba(0,0,0,0)',
                                   'width': '400px',
                                   'display': 'block',
                                   'margin': '0 auto',
                                   '.css-1wa3eu0-placeholder': {'color': 'red'}
                                   }  # initially set the dropdown to be invisible
                        )
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown_y',
                            options=[],
                            placeholder = 'Please select an Y value',
                            style={'display': 'none'}  # initially set the dropdown to be invisible
                        )
                    )
                ], style={'margin': '10px'}),
                dbc.Row(html.Div(id='warning-message', style={'color': 'red', 'fontSize': 20, 'text-align': 'center'}))
            ])
        ]),
        dbc.Tab(label='Kmeans Predictions', children=[
            html.H1('Kmeans Market Segmentation', style={'text-align': 'center'}),
            html.H3('''Kmeans Output Table''', style={'text-align': 'left'}),
            html.Br(),
            html.H5('''Please select the amount of clusters you would like to segment your data by:''',
                    style={'text-align': 'left'}),
            dcc.Dropdown(id='n-cluster',
                         options=[
                             {'label': '1', 'value': 1},
                             {'label': '2', 'value': 2},
                             {'label': '3', 'value': 3},
                             {'label': '4', 'value': 4},
                             {'label': '5', 'value': 5},
                             {'label': '6', 'value': 6},
                             {'label': '7', 'value': 7},
                             {'label': '8', 'value': 8},
                             {'label': '9', 'value': 9},
                             {'label': '10', 'value': 10}
                         ],
                         value=3
                         ),
            html.Br(),
            html.Div(id='n-cluster-container', children=[]),
            html.Br(),
            html.H3('''Kmeans Centroids Plot ''', style={'text-align': 'left'}),
            dcc.Graph(id='centroid-container', figure={}),
            html.Br(),

            html.Div(id='kmeans-table'),
            # Hidden div inside the app that stores the intermediate value
            # html.Div(id='intermediate-value', style={'display': 'none'})
        ])
    ], id="tabs",
        active_tab="tab-0",
        className="justify-content-center"
    ),
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
            filter_action='native',
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

###testing


@app.callback(Output('intermediate-value', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              #State('upload-data', 'last_modified'),
              prevent_initial_call=True)

def update_store_output(contents, filename ):
    if contents:
        df = parse_data(contents, filename)
        store = {
            "data": df.to_dict("records"),
            "columns": [{"name": i, "id": i} for i in df.columns],
        }
        #print(store)
        return store
        #return df.to_json(date_format='iso', orient='split')
    else:
        return dash.no_update






###testing



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



@app.callback(
    Output('dropdown_x', 'style'),
    [Input('button-1', 'n_clicks')]
)

def toggle_dropdown_visibility_x_var(n_clicks):
    if n_clicks is None:
        return {'display': 'none',
                'color': 'black',
                'textAlign': 'center'
                }
    if n_clicks % 2 == 0:
        # Make the dropdown invisible when the button is clicked an even number of times
        return {'display': 'none',
                'color': 'black',
                'textAlign': 'center'
                }
    else:
        # Make the dropdown visible when the button is clicked an odd number of times
        return {'display': 'block',
                'color': 'black',
                'textAlign': 'center'}

@app.callback(
    Output('dropdown_y', 'style'),
    [Input('button-1', 'n_clicks')]
)

def toggle_dropdown_visibility_y_var(n_clicks):
    if n_clicks is None:
        return {'display': 'none',
                'color': 'black',
                'textAlign': 'center'
                }
    if n_clicks % 2 == 0:
        # Make the dropdown invisible when the button is clicked an even number of times
        return {'display': 'none',
                'color': 'black',
                'textAlign': 'center'
                }
    else:
        # Make the dropdown visible when the button is clicked an odd number of times
        return {'display': 'block',
                'color': 'black',
                'textAlign': 'center'}


@app.callback(
    Output('dropdown_x', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_x_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []

@app.callback(
    Output('dropdown_y', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_y_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []


@app.callback(
    Output('warning-message', 'children'),
    [Input('dropdown_x', 'value'), Input('dropdown_y', 'value')]
)
def update_warning_message(dropdown_x_value, dropdown_y_value):
    if (dropdown_x_value is not None) or (dropdown_y_value is not None):
        if dropdown_x_value == dropdown_y_value:
            return f'''You have selected {dropdown_x_value} for both X and Y variables. Please change one of the variables chosen!'''
        else:
            return ''
    else:
        return ''


if __name__ == "__main__":
    app.run_server(debug=True)