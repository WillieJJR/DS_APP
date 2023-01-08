import dash
from dash import Dash, dcc, html, dash_table
from dash import dcc as dcc
from dash import html as html
#import dash_table
import dash_bootstrap_components as dbc
#from dash_core_components import set_theme
from dash.dependencies import Input, Output, State
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import plotly.graph_objects as go
import plotly.express as px
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
    'Distribution Plot',
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
    'Feature Importance',
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

button_reset = html.Button(
    'Reset Dropdown Options',
    id='button-reset',
    n_clicks=0,
    n_clicks_timestamp=0,
    style={
        'backgroundColor': 'rgba(0,0,0,0.3)',
        'color': 'red',
        'text-align': 'center',
        'margin-right': '10px',  # add a right margin to create a space between the buttons
        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
        'height': '60px',  # set the height of the buttons
        'width': '150px',  # set the width of the buttons
        'font-size': '16px'  # set the font size of the button labels
    }
)

button_regression = html.Button(
    'Regression',
    id='button-regression',
    n_clicks=0,
    n_clicks_timestamp=0,
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

button_classification = html.Button(
    'Classification',
    id='button-classification',
    n_clicks=0,
    n_clicks_timestamp=0,
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

button_placeholder = html.Button(
    'Placeholder',
    id='button-placeholder',
    n_clicks=0,
    n_clicks_timestamp=0,
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

button_linear = html.Button(
    'Linear',
    id='button_linear',
    n_clicks=0,
    n_clicks_timestamp=0,
    style={
        'backgroundColor': 'rgba(0,0,0,0.3)',
        'color': 'white',
        'text-align': 'center',
        'margin-right': '10px',  # add a right margin to create a space between the buttons
        'border-radius': '0%',  # set the border radius to 50% to make the buttons round
        'height': '30px',  # set the height of the buttons
        'width': '120px',  # set the width of the buttons
        'font-size': '16px'  # set the font size of the button labels
    }
)

button_nonlinear = html.Button(
    'Non Linear',
    id='button_nonlinear',
    n_clicks=0,
    n_clicks_timestamp=0,
    style={
        'backgroundColor': 'rgba(0,0,0,0.3)',
        'color': 'white',
        'text-align': 'center',
        'margin-right': '10px',  # add a right margin to create a space between the buttons
        'border-radius': '0%',  # set the border radius to 50% to make the buttons round
        'height': '30px',  # set the height of the buttons
        'width': '120px',  # set the width of the buttons
        'font-size': '16px'  # set the font size of the button labels
    }
)


# Use the html.Div component to create a container for the Feature Exploration buttons
button_container = html.Div(
    children=[button1, button2, button3],
    style={'display': 'flex', 'margin': 'auto', 'justify-content': 'center'}  # set the display property to flex to arrange the buttons horizontally
)

# Use the html.Div component to create a container for the Kmeans buttons
button_predictive_analytics_container = html.Div(
    children=[button_regression, button_classification, button_placeholder],
    style={'display': 'flex', 'margin': 'auto', 'justify-content': 'center'}
)




button_callout_1 = html.Button(children='Hover here', id='button_callout_1', n_clicks=0, style={
    'background-color': '#4CAF50',
    'border': 'none',
    'color': 'white',
    'padding': '15px 32px',
    'text-align': 'center',
    'text-decoration': 'none',
    'display': 'inline-block',
    'font-size': '16px',
    'margin': '4px 2px',
    'cursor': 'pointer',
})
callout_box = html.Div(children='This is a callout box.', id='callout', style = {
    'border': '2px solid black',
    'border-radius': '4px',
    'padding': '10px',
    'background-color': '#F3F3F3',
    'font-size': '16px',
    'line-height': '24px',
    'display': 'none',  # initially hidden
})


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
            html.Div(children='Note: Files containing over 10,000 records will be truncated to 10,000 records via Random Sampling for all features in the app.', style = {
                'border': '2px solid red',
                'border-radius': '4px',
                'padding': '10px',
                'backgroundColor': 'rgba(0,0,0,0)',
                'font-size': '16px',
                'line-height': '24px',
                'textAlign': 'center'})
        ]),
        dbc.Tab(label='Feature Exploration', children=[
            #html.H1('Understand Your variables! Please select a technique to better explore your data.', style={'text-align': 'center'}),

            html.Div([
                html.Div([
                    html.Center(
                        html.H2('Understand Your variables! Please select a technique to better explore your data.')
                    )
                ], className="row"),

                dbc.Row([button_container]),
                dbc.Row([dcc.Dropdown(
                            id='dropdown_featimp',
                            options=[],
                            placeholder = 'Please select target variable for Feature Importance',
                            style={'display': 'none'}  # initially set the dropdown to be invisible
                    )], style={'margin': '10px'}),
                dbc.Row([dcc.Dropdown(
                            id='dropdown_violin',
                            options=[],
                            placeholder = 'Please select values for Histogram and Distribution Plot',
                            style={'display': 'none'}  # initially set the dropdown to be invisible
                    )], style={'margin': '10px'}),
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
                dbc.Row(html.Div(id='warning-message', style={'color': 'red', 'fontSize': 20, 'text-align': 'center'})),
                dbc.Row(html.Div(id='scatterplot-div', children=[
                    html.Div(id='scatter-plot')
                ])),
                dbc.Row(html.Div(id='dist-plot-div', children=[
                    html.Div(id='violin-plot'),
                    html.Div(id = 'hist-plot')
                ])),
                dbc.Row(html.Div(id='featimp-div', children=[
                    html.Div(id='featimp-plot')
                ]))
            ])
        ]),
        dbc.Tab(label='Predictive Analytics', children=[
            html.Div([
                html.Div([
                    html.Center(
                        html.H2('Transform Your Data into Insights and Predictions!')
                    )
                ], className="row"),

                dbc.Row([button_predictive_analytics_container]),
                html.Div(id="additional-buttons"),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown_x_regression',
                            options=[],
                            placeholder = 'Please select input features ',
                            multi = True,
                            style={'display': 'none',
                                   'color': 'black',
                                    'textAlign': 'center',
                                   # 'backgroundColor': 'rgba(0,0,0,0)',
                                   'width': '400px',
                                   'margin': '0 auto',
                                   '.css-1wa3eu0-placeholder': {'color': 'red'}
                                   }  # initially set the dropdown to be invisible
                        )
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown_y_regression',
                            options=[],
                            placeholder = 'Please select a Target variable ',
                            style={'display': 'none'}  # initially set the dropdown to be invisible
                        )
                        )
                ], style={'margin': '10px'}),

                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown_x_class',
                            options=[],
                            placeholder = 'Please select input features ',
                            multi = True,
                            style={'display': 'none',
                                   'color': 'black',
                                    'textAlign': 'center',
                                   # 'backgroundColor': 'rgba(0,0,0,0)',
                                   'width': '400px',
                                   'margin': '0 auto',
                                   '.css-1wa3eu0-placeholder': {'color': 'red'}
                                   }  # initially set the dropdown to be invisible
                        )
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id='dropdown_y_class',
                            options=[],
                            placeholder = 'Please select a Target variable ',
                            style={'display': 'none'}  # initially set the dropdown to be invisible
                        )
                        )
                ], style={'margin': '10px'}),

                dbc.Row(html.Div(id='warning-message-regression', style={'color': 'red', 'fontSize': 20, 'text-align': 'center'})),
                dbc.Row(html.Div(id='regression-div', children=[
                    html.Div(id='regression-plot'),
                    html.Div(id="regression-results", style={"display": "none"}),
                    html.Div(id = "regression-tbl", style={"display": "none"})
                ])),
                dbc.Row(html.Div(id='classification-div', children=[
                    html.Div(id='classification-plot'),
                    html.Div(id="classification-results", style={"display": "none"}),
                    html.Div(id = "knn-output", style={"display": "none"})
                ]))

                ])


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


def standardize_inputs(df, target_column):
    # Make a copy of the DataFrame
    df_scaled = df.copy()

    # Standardize the numeric columns
    scaler = StandardScaler()
    numeric_columns = df_scaled.select_dtypes(include=["int", "float"]).columns
    numeric_columns = numeric_columns.drop(target_column)
    df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])

    return df_scaled


def standardize_inputs_class(df, target_column):
    # Make a copy of the DataFrame
    df_scaled = df.copy()

    # Standardize the numeric columns
    scaler = StandardScaler()
    numeric_columns = df_scaled.select_dtypes(include=["int", "float"]).columns
    df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])

    return df_scaled



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



@app.callback(Output('intermediate-value', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              #State('upload-data', 'last_modified'),
              prevent_initial_call=True)

def update_store_output(contents, filename ):
    if contents:
        df = parse_data(contents, filename)
        if len(df) > 10000:
            df = df.sample(n=10000)
        else:
            pass
        store = {
            "data": df.to_dict("records"),
            "columns": [{"name": i, "id": i} for i in df.columns],
        }
        #print(store)
        #return store
        return df.to_json(date_format='iso', orient='split')
    else:
        return dash.no_update




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


#######Button resets for Feature Exploration#########

@app.callback(
    dash.dependencies.Output('button-1', 'n_clicks'),
    dash.dependencies.Output('button-2', 'n_clicks'),
    dash.dependencies.Output('button-3', 'n_clicks'),
    [dash.dependencies.Input('button-1', 'n_clicks'),
     dash.dependencies.Input('button-2', 'n_clicks'),
     dash.dependencies.Input('button-3', 'n_clicks')]
)
def reset_button_clicks(n_clicks_1, n_clicks_2, n_clicks_3):
    if ((n_clicks_1 is not None and n_clicks_1 % 2 == 1 and
         n_clicks_2 is not None and n_clicks_2 % 2 == 1) or
        (n_clicks_1 is not None and n_clicks_1 % 2 == 1 and
         n_clicks_3 is not None and n_clicks_3 % 2 == 1) or
        (n_clicks_2 is not None and n_clicks_2 % 2 == 1 and
         n_clicks_3 is not None and n_clicks_3 % 2 == 1)):
        # Reset the n_clicks of all three buttons to 0
        return 0, 0, 0
    else:
        # Return the current n_clicks of all three buttons
        return n_clicks_1, n_clicks_2, n_clicks_3


#######Button resets for Predictive Analytics#########

@app.callback(
    dash.dependencies.Output('button-regression', 'n_clicks'),
    dash.dependencies.Output('button-classification', 'n_clicks'),
    dash.dependencies.Output('button-placeholder', 'n_clicks'),
    [dash.dependencies.Input('button-regression', 'n_clicks'),
     dash.dependencies.Input('button-classification', 'n_clicks'),
     dash.dependencies.Input('button-placeholder', 'n_clicks')]
)
def reset_button_clicks_kmeans(n_clicks_regression, n_clicks_classification, n_clicks_placeholder):
    if ((n_clicks_regression is not None and n_clicks_regression % 2 == 1 and
         n_clicks_classification is not None and n_clicks_classification % 2 == 1) or
        (n_clicks_regression is not None and n_clicks_regression % 2 == 1 and
         n_clicks_placeholder is not None and n_clicks_placeholder % 2 == 1) or
        (n_clicks_classification is not None and n_clicks_classification % 2 == 1 and
         n_clicks_placeholder is not None and n_clicks_placeholder % 2 == 1)):
        # Reset the n_clicks of all three buttons to 0
        return 0, 0, 0
    else:
        # Return the current n_clicks of all three buttons
        return n_clicks_regression, n_clicks_classification, n_clicks_placeholder


#######Button resets for Predictive Analytics - Linear Regression#########

@app.callback(
    [Output('linearregression-button', 'n_clicks'), Output('randomforest-button', 'n_clicks')],
    [Input('linearregression-button', 'n_clicks'), Input('randomforest-button', 'n_clicks')]
)
def reset_button_clicks_regression(n_clicks_linear, n_clicks_rf):
    if (n_clicks_linear is not None and n_clicks_linear % 2 == 1 and
        n_clicks_rf is not None and n_clicks_rf % 2 == 1):
        # Reset the n_clicks of both buttons to 0
        return 0, 0
    else:
        # Return the current n_clicks of both buttons
        return n_clicks_linear, n_clicks_rf

@app.callback(
    [Output('svm-button', 'n_clicks'), Output('rfclass-button', 'n_clicks')],
    [Input('svm-button', 'n_clicks'), Input('rfclass-button', 'n_clicks')]
)
def reset_button_clicks_classification(n_clicks_svm, n_clicks_rfclass):
    if (n_clicks_svm is not None and n_clicks_svm % 2 == 1 and
        n_clicks_rfclass is not None and n_clicks_rfclass % 2 == 1):
        # Reset the n_clicks of both buttons to 0
        return 0, 0
    else:
        # Return the current n_clicks of both buttons
        return n_clicks_svm, n_clicks_rfclass




#########Dynamic button actions for Feature Exploration tab (Scatterplot - Button 1, Distribution Plot - Button 2, Feature Importance plot - Button 3)##########
@app.callback(
    dash.dependencies.Output('button-1', 'style'),
    [dash.dependencies.Input('button-1', 'n_clicks'),
     dash.dependencies.Input('button-2', 'n_clicks'),
     dash.dependencies.Input('button-3', 'n_clicks')]
)
def update_button_styles(n_clicks_1, n_clicks_2, n_clicks_3):
    if n_clicks_1 is not None and n_clicks_1 % 2 == 1:
        # Change the style of button-1 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_2 is not None and n_clicks_2 % 2 == 1:
        # Change the style of button-1 to inactive when button-2 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_3 is not None and n_clicks_3 % 2 == 1:
        # Change the style of button-1 to inactive when button-3 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-1 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }


@app.callback(
    dash.dependencies.Output('button-2', 'style'),
    [dash.dependencies.Input('button-2', 'n_clicks'),
     dash.dependencies.Input('button-1', 'n_clicks'),
     dash.dependencies.Input('button-3', 'n_clicks')]
)
def update_correlationplot_style(n_clicks_2, n_clicks_1, n_clicks_3):
    if n_clicks_1 is not None and n_clicks_1 % 2 == 1:
        # Reset the style of button-2 to inactive when button-1 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_2 is not None and n_clicks_2 % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_3 is not None and n_clicks_3 % 2 == 1:
        # Reset the style of button-2 to inactive when button-3 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }


@app.callback(
    dash.dependencies.Output('button-3', 'style'),
    [dash.dependencies.Input('button-2', 'n_clicks'),
     dash.dependencies.Input('button-1', 'n_clicks'),
     dash.dependencies.Input('button-3', 'n_clicks')]
)
def update_featureimp_plot_style(n_clicks_2, n_clicks_1, n_clicks_3):
    if n_clicks_1 is not None and n_clicks_1 % 2 == 1:
        # Reset the style of button-2 to inactive when button-1 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_2 is not None and n_clicks_2 % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_3 is not None and n_clicks_3 % 2 == 1:
        # Reset the style of button-2 to inactive when button-3 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }



#########Dynamic button actions for Predictive Analytics tab (Regression - Button 1, Classification - Button 2, Placeholder - Button 3)##########

@app.callback(
    dash.dependencies.Output('button-regression', 'style'),
    [dash.dependencies.Input('button-regression', 'n_clicks'),
     dash.dependencies.Input('button-classification', 'n_clicks'),
     dash.dependencies.Input('button-placeholder', 'n_clicks')]
)
def update_button_styles_regression(n_clicks_regression, n_clicks_classification, n_clicks_placeholder):
    if n_clicks_regression is not None and n_clicks_regression % 2 == 1:
        # Change the style of button-1 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_classification is not None and n_clicks_classification % 2 == 1:
        # Change the style of button-1 to inactive when button-2 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_placeholder is not None and n_clicks_placeholder % 2 == 1:
        # Change the style of button-1 to inactive when button-3 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-1 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }


@app.callback(
    dash.dependencies.Output('button-classification', 'style'),
    [dash.dependencies.Input('button-classification', 'n_clicks'),
     dash.dependencies.Input('button-regression', 'n_clicks'),
     dash.dependencies.Input('button-placeholder', 'n_clicks')]
)
def update_classification_style(n_clicks_classification, n_clicks_regression, n_clicks_placeholder):
    if n_clicks_regression is not None and n_clicks_regression % 2 == 1:
        # Reset the style of button-2 to inactive when button-1 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_classification is not None and n_clicks_classification % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_placeholder is not None and n_clicks_placeholder % 2 == 1:
        # Reset the style of button-2 to inactive when button-3 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }


@app.callback(
    dash.dependencies.Output('button-placeholder', 'style'),
    [dash.dependencies.Input('button-classification', 'n_clicks'),
     dash.dependencies.Input('button-regression', 'n_clicks'),
     dash.dependencies.Input('button-placeholder', 'n_clicks')]
)
def update_placeholder_plot_style(n_clicks_classification, n_clicks_regression, n_clicks_placeholder):
    if n_clicks_regression is not None and n_clicks_regression % 2 == 1:
        # Reset the style of button-2 to inactive when button-1 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_classification is not None and n_clicks_classification % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_placeholder is not None and n_clicks_placeholder % 2 == 1:
        # Reset the style of button-2 to inactive when button-3 is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '60px',  # set the height of the buttons
            'width': '150px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }



#########Dynamic button actions for model selection##########


@app.callback(
    Output('linearregression-button', 'style'),
    [Input('linearregression-button', 'n_clicks'), Input('randomforest-button', 'n_clicks')]
)
def update_button_style_lr(n_clicks_lr, n_clicks_rf):
    if n_clicks_rf is not None and n_clicks_rf % 2 == 1:
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_lr is not None and n_clicks_lr % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }



@app.callback(
    Output('randomforest-button', 'style'),
    [Input('linearregression-button', 'n_clicks'), Input('randomforest-button', 'n_clicks')]
)
def update_button_style_svm(n_clicks_lr, n_clicks_rf):
    if n_clicks_lr is not None and n_clicks_lr % 2 == 1:
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_rf is not None and n_clicks_rf % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }


@app.callback(
    Output('svm-button', 'style'),
    [Input('svm-button', 'n_clicks'), Input('rfclass-button', 'n_clicks')]
)
def update_button_style_svm(n_clicks_svm, n_clicks_rfclass):
    if n_clicks_rfclass is not None and n_clicks_rfclass % 2 == 1:
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_svm is not None and n_clicks_svm % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }


@app.callback(
    Output('rfclass-button', 'style'),
    [Input('svm-button', 'n_clicks'), Input('rfclass-button', 'n_clicks')]
)
def update_button_style_rfclass(n_clicks_svm, n_clicks_rfclass):
    if n_clicks_svm is not None and n_clicks_svm % 2 == 1:
        return {
            'backgroundColor': 'rgba(211, 211, 211, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    elif n_clicks_rfclass is not None and n_clicks_rfclass % 2 == 1:
        # Change the style of button-2 to active when it is clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0, 255, 0, 0.5)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '0%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }
    else:
        # Reset the style of button-2 to the default when none of the other buttons are clicked an odd number of times
        return {
            'backgroundColor': 'rgba(0,0,0,0.3)',
            'color': 'white',
            'text-align': 'center',
            #'margin-right': '10px',  # add a right margin to create a space between the buttons
            'border-radius': '7%',  # set the border radius to 50% to make the buttons round
            'height': '30px',  # set the height of the buttons
            'width': '200px',  # set the width of the buttons
            'font-size': '16px'  # set the font size of the button labels
        }


##########Dynamic Dropdowns for Feature Exploration Tab############

#### Scatterplot dropdowns
@app.callback(
    [
        dash.dependencies.Output('dropdown_x', 'style'),
        dash.dependencies.Output('dropdown_x', 'value'),
    ],
    [
        dash.dependencies.Input('button-1', 'n_clicks'),
        dash.dependencies.Input('button-2', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_x_var(button_1_click, button_2_click):

    if (button_2_click is None) or (button_2_click % 2 == 0):

        if button_1_click is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if button_1_click is not None and button_1_click % 2 == 1:
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]


@app.callback(
    [
        dash.dependencies.Output('dropdown_y', 'style'),
        dash.dependencies.Output('dropdown_y', 'value'),
    ],
    [
        dash.dependencies.Input('button-1', 'n_clicks'),
        dash.dependencies.Input('button-2', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_y_var(button_1_click, button_2_click):

    if (button_2_click is None) or (button_2_click % 2 == 0):

        if button_1_click is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if button_1_click is not None and button_1_click % 2 == 1:
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]


#### Distribution Plot Dropdown
@app.callback(
    [
        dash.dependencies.Output('dropdown_violin', 'style'),
        dash.dependencies.Output('dropdown_violin', 'value'),
    ],
    [
        dash.dependencies.Input('button-2', 'n_clicks'),
        dash.dependencies.Input('button-1', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_corr_var(button_corr_click, button_scatter_click):

    if (button_scatter_click is None) or (button_scatter_click % 2 == 0):

        if button_corr_click is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if button_corr_click is not None and button_corr_click % 2 == 1:
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]


#### Feature Importance plot dropdown
@app.callback(
    [
        dash.dependencies.Output('dropdown_featimp', 'style'),
        dash.dependencies.Output('dropdown_featimp', 'value'),
    ],
    [
        dash.dependencies.Input('button-3', 'n_clicks'),
        dash.dependencies.Input('button-1', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_featimp_var(button_featimp_click, button_scatter_click):

    if (button_scatter_click is None) or (button_scatter_click % 2 == 0):

        if button_featimp_click is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if button_featimp_click is not None and button_featimp_click % 2 == 1:
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]



##########Dynamic Dropdowns for Predictive Analytics Tab############

@app.callback(
    [
        dash.dependencies.Output('dropdown_x_regression', 'style'),
        dash.dependencies.Output('dropdown_x_regression', 'value'),
    ],
    [
        dash.dependencies.Input('linearregression-button', 'n_clicks'),
        dash.dependencies.Input('randomforest-button', 'n_clicks'),
        dash.dependencies.Input('button-classification', 'n_clicks')
        #dash.dependencies.Input('button-regression', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_x_var_regression(lr_button, rf_button, button_classification_click):

    if (button_classification_click is None) or (button_classification_click % 2 == 0):

        if lr_button is None and rf_button is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if (lr_button is not None or rf_button is not None) and (lr_button % 2 == 1 or rf_button % 2 == 1):
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]


@app.callback(
    [
        dash.dependencies.Output('dropdown_y_regression', 'style'),
        dash.dependencies.Output('dropdown_y_regression', 'value'),
    ],
    [
        dash.dependencies.Input('linearregression-button', 'n_clicks'),
        dash.dependencies.Input('randomforest-button', 'n_clicks'),
        dash.dependencies.Input('button-classification', 'n_clicks')
        #dash.dependencies.Input('button-regression', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_y_var_regression(lr_button, rf_button, button_classification_click):

    if (button_classification_click is None) or (button_classification_click % 2 == 0):

        if lr_button is None and rf_button is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if (lr_button is not None or rf_button is not None) and (lr_button % 2 == 1 or rf_button % 2 == 1):
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]



@app.callback(
    [
        dash.dependencies.Output('dropdown_x_class', 'style'),
        dash.dependencies.Output('dropdown_x_class', 'value'),
    ],
    [
        dash.dependencies.Input('button-regression', 'n_clicks'),
        dash.dependencies.Input('svm-button', 'n_clicks'),
        dash.dependencies.Input('rfclass-button', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_x_var_classification(button_regression_click, svm_button, rfclass_button):

    if (button_regression_click is None) or (button_regression_click % 2 == 0):

        if svm_button is None and rfclass_button is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if (svm_button is not None or rfclass_button is not None) and (svm_button % 2 == 1 or rfclass_button % 2 == 1):
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]


@app.callback(
    [
        dash.dependencies.Output('dropdown_y_class', 'style'),
        dash.dependencies.Output('dropdown_y_class', 'value'),
    ],
    [
        dash.dependencies.Input('button-regression', 'n_clicks'),
        dash.dependencies.Input('svm-button', 'n_clicks'),
        dash.dependencies.Input('rfclass-button', 'n_clicks')
    ]
)
def toggle_dropdown_visibility_y_var_classification(button_regression_click, svm_button, rfclass_button):

    if (button_regression_click is None) or (button_regression_click % 2 == 0):

        if svm_button is None and rfclass_button is None:
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]

        if (svm_button is not None or rfclass_button is not None) and (svm_button % 2 == 1 or rfclass_button % 2 == 1):
            # Make the dropdown visible when Button 1 is clicked an odd number of times
            return [
                {'display': 'block', 'color': 'black', 'textAlign': 'center'},
                None
            ]
        else:
            # Make the dropdown invisible and reset the value when either button is clicked an even number of times
            return [
                {'display': 'none', 'color': 'black', 'textAlign': 'center'},
                None
            ]
    else:
        return [
            {'display': 'none', 'color': 'black', 'textAlign': 'center'},
            None
        ]

##########Dynamic Charts for Feature Exploration Tab############

#### Scatterlplot Div behavior
@app.callback(
    Output(component_id='scatterplot-div', component_property='style'),
    [Input('button-1', 'n_clicks'), Input('button-2', 'n_clicks')]
)
def update_scatterplot_visibility(n_clicks_1, n_clicks_2):

    if (n_clicks_2 is None) or (n_clicks_2 % 2 == 0) :
        if n_clicks_1 is None:
            return {'display': 'none'}
        if n_clicks_1 % 2 == 0:
            # Make the scatterplot invisible when the button is clicked an even number of times
            return {'display': 'none'}
        else:
            # Make the scatterplot visible when the button is clicked an odd number of times
            return {'display': 'block'}
    else:
        return {'display': 'none'}


#### Distribution plot Div behavior
@app.callback(
    Output(component_id='dist-plot-div', component_property='style'),
    [Input('button-2', 'n_clicks'), Input('button-1', 'n_clicks')]
)
def update_corplot_visibility(corr_plot_clicks, scatter_plot_clicks):

    if (scatter_plot_clicks is None) or (scatter_plot_clicks % 2 == 0):

        if corr_plot_clicks is None:
            return {'display': 'none'}
        if corr_plot_clicks % 2 == 0:
            # Make the correlation plot invisible when the button is clicked an even number of times
            return {'display': 'none'}
        else:
            # Make the correlation plot visible when the button is clicked an odd number of times
            return {'display': 'block'}
    else:
        return {'display': 'none'}



#### Feature Importance Plot Div behavior

@app.callback(
    Output(component_id='featimp-div', component_property='style'),
    [Input('button-1', 'n_clicks'), Input('button-2', 'n_clicks'), Input('button-3', 'n_clicks')]
)
def update_scatterplot_visibility(n_clicks_1, n_clicks_2, n_clicks_3):
    if (n_clicks_3 is not None and n_clicks_3 % 2 == 1) and (n_clicks_2 is None or n_clicks_2 % 2 == 0) and (n_clicks_1 is None or n_clicks_1 % 2 == 0):
        # Make the featimp visible when button 3 is clicked an odd number of times and all other buttons are clicked an even number of times
        return {'display': 'block'}
    else:
        # Make the scatterplot invisible in all other cases
        return {'display': 'none'}



##########Dynamic Charts for Predictive Analytics Tab############
#### Regression Div behavior
@app.callback(
    Output(component_id='regression-div', component_property='style'),
    [Input('button-regression', 'n_clicks'), Input('button-classification', 'n_clicks')]
)
def update_regression_visibility(n_clicks_regression, n_clicks_classification):

    if (n_clicks_classification is None) or (n_clicks_classification % 2 == 0):
        if n_clicks_regression is None:
            return {'display': 'none'}
        if n_clicks_regression % 2 == 0:
            # Make the scatterplot invisible when the button is clicked an even number of times
            return {'display': 'none'}
        else:
            # Make the scatterplot visible when the button is clicked an odd number of times
            return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output(component_id='classification-div', component_property='style'),
    [Input('button-regression', 'n_clicks'), Input('button-classification', 'n_clicks')]
)
def update_classification_visibility(n_clicks_regression, n_clicks_classification):

    if (n_clicks_regression is None) or (n_clicks_regression % 2 == 0):
        if n_clicks_classification is None:
            return {'display': 'none'}
        if n_clicks_classification % 2 == 0:
            # Make the scatterplot invisible when the button is clicked an even number of times
            return {'display': 'none'}
        else:
            # Make the scatterplot visible when the button is clicked an odd number of times
            return {'display': 'block'}
    else:
        return {'display': 'none'}



########Dropdowns for Feature Exploration

#### Dropdowns for Scatterplot
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


#### Dropdowns for Distribution Plots
@app.callback(
    Output('dropdown_violin', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_corr_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []


#### Function to impute Null values for Random Forest Feature Importance
def impute_and_remove(df):
    # Calculate the percentage of missing values in each column
    missing_values_perc = df.isnull().mean()

    # Drop the columns with more than 20% missing values
    #df = df.drop(columns=missing_values_perc[missing_values_perc > 0.2].index)

    # Iterate over the remaining columns
    for col in df.columns:
        # Determine the data type of the column
        if np.issubdtype(df[col].dtype, np.number):
            # Continuous column, use SimpleImputer with strategy 'mean'
            imputer = SimpleImputer(strategy='mean')
        elif df[col].dtype == 'bool':
            # Convert boolean column to string
            df[col] = df[col].astype('category')
            # Use SimpleImputer with strategy 'most_frequent' for categorical column
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            # Categorical column, use SimpleImputer with strategy 'mode'
            imputer = SimpleImputer(strategy='most_frequent')

        # Impute the missing values in the column using the SimpleImputer object
        df[col] = imputer.fit_transform(df[[col]])

    return df


#### Dropdown for Feature Importance plot
@app.callback(
    Output('dropdown_featimp', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_featimp_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)

        df = impute_and_remove(df)

        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []


########Dropdowns for Predictive Analytics

#### Dropdowns for Scatterplot
@app.callback(
    Output('dropdown_x_regression', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_x_regress_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []

@app.callback(
    Output('dropdown_y_regression', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_y_regress_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)
        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []


@app.callback(
    Output('dropdown_x_class', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_x_class_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)

        df = impute_and_remove(df)

        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []

@app.callback(
    Output('dropdown_y_class', 'options'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')])
def update_x_class_options(contents, filename):
    if contents:
        df = parse_data(contents, filename)

        df = impute_and_remove(df)

        df = df.set_index(df.columns[0])
        lst = [{'label': i, 'value': i} for i in df.columns]
        return lst
    else:
        return []

#######Warning Messages########
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

@app.callback(
    Output('warning-message-regression', 'children'),
    [Input('dropdown_x_regression', 'value'), Input('dropdown_y_regression', 'value')]
)
def update_warning_message(dropdown_x_value, dropdown_y_value):
    if (dropdown_x_value is not None) or (dropdown_y_value is not None):
        if dropdown_y_value in dropdown_x_value:
            return f'''One of your selected features ({dropdown_y_value}) is also your target variable. Please remove the target variable from your feature inputs!'''
        else:
            return ''
    else:
        return ''


##############testing nested buttons

@app.callback(
    Output("additional-buttons", "children"),
    [Input("button-regression", "n_clicks"), Input("button-classification", "n_clicks"), Input("button-placeholder", "n_clicks")],
)
def render_additional_buttons(btn1_clicks, btn2_clicks, btn3_clicks):
    ctx = dash.callback_context
    if btn1_clicks is None and btn2_clicks is None and btn3_clicks is None:
        return []
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if btn1_clicks % 2 == 1:
            return [
                dbc.Row(
                    [
                        html.Div([
                        html.Button("Linear Regression Model", id="linearregression-button",
                                    n_clicks=0,
                                    n_clicks_timestamp=0,
                                    style={
                                        'backgroundColor': 'rgba(0,0,0,0.3)',
                                        'color': 'white',
                                        'text-align': 'center',
                                        #'margin-right': '10px',
                                        # add a right margin to create a space between the buttons
                                        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
                                        'height': '30px',  # set the height of the buttons
                                        'width': '200px',  # set the width of the buttons
                                        'font-size': '16px'  # set the font size of the button labels
                                    }
                                    ),
                        html.Button("Random Forest Model", id="randomforest-button",
                                    n_clicks=0,
                                    n_clicks_timestamp=0,
                                    style={
                                        'backgroundColor': 'rgba(0,0,0,0.3)',
                                        'color': 'white',
                                        'text-align': 'center',
                                        #'margin-right': '10px',
                                        # add a right margin to create a space between the buttons
                                        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
                                        'height': '30px',  # set the height of the buttons
                                        'width': '200px',  # set the width of the buttons
                                        'font-size': '16px'  # set the font size of the button labels
                                    }
                                    )
                            ],
                            style={"display": "flex", "justify-content": "center"}, className="p-4"),
                    ]
                )
            ]
        elif btn2_clicks % 2 == 1:
            return [
                dbc.Row(
                    [
                        html.Div([
                        html.Button("SVM Model", id="svm-button",
                                    n_clicks=0,
                                    n_clicks_timestamp=0,
                                    style={
                                        'backgroundColor': 'rgba(0,0,0,0.3)',
                                        'color': 'white',
                                        'text-align': 'center',
                                        #'margin-right': '10px',
                                        # add a right margin to create a space between the buttons
                                        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
                                        'height': '30px',  # set the height of the buttons
                                        'width': '200px',  # set the width of the buttons
                                        'font-size': '16px'  # set the font size of the button labels
                                    }
                                    ),
                        html.Button("Random Forest Model", id="rfclass-button",
                                    n_clicks=0,
                                    n_clicks_timestamp=0,
                                    style={
                                        'backgroundColor': 'rgba(0,0,0,0.3)',
                                        'color': 'white',
                                        'text-align': 'center',
                                        #'margin-right': '10px',
                                        # add a right margin to create a space between the buttons
                                        'border-radius': '7%',  # set the border radius to 50% to make the buttons round
                                        'height': '30px',  # set the height of the buttons
                                        'width': '200px',  # set the width of the buttons
                                        'font-size': '16px'  # set the font size of the button labels
                                    }
                                    ),
                        ],
                            style={"display": "flex", "justify-content": "center"}, className="p-4")
                    ]
                )
            ]
        else:
            return []



########Callbacks for all features in the App#########
@app.callback(
    Output(component_id='scatter-plot', component_property='children'),
    [Input(component_id='dropdown_x', component_property='value'),
     Input(component_id='dropdown_y', component_property='value'),
     Input('intermediate-value', 'data'),
     Input('button-1', 'n_clicks')
     ]
)
def update_scatterplot(x_axis, y_axis, jsonified_cleaned_data, n_clicks):
    if n_clicks is not None:
        if (jsonified_cleaned_data is not None) and (x_axis is not None) and (y_axis is not None):
            df = pd.read_json(jsonified_cleaned_data, orient='split')

            slope, intercept, r_value, p_value, std_err = linregress(df[x_axis], df[y_axis])

            figure = px.scatter(df, x=x_axis, y=y_axis, title=f'''Scatter Plot: Relationship between {x_axis} and {y_axis}''')
            figure.update_traces(marker=dict(color='red'))
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)
            figure.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            figure.update_layout(
                title={
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            figure.update_layout(title_font_color="white",
                                 font_color="white")

            figure.add_scatter(x=df[x_axis], y=intercept + slope * df[x_axis], mode='lines',
                            name='Regression Line')
            r_squared = r_value ** 2
            figure.add_annotation(x=0.9, y=0.9, text=f'R-squared = {r_squared:.3f}')

            return dcc.Graph(id='scatterplot', figure=figure)
        elif x_axis is None and y_axis is None and n_clicks % 2 ==1:
            return html.Div([
                html.Center(html.H4('Please make sure to select BOTH an X and Y variable to display Scatterplot')),
                dcc.Loading(type = 'circle', children=[html.Div(id='loading-scatterplot')])
            ])
    #else:
    #    return html.Div([
    #            html.Center(html.H4('Please make sure Scatter Plot button is active!'))
    #        ])


@app.callback(
    Output('dist-plot-div', 'children'),
    [Input(component_id='dropdown_violin', component_property='value'),
     Input('intermediate-value', 'data'),
     Input('button-2', 'n_clicks')
     ]
)
def update_hist_plots(violin_value, jsonified_cleaned_data, n_clicks):
    if n_clicks is not None:
        if (jsonified_cleaned_data is not None) and (violin_value is not None):
            df = pd.read_json(jsonified_cleaned_data, orient='split')

            # Create density trace
            density_trace = go.Violin(
                x=df[violin_value],
                points='all',
                name=violin_value,
                box=dict(visible=False),
            )

            # Create histogram trace
            histogram_trace = go.Histogram(
                x=df[violin_value],
                name=violin_value
            )

            # Create figure object for the violin plot
            violin_fig = go.Figure(data=[density_trace])

            # Set layout properties for the violin plot
            violin_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',  # set background color to transparent
                font=dict(color='white'),  # set font color to white
                xaxis=dict(showgrid=False),  # remove x-axis gridlines
                yaxis=dict(showgrid=False),  # remove y-axis gridlines
            )

            # Create figure object for the histogram plot
            hist_fig = go.Figure(data=[histogram_trace])

            # Set layout properties for the histogram plot
            hist_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',  # set background color to transparent
                font=dict(color='white'),  # set font color to white
                xaxis=dict(showgrid=False),  # remove x-axis gridlines
                yaxis=dict(showgrid=False),  # remove y-axis gridlines
            )

            return [
                dcc.Graph(id='violin-plot', figure=violin_fig),
                dcc.Graph(id='hist-plot', figure=hist_fig)
            ]
        elif violin_value is None and n_clicks % 2 == 1:
            return html.Div([
                html.Center(html.H4('Please make sure to select values to display Correlation Plot')),
                dcc.Loading(type='circle', children=[html.Div(id='loading-corrplot')])
            ])
        else:
            return html.Div([
                html.Center(html.H4('testing'))
            ])


@app.callback(
    Output('featimp-div', 'children'),
    [Input(component_id='dropdown_featimp', component_property='value'),
     Input('intermediate-value', 'data'),
     Input('button-3', 'n_clicks')
     ]
)
def update_featimp_plots(target_value, jsonified_cleaned_data, n_clicks):
    if n_clicks is not None:
        if (jsonified_cleaned_data is not None) and (target_value is not None):
            df = pd.read_json(jsonified_cleaned_data, orient='split')


            #encode target variable if needed
            if df[target_value].dtype == 'object':
                if df[target_value].nunique() > 5:
                    return html.Div([
                html.Center(html.H2('This target variable has a high degree of cardinality. This will not likely derive any meaningful insights.')),
                dcc.Loading(type='circle', children=[html.Div(id='loading-corrplot')])
            ])
                else:
                    le_encoder = LabelEncoder()
                    df[target_value] = le_encoder.fit_transform(df[target_value])
            else:
                pass


            # Get a list of all features, excluding the target variable
            features = df.columns.drop(target_value)

            # Select all non-numeric features
            non_numeric_features = df[features].select_dtypes(exclude='number').columns

            # Create an instance of the OneHotEncoder class
            encoder = OneHotEncoder()

            # Encode or drop the non-numeric features based on the degree of cardinality
            for feature in non_numeric_features:
                if df[feature].nunique() < 4:
                    one_hot = encoder.fit_transform(df[[feature]])
                    # Generate a list of names for the one-hot encoded columns based on the unique values of the feature
                    column_names = [f"{feature}_{value}" for value in df[feature].unique()]
                    one_hot_df = pd.DataFrame(one_hot.toarray(), columns=column_names)
                    df = pd.concat([df, one_hot_df], axis=1)
                    df = df.drop(columns=[feature])
                else:
                    df = df.drop(columns=[feature])


            df = impute_and_remove(df)


            X = df.drop(columns=[target_value])

            y = df[target_value]


            n_unique_values = y.nunique()

            if n_unique_values > 2:
                # Use RandomForestRegressor
                model = RandomForestRegressor()

            else:
                # Use RandomForestClassifier
                model = RandomForestClassifier()


            model.fit(X, y)

            importance = model.feature_importances_

            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importance})

            # Sort the dataframe by importance

            feature_importance = feature_importance.sort_values(by='importance', ascending=False)

            fig = go.Figure([go.Bar(x=feature_importance['feature'], y=feature_importance['importance'],
                                    name='Feature Importance')])

            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            fig.update_layout(title_font_color="white",
                                 font_color="white")

            return [
                dcc.Graph(id='featimp-plot', figure=fig)
            ]
        elif target_value is None and n_clicks % 2 == 1:
            return html.Div([
                html.Center(html.H4('Please make sure to select values to display Feature Importance Plot')),
                dcc.Loading(type='circle', children=[html.Div(id='loading-corrplot')])
            ])
        else:
            return html.Div([
                html.Center(html.H4('testing'))
            ])



@app.callback(
    Output("regression-div", "children"),
    [Input("button-regression", "n_clicks"),
     Input("linearregression-button", "n_clicks"),
     Input("randomforest-button", "n_clicks"),
     Input('intermediate-value', 'data')],
    [Input("dropdown_x_regression", "value"),
     Input("dropdown_y_regression", "value")]
)
def update_regression_graph(n_clicks_regress, n_clicks_linear, n_clicks_rf, jsonified_cleaned_data, feature_columns, target_column):
    if n_clicks_regress is not None and n_clicks_linear % 2 == 1:
        if (jsonified_cleaned_data is not None) and (target_column is not None) and (feature_columns is not None):
            df = pd.read_json(jsonified_cleaned_data, orient='split')

            # encode target variable if needed
            if df[target_column].dtype == 'object':
                return html.Div([
                    html.Center(html.H2(
                        'This target variable is NOT a continuous variable and therefore not appropriate to use for Regression.')),
                    #dcc.Loading(type='circle', children=[html.Div(id='loading-corrplot')])
                ])
            else:
                pass

            non_numeric_features = df[feature_columns].select_dtypes(exclude='number').columns

            # Create an instance of the OneHotEncoder class
            encoder = OneHotEncoder()

            # Encode or drop the non-numeric features based on the degree of cardinality - if a categorical feature has more than 4 unique classes then drop the variable as it may add too many dimensions
            for feature in non_numeric_features:
                if df[feature].nunique() < 4:
                    one_hot = encoder.fit_transform(df[[feature]])
                    # Generate a list of names for the one-hot encoded columns based on the unique values of the feature
                    column_names = [f"{feature}_{value}" for value in df[feature].unique()]
                    one_hot_df = pd.DataFrame(one_hot.toarray(), columns=column_names)
                    df = pd.concat([df, one_hot_df], axis=1)
                    df = df.drop(columns=[feature])
                else:
                    df = df.drop(columns=[feature])

            #if not feature_columns:
            #    return "Please select features"
            #if not target_column:
            #    return "Please select a target"

            #preprocessing step 1 - impute missing values
            df = impute_and_remove(df)


            df = standardize_inputs(df, target_column)


            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            predictions = model.predict(X_test)
            fig = px.scatter(x=y_test, y=predictions)
            fig.add_scatter(x=y_test, y=y_test, mode="lines", name="Line of Best Fit")
            fig.update_layout(title="Actual vs Predicted")
            fig.update_traces(marker=dict(color='red'))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            fig.update_layout(
                title={
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            fig.update_layout(title_font_color="white",
                                 font_color="white")
            fig.update_layout(xaxis={"title": "Actual"}, yaxis={"title": "Predicted"})

            results = pd.DataFrame({
                "Actual": y_test,
                "Predicted": predictions
            })
            table = dash_table.DataTable(
                id="table",
                columns=[{"name": col, "id": col} for col in results.columns],
                data=results.to_dict("records"),
                style_header={'backgroundColor': 'rgba(0,0,0,0)',
                              'color': 'white',
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
                editable=False,
                page_size=5,
                sort_mode='multi',
                sort_action='native',
                filter_action='native',
            )




            return dcc.Graph(id='regression-plot', figure=fig), f"R2 Score: {score}", table

    elif n_clicks_regress is not None and n_clicks_rf % 2 == 1:
        if (jsonified_cleaned_data is not None) and (target_column is not None) and (feature_columns is not None):
            df = pd.read_json(jsonified_cleaned_data, orient='split')



            # encode target variable if needed
            if df[target_column].dtype == 'object':
                return html.Div([
                    html.Center(html.H2(
                        'This target variable is labeled as a Categorical Variable and therefore not appropriate to use for Regression.')),
                    # dcc.Loading(type='circle', children=[html.Div(id='loading-corrplot')])
                ])
            else:
                pass

            non_numeric_features = df[feature_columns].select_dtypes(exclude='number').columns

            # Create an instance of the OneHotEncoder class
            encoder = OneHotEncoder()

            # Encode or drop the non-numeric features based on the degree of cardinality - if a categorical feature has more than 4 unique classes then drop the variable as it may add too many dimensions
            for feature in non_numeric_features:
                if df[feature].nunique() < 4:
                    one_hot = encoder.fit_transform(df[[feature]])
                    # Generate a list of names for the one-hot encoded columns based on the unique values of the feature
                    column_names = [f"{feature}_{value}" for value in df[feature].unique()]
                    one_hot_df = pd.DataFrame(one_hot.toarray(), columns=column_names)
                    df = pd.concat([df, one_hot_df], axis=1)
                    df = df.drop(columns=[feature])
                else:
                    df = df.drop(columns=[feature])

            # if not feature_columns:
            #    return "Please select features"
            # if not target_column:
            #    return "Please select a target"

            # preprocessing step 1 - impute missing values
            df = impute_and_remove(df)

            df = standardize_inputs(df, target_column)

            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

            model = RandomForestRegressor()

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            score = model.score(X_test, y_test)
            predictions = model.predict(X_test)
            fig = px.scatter(x=y_test, y=predictions)
            fig.add_scatter(x=y_test, y=y_test, mode="lines", name="Line of Best Fit")
            fig.update_layout(title="Actual vs Predicted (Random Forest)")
            fig.update_traces(marker=dict(color='red'))
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            fig.update_layout(
                title={
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'})
            fig.update_layout(title_font_color="white",
                              font_color="white")
            fig.update_layout(xaxis={"title": "Actual"}, yaxis={"title": "Predicted"})

            results = pd.DataFrame({
                "Actual": y_test,
                "Predicted": predictions
            })
            table = dash_table.DataTable(
                id="table",
                columns=[{"name": col, "id": col} for col in results.columns],
                data=results.to_dict("records"),
                style_header={'backgroundColor': 'rgba(0,0,0,0)',
                              'color': 'white',
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
                editable=False,
                page_size=5,
                sort_mode='multi',
                sort_action='native',
                filter_action='native',
            )

            return dcc.Graph(id='regression-plot', figure=fig), f"R2 Score: {score}", table

####NEED TO CREATE OUTPUTS FOR CLASSIFICATION MODELS


@app.callback(
    Output("classification-div", "children"),
    [Input("button-classification", "n_clicks"),
     Input("svm-button", "n_clicks"),
     Input("rfclass-button", "n_clicks"),
     Input('intermediate-value', 'data')],
    [Input("dropdown_x_class", "value"),
     Input("dropdown_y_class", "value")]
)
def update_classification_graph(n_clicks_class, n_clicks_svm, n_clicks_rfclass, jsonified_cleaned_data, feature_columns, target_column):
    if n_clicks_class is not None and n_clicks_svm% 2 == 1:
        if (jsonified_cleaned_data is not None) and (target_column is not None) and (feature_columns is not None):
            print("check")
            df = pd.read_json(jsonified_cleaned_data, orient='split')


            non_numeric_features = df[feature_columns].select_dtypes(exclude='number').columns



            # Create an instance of the OneHotEncoder class
            encoder = OneHotEncoder()

            selected_features = feature_columns
            new_column_names = []
            dropped_columns = []

            # Encode or drop the non-numeric features based on the degree of cardinality - if a categorical feature has more than 4 unique classes then drop the variable as it may add too many dimensions
            for feature in non_numeric_features:
                if df[feature].nunique() < 4:
                    #selected_features.append(feature)
                    one_hot = encoder.fit_transform(df[[feature]])
                    # Generate a list of names for the one-hot encoded columns based on the unique values of the feature
                    column_names = [f"{feature}_{value}" for value in df[feature].unique()]
                    new_column_names.extend(column_names)
                    dropped_columns.append(feature)
                    one_hot_df = pd.DataFrame(one_hot.toarray(), columns=column_names)
                    df = pd.concat([df, one_hot_df], axis=1)
                    df = df.drop(columns=[feature])
                else:
                    df = df.drop(columns=[feature])

            combined_columns = selected_features + new_column_names
            for column in combined_columns:
                if column in dropped_columns:
                    combined_columns.remove(column)


            df = impute_and_remove(df)


            df = standardize_inputs_class(df, target_column)



            X = df[combined_columns] #this already has the chosen columns just named after the OHE
            y = df[target_column]


            #encode target vars
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            #split data
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=10)

            # Model complexity
            neighboors = np.arange(1, 30)
            train_accuracy = []
            test_accuracy = []

            # Loop over different values of k
            for i, k in enumerate(neighboors):
                # k from 1 to 30(exclude)
                knn = KNeighborsClassifier(n_neighbors=k)
                # fit with knn
                knn.fit(X_train, y_train)
                train_accuracy.append(knn.score(X_train, y_train))  # train accuracy
                test_accuracy.append(knn.score(X_test, y_test))

            max_test_accuracy = np.max(test_accuracy)
            best_k = neighboors[np.argmax(test_accuracy)]
            print(best_k)

            best_knn = KNeighborsClassifier(n_neighbors=best_k)
            best_knn.fit(X_train, y_train)

            y_pred = best_knn.predict(X_test)
            #print(predictions)




            trace1 = go.Scatter(
                x=neighboors,
                y=train_accuracy,
                mode="lines",
                name="train_accuracy",
                marker=dict(color='rgb(255, 0, 0)'),
                text="train_accuracy")
            # Creating trace2
            trace2 = go.Scatter(
                x=neighboors,
                y=test_accuracy,
                mode="lines+markers",
                name="test_accuracy",
                marker=dict(color='rgb(104,111,254)'),
                text="test_accuracy")
            data = [trace1, trace2]
            layout = dict(
                title='K Value vs Accuracy',
                xaxis=dict(title='Number of Neighbors', ticklen=10, zeroline=False, showgrid=False,
                           tickfont=dict(color='white'), showline=False),
                yaxis=dict(linecolor='white', tickcolor='white', showgrid=False, tickfont=dict(color='white'),
                           zeroline=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='white')
            )
            fig = dict(data=data, layout=layout)



            acc = "Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy)))



            #accuracy = accuracy_score(y_test, predictions)
            #print('Accuracy:', accuracy)
            print("check:", y_test)
            var = 'var'
            print(var)
            print(len(y_pred))
            print(len(y_test))
            results = pd.DataFrame(columns=["Actual", "Predicted"])
            results["Actual"] = y_test
            results["Predicted"] = y_pred
            print(results)


            # Create a dataframe from the test features and test targets
            #X_test_unscaled = scaler.inverse_transform(X_test)
            #print(X_test_unscaled)

            test_df = pd.DataFrame(data=np.column_stack((X_test, y_test)), columns=list(X_test.columns) + ["target"])

            # Add a new column with the predicted values
            test_df["predictions"] = y_pred

            table = dash_table.DataTable(
                id="table",
                columns=[{"name": col, "id": col} for col in test_df.columns],
                data=test_df.to_dict("records"),
                style_header={'backgroundColor': 'rgba(0,0,0,0)',
                              'color': 'white',
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
                editable=False,
                page_size=5,
                sort_mode='multi',
                sort_action='native',
                filter_action='native',
            )


            return dcc.Graph(id='classification-plot', figure=fig), acc, table




if __name__ == "__main__":
    app.run_server(debug=True)