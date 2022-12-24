import dash
from dash import Dash, dcc, html, dash_table
from dash import dcc as dcc
from dash import html as html
#import dash_table
import dash_bootstrap_components as dbc
import plotly.express as px

dash.register_page(__name__,
                   path='/EDA',
                   name='Exploratory Analysis',
                   title='EDA',
                   description='EDA Page to better understand your data.'
                   )

df = px.data.tips()

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

layout = html.Div([
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
                style={'display': 'none'}  # initially set the dropdown to be invisible
            )
        ),
        dbc.Col(
            dcc.Dropdown(
                id='dropdown_y',
                options=[],
                style={'display': 'none'}  # initially set the dropdown to be invisible
            )
        )
    ], style={'margin': '10px'})
])