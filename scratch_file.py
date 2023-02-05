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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.subplots as subplots
import plotly.express as px
import base64
import datetime
import numpy as np
import io
from sklearn import datasets

# Set up the Dash app
app = dash.Dash()

# Load the data
#df = pd.read_csv(r'C:/Users/willi/Desktop/misc_data/house-prices-advanced-regression-techniques/train.csv')
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Rename the columns
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
feature_columns = list(df.columns)

# Create a dropdown component for the features
feature_dropdown = dcc.Dropdown(
    id='feature-dropdown',
    options=[{'label': col, 'value': col} for col in feature_columns],
    value=feature_columns[0],
    multi=True
)

# Create a dropdown component for the target variable
target_dropdown = dcc.Dropdown(
    id='target-dropdown',
    options=[{'label': col, 'value': col} for col in feature_columns],
    value=feature_columns[0]
)

# Create a button component to run the model
button = html.Button('Run model', id='button')

# Create an output component to display the bar chart
output = dcc.Graph(id='plot')

app.layout = html.Div([
    html.H1('Multinomial Logistic Regression'),
    html.Div([
        html.Label('Features:'),
        feature_dropdown,
        html.Br(),
        html.Label('Target:'),
        target_dropdown,
        html.Br(),
        button
    ]),
    output
])

@app.callback(
    dash.dependencies.Output('plot', 'figure'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('feature-dropdown', 'value'),
     dash.dependencies.State('target-dropdown', 'value')]
)
def update_plot(n_clicks, features, target):
    # Train a multinomial logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    X = df[features]
    y = df[target]
    model.fit(X, y)

    # Obtain the probability of each class for a sample
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    proba = model.predict_proba(sample)

    # Create a data frame with the probability of each class
    df_proba = pd.DataFrame({'class': range(len(proba[0])), 'probability': proba[0]})

    # Create the bar chart
    figure = px.bar(df_proba, x='class', y='probability')

    return figure

if __name__ == '__main__':
    app.run_server()