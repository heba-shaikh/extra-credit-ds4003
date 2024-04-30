# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from dash import Dash, dcc, html, Input, Output, dash_table, State
import plotly.graph_objs as go

# %%
df=pd.read_csv("match_data.csv", encoding='utf-8')
df = df.drop(df.columns[0:6], axis=1)
df = df.drop(df.columns[83:92], axis=1)
df = df.drop(df.columns[5:8], axis=1)

df.head()


# %%
#Decide which team won
df['Home_Win'] = (df['Basketball Matches - Match → Home Score'] > df['Basketball Matches - Match → Away Score']).astype(int)

# Extract integer features 
int_features = df.iloc[:, 0:78].select_dtypes(include=['int64', 'int32']).columns

# Normalize integer features
scaler = MinMaxScaler()
df[int_features] = scaler.fit_transform(df[int_features])

# Define features and target variable
features = df.columns[0:78]
X = df[features]
y = df['Home_Win']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train) # Train the classifier on the training data

# %%
stylesheets= ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=stylesheets) # initialize the app
server=app.server

teams=df["Team Name"].unique() #Get the team names from the column

app.layout = html.Div(style={'backgroundColor': 'orange'}, children=[
    html.Div([
        html.Div([
            html.H1("NCAA Tournament Team Probabilities", style={"font-family": "Montserrat, sans-serif", "text-align": "center", "justify-content": "center"}),
            html.Br(),
            html.Br(),
            html.P("Pick the teams that you want to view probability for if they were to vs each other in a round in the NCAA tournament. You can start off with the round of 64 by putting two teams against each other. Based on the team that wins that probability, you can predict it's chances of beating another team in the round of 32 and so on...", style={"font-family": "Montserrat, sans-serif","font-size": "17px"}),
            html.Div([
                html.Div([
                    # Dropdown for home team
                    dcc.Dropdown(
                        id='team-dropdown',
                        placeholder="Select Home Team",
                        options=[{'label': team, 'value': team} for team in teams],
                    )
                ], className="six columns"),
                html.Div([
                    # Dropdown for away team
                    dcc.Dropdown(
                        id='opponent-dropdown',
                        placeholder="Select Away Team",
                        options=[{'label': team, 'value': team} for team in teams],
                    )
                ], className="six columns")
            ], className="row")
        ], className="six columns"),  

        # Chart for probability of either team to win
        html.Div([
            dcc.Graph(id='pie-chart')
        ], className="six columns"),  
    ], className="row")
])



@app.callback(
    Output('pie-chart', 'figure'),
    [Input('team-dropdown', 'value'),
     Input('opponent-dropdown', 'value')]
)
def update_pie_chart(selected_team, selected_opponent):
    # Empty figure
    if not selected_team or not selected_opponent:
        return {'data': [], 'layout': {}}
    
    # features for prediction for both teams
    selected_team_data = df[df['Team Name'] == selected_team][features]
    opponent_data = df[df['Team Name'] == selected_opponent][features]

    # Predict probabilities of winning for both teams
    selected_team_probability = rf_classifier.predict_proba(selected_team_data)[:, 1][0]
    opponent_team_probability = rf_classifier.predict_proba(opponent_data)[:, 1][0]

    # Create pie chart data
    labels = [selected_team, selected_opponent]
    values = [selected_team_probability, opponent_team_probability]

    #pie chart details
    data = go.Pie(
        labels=labels,
        values=values,
        hoverinfo='label+percent'
    )

    layout = go.Layout(
        title='Probability of Winning',
        margin=dict(t=50) 
    )

    return {'data': [data], 'layout': layout}



if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check_interval=60000)


