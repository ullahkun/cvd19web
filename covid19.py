import cgitb
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc

import warnings
warnings.filterwarnings("ignore")



LOGO = "https://ipb.ac.id//assets/assert/logo/Logo_web.png"
IMGJBT = "https://designrevision.com/demo/shards/extra/images/app-promo/iphone-app-mockup.png"
ILUSTRASI = "https://i.ibb.co/Tw7cpVX/covid19-ilustrasi.png"
BG = "https://i.ibb.co/g7xH5JK/bg.png"

baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
tickFont = {'size':12, 'color':"rgb(30,30,30)", 'family':"Heebo, monospace"}

def loadData(fileName, columnName): 
    data = pd.read_csv(baseURL + fileName) \
             .drop(['Lat', 'Long'], axis=1) \
             .melt(id_vars=['Province/State', 'Country/Region'], var_name='date', value_name=columnName) \
             .fillna('<all>')
    data['date'] = data['date'].astype('datetime64[ns]')
    return data

allData = loadData("time_series_covid19_confirmed_global.csv", "CumConfirmed") \
    .merge(loadData("time_series_covid19_deaths_global.csv", "CumDeaths")) \
    .merge(loadData("time_series_covid19_recovered_global.csv", "CumRecovered"))

countries = allData['Country/Region'].unique()
countries.sort()

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


#BARPLOT CODE

# Time
import time
import datetime
from datetime import datetime
from time import gmtime, strftime
from pytz import timezone

# System
import sys
import os
import pandas as pd
import numpy as np


# Graph/ Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
from matplotlib.ticker import ScalarFormatter


#from fbprophet import Prophet
#from fbprophet.plot import add_changepoints_to_plot

#%matplotlib inline

print(os.listdir("./input/"))


plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (9, 6)


def print_time():
    fmt = "%a, %d %B %Y %H:%M:%S %Z%z"
    
    pacific = timezone('US/Pacific')
    
    loc_dt = datetime.now(pacific)
    
    time_str = loc_dt.strftime(fmt)
    
    print("Pacific Time" + " : " + time_str)
    
    return time_str

def format_date_columns_indo(data_cols):
    data_cols_new_format = []
    data_cols_map = {}
    
    for d in data_cols:
        new_d = datetime.strftime(datetime.strptime(d, '%d/%m/%Y'),'%b %d')
        data_cols_map[d] = new_d
        data_cols_new_format.append(new_d)
    
    return data_cols_new_format, data_cols_map

def format_date_columns(data_cols):
    data_cols_new_format = []
    data_cols_map = {}
    
    for d in data_cols:
        new_d = datetime.strftime(datetime.strptime(d, '%m/%d/%y'),'%b %d')
        data_cols_map[d] = new_d
        data_cols_new_format.append(new_d)
    
    return data_cols_new_format, data_cols_map
def get_all_cases(tmp, col):
    sum_list = []
    
    df = tmp.copy()

    for i, day in enumerate(df.sort_values('Date').Date.unique()):    
        tmp_df = df[df.Date == day]
        
        sum_list.append(tmp_df[col].sum())
    
    return sum_list


def get_new_cases(tmp, col):
    diff_list = []
    tmp_df_list = []
    df = tmp.copy()

    for i, day in enumerate(df.sort_values('Date').Date.unique()):    
        tmp_df = df[df.Date == day]
        tmp_df_list.append(tmp_df[col].sum())
        
        if i == 0:
            diff_list.append(tmp_df[col].sum())
        else:
            diff_list.append(tmp_df[col].sum() - tmp_df_list[i-1])
        
    return diff_list

def get_moving_average(tmp, col):
    df = tmp.copy()
    return df[col].rolling(window=2).mean()

def get_exp_moving_average(tmp, col):
    df = tmp.copy()
    return df[col].ewm(span=2, adjust=True).mean()
def fatality_rate_func(country="Total", state=""):
    if not state == "":
        return (time_series_covid_19_deaths[time_series_covid_19_deaths["Province/State"]==state][data_cols].sum()/time_series_covid_19_confirmed[time_series_covid_19_confirmed["Province/State"]==state][data_cols].sum())*100
    elif country == "Total":
        return (time_series_covid_19_deaths[data_cols].sum()/time_series_covid_19_confirmed[data_cols].sum())*100
    else:
        return (time_series_covid_19_deaths[time_series_covid_19_deaths["Country/Region"]==country][data_cols].sum()/time_series_covid_19_confirmed[time_series_covid_19_confirmed["Country/Region"]==country][data_cols].sum())*100

def line_plot(df, title, ylabel="Cases", h=None, v=None,
              xlim=(None, None), ylim=(0, None), math_scale=True, y_logscale=False, y_integer=False):
    """
    Show chlonological change of the data.
    """
    ax = df.plot()
    if math_scale:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci",  axis="y",scilimits=(0, 0))
    
    if y_logscale:
        ax.set_yscale("log")
    
    if y_integer:
        fmt = matplotlib.ticker.ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
    ax.set_title(title)
    ax.set_xlabel(None)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0)
    if h is not None:
        ax.axhline(y=h, color="black", linestyle="--")
    if v is not None:
        if not isinstance(v, list):
            v = [v]
        for value in v:
            ax.axvline(x=value, color="black", linestyle="--")
    plt.tight_layout()


def show_trend(ncov_df, variable="Confirmed", n_changepoints=2, places=None, excluded_places=None):
    """
    Show trend of log10(@variable) using fbprophet package.
    @ncov_df <pd.DataFrame>: the clean data
    @variable <str>: variable name to analyse
        - if Confirmed, use Infected + Recovered + Deaths
    @n_changepoints <int>: max number of change points
    @places <list[tuple(<str/None>, <str/None>)]: the list of places
        - if the list is None, all data will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    @excluded_places <list[tuple(<str/None>, <str/None>)]: the list of excluded places
        - if the list is None, all data in the "places" will be used
        - (str, str): both of country and province are specified
        - (str, None): only country is specified
        - (None, str) or (None, None): Error
    """
#     Data arrangement
#     df = select_area(ncov_df, places=places, excluded_places=excluded_places)
    df = ncov_df.copy()
    if variable == "Confirmed":
        df["Confirmed"] = df[["Infected", "Recovered", "Deaths"]].sum(axis=1)
    df = df.loc[:, ["Date", variable]]
    
    df.columns = ["ds", "y"]
    
#     # Log10(x)
#     warnings.resetwarnings()
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         df["y"] = np.log10(df["y"]).replace([np.inf, -np.inf], 0)
    
    # fbprophet
#     model = Prophet(growth="logistic", daily_seasonality=False, n_changepoints=n_changepoints)
    model = Prophet(daily_seasonality=False, n_changepoints=n_changepoints)
    model.fit(df)
    future = model.make_future_dataframe(periods=14)
    
    
    forecast = model.predict(future)
    
    # Create figure
    fig = model.plot(forecast)
    _ = add_changepoints_to_plot(fig.gca(), model, forecast)
    
    plt.title(f"Cases ({variable}) over time and change points ")
    plt.ylabel(f"Number of cases")
    plt.xlabel("")
    
    fig = model.plot_components(forecast)
    return forecast, df

#Input data indonesia 
input_dir_indo = "./input/"
df_indonesia = pd.read_csv(input_dir_indo + "data-kumulatif-harian-corona-indonesia-20200410.csv",delimiter=";")
df_indonesia['Date'] = pd.to_datetime(df_indonesia['Date'], format="%d/%m/%Y")

#------------------- get date unique and change format to Month Day ---------------------
date = df_indonesia['Date'].unique()
df_date = pd.DataFrame(data=date, columns=["Date"])
df_date['Date'] = pd.to_datetime(df_date['Date'], format="%d/%m/%Y")

# ----------------------
df = df_indonesia.copy()
def daily_cases(df):
    daily_cases_df = pd.DataFrame([])

    date = df['Date'].unique()
    df_date = pd.DataFrame(data=date, columns=["Date"])
    df_date['Date'] = pd.to_datetime(df_date['Date'], format="%d/%m/%Y")

    daily_cases_df['Date'] = df_date["Date"]

    #sum of all cases on the same date
    daily_cases_df['Confirmed'] = get_all_cases(df, "Confirmed")
    daily_cases_df['Deaths'] = get_all_cases(df, "Deaths")
    daily_cases_df['Recovered'] = get_all_cases(df, "Recovered")

    daily_cases_df["Infected"] = daily_cases_df["Confirmed"] - daily_cases_df["Deaths"] - daily_cases_df["Recovered"]
    # # #get new cases
    daily_cases_df['New Confirmed'] = get_new_cases(df, 'Confirmed')
    daily_cases_df['New Death'] = get_new_cases(df, 'Deaths')
    daily_cases_df['New Recovered'] = get_new_cases(df, 'Recovered')

    # # #Moving average
    # daily_cases_df['confirmed_MA'] = get_moving_average(daily_cases_df, 'New Confirmed')
    # daily_cases_df['deaths_MA'] = get_moving_average(daily_cases_df, 'New Death')
    # daily_cases_df['recovered_MA'] = get_moving_average(daily_cases_df, 'New Recovered')

    # #Exponential moving average
    # daily_cases_df['confirmed_exp_MA'] = get_exp_moving_average(daily_cases_df, 'New Confirmed')
    # daily_cases_df['deaths_exp_MA'] = get_exp_moving_average(daily_cases_df, 'New Death')
    # daily_cases_df['recovered_exp_MA'] = get_exp_moving_average(daily_cases_df, 'New Recovered')

    daily_cases_df ["Active"] = daily_cases_df["Confirmed"] - (daily_cases_df["Recovered"] + daily_cases_df["Deaths"])

    daily_cases_df["Persistence Rate"] = ((daily_cases_df["Active"]/daily_cases_df["Confirmed"])*100)
    daily_cases_df["Fatality Rate"] = ((daily_cases_df["Deaths"]/daily_cases_df["Confirmed"])*100)
    daily_cases_df["Recovery Rate"] = ((daily_cases_df["Recovered"]/daily_cases_df["Confirmed"])*100)
    


    return daily_cases_df
daily_cases_df = daily_cases(df)
last_day = df['Date'].max() #df is cumulatif table, so we take last day for total cases
total_indonesia_df = daily_cases_df[daily_cases_df['Date']==last_day]
total_confirmed_indonesia = total_indonesia_df['Confirmed'].values[0]
total_active_indonesia = total_indonesia_df['Active'].values[0]
total_recovered_indonesia = total_indonesia_df['Recovered'].values[0]
total_death_indonesia = total_indonesia_df['Deaths'].values[0]
indonesia_persistence_rate = total_indonesia_df['Persistence Rate'].values[0]
indonesia_recovery_rate = total_indonesia_df['Recovery Rate'].values[0]
indonesia_fatality_rate = total_indonesia_df['Fatality Rate'].values[0]

data = [indonesia_persistence_rate, indonesia_recovery_rate, indonesia_fatality_rate]

m_sum = {"Total Confirmed (Indonesia)": [total_confirmed_indonesia], 
         "Total Active (Indonesia)"   : [total_active_indonesia],
         "Total Recovered (Indonesia)": [total_recovered_indonesia], 
         "Total Deaths (Indonesia)"   : [total_death_indonesia]}


indonesia_sum_df = pd.DataFrame.from_dict(m_sum)

data = [total_confirmed_indonesia, total_active_indonesia, total_recovered_indonesia, total_death_indonesia]
labels = ["Confirmed", "Active", "Recovered", "Deaths"]


#graph = sns.barplot(x=labels, y=data)
data = [go.Bar(
    x=labels,
    y=data
)]

layout = go.Layout(
    xaxis = {'title': 'Total Confimed, Active, Recovered and Deaths (Current indonesia)'},
)



#for p in graph.patches:
#        graph.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),
#                    ha='center', va='bottom',
#                    color= 'black')
        
my_fig = go.Figure(data=data, layout=layout)


data1 = [indonesia_persistence_rate, indonesia_recovery_rate, indonesia_fatality_rate]
labels1 = ["Persistence Rate", "Recovery Rate", "Fatality Rate"]

data1 = [go.Bar(
    x=labels1,
    y=data1
)]

layout1 = go.Layout(
    xaxis = {'title': 'Current indonesia Persistence, Recovery, Fatality Rate (%)'},
)
my_fig2 = go.Figure(data=data1, layout=layout1)




#Fatality Rate of Corona Virus Cases
layout_fatality = go.Layout(
    title = 'Fatality Rate of Corona Virus Cases',
    xaxis = {'title': 'Date'},
    yaxis = {'title': 'Fatality Rate'},
)
fig_fatality = go.Figure(data=go.Scatter(x=daily_cases_df['Date'], y=daily_cases_df['Fatality Rate']), layout = layout_fatality)

#/BARPLOT CODE


#DATA PROVINSI
last_day = df['Date'].max() #df is cumulatif table, so we take last day for total cases
df_province = df[df["Date"]==last_day]

df_province ["Active"] = df_province["Confirmed"] - (df_province["Recovered"] + df_province["Deaths"])
df_province["Persistence Rate"] = ((df_province["Active"]/df_province["Confirmed"])*100)
df_province["Fatality Rate"] = ((df_province["Deaths"]/df_province["Confirmed"])*100)
df_province["Recovery Rate"] = ((df_province["Recovered"]/df_province["Confirmed"])*100)

data_prov = [go.Bar(
    x=df_province['Province'],
    y=df_province['Confirmed']
)]
layout = go.Layout(
    xaxis = {'title': 'Province'},
    yaxis = {'title': 'Confirmed'},
    title = 'Provincial Case Data'
)
fig_prov = go.Figure(data=data_prov, layout=layout)
#/DATA PROVINSI

#DATA NASIONAL
data_cols = ["Infected", "Deaths", "Recovered"]
rate_cols = ["Fatality Rate", "Persistence Rate", "Recovery Rate"]
variable_dict = {"Susceptible": "S", "Infected": "I", "Recovered": "R", "Deaths": "D"}
ncov_df = daily_cases_df.copy()
ncov_df["Infected"] = ncov_df["Confirmed"] - ncov_df["Deaths"] - ncov_df["Recovered"]
ncov_df[data_cols] = ncov_df[data_cols].astype(np.int64)
ncov_df = ncov_df.loc[:, ["Date", *data_cols]]
ncov_df = ncov_df.set_index(['Date'])


fig_nasional = go.Figure()
fig_nasional.add_trace(go.Scatter(x=ncov_df.index, y=ncov_df["Infected"],
                    mode='lines',
                    name='Infected'))
fig_nasional.add_trace(go.Scatter(x=ncov_df.index, y=ncov_df["Deaths"],
                    mode='lines',
                    name='Deaths'))
fig_nasional.add_trace(go.Scatter(x=ncov_df.index, y=ncov_df["Recovered"],
                    mode='lines', name='Recovered'))






#/DATA NASIONAL

nav_item = dbc.NavItem(dbc.NavLink("Home", href="#", className="active"))

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Epidemiology Scenario", href='#'),
        dbc.DropdownMenuItem("Intervention Scenario Impact", href='#'),
        dbc.DropdownMenuItem("Herd Immunity Model", href='#'),
        dbc.DropdownMenuItem("Personal Protective Equipment", href='#'),
        dbc.DropdownMenuItem("Viral Transport Medium", href='#'),
        dbc.DropdownMenuItem("Economic Impact", href='#'),
    ],

    nav=True,
    in_navbar=True,
    label="Information Center",
)
about = dbc.NavItem(dbc.NavLink("About", href="#"))

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=LOGO, height="75px")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="#",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    [nav_item,
                     dropdown,about,
                     ], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="#06115C",
    dark=True,
    )

jumbotron = dbc.Jumbotron(
    dbc.Container(
        [
            dbc.Row(

                    [
                        dbc.Col(
                            children=[
                                html.H3("Coronavirus Resource Center", className="welcome-heading display-4 sm-1  text-sm-left text-center", style={'color': '#06115C'}),
                                html.P("This website is a resource to help advance the understanding of the virus, inform the public, and brief policy makers in order to guide a response, improve care and save lives.", className="text-muted text-sm-left text-center"),
                                dbc.Button("More Info", color="success", className="mr-1"),
                                ],
                                className="col-lg-6 col-md-6 col-sm-12 mt-auto mb-auto mr-3",

                            ),
                        dbc.Col(
                                html.Img(src=ILUSTRASI, width="90%"),
                                className="col-lg-4 col-md-5 ml-auto mb-auto col-sm-3 d-none d-md-block",
                                
                            )
                    ],
                    align="center",
                    no_gutters=True,
                    style={'background-image': 'url("/assets/bg.png")'}

                ),
        ]
    ), 
    
    )

##CARD
card = dbc.Container (
        [
            html.H3('Actual Monitoring', className="section-title text-center m-5", style={"color":"#06115C"}),

         dbc.Row(
            [
            dbc.Card(
    
                [
                dbc.CardBody(
                    [
                html.H6("Epidemiology Scenario", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Read more", color="success"),
            ]
        ),
    ]
    ,className="col-md-12 col-lg-3 mb-3 mx-3"),
            dbc.Card(
    
                [
                dbc.CardBody(
                    [
                html.H6("Intervention Scenario Impact", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Read more", color="success"),
            ]
       ),
    ]
    ,className="col-md-12 col-lg-3 mb-3 mx-3"),
            dbc.Card(
    
                [
                dbc.CardBody(
                    [
                html.H6("Herd Immunity Model", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Read more", color="success"),
            ]
         ),
    ]
    ,className="col-md-12 col-lg-3 mb-3 mx-3")
            ]
    ,className="justify-content-center")
]
)

carddua = dbc.Container (

         dbc.Row(
            [
            dbc.Card(
    
                [
                dbc.CardBody(
                    [
                html.H6("Epidemiology Scenario", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Read more", color="success"),
            ]
        ),
    ]
    ,className="col-md-12 col-lg-3 mb-3 mx-3"),
            dbc.Card(
    
                [
                dbc.CardBody(
                    [
                html.H6("Intervention Scenario Impact", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Read more", color="success"),
            ]
       ),
    ]
    ,className="col-md-12 col-lg-3 mb-3 mx-3"),
            dbc.Card(
    
                [
                dbc.CardBody(
                    [
                html.H6("Herd Immunity Model", className="card-title"),
                html.P(
                    "Some quick example text to build on the card title and "
                    "make up the bulk of the card's content.",
                    className="card-text",
                ),
                dbc.Button("Read more", color="success"),
            ]
         ),
    ]
    ,className="col-md-12 col-lg-3 mb-3 mx-3")
            ]
    ,className="justify-content-center mb-4")

)

case_data = html.Div(
   dbc.Container(
    children=[

        html.H3('Cases History of the Coronavirus (COVID-19)', className="section-title text-center m-5", style={"color":"#06115C"}),
            html.Div(className="row", children=[
                html.Div(className="col-lg-4", children=[
                    html.H6('Country'),
                    dcc.Dropdown(
                        id='country',
                        options=[{'label':c, 'value':c} for c in countries],
                        value='Italy'
                    )
                ]),
                html.Div(className="col-lg-4", children=[
                    html.H6('State / Province'),
                    dcc.Dropdown(
                        id='state'
                    )
                ]),
                html.Div(className="col-lg-4", children=[
                    html.H6('Selected Metrics'),
                    dcc.Checklist(
                        id='metrics',
                        options=[{'label': m, 'value': m} for m in ['Confirmed', 'Deaths', 'Recovered']],
                        value=['Confirmed', 'Deaths'],
    

                    )
                ])
            ]),
            dcc.Graph(
                id="plot_new_metrics",
                config={ 'displayModeBar': False }
            ),
            dcc.Graph(
                id="plot_cum_metrics",
                config={ 'displayModeBar': False }
            )
        ]))



barplot = dbc.Jumbotron(
    dbc.Container(
        [
            dbc.Row(
                    [
                        dbc.Col(
                                dcc.Graph(id='scatterplot2', figure = my_fig)
                               
                                ),
                        dbc.Col(
                                dcc.Graph(id='scatterplot3', figure = my_fig2)
                                
                            )
                    ],
                    #style
                ),
            dbc.Row(
                    [
                        dbc.Col(
                                dcc.Graph(id='scatterplot4', figure = fig_fatality)
                             )
                        
                    ],
                    className="mt-3"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                                dcc.Graph(id='scatterplot5', figure = fig_nasional)
                             )
                        
                    ],
                    className="mt-3"
                ),       
                dbc.Row(
                    [
                        dbc.Col(
                                dcc.Graph(id='scatterplot6', figure = fig_prov)
                             )
                        
                    ],
                    className="mt-3"
                ),                                      
               
        ]
    ), 
    
    )


#FOOTER
footer = dbc.NavbarSimple(
    brand="© 2020 IPB University, All Rights Reserved.",
    className="navbar navbar-expand-lg navbar-dark bg-dark",
)


app.layout = html.Div(
    [navbar, jumbotron, case_data, barplot,card, carddua, footer]
)

@app.callback(
    [Output('state', 'options'), Output('state', 'value')],
    [Input('country', 'value')])

def update_states(country):
    states = list(allData.loc[allData['Country/Region'] == country]['Province/State'].unique())
    states.insert(0, '<all>')
    states.sort()
    state_options = [{'label':s, 'value':s} for s in states]
    state_value = state_options[0]['value']
    return state_options, state_value

def nonreactive_data(country, state):
    data = allData.loc[allData['Country/Region'] == country]
    if state == '<all>':
        data = data.drop('Province/State', axis=1).groupby("date").sum().reset_index()
    else:
        data = data.loc[data['Province/State'] == state]
    newCases = data.select_dtypes(include='int64').diff().fillna(0)
    newCases.columns = [column.replace('Cum', 'New') for column in newCases.columns]
    data = data.join(newCases)
    data['dateStr'] = data['date'].dt.strftime('%b %d, %Y')
    return data

def barchart(data, metrics, prefix="", yaxisTitle=""):
    figure = go.Figure(data=[
        go.Bar( 
            name=metric, x=data.date, y=data[prefix + metric],
            marker_line_color='rgb(0,0,0)', marker_line_width=1,
            marker_color={ 'Deaths':'rgb(200,30,30)', 'Recovered':'rgb(30,200,30)', 'Confirmed':'rgb(100,140,240)'}[metric]
        ) for metric in metrics
    ])
    figure.update_layout( 
              barmode='group', legend=dict(x=.05, y=0.95, font={'size':15}, bgcolor='rgba(240,240,240,0.5)'), 
              plot_bgcolor='#FFFFFF', font=tickFont) \
          .update_xaxes( 
              title="", tickangle=-90, type='category', showgrid=True, gridcolor='#DDDDDD', 
              tickfont=tickFont, ticktext=data.dateStr, tickvals=data.date) \
          .update_yaxes(
              title=yaxisTitle, showgrid=True, gridcolor='#DDDDDD')
    return figure

@app.callback(
    Output('plot_new_metrics', 'figure'), 
    [Input('country', 'value'), Input('state', 'value'), Input('metrics', 'value')]
)

def update_plot_new_metrics(country, state, metrics):
    data = nonreactive_data(country, state)
    return barchart(data, metrics, prefix="New", yaxisTitle="New Cases per Day")

@app.callback(
    Output('plot_cum_metrics', 'figure'), 
    [Input('country', 'value'), Input('state', 'value'), Input('metrics', 'value')]
)
def update_plot_cum_metrics(country, state, metrics):
    data = nonreactive_data(country, state)
    return barchart(data, metrics, prefix="Cum", yaxisTitle="Cumulated Cases")


def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)



if __name__ == "__main__":
    app.run_server(debug=True, port=2323)