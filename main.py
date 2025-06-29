import taipy as  tp 
import taipy.gui.builder as tgb
from taipy.gui import Icon
from taipy import Config
import datetime
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np

company_data = pd.read_csv("data/sp500_companies.csv")
stock_data = pd.read_csv("data/sp500_stocks.csv")
stock_data['Adj Close'] = pd.to_numeric(stock_data['Adj Close'], errors='coerce')
stock_data.dropna(subset=['Adj Close'], inplace=True)
country_names = company_data["Country"].unique().tolist()
country_names = [(name, Icon("images/flags/"+name+".png", name)) for name in country_names]
country_names.insert(0, ("All Countries", Icon("images/flags/all.png", "All Countries")))
company_symbols_stock_data= set(stock_data["Symbol"].unique().tolist())
company_data= company_data[company_data["Symbol"].isin(company_symbols_stock_data)]
company_names = company_data[
    ["Symbol", "Shortname"]
    ].sort_values("Shortname").values.tolist()

#################contsants######################
dates= [
    datetime.date(2013,1,1),
    datetime.date(2024,1,1)
    ]
country = "United States"
company = ['AOS']
graph_data = None
figure = None

lin_pred = 0
knn_pred = 0
rnn_pred = 0

with tgb.Page() as page:
    with tgb.part("text-center"):
        tgb.image("images/icons/logo.png",width = "10vw")
        tgb.text(
            "## S&P stock value over time", 
            mode = "md"
            )
        tgb.date_range( 
            "{dates}",
            label_start = "Start Date",
            label_end = "End Date"
        )
    with tgb.layout("20 80"):
        tgb.selector(
            label = "Country",
            class_name = "fullwidth",
            value="{country}",
            lov = "{country_names}",
                dropdown = True,
                value_by_id=True
        )
        
        tgb.selector(
            label = "Company",
            class_name = "fullwidth",
            value="{company}",
            lov = "{company_names}",
                dropdown = True,
                value_by_id=True,
                multiple = True
        )
    tgb.chart(figure = "{figure}")
    with tgb.part("text-left"):
        with tgb.layout( " 4 72 4 4 4 4 4 4 "):
            tgb.image(
                "images/icons/id-card.png",
                width = "3vw"
            )
            tgb.text ( "{company[-1]}",mode = "md")
            tgb.image(
                "images/icons/lin.png",
                width = "3vw"
            )
            tgb.text ( "{lin_pred}",mode = "md")
            tgb.image(
                "images/icons/knn.png",
                width = "3vw"
            )
            tgb.text ( "{knn_pred}",mode = "md")
            tgb.image(
                "images/icons/rnn.png",
                width = "3vw"
            )
            tgb.text ( "{rnn_pred}",mode = "md")


def build_company_names(country):
    if country == "All Countries":
        company_names = company_data[
            ["Symbol", "Shortname"]].sort_values("Shortname").values.tolist()
        return company_names               
    else:
        company_names = company_data[
        ["Symbol", "Shortname"]][
            company_data["Country"]==country
        ].sort_values("Shortname").values.tolist()
    return company_names

def build_graph_data(dates, company):
    temp_data = stock_data[["Date","Adj Close", "Symbol"]][
            (stock_data["Date"] > str(dates[0]) ) & 
            (stock_data["Date"] < str (dates[1]))&
            (stock_data["Symbol"].isin(company))
                ]
    graph_data = temp_data["Date"]
    for i in company:
        graph_data_temp = pd.DataFrame()
        tempdf= temp_data[temp_data["Symbol"]==i]
        graph_data_temp["Date"]=tempdf["Date"]
        graph_data_temp[i]=tempdf["Adj Close"]
        graph_data = pd.concat([graph_data, graph_data_temp[i]], axis=1)
    return graph_data

def display_graph(graph_data):
    figure = go.Figure()
    symbols = graph_data.columns[1:]
    for i in symbols:
        figure.add_trace(go.Scatter(
            x=graph_data["Date"],
            y=graph_data[i],
            name = i,
            showlegend=True
        ))
    figure.update_layout(
        xaxis_title = "Date",
        yaxis_title = "Stock Value"
    )
    return figure

def split_data(stock_data, dates, symbols):
    temp_data = stock_data[
            (stock_data["Date"] > str(dates[0]) ) & 
            (stock_data["Date"] < str (dates[1]))&
            (stock_data["Symbol"]==symbols)
                ].drop(["Date", "Symbol"], axis=1)
    eval_features = temp_data.values[-1]
    eval_features = eval_features.reshape(1, -1)
    features = temp_data.values[:-1]
    targets= temp_data["Adj Close"].shift(-1).values[:-1]
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / std
    eval_features = (eval_features - mean) / std
    return features, targets, eval_features

def get_lin(dates, company):
    print(company[-1])
    x, y, eval_x = split_data(stock_data, dates, company[-1])
    lin_model.fit(x, y)
    lin_pred = lin_model.predict(eval_x)
    return round(lin_pred[0],3)

def get_knn(dates, company):
    print(company)
    x, y, eval_x = split_data(stock_data, dates, company[-1])
    knn_model.fit(x, y)
    knn_pred = knn_model.predict(eval_x)
    return round(knn_pred[0],3)

def get_rnn(dates, company):
    print(company)
    x, y, eval_x = split_data(stock_data, dates, company[-1])
    # x=np.reshape(x, ( x.shape[0], x.shape[1]), 1)  # Reshape for RNN input
    print(x)
    rnn_model.fit(x, y, batch_size=32, epochs=10, verbose=0)
    rnn_pred = rnn_model.predict(eval_x)
    print(type(rnn_pred[0][0]))
    return round(float(rnn_pred[0][0]),3)

def on_init(state, name, value):
    state.scenario.country.write(state.country)
    state.scenario.dates.write(state.dates)
    state.scenario.company.write(state.company)
    state.scenario.submit(wait=True)
    state.graph_data = state.scenario.graph_data.read()
    state.company_names = state.scenario.company_names.read()
    state.lin_pred = state.scenario.lin_pred.read()
    state.knn_pred = state.scenario.knn_pred.read()
    state.rnn_pred = state.scenario.rnn_pred.read()

def on_change(state, name, value):
    if name == "country":
        state.scenario.country.write(state.country)
        state.scenario.submit(wait=True)
        state.company_names = state.scenario.company_names.read()
    if name == "company" or name == "dates":
        state.scenario.dates.write(state.dates)
        state.scenario.company.write(state.company)
        state.scenario.submit(wait=True)
        state.graph_data = state.scenario.graph_data.read()
        state.lin_pred = state.scenario.lin_pred.read()
        state.knn_pred = state.scenario.knn_pred.read()
        state.rnn_pred = state.scenario.rnn_pred.read()
    if name == "graph_data":
        state.figure = display_graph(state.graph_data)

def build_RNN(n_features):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(n_features,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


country_cfg=Config.configure_data_node(
    id="country"
    )
company_names_cfg=Config.configure_data_node(
    id="company_names"
    )

dates_cfg=Config.configure_data_node(
    id="dates"
    )
company_cfg=Config.configure_data_node(
    id="company"
    )
graph_data_cfg=Config.configure_data_node(
    id="graph_data"
    )
lin_pred_cfg=Config.configure_data_node(
    id="lin_pred"
    )
knn_pred_cfg=Config.configure_data_node(
    id="knn_pred"
    )
rnn_pred_cfg=Config.configure_data_node(
    id="rnn_pred"
    )


build_company_names_cfg= Config.configure_task(
    input = country_cfg,
    output = company_names_cfg,
    function = build_company_names,
    id = "build_company_names",
    skippable = True
)


build_graph_data_cfg= Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = graph_data_cfg,
    function = build_graph_data,
    id = "build_graph_data",
    skippable = True
)

get_lin_cfg= Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = lin_pred_cfg,
    function = get_lin,
    id = "get_lin",
    skippable = True
)
get_knn_cfg= Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = knn_pred_cfg,
    function = get_knn,
    id = "get_knn",
    skippable = True
)
get_rnn_cfg= Config.configure_task(
    input = [dates_cfg, company_cfg],
    output = rnn_pred_cfg,
    function = get_rnn,
    id = "get_rnn",
    skippable = True
)

scenario_cfg = Config.configure_scenario(
    task_configs=[build_company_names_cfg,
                build_graph_data_cfg,
                get_lin_cfg,
                get_knn_cfg,
                get_rnn_cfg],
    id = "scenario"
)

if __name__ == "__main__":
    lin_model = LinearRegression()
    knn_model = KNeighborsRegressor(n_neighbors=5)
    rnn_model = build_RNN(6)
    tp.Orchestrator().run()
    scenario = tp.create_scenario(scenario_cfg)
    gui = tp.Gui(page)
    gui.run(title="S&P Stock Value", use_reloader=True)