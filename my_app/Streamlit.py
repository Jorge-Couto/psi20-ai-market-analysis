import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

from Forecasting import run_forecast_from_csv
from Agents_Runner import run_agents_for_company  # ✅ FIXED NAME

# -----------------------
# CONFIG
# -----------------------
PSI20_TICKERS = {
    "EDP": "EDP.LS",
    "EDP_Renovaveis": "EDPR.LS",
    "Galp_Energia": "GALP.LS",
    "Banco_Comercial_Portugues": "BCP.LS",
    "Jeronimo_Martins": "JMT.LS",
    "Sonae": "SON.LS",
    "NOS_SGPS": "NOS.LS",
    "REN": "RENE.LS",
    "The_Navigator": "NVG.LS",
    "Semapa": "SEM.LS",
    "Mota_Engil": "EGL.LS",
    "Altri": "ALTR.LS",
    "CTT_Correios_de_Portugal": "CTT.LS",
    "Corticeira_Amorim": "COR.LS",
    "Ibersol": "IBS.LS",
    "Novabase": "NBA.LS",
    "Pharol": "PHR.LS"
}

# -----------------------
# HELPERS
# -----------------------
def clean_yfinance_df(df):
    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])

    return df


@st.cache_data
def fetch_data(ticker):
    df = yf.download(
        ticker,
        start="2010-01-01",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return None

    return clean_yfinance_df(df)


# Cache FORECAST (heavy)
@st.cache_resource
def run_forecast_cached(company, path):
    return run_forecast_from_csv(path)


# Cache AGENTS (heavy)
@st.cache_resource
def run_agents_cached(company):
    return run_agents_for_company(company)


# -----------------------
# SESSION STATE INIT
# -----------------------
if "forecast_cache" not in st.session_state:
    st.session_state["forecast_cache"] = {}

if "agents_cache" not in st.session_state:
    st.session_state["agents_cache"] = {}

if "agents_result" not in st.session_state:
    st.session_state["agents_result"] = None


# -----------------------
# UI
# -----------------------
st.title("PSI-20 Stock Data Viewer")

selected_company = st.selectbox(
    "Select a company:",
    list(PSI20_TICKERS.keys())
)

ticker = PSI20_TICKERS[selected_company]
df = fetch_data(ticker)

if df is None:
    st.error("No data found.")
    st.stop()

st.success(f"{selected_company} data loaded")

# -----------------------
# TIMEFRAME
# -----------------------
timeframe = st.selectbox(
    "Select timeframe:",
    ["1W", "1M", "3M", "6M", "1Y", "5Y", "All", "Custom"]
)

latest_date = df["Date"].max()

if timeframe == "Custom":
    start_date = st.date_input("Start date", df["Date"].min())
    end_date = st.date_input("End date", latest_date)

    filtered_df = df[
        (df["Date"] >= pd.to_datetime(start_date)) &
        (df["Date"] <= pd.to_datetime(end_date))
    ]
else:
    if timeframe == "1W":
        start_date = latest_date - timedelta(weeks=1)
    elif timeframe == "1M":
        start_date = latest_date - timedelta(days=30)
    elif timeframe == "3M":
        start_date = latest_date - timedelta(days=90)
    elif timeframe == "6M":
        start_date = latest_date - timedelta(days=180)
    elif timeframe == "1Y":
        start_date = latest_date - timedelta(days=365)
    elif timeframe == "5Y":
        start_date = latest_date - timedelta(days=365 * 5)
    else:
        start_date = df["Date"].min()

    filtered_df = df[df["Date"] >= start_date]

# -----------------------
# DISPLAY DATA
# -----------------------
st.subheader("Filtered Data")
st.dataframe(filtered_df, use_container_width=True)

fig = px.line(
    filtered_df,
    x="Date",
    y="Close",
    title=f"{selected_company} Stock Price"
)
st.plotly_chart(fig, use_container_width=True)


# -----------------------
# FORECAST BUTTON
# -----------------------
if st.button("Run Forecast", key=f"forecast_{selected_company}"):

    if selected_company in st.session_state["forecast_cache"]:
        st.info("Using cached forecast")
        forecast_df, df_ready = st.session_state["forecast_cache"][selected_company]

    else:
        with st.spinner("Training model... this may take a while"):
            forecast_df, df_ready = run_forecast_cached(
                selected_company,
                f"data/{selected_company}_Stock_Price.csv"
            )

        # ✅ SAVE to cache
        st.session_state["forecast_cache"][selected_company] = (forecast_df, df_ready)

        # ✅ ALSO save to CSV for agents
        forecast_df.to_csv(f"forecasts/{selected_company}_forecast.csv", index=False)

    # active session
    st.session_state["forecast_df"] = forecast_df
    st.session_state["df_ready"] = df_ready


# -----------------------
# SHOW FORECAST
# -----------------------
if "forecast_df" in st.session_state:

    forecast_df = st.session_state["forecast_df"]
    df_ready = st.session_state["df_ready"]

    st.subheader("Forecast Data")
    st.dataframe(forecast_df)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_ready.index[-60:],
        y=df_ready["Close"].iloc[-60:],
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecast_price"],
        name="Forecast"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["upper_ci"],
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["lower_ci"],
        fill='tonexty',
        name="Confidence Interval"
    ))

    st.plotly_chart(fig, use_container_width=True)


# -----------------------
# AI AGENTS
# -----------------------
st.subheader("AI Investment Analysis")

if st.button("Run AI Analysis", key=f"agents_{selected_company}"):

    if selected_company not in st.session_state["forecast_cache"]:
        st.error("Run forecast first before AI analysis.")
        st.stop()

    if selected_company in st.session_state["agents_cache"]:
        st.info("Using cached AI analysis")
        result = st.session_state["agents_cache"][selected_company]

    else:
        with st.spinner("Running AI agents (30–60s)..."):
            result = run_agents_cached(selected_company)

        st.session_state["agents_cache"][selected_company] = result

    st.session_state["agents_result"] = result


# -----------------------
# DISPLAY AI RESULT
# -----------------------
if st.session_state["agents_result"] is not None:

    st.subheader("Investment Report")

    st.text_area(
        "AI Output",
        st.session_state["agents_result"],
        height=500
    )

    # Optional formatted version
#    def format_report(text):
#        sections = text.split("\n\n")
#        for section in sections:
#            st.markdown(section)

#    format_report(st.session_state["agents_result"])