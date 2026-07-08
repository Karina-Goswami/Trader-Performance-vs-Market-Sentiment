import streamlit as st
import pandas as pd
import plotly.express as px
import pickle

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Trader Performance Dashboard",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():

    trades = pd.read_csv("historical_data.csv")
    sentiment = pd.read_csv("fear_greed_index.csv")

    trades["Timestamp IST"] = pd.to_datetime(
        trades["Timestamp IST"],
        dayfirst=True
    )

    trades["Date"] = trades["Timestamp IST"].dt.date

    sentiment["Date"] = pd.to_datetime(
        sentiment["date"]
    ).dt.date

    df = trades.merge(
        sentiment[["Date", "classification", "value"]],
        on="Date",
        how="left"
    )

    df["Win"] = df["Closed PnL"] > 0
    df["Profitable"] = (
        df["Closed PnL"] > 0
    ).astype(int)

    return df


df = load_data()

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():

    with open("profitability_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    with open("model_accuracy.pkl", "rb") as f:
        accuracy = pickle.load(f)

    return model, model_columns, accuracy


model, model_columns, accuracy = load_model()

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", " Prediction"]
)

# ==========================================================
# OVERVIEW PAGE
# ==========================================================
if page == "Overview":

    st.title(" Trader Performance vs Market Sentiment")

    sentiment_options = ["All"] + sorted(
        df["classification"].dropna().unique().tolist()
    )

    coin_options = ["All"] + sorted(
        df["Coin"].dropna().unique().tolist()
    )

    st.sidebar.header("Filters")

    selected_sentiment = st.sidebar.selectbox(
        "Market Sentiment",
        sentiment_options
    )

    selected_coin = st.sidebar.selectbox(
        "Coin",
        coin_options
    )

    filtered_df = df.copy()

    if selected_sentiment != "All":
        filtered_df = filtered_df[
            filtered_df["classification"]
            == selected_sentiment
        ]

    if selected_coin != "All":
        filtered_df = filtered_df[
            filtered_df["Coin"]
            == selected_coin
        ]

    # ======================================================
    # KPI CARDS
    # ======================================================
    total_trades = len(filtered_df)
    total_pnl = filtered_df["Closed PnL"].sum()
    win_rate = filtered_df["Win"].mean() * 100
    avg_trade = filtered_df["Size USD"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Trades",
        f"{total_trades:,}"
    )

    c2.metric(
        "Total PnL",
        f"${total_pnl:,.2f}"
    )

    c3.metric(
        "Win Rate",
        f"{win_rate:.2f}%"
    )

    c4.metric(
        "Avg Trade Size",
        f"${avg_trade:,.2f}"
    )

    # ======================================================
    # PNL DISTRIBUTION
    # ======================================================
    st.subheader("PnL Distribution by Market Sentiment")

    fig = px.box(
        filtered_df,
        x="classification",
        y="Closed PnL",
        color="classification"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================================
    # TRADE FREQUENCY
    # ======================================================
    st.subheader("Trade Frequency")

    trade_freq = (
        filtered_df.groupby("Date")
        .size()
        .reset_index(name="Trades")
    )

    fig = px.line(
        trade_freq,
        x="Date",
        y="Trades",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================================
    # LONG VS SHORT
    # ======================================================
    st.subheader("Long vs Short Positions")

    long_short = (
        filtered_df.groupby(
            ["classification", "Side"]
        )
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        long_short,
        x="classification",
        y="Count",
        color="Side",
        barmode="group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================================
    # TOP TRADERS
    # ======================================================
    st.subheader("Top 10 Traders")

    top_traders = (
        filtered_df.groupby("Account")
        ["Closed PnL"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.dataframe(
        top_traders,
        use_container_width=True
    )

    # ======================================================
    # DOWNLOAD BUTTON
    # ======================================================
    csv = filtered_df.to_csv(index=False)

    st.download_button(
        "Download Filtered Data",
        csv,
        "filtered_data.csv",
        "text/csv"
    )

# ==========================================================
# PREDICTION PAGE
# ==========================================================
elif page == " Prediction":

    st.title(" Trade Profitability Prediction")

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

    col1, col2 = st.columns(2)

    with col1:

        sentiment_value = st.slider(
            "Fear & Greed Value",
            0,
            100,
            50
        )

        trade_size = st.number_input(
            "Trade Size (USD)",
            min_value=0.0,
            value=1000.0
        )

        execution_price = st.number_input(
            "Execution Price",
            min_value=0.0,
            value=100000.0
        )

    with col2:

        fee = st.number_input(
            "Fee",
            min_value=0.0,
            value=1.0
        )

        sentiment = st.selectbox(
            "Market Sentiment",
            sorted(
                df["classification"]
                .dropna()
                .unique()
            )
        )

        side = st.selectbox(
            "Side",
            sorted(
                df["Side"]
                .dropna()
                .unique()
            )
        )

        direction = st.selectbox(
            "Direction",
            sorted(
                df["Direction"]
                .dropna()
                .unique()
            )
        )

        coin = st.selectbox(
            "Coin",
            sorted(
                df["Coin"]
                .dropna()
                .unique()
            )
        )

    if st.button("Predict Profitability"):

        input_df = pd.DataFrame({
            "value": [sentiment_value],
            "Size USD": [trade_size],
            "Execution Price": [execution_price],
            "Fee": [fee],
            "classification": [sentiment],
            "Side": [side],
            "Direction": [direction],
            "Coin": [coin]
        })

        input_df = pd.get_dummies(input_df)

        input_df = input_df.reindex(
            columns=model_columns,
            fill_value=0
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.success(
                f"Likely Profitable ✅\n\nProbability: {probability[1]*100:.2f}%"
            )
        else:
            st.error(
                f"Likely Not Profitable ❌\n\nProbability: {probability[0]*100:.2f}%"
            )