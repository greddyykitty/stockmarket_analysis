import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# App title
st.set_page_config(page_title="Stock Market Analysis Tool", layout="wide")
st.title("Stock Price & Prediction Tool")

# Stock selection moved to main page
stock_symbol_input = st.text_input("Enter Stock Ticker", "AAPL").upper()

# --- Stock Data Function ---
def fetch_stock_history(ticker_symbol, time_period="5y"):
    stock_ticker_obj = yf.Ticker(ticker_symbol)
    stock_history_data = stock_ticker_obj.history(period=time_period, auto_adjust=True)
    if stock_history_data.empty:
        st.warning(f"No data found for {ticker_symbol} for the period {time_period}. Check ticker.")
    return stock_history_data

# --- Display Current Price ---
st.header(f"Current Stock Price for {stock_symbol_input}")
stock_info = yf.Ticker(stock_symbol_input)
latest_price = None
try:
    # Fetch last 2 days to increase chances of getting data
    price_data_2d = stock_info.history(period='2d', auto_adjust=True)
    # st.dataframe(current_data) # DEBUG - Removed for now
    if not price_data_2d.empty and 'Close' in price_data_2d.columns and not price_data_2d['Close'].isnull().all():
        # Get the last available closing price
        latest_price = price_data_2d['Close'].dropna().iloc[-1]
        st.write(f"Latest Closing Price: ${latest_price:.2f}")
    else:
        st.error(f"Could not fetch current closing price data for {stock_symbol_input}. Check ticker or data availability.")
# Catch potential errors during data fetching or processing
except Exception as error_msg:
    st.error(f"An error occurred while fetching current price for {stock_symbol_input}: {error_msg}")


# --- Historical Price Chart ---
st.header(f"Historical Price Chart for {stock_symbol_input}")
# Fetch 5 years of data for the historical chart and prediction
historical_data = fetch_stock_history(stock_symbol_input, time_period="5y")

if not historical_data.empty and 'Close' in historical_data.columns:
    history_figure = go.Figure()
    history_figure.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Closing Price'))
    history_figure.update_layout(title=f"{stock_symbol_input} Historical Closing Prices (5Y)", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(history_figure)
else:
    st.warning("Could not display historical price chart due to missing data.")


# --- Stock Price Prediction ---
st.header("Stock Price Prediction (using Linear Regression)")

# Use the already fetched historical_data
if not historical_data.empty:
    # Feature Engineering
    prediction_input_data = historical_data.copy() # Work on a copy
    prediction_input_data['Date_Num'] = prediction_input_data.index.map(pd.Timestamp.toordinal)
    prediction_input_data['MA5'] = prediction_input_data['Close'].rolling(window=5).mean()
    prediction_input_data['MA20'] = prediction_input_data['Close'].rolling(window=20).mean()
    prediction_input_data.dropna(inplace=True)

    # Prepare data for Linear Regression
    feature_columns = ['Date_Num', 'MA5', 'MA20']
    feature_set = prediction_input_data[feature_columns]
    target_variable = prediction_input_data['Close']

    if len(feature_set) > 1: # Need at least 2 samples for train_test_split
        X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(feature_set, target_variable, test_size=0.2, random_state=42, shuffle=False) # Keep chronological order

        # Model Training
        regression_model = LinearRegression()
        regression_model.fit(X_train_data, y_train_data)

        # Evaluation
        test_predictions = regression_model.predict(X_test_data)
        root_mean_sq_error = np.sqrt(mean_squared_error(y_test_data, test_predictions))

        st.subheader("Model Evaluation")
        st.write(f"Model Used: Linear Regression")
        st.write(f"Features Used: {', '.join(feature_columns)}")
        st.write(f"Root Mean Squared Error (RMSE) on Test Set: ${root_mean_sq_error:.2f}")

        # Prediction
        prediction_days = st.slider("Days to Predict into the Future", 1, 30, 10)
        last_date_num = feature_set['Date_Num'].iloc[-1]
        last_ma5_val = prediction_input_data['MA5'].iloc[-1]
        last_ma20_val = prediction_input_data['MA20'].iloc[-1]

        future_ordinal_dates = [last_date_num + i + 1 for i in range(prediction_days)]
        future_feature_data = pd.DataFrame({
            'Date_Num': future_ordinal_dates,
            'MA5': [last_ma5_val] * prediction_days,
            'MA20': [last_ma20_val] * prediction_days
        })

        future_predicted_prices = regression_model.predict(future_feature_data[feature_columns])
        future_actual_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(date_num)) for date_num in future_ordinal_dates])

        future_predictions_df = pd.DataFrame({'Date': future_actual_dates, 'Predicted Price': future_predicted_prices})

        st.subheader("Predicted Prices Table")
        st.dataframe(future_predictions_df.set_index('Date'))

        # Plotting ONLY Predicted Prices
        st.subheader("Prediction Graph (Future Only)")
        prediction_figure = go.Figure()
        prediction_figure.add_trace(go.Scatter(x=future_predictions_df['Date'], y=future_predictions_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(dash='dash')))
        prediction_figure.update_layout(title=f"{stock_symbol_input} Future Price Prediction", xaxis_title="Date", yaxis_title="Predicted Price")
        st.plotly_chart(prediction_figure)

    else:
        st.warning("Not enough data to train the prediction model after feature engineering.")
else:
    st.error("Cannot perform prediction or display historical chart due to missing stock data.")

# Removed other menu sections
