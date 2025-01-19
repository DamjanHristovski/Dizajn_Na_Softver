import pandas as pd
from sqlalchemy import create_engine
import pandas_ta as ta
import streamlit as st
from ta.trend import MACD
from ta.momentum import StochasticOscillator
import plotly.graph_objects as go
import numpy as np
import subprocess

import os
from django.conf import settings

#Make the data into a dataframe for each company_id / code
#db_engine = create_engine('mysql+pymysql://root:#Sedi_Madro_Da_Ne$BudeModro69@localhost/berza_data')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# def fetch_data_by_code(company_code):
#     query = f"""
#     SELECT
#         d.date, d.last_transaction, d.max_price, d.min_price, d.average_price, d.volume
#     FROM
#         daily_data d
#     JOIN
#         company c
#     ON
#         d.company_id = c.company_id
#     WHERE
#         c.name = '{company_code}';
#     """
#     return pd.read_sql(query, db_engine)



# Read query parameters
query_params = st.query_params
code = query_params.get("company_code")

# Ensure no unintended trimming or splitting
if code is not None:
    code = str(code).strip()  # Remove unnecessary spaces

# Validate the company code
if not code:
    st.error("No company code provided in the query parameters.")
else:
    st.write(f"Company code: {code}")

#daily_data_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'testing_table.csv')
# = pd.read_csv(daily_data_path)
#daily_data = pd.read_csv(r"C:\Users\matej\Desktop\dias_project\stock_predictor\stockapp\data\testing_table.csv")\
    daily_data_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'testing_table.csv')
    daily_data = pd.read_csv(daily_data_path)
daily_data = daily_data[daily_data['name'] == code]
daily_data.last_transaction = daily_data.last_transaction.fillna(daily_data.last_transaction.median())
daily_data.max_price = daily_data.max_price.fillna(daily_data.max_price.median())
daily_data.min_price = daily_data.min_price.fillna(daily_data.min_price.median())
daily_data.average_price = daily_data.average_price.fillna(daily_data.average_price.median())

daily_data.rename(columns={
    'last_transaction': 'close',
    'max_price': 'high',
    'min_price': 'low',
    'average_price': 'open',
    'volume': 'volume'
}, inplace=True)


def calculate_indicators(data):
    data['RSI'] = ta.rsi(data['close'], length=14)
    macd_object = MACD(data['close'])
    data['MACD line'] = macd_object.macd()
    data['MACD signal'] = macd_object.macd_signal()
    stochastic = StochasticOscillator(
        high=data['high'],
        low=data['low'],
        close=data['close'],
        window=14,
        smooth_window=3
    )
    data['Stochastic_K'] = stochastic.stoch()
    data['Stochastic_D'] = stochastic.stoch_signal()
    data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=20)
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)

    # Moving Averages
    data['SMA_20'] = ta.sma(data['close'], length=20)
    data['EMA_10'] = ta.ema(data['close'], length=10)
    data['WMA_15'] = ta.wma(data['close'], length=15)
    data['HMA_30'] = ta.hma(data['close'], length=30)
    data['TEMA_9'] = ta.tema(data['close'], length=9)
    return data


daily_data = daily_data.sort_values(by="date", ascending=True)
daily_data = calculate_indicators(daily_data)
daily_data = daily_data.fillna(0)


daily_data['date'] = pd.to_datetime(daily_data['date'])
daily_data = daily_data.set_index('date')
weekly_data = daily_data.resample('W-MON').agg({
    'close': 'last',
    'high': 'max',
    'low': 'min',
    'open': 'first',
    'volume': 'sum'
})

weekly_data = calculate_indicators(weekly_data)
daily_data = daily_data.fillna(0)

monthly_data = daily_data.resample('ME').agg({
    'close': 'last',
    'high': 'max',
    'low': 'min',
    'open': 'first',
    'volume': 'sum'
})

monthly_data = calculate_indicators(monthly_data)
daily_data = daily_data.fillna(0)



def make_detailed_decision(data):
    last_row = data.iloc[-1]
    if pd.isna(last_row['ATR']):
        last_row['ATR'] = 0
    if pd.isna(last_row['RSI']):
        last_row['RSI'] = 0
    if pd.isna(last_row['MACD line']):
        last_row['MACD line'] = 0
    if pd.isna(last_row['MACD signal']):
        last_row['MACD signal'] = 0
    if pd.isna(last_row['Stochastic_K']):
        last_row['Stochastic_K'] = 0
    if pd.isna(last_row['CCI']):
        last_row['CCI'] = 0
    if pd.isna(last_row['EMA_10']):
        last_row['EMA_10'] = 0
    if pd.isna(last_row['SMA_20']):
        last_row['SMA_20'] = 0
    if pd.isna(last_row['WMA_15']):
        last_row['WMA_15'] = 0
    if pd.isna(last_row['HMA_30']):
        last_row['HMA_30'] = 0
    if pd.isna(last_row['TEMA_9']):
        last_row['TEMA_9'] = 0

    if not np.isfinite(last_row['Stochastic_K']):
        last_row['Stochastic_K'] = 0

    atr_threshold_high = last_row['ATR'] * 1.5
    atr_threshold_low = last_row['ATR'] * 0.5

    decisions = {'RSI': f"{last_row['RSI']:.2f} - " + (
        'Buy' if last_row['RSI'] < 30 else ('Sell' if last_row['RSI'] > 70 else 'Neutral')),
                 'MACD': f"{last_row['MACD line']:.2f} (Signal: {last_row['MACD signal']:.2f}) - " + (
                     'Buy' if last_row['MACD line'] > last_row['MACD signal'] else (
                         'Sell' if last_row['MACD line'] < last_row['MACD signal'] else 'Neutral')),
                 'Stochastic': f"{last_row['Stochastic_K']:.2f} - " + (
                     'Buy' if last_row['Stochastic_K'] < 20 and last_row['Stochastic_K'] != 0 else (
                         'Sell' if last_row['Stochastic_K'] > 80 else 'Neutral')),
                 'CCI': f"{last_row['CCI']:.2f} - " + (
                     'Buy' if last_row['CCI'] < -100 else ('Sell' if last_row['CCI'] > 100 else 'Neutral')),
                 'ATR': f"{last_row['ATR']:.2f} - " + (
                     'High Volatility' if last_row['ATR'] > atr_threshold_high else (
                         'Low Volatility' if last_row['ATR'] < atr_threshold_low else 'Neutral')),
                 'EMA vs SMA': f"EMA_10: {last_row['EMA_10']:.2f}, SMA_20: {last_row['SMA_20']:.2f} - " + (
                     'Buy' if last_row['EMA_10'] > last_row['SMA_20'] else (
                         'Sell' if last_row['EMA_10'] < last_row['SMA_20'] else 'Neutral')),
                 'WMA': f"{last_row['WMA_15']:.2f} - Neutral", 'HMA': f"{last_row['HMA_30']:.2f} - Neutral",
                 'TEMA': f"{last_row['TEMA_9']:.2f} - Neutral"}

    buy_count = sum(1 for decision in decisions.values() if 'Buy' in decision)
    sell_count = sum(1 for decision in decisions.values() if 'Sell' in decision)
    neutral_count = sum(1 for decision in decisions.values() if 'Neutral' in decision)

    if buy_count > sell_count and buy_count > neutral_count:
        final_decision = 'Buy'
    elif sell_count > buy_count and sell_count > neutral_count:
        final_decision = 'Sell'
    elif neutral_count == buy_count and neutral_count > sell_count:
        final_decision = 'Neutral towards Buy'
    elif neutral_count == sell_count and neutral_count > buy_count:
        final_decision = 'Neutral towards Sell'
    else:
        final_decision = 'Neutral'

    return decisions, final_decision


daily_decisions, daily_final = make_detailed_decision(daily_data)
weekly_decisions, weekly_final = make_detailed_decision(weekly_data)
monthly_decisions, monthly_final = make_detailed_decision(monthly_data)

st.title("Stock Analysis Dashboard")

option = st.radio("Select Timeframe:", ["Daily", "Weekly", "Monthly"])

if option == "Daily":
    selected_decisions = daily_decisions
    final_decision = daily_final
elif option == "Weekly":
    selected_decisions = weekly_decisions
    final_decision = weekly_final
else:
    selected_decisions = monthly_decisions
    final_decision = monthly_final

# Display Oscillators and Indicators
st.header(f"Decisions for {option}")
st.write("### Individual Oscillator/Indicator Decisions")
for key, value in selected_decisions.items():
    st.write(f"- {key}: {value}")

fig = go.Figure(data=[go.Candlestick(
    x=daily_data.index,
    open=daily_data['open'],
    high=daily_data['high'],
    low=daily_data['low'],
    close=daily_data['close'],
    name='Candlestick'
)])

fig.update_layout(
    title=f"Price Chart for {code}",
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig)

if final_decision == "Buy":
    box_color = "green"
    box_text = f"<div style='background-color:{box_color}; padding:20px; text-align:center; font-size:20px; color:white; border-radius:10px; font-weight:bold;'>Final Decision: {final_decision}</div>"
elif final_decision == "Sell":
    box_color = "red"
    box_text = f"<div style='background-color:{box_color}; padding:20px; text-align:center; font-size:20px; color:white; border-radius:10px; font-weight:bold;'>Final Decision: {final_decision}</div>"
else:
    box_color = "orange"
    box_text = f"<div style='background-color:{box_color}; padding:20px; text-align:center; font-size:20px; color:white; border-radius:10px; font-weight:bold;'>Final Decision: {final_decision}</div>"

st.markdown(box_text, unsafe_allow_html=True)

