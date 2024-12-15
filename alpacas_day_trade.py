import asyncio
import pandas as pd
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca_trade_api.rest import REST, TimeFrame as TimeFrameRest
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import holidays
import smtplib
import os
import time
from pytz import timezone
import yfinance as yf

email_user = 'jevanchahal1@yahoo.com'
email_password = 'plhokhnqrhzaaegu'
email_send = 'jevanchahal1@gmail.com'
sms_send = '9168492930@txt.att.net'  # AT&T's email-to-SMS gateway
smtp_server = 'smtp.mail.yahoo.com'
smtp_port = 465

# Replace with your Alpaca API key and secret
API_KEY = "PKT37W3VOPN2OJJJ6BW7"
API_SECRET = "aeI3lbI6cWVNLq4AkrGf4MhU0rxUDJm2HSGC0Nju"
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(API_KEY, API_SECRET, BASE_URL)
stream = StockDataStream(API_KEY, API_SECRET)
historical_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Initialize a dictionary to hold the live data for multiple stocks
live_data = {}
last_update_time = datetime.now()
invested = {}

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the RSI calculation function
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Define the Bollinger Bands calculation function
def calculate_bollinger_bands(data, window=9, std_multiplier=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    upper_band = rolling_mean + (rolling_std * std_multiplier)
    lower_band = rolling_mean - (rolling_std * std_multiplier)

    return upper_band, lower_band

# Define the EMA calculation function
def calculate_ema(data, short_window=9, long_window=20):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()

    return short_ema, long_ema

# Function to fetch historical data
def fetch_historical_data(symbol, start_date, end_date):
    return (yf.download(symbol, start=start_date, end='2024-06-26', period='1d', interval='1m')
            ).reset_index()[['Open', 'Close']]
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )
    bars = historical_client.get_stock_bars(request_params)
    df = bars.df
    logging.info(f"Fetched data for {symbol}: {df.head()}")
    
    # Check if the necessary columns are present
    required_columns = ['open', 'close']
    if not all(column.strip() in df.columns for column in required_columns):
        logging.error(f"Data for {symbol} does not contain required columns: {required_columns}, it only has {df.columns}")
        return pd.DataFrame(columns=required_columns)
    
    df = df.reset_index()[required_columns]
    return df

def send_email(subject, body, recipient, retries=3):
    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = recipient
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    text = msg.as_string()
    
    for attempt in range(retries):
        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(email_user, email_password)
                server.sendmail(email_user, recipient, text)
            logging.info(f"Email sent to {recipient} with subject: {subject}")
            print(f"Email sent to {recipient} with subject: {subject}")
            return True
        except smtplib.SMTPException as e:
            logging.error(f"Failed to send email to {recipient} (attempt {attempt + 1}): {e}")
            print(f"Failed to send email to {recipient} (attempt {attempt + 1}): {e}")
    return False

# Function to check the timestamp of the last entry in the CSV file
def get_last_timestamp_from_csv(symbol):
    file_path = f'/Users/jevanchahal/Desktop/Stocks/CSV/stock_data_{symbol}.csv'
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None

    df = pd.read_csv(file_path)
    if df.empty:
        return None
    
    last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
    if last_timestamp.tzinfo is None:
        last_timestamp = last_timestamp.tz_localize('UTC')
    return last_timestamp

# Handler function to process trade updates
async def handle_trades(data):
    global live_data
    global last_update_time

    # Append new data to the DataFrame
    ticker = data.symbol
    if ticker not in live_data:
        live_data[ticker] = pd.DataFrame(columns=['timestamp', 'open', 'close', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'Short_EMA', 'Long_EMA'])

    new_row = {'timestamp': data.timestamp, 'open': data.price, 'close': data.price}
    new_row_df = pd.DataFrame([new_row])
    live_data[ticker] = pd.concat([live_data[ticker], new_row_df])

    # Calculate technical indicators
    live_data[ticker]['RSI'] = calculate_rsi(live_data[ticker]['close'])
    live_data[ticker]['Bollinger_Upper'], live_data[ticker]['Bollinger_Lower'] = calculate_bollinger_bands(live_data[ticker]['close'])
    live_data[ticker]['Short_EMA'], live_data[ticker]['Long_EMA'] = calculate_ema(live_data[ticker]['close'])

    # Check if one minute has passed since the last update
    last_csv_timestamp = get_last_timestamp_from_csv(ticker)
    current_time = datetime.now().replace(tzinfo=None)  # Make current_time timezone-naive
    if last_csv_timestamp is None or (current_time - last_csv_timestamp.replace(tzinfo=None)) >= timedelta(minutes=1):
        # Append the latest data to the CSV file
        live_data[ticker].to_csv(f'/Users/jevanchahal/Desktop/Stocks/CSV/stock_data_{ticker}.csv', mode='a', header=False, index=False)
        last_update_time = datetime.now()

    # Print the latest row with indicators
    print(f"Latest data for {ticker}:")
    print(live_data[ticker].tail(1))

    # Call the trading function
    start_index = len(live_data[ticker]) - len(new_row_df)  # Assuming you want to start from the index of the new row
    initial_money = 10000  # Define the initial money to invest
    trading(live_data[ticker], start_index, ticker, initial_money)

# Trading function
def trading(stock_data, start_index, ticker, initial_money):
    global invested
    transactions = 0

    for idx in range(start_index, len(stock_data)):
        rsi_value = stock_data['RSI'].iloc[idx]
        current_price = stock_data['open'].iloc[idx]
        lower_bb = stock_data['Bollinger_Lower'].iloc[idx]
        upper_bb = stock_data['Bollinger_Upper'].iloc[idx]
        short_ema = stock_data['Short_EMA'].iloc[idx]
        long_ema = stock_data['Long_EMA'].iloc[idx]

        # Updated buying condition
        if rsi_value <= 35 and ticker not in invested and short_ema < long_ema and current_price < upper_bb:
            num_shares = initial_money // current_price
            if num_shares > 0:
                try:
                    api.submit_order(
                        symbol=ticker,
                        qty=num_shares,
                        side='buy',
                        type='market',
                        time_in_force='day'  # Good 'Til Canceled
                    )
                    invested[ticker] = {
                        'quantity': num_shares,
                        'purchase_price': current_price
                    }
                    transactions += 1
                    subject = f"Buy Order Executed for {ticker}"
                    body = (f"Buy order executed for {ticker} at ${current_price:.2f}.\n"
                            f"RSI: {rsi_value}, Short EMA: {short_ema}, Long EMA: {long_ema}, Upper BB: {upper_bb}\n")
                    send_email(subject, body, 'your_email@example.com')  # Replace with actual email recipient
                    print(f'Successfully submitted order to buy {ticker}.')
                except Exception as e:
                    print(f'Error submitting order: {e}')

        # Updated selling condition
        elif rsi_value >= 75 and ticker in invested and short_ema > long_ema and current_price > lower_bb:
            num_shares = invested[ticker]['quantity']
            if num_shares > 0:
                try:
                    api.submit_order(
                        symbol=ticker,
                        qty=num_shares,
                        side='sell',
                        type='market',
                        time_in_force='day'  # Good 'Til Canceled
                    )
                    del invested[ticker]
                    transactions += 1
                    subject = f"Sell Order Executed for {ticker}"
                    body = (f"Sell order executed for {ticker} at ${current_price:.2f}.\n"
                            f"RSI: {rsi_value}, Short EMA: {short_ema}, Long EMA: {long_ema}, Lower BB: {lower_bb}\n")
                    send_email(subject, body, 'your_email@example.com')  # Replace with actual email recipient
                    print(f'Successfully submitted order to sell {ticker}.')
                except Exception as e:
                    print(f'Error submitting order: {e}')

    return transactions

def get_previous_close(ticker, max_attempts=5):
    end_date = datetime.now()
    for attempt in range(1, max_attempts + 1):
        try:
            start_date = end_date - timedelta(days=attempt)
            data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)

            if not data.empty:
                prev_close = data['Close'].iloc[-1]
                return prev_close
            logging.info(f"No historical data found for {ticker} on attempt {attempt}, trying previous day.")

        except Exception as e:
            logging.error(f"Error fetching previous close for {ticker}: {e}")
    logging.info(f"No historical data found for {ticker} after {max_attempts} attempts, marking as delisted.")
    return None

def get_ticker_decrease(ticker, prev_close):
    try:
        # Fetch current data with progress bar suppressed
        current_data = yf.download(ticker, period='1d', interval='1m', progress=False)
        
        if current_data.empty:
            logging.info(f"No price data found for {ticker}, marking as delisted.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Calculate percentage decrease from previous close
        current_price = current_data['Close'].iloc[-1]
        percent_decrease_prev_close = ((current_price - prev_close) / prev_close) * 100

        # Calculate percentage decrease from today's open
        open_price = current_data['Open'].iloc[0]
        percent_decrease_open = ((current_price - open_price) / open_price) * 100
        
        result_prev_close = pd.DataFrame()
        result_open = pd.DataFrame()
        
        if percent_decrease_prev_close < -5:
            result_prev_close = pd.DataFrame({
                'Ticker': [ticker],
                'Previous Close': [prev_close],
                'Current Price': [current_price],
                'Percent Change from Previous Close': [percent_decrease_prev_close]
            })
        
        if percent_decrease_open < -5:
            result_open = pd.DataFrame({
                'Ticker': [ticker],
                'Open Price': [open_price],
                'Current Price': [current_price],
                'Percent Change from Open': [percent_decrease_open]
            })
        
        return result_prev_close, result_open
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Main function to run the stream
def main():
    stocks = ['AZEK', 'BMEA', 'DQ', 'NVDA', 'EC', 'FLYE', 'GPCR', 'ICU', 'MAX', 'OMI',
              'SEDG', 'SPR', 'TEM', 'TROX', ]  # List of stocks to trade
    print(f"Subscribing to trades for: {', '.join(stocks)}")

    # Fetch historical data for today
    today = datetime.now()
    start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = today

    global live_data
    for stock in stocks:
        historical_data = fetch_historical_data(stock, start_date, end_date)
        live_data[stock] = pd.DataFrame(columns=['timestamp', 'open', 'close', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'Short_EMA', 'Long_EMA'])
        live_data[stock] = pd.concat([live_data[stock], historical_data], ignore_index=True)

        # Calculate initial technical indicators
        live_data[stock]['RSI'] = calculate_rsi(live_data[stock]['close'])
        live_data[stock]['Bollinger_Upper'], live_data[stock]['Bollinger_Lower'] = calculate_bollinger_bands(live_data[stock]['close'])
        live_data[stock]['Short_EMA'], live_data[stock]['Long_EMA'] = calculate_ema(live_data[stock]['close'])

        # Save the initial data to the CSV file
        live_data[stock].to_csv(f'/Users/jevanchahal/Desktop/Stocks/CSV/stock_data_{stock}.csv', index=False)

    # Subscribe to live trades
    for stock in stocks:
        stream.subscribe_trades(handle_trades, stock)

    # Keep the stream running
    stream.run()

if __name__ == "__main__":
    main()