import pandas as pd
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(filename='stock_alerts.log', level=logging.INFO,
                    format=('%(asctime)s:%(levelname)s:%(message)s'))

# Load the tickers that met the criteria from the previous day
valid_stocks = pd.read_csv('/Users/jevanchahal/Desktop/Stocks/CSV/valid_stocks_07-02-24.csv')

# Email configuration
email_user = 'jevanchahal1@yahoo.com'
email_password = 'plhokhnqrhzaaegu'  # Use app-specific password if 2FA is enabled
email_send = 'jevanchahal1@gmail.com'
sms_send = '9168492930@txt.att.net'  # AT&T's email-to-SMS gateway
smtp_server = 'smtp.mail.yahoo.com'
smtp_port = 465

# Sets to keep track of already printed stocks based on different criteria
printed_stocks_high = set()
printed_stocks_low = set()
delisted_stocks = set()  # Track delisted stocks

def get_ticker_change(ticker):
    try:
        # Fetch current data with progress bar suppressed
        time.sleep(0.2)
        high_low = yf.download(ticker, period='1d', interval='1d', progress=False, prepost=True)
        if high_low.empty:
            logging.info(f"No price data found for {ticker}, marking as delisted.")
            delisted_stocks.add(ticker)  # Add to delisted stocks set
            return pd.DataFrame(), pd.DataFrame()
        current_data = yf.download(ticker, period='1d', interval='1m', progress=False, prepost=True)
        # Calculate percentage change and $5 swing from high/low
        current_price = current_data['Close'].iloc[-1]
        high_price = high_low['High'].iloc[-1]
        low_price = high_low['Low'].iloc[-1]
        percent_change_high = ((current_price - high_price) / high_price) * 100
        percent_change_low = ((current_price - low_price) / low_price) * 100
        dollar_swing_high = current_price - high_price
        dollar_swing_low = current_price - low_price
        print(f"Checking {ticker}; High: {high_price}, Low: {low_price}, Current: {current_price}")
        result_high = pd.DataFrame()
        result_low = pd.DataFrame()

        if np.abs(percent_change_high) > 10 or np.abs(dollar_swing_high) > 5:
            result_high = pd.DataFrame({
                'Ticker': [ticker],
                'High Price': [high_price],
                'Current Price': [current_price],
                'Percent Change from High': [percent_change_high],
                'Dollar Swing from High': [dollar_swing_high]
            })

        if np.abs(percent_change_low) > 10 or np.abs(dollar_swing_low) > 5:
            result_low = pd.DataFrame({
                'Ticker': [ticker],
                'Low Price': [low_price],
                'Current Price': [current_price],
                'Percent Change from Low': [percent_change_low],
                'Dollar Swing from Low': [dollar_swing_low]
            })

        return result_high, result_low
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame(), pd.DataFrame()

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
            time.sleep(5)  # Wait before retrying
    return False

# Initialize an empty DataFrame to hold the results
change_stocks_high = pd.DataFrame()
change_stocks_low = pd.DataFrame()
stocks_to_invest = pd.DataFrame(columns=['Ticker', 'Timestamp', 'Price at Found', 'Percent Change from High', 'Percent Change from Low'])

# Function to check stocks in real-time
def check_stocks_realtime(send_email_bool=True, send_text_bool=True):
    global change_stocks_high, change_stocks_low, printed_stocks_high, delisted_stocks, stocks_to_invest
    
    for _, row in valid_stocks.iterrows():
        ticker = row['Ticker']
        
        # Skip delisted stocks
        if ticker in delisted_stocks:
            continue
        
        if ticker not in printed_stocks_high or ticker not in printed_stocks_low:
            result_high, result_low = get_ticker_change(ticker)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if not result_high.empty and ticker not in printed_stocks_high:
                change_stocks_high = pd.concat([change_stocks_high, result_high])
                subject = f"Stock Alert: {ticker} has Decreased significantly from today's high"
                body = (f"Stock {ticker} has Decreased by more than 10% or $5 from today's high price.\n"
                        f"{result_high.to_string(index=False)}\nTimestamp: {timestamp}")
                
                if send_email_bool:
                    if send_email(subject, body, email_send):
                        print(f"Email sent for {ticker} based on today's high with message:\n{body}")
                    else:
                        logging.info(f"Email alert for {ticker} based on today's high failed")
                if send_text_bool:
                    if send_email(subject, body, sms_send):
                        print(f"SMS sent for {ticker} based on today's high with message:\n{body}")
                    else:
                        logging.info(f"SMS alert for {ticker} based on today's high failed")
                
                logging.info(f"Sent alert for {ticker} based on today's high")
                printed_stocks_high.add(ticker)
                print(body)
                stock_info = {
                    'Ticker': ticker,
                    'Timestamp': timestamp,
                    'Price at Found': result_high['Current Price'].iloc[0],
                    'Percent Change from High': result_high['Percent Change from High'].iloc[0],
                    'Percent Change from Low': None
                }
                stocks_to_invest = pd.concat([stocks_to_invest, pd.DataFrame([stock_info])], ignore_index=True)
                
            if not result_low.empty and ticker not in printed_stocks_low:
                change_stocks_low = pd.concat([change_stocks_low, result_low])
                subject = f"Stock Alert: {ticker} has Increased significantly from today's low"
                body = (f"Stock {ticker} has Increased by more than 10% or $5 from today's low price.\n"
                        f"{result_low.to_string(index=False)}\nTimestamp: {timestamp}")
                
                if send_email_bool:
                    if send_email(subject, body, email_send):
                        print(f"Email sent for {ticker} based on today's low with message:\n{body}")
                    else:
                        logging.info(f"Email alert for {ticker} based on today's low failed")
                if send_text_bool:
                    if send_email(subject, body, sms_send):
                        print(f"SMS sent for {ticker} based on today's low with message:\n{body}")
                    else:
                        logging.info(f"SMS alert for {ticker} based on today's low failed")
                
                logging.info(f"Sent alert for {ticker} based on today's low")
                printed_stocks_low.add(ticker)
                print(body)
                stock_info = {
                    'Ticker': ticker,
                    'Timestamp': timestamp,
                    'Price at Found': result_low['Current Price'].iloc[0],
                    'Percent Change from High': None,
                    'Percent Change from Low': result_low['Percent Change from Low'].iloc[0]
                }
                stocks_to_invest = pd.concat([stocks_to_invest, pd.DataFrame([stock_info])], ignore_index=True)
    
    # Save the current state to CSV files after each iteration
    change_stocks_high.to_csv('/Users/jevanchahal/Desktop/Stocks/CSV/change_stocks_high.csv', index=False)
    change_stocks_low.to_csv('/Users/jevanchahal/Desktop/Stocks/CSV/change_stocks_low.csv', index=False)
    stocks_to_invest.to_csv('/Users/jevanchahal/Desktop/Stocks/CSV/stocks_to_invest.csv', index=False)

# Clear the sets at the start of each run
printed_stocks_high.clear()
printed_stocks_low.clear()

# Run the loop to check for updates continuously
while True:
    check_stocks_realtime(send_email_bool=True, send_text_bool=False)
