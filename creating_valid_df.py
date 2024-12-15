import pandas as pd
import yfinance as yf
from tqdm import tqdm
import requests
import os
from bs4 import BeautifulSoup
import logging
from lxml import html
from lxml import etree
from requests_html import HTMLSession

# Load the tickers from the CSV file
csv_file_path = '/Users/jevanchahal/Desktop/Stocks/CSV/us_symbols.csv'
every_ticker = pd.read_csv(csv_file_path)['ticker']
    
def get_short_interest(ticker):
    session = HTMLSession()
    url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics?p={ticker}"
    response = session.get(url)
    
    # Render the JavaScript
    response.html.render()
    
    # Parse the content with lxml
    tree = etree.HTML(response.html.html)
    
    # Example XPath to find short interest data (this might need to be adjusted based on the actual structure of the page)
    short_interest_xpath = '//*[@id="nimbus-app"]/section/section/section/article/article/div/section[2]/div/section[2]/table/tbody/tr[10]/td[2]'
    short_interest = tree.xpath(short_interest_xpath)
    
    if short_interest:
        return float(short_interest[0].text.replace('%', ''))
    else:
        return "Short interest data not found"
    
def check_eps_beat(ticker):
    try:
        # URL of the website you want to scrape
        url = f'https://finance.yahoo.com/quote/{ticker}/analysis/'

        # Send a GET request to the website with headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the section with data-testid="earningsHistory"
            earnings_history_section = soup.find('section', {'data-testid': 'earningsHistory'})

            # Check if the section was found
            if earnings_history_section:
                # Locate the table rows
                table_rows = earnings_history_section.find_all('tr')

                # Check if there are enough rows
                if len(table_rows) > 2:
                    # Extract EPS estimate and actual EPS values
                    eps_est = float(table_rows[1].find_all('td')[1].text.strip())
                    eps_act = float(table_rows[2].find_all('td')[1].text.strip())

                    return f"act:{eps_act} est: {eps_est}"
                else:
                    return f"act:{eps_act} est: {eps_est}"
            else:
                return f"section not found"
        else:
            return f"request unsuccesful"
    except Exception as e:
        logging.error(f"Error checking EPS beat for {ticker}: {e}")
        return e
    
def get_ticker_summary(ticker, date):
    try:
        # Fetch data for the entire day (start date inclusive, end date exclusive)
        data = yf.download(ticker, start=date, end=pd.to_datetime(date) + pd.Timedelta(days=1), interval='1d')
        
        if data.empty:
            print(f"No data found for {ticker} on {date}")
            return None

        closing_price = data['Close'].iloc[0]
        total_volume = data['Volume'].iloc[0]
        
        if 1.00 <= closing_price and total_volume >= 1000000:# and beat_eps == True:
            #beat_eps = check_eps_beat(ticker)
            #short_interest_percent = get_short_interest(ticker)
            return {
                'Ticker': ticker,
                'Closing Price': closing_price,
                'Total Volume': total_volume
                #'Beat EPS': beat_eps
                #'Short Interest': short_interest_percent
            }
        else:
            print(f"{ticker} does not meet the price or volume criteria")
            return None
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Initialize a list to collect valid stock data
valid_stocks_list = []

# Iterate over each ticker and fetch the data with a progress bar
for i, ticker in enumerate(tqdm(every_ticker, desc="Processing tickers")):
    valid_result = get_ticker_summary(ticker, '2024-07-01')
    if valid_result is not None:
        valid_stocks_list.append(valid_result)

# Convert the list of dictionaries to a DataFrame
if valid_stocks_list:
    valid_stocks = pd.DataFrame(valid_stocks_list)
    # Save the final DataFrame to a CSV file
    penny_output_file_path = '/Users/jevanchahal/Desktop/Stocks/CSV/penny_valid_stocks_07-02-24.csv'
    medium_output_file_path = '/Users/jevanchahal/Desktop/Stocks/CSV/medium_valid_stocks_07-02-24.csv'
    output_file_path = '/Users/jevanchahal/Desktop/Stocks/CSV/valid_stocks_07-02-24.csv'

    valid_penny_stocks = valid_stocks[valid_stocks['Closing Price'] <= 5.00]
    valid_medium_stocks = valid_stocks[valid_stocks['Closing Price'] > 5.00]
    valid_stocks.to_csv(output_file_path, index=False)
    valid_penny_stocks.to_csv(penny_output_file_path, index=False)
    valid_medium_stocks.to_csv(medium_output_file_path, index=False)
    print("Valid stocks saved to valid_stocks.csv")
else:
    print("No valid stocks found")
