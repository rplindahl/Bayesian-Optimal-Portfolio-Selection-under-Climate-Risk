import os
import pandas as pd
import numpy as np
import alpha_vantage
import financial_modeling_prep
import yahoo_finance
from dotenv import load_dotenv
import logging

load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)


## Set up directories for various data sources
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the data directory
data_dir = os.path.join(parent_dir, 'data')

# Construct the path to the intraday stock prices directory inside the data directory
stock_intraday_prices_dir = os.path.join(data_dir, 'stock_intraday_prices')

# Construct the path to the stock prices directory inside the data directory
stock_prices_dir = os.path.join(data_dir, 'stock_prices')

# Construct the path to the market caps directory inside the data directory
stock_market_caps_dir = os.path.join(data_dir, 'stock_market_caps')

# Construct the path to the vix directory inside the data directory
vix_dir = os.path.join(data_dir, 'vix_prices')

# Construct the path to the epu directory inside the data directory
epu_dir = os.path.join(data_dir, 'epu_prices')

# Construct the path to the S&P500 prices directory inside the data directory
sp500tr_dir = os.path.join(data_dir, 'sp500_prices')

# Construct the path to the S&P500 prices directory inside the data directory
risk_free_rate_dir = os.path.join(data_dir, 'risk_free_rate')

# Construct the path to the S&P500 components directory inside the data directory
sp500_components_dir = os.path.join(data_dir, 'sp500_components')

#### New - Placeholder names
# Create directories to climate risk 1 and 2 (and further)
climate_risk1_dir = os.path.join(data_dir, "climate_risk1") # adjust name of folder
climate_risk2_dir = os.path.join(data_dir, "climate_risk2") # adjust name of folder

intraday_frequency = os.environ.get("INTRADAY_FREQUENCY")


### New Functions. Get climate risk data. 
# Climate risk 1
## These will not be downloaded through API but rather manually added into the data-folder. 
def get_climate_risk1():
    csv_file = os.path.join(climate_risk1_dir, 'ClimateRisk1.csv') # Adjust name of csv file

    if os.path.exists(csv_file):
        climate_risk1_df = pd.read_csv(csv_file, index_col = 0, parse_dates = True, na_values = ["."])
    else:
        raise ValueError("Climate Risk 1 Data does not exist in data folder.")
    
    # Ensure index is datetime
    climate_risk1_df.index = pd.to_datetime(climate_risk1_df.index)

    # **Fix: Remove commas before converting to float**
    climate_risk1_df = climate_risk1_df.replace(",", "", regex=True).astype(float)

    # Sort index
    climate_risk1_df = climate_risk1_df.sort_index()

    return climate_risk1_df

def get_climate_risk2():
    csv_file = os.path.join(climate_risk2_dir, 'ClimateRisk2.csv') # Adjust name of csv file

    if os.path.exists(csv_file):
        climate_risk2_df = pd.read_csv(csv_file, index_col = 0, parse_dates = True, na_values = ["."])
    else:
        raise ValueError("Climate Risk 2 Data does not exist in data folder.")
    
    # Ensure index is datetime
    climate_risk2_df.index = pd.to_datetime(climate_risk2_df.index)

    # **Fix: Remove commas before converting to float**
    climate_risk2_df = climate_risk2_df.replace(",", "", regex=True).astype(float)

    # Sort index
    climate_risk2_df = climate_risk2_df.sort_index()

    return climate_risk2_df


### Existing functions
def check_directory_for_csv(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        logger.info(f"The directory '{directory}' does not exist.")
        return False

    # Check if the directory contains any CSV files
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        logger.info(f"The directory '{directory}' does not contain any CSV files.")
        return False

    return True

def load_all_csv_to_dataframe(directory):
    logger.info(f"Loading csv from {directory} into dataframe.")
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    dataframes = []

    for f in files:
        df = pd.read_csv(os.path.join(directory, f), index_col=0, parse_dates=True)
        df.columns = [f.replace('.csv', '')]  # renaming the column as the file name (without the .csv extension)
        dataframes.append(df)

    large_df = pd.concat(dataframes, axis=1)  # concatenate the dataframes horizontally
    large_df = large_df.ffill() # Fill missing values with the previous value

    return large_df

def extract_unique_tickers(start_date_ts, end_date_ts):
    if check_directory_for_csv(sp500_components_dir):
        tickers_df = load_all_csv_to_dataframe(sp500_components_dir)
    else:
        raise ValueError(f"S&P 500 historical components must be downloaded and located in /data/sp500_components. Please visit https://github.com/fja05680/sp500")

    # Convert index to datetime
    tickers_df.index = pd.to_datetime(tickers_df.index)

    # Find the closest date before the specified start date
    closest_start_date = tickers_df[tickers_df.index <= start_date_ts].index.max()
    if pd.isnull(closest_start_date):
        logger.error(f"No S&P 500 historical components available before the specified start date: {start_date_ts}")
        raise ValueError("No S&P 500 historical components.")

    # Filter rows between start_date and end_date
    filtered_tickers_df = tickers_df[(tickers_df.index >= closest_start_date) & (tickers_df.index <= end_date_ts)].copy()

    # Check if the column exists
    column_name = 'S&P 500 Historical Components & Changes(08-17-2024)'
    if column_name not in filtered_tickers_df.columns:
        raise KeyError(f"Column '{column_name}' not found in the data.")

    # Ensure there are no NaN values before splitting
    filtered_tickers_df = filtered_tickers_df.dropna(subset=[column_name])

    # Split tickers strings into lists
    filtered_tickers_df['split_tickers'] = filtered_tickers_df[column_name].str.split(',')

    # Flatten list of lists and get unique tickers
    unique_tickers = pd.unique(pd.Series([
        ticker.strip() for sublist in filtered_tickers_df['split_tickers'] for ticker in sublist
    ]))

    return list(unique_tickers) if unique_tickers.size > 0 else []

def get_stock_intraday_prices():
    str_start_year_month = "2000-01"
    str_end_year_month = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m')
    max_calls_per_minute = 75

    if check_directory_for_csv(stock_intraday_prices_dir):
        stock_intraday_prices_df = load_all_csv_to_dataframe(stock_intraday_prices_dir)
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save intraday stock prices to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            unique_tickers_list = extract_unique_tickers(pd.to_datetime("1999-12-31"),
                                                         pd.to_datetime("2022-12-31"))
            alpha_vantage.save_stock_intraday_prices_to_csv(unique_tickers_list, str_start_year_month, str_end_year_month, intraday_frequency, max_calls_per_minute)
            stock_intraday_prices_df = load_all_csv_to_dataframe(stock_intraday_prices_dir)
        else:
            stock_intraday_prices_df = None

    # Use pd.Timestamp date
    stock_intraday_prices_df.index = pd.to_datetime(stock_intraday_prices_df.index)

    return stock_intraday_prices_df

def get_stock_prices():
    str_start_date = "2000-01-01"
    str_end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    max_calls_per_minute = 75

    if check_directory_for_csv(stock_prices_dir):
        stock_prices_df = load_all_csv_to_dataframe(stock_prices_dir)
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save stock prices to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            unique_tickers_list = extract_unique_tickers(pd.to_datetime("1999-12-31"),
                                                         pd.to_datetime("2022-12-31"))
            alpha_vantage.save_stock_prices_to_csv(unique_tickers_list, str_start_date, str_end_date, max_calls_per_minute)
            stock_prices_df = load_all_csv_to_dataframe(stock_prices_dir)
        else:
            stock_prices_df = None

    # Use pd.Timestamp date
    stock_prices_df.index = pd.to_datetime(stock_prices_df.index)

    return stock_prices_df

def get_stock_market_caps():
    str_start_date = "2000-01-01"
    str_end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    if check_directory_for_csv(stock_market_caps_dir):
        stock_market_caps_df = load_all_csv_to_dataframe(stock_market_caps_dir)
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save market caps to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            unique_tickers_list = extract_unique_tickers(pd.to_datetime("1999-12-31"),
                                                         pd.to_datetime("2022-12-31"))
            financial_modeling_prep.save_stock_market_caps_to_csv(unique_tickers_list, str_start_date, str_end_date)
            stock_market_caps_df = load_all_csv_to_dataframe(stock_market_caps_dir)
        else:
            stock_market_caps_df = None

    # Use pd.Timestamp date
    stock_market_caps_df.index = pd.to_datetime(stock_market_caps_df.index)

    return stock_market_caps_df


def get_vix_prices():
    str_start_date = "2000-01-01"
    str_end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    csv_file = os.path.join(vix_dir, 'VIX.csv')

    if os.path.exists(csv_file):
        vix_prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)  # Assuming the first column is a date index
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save VIX data to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            yahoo_finance.save_vix_prices_to_csv(str_start_date, str_end_date)
            vix_prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)  # Reload the saved CSV
        else:
            vix_prices_df = None

    # Use pd.Timestamp date
    vix_prices_df.index = pd.to_datetime(vix_prices_df.index)

    return vix_prices_df

def get_epu_prices():
    csv_file = os.path.join(epu_dir, 'EPU.csv')

    if os.path.exists(csv_file):
        # Read CSV and treat '.' as NaN
        epu_prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True, na_values=['.'])
    else:
        raise ValueError(f"EPU prices csv must be downloaded. Please visit https://fred.stlouisfed.org/series/USEPUINDXD")

    # Use pd.Timestamp date
    epu_prices_df.index = pd.to_datetime(epu_prices_df.index)

    # Convert 'EPU' column to float
    epu_prices_df['EPU'] = epu_prices_df['EPU'].astype(float)

    # Sort the index
    risk_free_rate = epu_prices_df.sort_index()

    return risk_free_rate


def get_sp500tr_prices():
    str_start_date = "2000-01-01"
    str_end_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    csv_file = os.path.join(sp500tr_dir, 'SP500TR.csv')

    if os.path.exists(csv_file):
        sp500tr_prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)  # Assuming the first column is a date index
    else:
        # Prompt the user for a response
        user_response = input("Do you want to save SP500TR data to CSV? (Y/N): ").strip().upper()

        if user_response == 'Y':
            yahoo_finance.save_sp500tr_prices_to_csv(str_start_date, str_end_date)
            sp500tr_prices_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)  # Reload the saved CSV
        else:
            sp500tr_prices_df = None

    # Use pd.Timestamp date
    sp500tr_prices_df.index = pd.to_datetime(sp500tr_prices_df.index)

    return sp500tr_prices_df


def get_risk_free_rate():
    csv_file = os.path.join(risk_free_rate_dir, 'DTB3.csv')

    if os.path.exists(csv_file):
        # Read CSV and treat '.' as NaN
        risk_free_rate_df = pd.read_csv(csv_file, index_col=0, parse_dates=True, na_values=['.'])
    else:
        raise ValueError(f"Risk-free rate csv must be downloaded. Please visit https://fred.stlouisfed.org/series/DTB3")

    # Use pd.Timestamp date
    risk_free_rate_df.index = pd.to_datetime(risk_free_rate_df.index)

    # Convert 'DTB3' column to float
    risk_free_rate_df['DTB3'] = risk_free_rate_df['DTB3'].astype(float)

    # Divide 'DTB3' values by 100
    risk_free_rate_df['DTB3'] = risk_free_rate_df['DTB3'] / 100.0

    # Sort the index
    risk_free_rate_df = risk_free_rate_df.sort_index()

    return risk_free_rate_df




### Add the climate risk data functions to this function
def get_market_data():
    stock_prices_df = get_stock_prices()
    stock_simple_returns_df = stock_prices_df.pct_change()
    stock_log_returns_df = np.log(stock_prices_df / stock_prices_df.shift(1))
    stock_intraday_prices_df = get_stock_intraday_prices()
    stock_market_caps_df = get_stock_market_caps()
    vix_prices_df = get_vix_prices()
    epu_prices_df = get_epu_prices()
    sp500_prices_df = get_sp500tr_prices()
    sp500_simple_returns_df = sp500_prices_df.pct_change()
    risk_free_rate_df = get_risk_free_rate()
    climate_risk1_df = get_climate_risk1()
    climate_risk2_df = get_climate_risk2()

    return {"stock_prices_df": stock_prices_df,
            "stock_simple_returns_df": stock_simple_returns_df,
            "stock_log_returns_df": stock_log_returns_df,
            "stock_intraday_prices_df": stock_intraday_prices_df,
            "stock_market_caps_df": stock_market_caps_df,
            "vix_prices_df": vix_prices_df,
            "epu_prices_df": epu_prices_df,
            "sp500_prices_df": sp500_prices_df,
            "sp500_simple_returns_df": sp500_simple_returns_df,
            "risk_free_rate_df": risk_free_rate_df,
            "climate_risk1_df": climate_risk1_df,
            "climate_risk2_df": climate_risk2_df}
