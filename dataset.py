import numpy as np
import pandas as pd
import yfinance as yf
import os

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    return data

def preprocess_data(data):
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    return data

def create_stocks_data(data):
    # Assuming 'Adj Close' is used for asset scoring unit
    stocks_data = data['Adj Close'].values
    num_days, num_stocks = stocks_data.shape
    num_features = 6  # As per ASU.py
    stocks_data = np.expand_dims(stocks_data.T, axis=-1)  # Transpose to match required shape
    stocks_data = np.repeat(stocks_data, num_features, axis=-1)
    stocks_data = stocks_data.reshape(num_stocks, num_days, num_features)  # Reshape to (num_stocks, num_days, num_features)
    return stocks_data

def create_market_data(data):
    # Assuming 'Adj Close' of the index itself is used for market scoring unit
    market_data = data['Adj Close'].mean(axis=1).values
    num_days = market_data.shape[0]
    num_features = 4  # As per MSU.py
    market_data = np.expand_dims(market_data, axis=-1)
    market_data = np.repeat(market_data, num_features, axis=-1)
    return market_data

def create_ror_data(data):
    # Rate of return calculation
    ror_data = data['Adj Close'].pct_change().fillna(0).values
    return ror_data.T  # Transpose to match required shape

def create_industry_classification(num_stocks):
    # Dummy industry classification matrix
    industry_classification = np.eye(num_stocks)
    return industry_classification

def preprocess_for_asu(stocks_data, batch_size, window_len):
    num_stocks, num_days, num_features = stocks_data.shape
    batches = []
    for i in range(0, num_days - window_len + 1, window_len):
        batch = stocks_data[:, i:i + window_len, :]
        if batch.shape[1] == window_len:
            batches.append(batch)
    batches = np.array(batches)  # (num_batches, num_stocks, window_len, num_features)
    num_batches = len(batches)
    if num_batches % batch_size != 0:
        # Adjust num_batches to be divisible by batch_size
        num_batches = (num_batches // batch_size) * batch_size
        batches = batches[:num_batches]
    batches = batches.reshape(num_batches // batch_size, batch_size, num_stocks, window_len, num_features)
    return batches  # (num_batches // batch_size, batch_size, num_stocks, window_len, num_features)

def preprocess_for_msu(market_data, batch_size, window_len):
    num_days, num_features = market_data.shape
    batches = []
    for i in range(0, num_days - window_len + 1, window_len):
        batch = market_data[i:i + window_len, :]
        if batch.shape[0] == window_len:
            batches.append(batch)
    batches = np.array(batches)  # (num_batches, window_len, num_features)
    num_batches = len(batches)
    if num_batches % batch_size != 0:
        # Adjust num_batches to be divisible by batch_size
        num_batches = (num_batches // batch_size) * batch_size
        batches = batches[:num_batches]
    batches = batches.reshape(num_batches // batch_size, batch_size, window_len, num_features)
    return batches  # (num_batches // batch_size, batch_size, window_len, num_features)

def save_data(stocks_data, market_data, ror_data, industry_classification, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'stocks_data.npy'), stocks_data)
    np.save(os.path.join(output_dir, 'market_data.npy'), market_data)
    np.save(os.path.join(output_dir, 'ror.npy'), ror_data)
    np.save(os.path.join(output_dir, 'industry_classification.npy'), industry_classification)
    
    # Save as CSV for verification
    pd.DataFrame(stocks_data.reshape(stocks_data.shape[0], -1)).to_csv(os.path.join(output_dir, 'stocks_data.csv'), index=False)
    pd.DataFrame(market_data).to_csv(os.path.join(output_dir, 'market_data.csv'), index=False)
    pd.DataFrame(ror_data).to_csv(os.path.join(output_dir, 'ror.csv'), index=False)
    pd.DataFrame(industry_classification).to_csv(os.path.join(output_dir, 'industry_classification.csv'), index=False)

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "V", "JNJ", "WMT", "PG", "UNH", "DIS", "HD", "VZ", "CVX", "KO", "MRK", "INTC", "CSCO", "PFE", "BA", "IBM", "XOM", "GS", "AXP", "CAT", "MMM", "NKE", "TRV", "MCD", "WBA", "DOW", "AMGN"]
    start_date = "2000-01-01"  # Adjusted start date to ensure sufficient data
    end_date = "2024-12-31"
    output_dir = "./data/DJIA"

    data = download_data(tickers, start_date, end_date)
    data = preprocess_data(data)

    stocks_data = create_stocks_data(data)
    market_data = create_market_data(data)
    ror_data = create_ror_data(data)
    industry_classification = create_industry_classification(len(tickers))

    print("Stocks Data Shape:", stocks_data.shape)
    print("Market Data Shape:", market_data.shape)
    print("Rate of Return Data Shape:", ror_data.shape)
    print("Industry Classification Shape:", industry_classification.shape)

    # Ensure stocks_data and rate_of_return have matching shapes
    assert stocks_data.shape[0] == ror_data.shape[0], "Number of stocks must match"
    assert stocks_data.shape[1] == ror_data.shape[1], "Number of days must match"

    # Preprocess for ASU and MSU
    batch_size = 32
    window_len = 20
    asu_input = preprocess_for_asu(stocks_data, batch_size, window_len)
    msu_input = preprocess_for_msu(market_data, batch_size, window_len)

    save_data(stocks_data, market_data, ror_data, industry_classification, output_dir)