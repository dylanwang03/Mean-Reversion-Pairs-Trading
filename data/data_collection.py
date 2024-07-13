from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()

def fetch_data(tickers, start_date, end_date):
    dfs = []
    invalids = []
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        data["Symbol"] = ticker
        if data.isnull().values.any():
            invalids.append(ticker)
        else:
            dfs.append(data)
    return dfs, invalids
