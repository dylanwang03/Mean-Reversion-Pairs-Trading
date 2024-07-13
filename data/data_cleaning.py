import pandas as pd

def clean_data(dfs):
    df = pd.concat(dfs)
    df = df.reset_index()
    df = df[["Date", "Close", "Symbol"]]
    df_pivot = df.pivot('Date', 'Symbol', 'Close').reset_index()
    return df_pivot
