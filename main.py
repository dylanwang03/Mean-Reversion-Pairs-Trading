from analysis.correlation_analysis import calculate_kendall_correlation, cointegration, find_cointegrated_pairs, get_sharpe
from analysis.pairs_trading_strategy import run_strategy
from data.data_collection import fetch_data
from data.data_cleaning import clean_data
from visualization.heatmap import plot_heatmap
from visualization.pairs_trading_plot import plot_pairs_trading_results
import pandas as pd

def main():
    tickers = [
        "SO", "EOG", "CNQ", "EPD", "OXY", "E", "VLO",
        "SRE", "PXD", "NGG", "AEP", "WDS", "COP", "BP",
        "EQNR", "ENB", "DUK", "SLB", "PBR",
        "PCG", "EIX", "CVX", "SHEL", "TTE", "NEE",
        'SNMP', 'MIND', 'ENG', 'GTE', 'GVP',
        'ENSV', 'PED', 'DNN', 'DWSN', 'TELL',
        'CREG', 'CIG', 'REI', 'USEG', 'BRN', 'MMLP',
        'GASS', 'HUSA', 'FCEL', 'NAT', 'UEC', 'TTI',
        'GIFI', 'SLNG', 'NR', 'EGY',
        'SWN', 'CPYYY', 'TK', 'HGKGY', 'GEOS', 'WTI',
        'CLNE' ,'OBE'
    ]
    start_date = "2007-12-31"
    end_date = "2017-12-31"

    # Data Collection
    dfs, invalids = fetch_data(tickers, start_date, end_date)
    df_pivot = clean_data(dfs)
    df_pivot.to_csv("cleaned_data.csv", index=False)
    print("Invalid tickers:", invalids)

    # Data Analysis
    non_self_dup = calculate_kendall_correlation(df_pivot)
    non_self_dup.to_csv("kendall_correlation.csv", index=False)
    
    

if __name__ == "__main__":
    main()


def main():
    tickers = [
        "SO", "EOG", "CNQ", "EPD", "OXY", "E", "VLO",
        "SRE", "PXD", "NGG", "AEP", "WDS", "COP", "BP",
        "EQNR", "ENB", "DUK", "SLB", "PBR",
        "PCG", "EIX", "CVX", "SHEL", "TTE", "NEE",
        'SNMP', 'MIND', 'ENG', 'GTE', 'GVP',
        'ENSV', 'PED', 'DNN', 'DWSN', 'TELL',
        'CREG', 'CIG', 'REI', 'USEG', 'BRN', 'MMLP',
        'GASS', 'HUSA', 'FCEL', 'NAT', 'UEC', 'TTI',
        'GIFI', 'SLNG', 'NR', 'EGY',
        'SWN', 'CPYYY', 'TK', 'HGKGY', 'GEOS', 'WTI',
         'CLNE', 'OBE'
    ]
    start_date = "2007-12-31"
    end_date = "2017-12-31"

    # Step 1: Data Collection
    dfs, invalids = fetch_data(tickers, start_date, end_date)
    print("Invalid tickers:", invalids)

    # Step 2: Data Cleaning
    df_pivot = clean_data(dfs)

    # Step 3: Data Analysis
    non_self_dup = calculate_kendall_correlation(df_pivot)
    consolidated = pd.concat(dfs, axis=1)
    consolidated.columns = tickers
    corr_coint = find_cointegrated_pairs(consolidated, non_self_dup)

    # Step 4: Run Strategy
    signals = run_strategy(corr_coint, df_pivot)

    # Step 5: Test Strategy
    sharpe_ratio = get_sharpe(signals)
    print(f"Sharpe Ratio: {sharpe_ratio}")


if __name__ == "__main__":
    main()


