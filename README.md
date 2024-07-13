
## How to Run

1. Clone the repository:
    ```
    git clone https://github.com/your-username/PairsTradingAnalysis.git
    cd PairsTradingAnalysis
    ```

2. Run the main script:
    ```
    python main.py
    ```

## Files Description

- `data/`: Contains scripts for data collection and cleaning.
- `analysis/`: Contains scripts for performing correlation analysis and pairs trading strategy.
- `visualization/`: Contains scripts for generating visualizations.
- `main.py`: Main script to run the entire analysis.

## Dependencies

- pandas
- pandas_datareader
- yfinance
- matplotlib
- seaborn


This repository provides an implementation of a mean reversion pairs trading algorithm using Python. Pairs trading is a quantitative trading strategy that involves exploiting the price divergence of two correlated assets by taking opposite positions when they deviate from their historical relationship.


Data Retrieval: Download historical price data for the two assets under consideration.
Data Preprocessing: Clean, normalize, and align the data for analysis.
Pair Selection: Identify suitable pairs of assets for trading using statistical measures.
Spread Calculation: Calculate the spread between the two assets and identify trading signals.
Backtesting: Simulate the trading strategy on historical data to evaluate its performance.
Visualizations: Create various visualizations to understand the strategy's outcomes.
Prerequisites
To run this notebook, you will need the following libraries:

pandas
numpy
matplotlib
yfinance (for downloading historical price data)
seaborn (for enhanced visualizations)
You can install these libraries using the following command:

bash
Copy code
pip install pandas numpy matplotlib yfinance seaborn

Make sure you have an active internet connection for data retrieval using yfinance.
