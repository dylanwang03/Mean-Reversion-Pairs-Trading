# Mean-Reversion-Pairs-Trading

This Jupyter Notebook provides an implementation of a mean reversion pairs trading algorithm using Python. Pairs trading is a quantitative trading strategy that involves exploiting the price divergence of two correlated assets by taking opposite positions when they deviate from their historical relationship.


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
Getting Started
Clone this repository to your local machine or download the notebook directly.
Open the notebook using Jupyter Notebook or Jupyter Lab.
Make sure you have an active internet connection for data retrieval using yfinance.
Notebook Structure
The notebook is organized into the following sections:

1. Data Retrieval
This section covers how to download historical price data for the two assets using the yfinance library.

2. Data Preprocessing
Learn how to clean and preprocess the downloaded data, handling missing values, and normalizing the prices.

3. Pair Selection
Discover how to identify suitable pairs of assets for trading using correlation analysis and statistical measures.

4. Spread Calculation
Calculate the spread between the two assets and determine entry and exit signals based on mean reversion.

5. Backtesting
Simulate the trading strategy using historical data to evaluate its performance and calculate key performance metrics.

6. Visualizations
Create visualizations such as price charts, spread plots, and equity curves to better understand the strategy's behavior.
