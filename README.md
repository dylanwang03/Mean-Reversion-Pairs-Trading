# Pairs Trading Strategy

In this project, I dealt with two main tasks:

- **Data Collection and Cleaning**: Collects and preprocesses historical price data for the selected assets.
- **Statistical Analysis and Trading Strategy**: Identifies pairs of assets using statistical methods and implements the trading strategy based on these pairs.

## Data Collection and Cleaning

The project starts by collecting historical price data for the selected assets using the `data_collection.py` script. This data is then cleaned and preprocessed with the `data_cleaning.py` script, which handles missing values and normalizes the data to ensure accuracy and consistency for the subsequent analysis.

## Statistical Analysis and Trading Strategy

### Correlation Analysis

The `correlation_analysis.py` script performs correlation analysis to identify pairs of assets with high correlation. The Pearson Correlation Coefficient is used to measure the linear relationship between the assets' price movements, ensuring that the selected pairs move together, which is crucial for pairs trading.

### Stationarity and Cointegration Tests

#### Augmented Dickey-Fuller (ADF) Test

To ensure the time series data is suitable for pairs trading, we apply the Augmented Dickey-Fuller (ADF) test. This test checks for stationarity in the time series data, which is essential for pairs trading as it ensures that the statistical properties of the spread between asset prices do not change over time. A stationary series will have a mean-reverting behavior, which is a critical assumption for pairs trading.

#### Cointegration Test

The Engle-Granger two-step method is used to test for cointegration between the pairs. Cointegration indicates that a linear combination of the two non-stationary time series is stationary, suggesting a long-term equilibrium relationship. This ensures that while individual asset prices may drift, their relative pricing remains stable over time.

### Pairs Trading Strategy

The `pairs_trading_strategy.py` script implements the pairs trading strategy through the following steps:

1. **Spread Calculation**: Calculate the spread between the prices of the paired assets. The spread is expected to be mean-reverting.
   
2. **Z-Score Calculation**: Compute the z-score of the spread to standardize the deviation from its mean. The z-score helps in determining the entry and exit points for trades.

3. **Trading Signals**:
   - **Entry Signal**: Generated when the spread deviates significantly from the mean, indicating a potential trading opportunity.
   - **Exit Signal**: Generated when the spread reverts to the mean, suggesting closing the positions.

4. **Execution**: Trades are executed based on the generated signals, aiming to profit from the convergence of the spread back to the mean.

### Gradient Descent

In addition to the above steps, gradient descent is used to optimize the parameters of the trading strategy. Gradient descent helps in minimizing the prediction error by iteratively adjusting the model parameters. This optimization process ensures that the model effectively captures the relationship between the asset prices and improves the accuracy of the trading signals.


## Summary

1. Collect and preprocess historical price data for the selected assets.
2. Perform correlation analysis to identify pairs of assets with high correlation.
3. Conduct stationarity and cointegration tests to confirm the mean-reverting behavior of the spread between the pairs.
4. Implement the pairs trading strategy based on the identified pairs and statistical tests.

## To Run

1. **Data Collection**:
   - Run `data_collection.py` to collect historical price data for the selected assets.

2. **Data Cleaning**:
   - Run `data_cleaning.py` to clean and preprocess the collected data.

3. **Correlation Analysis**:
   - Run `correlation_analysis.py` to perform correlation analysis and identify pairs of assets.

4. **Pairs Trading Strategy**:
   - Run `pairs_trading_strategy.py` to implement the pairs trading strategy.