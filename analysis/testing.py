from pandas_datareader import data as pdr
import pandas as pd
import yfinance as yf
from datetime import datetime
# To visualize the results
import matplotlib.pyplot as plt
import seaborn
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np

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

#clean data and check for untradable stocks
dfs = []
invalids = []
for ticker in tickers:
  data = pdr.get_data_yahoo(ticker, start="2017-12-31", end="2022-12-31")
  data["Symbol"] = ticker
  if data.isnull().values.any():
    invalids.append(ticker)
  else:
    dfs.append(data)

print(invalids)
dataframe = dfs.copy()

#restructure dataframe
df = pd.concat(dfs)
df = df.reset_index()
df = df[["Date", "Close", "Symbol"]]
df_pivot = df.pivot('Date','Symbol','Close').reset_index()

#perform kendall correlation test
corr_df = df_pivot.corr(method='kendall')
corr_df.reset_index()
stacked = corr_df.rename_axis(None).rename_axis(None, axis = 1)
stacked = stacked.stack().reset_index()
stacked = stacked.sort_values(0)

non_self = stacked[stacked[0] != 1]
non_self = non_self.rename(columns={'level_0':'Stock 1', 'level_1':'Stock 2', 0: "Kendall Corr"})
non_self_dup = non_self[-1::-2]
print(non_self_dup.head(20))

plt.figure(figsize=(26, 16))
seaborn.heatmap(corr_df, annot=True, cmap= "RdYlGn")
plt.figure()


for df in dataframe:
  df = df.reset_index()
  df["Date"] = pd.to_datetime(df["Date"])

consolidated = pd.DataFrame()
for df in dataframe:
  consolidated[df["Symbol"][0]] = df["Close"]

consolidated.plot(figsize=(30, 18))

#function to find cointegration
def cointegration(x,y):
  regr = linear_model.LinearRegression()
  x_constant = pd.concat([x,pd.Series([1]*len(x),index = x.index)], axis=1)
  regr.fit(x_constant, y)
  beta = regr.coef_[0]
  alpha = regr.intercept_
  spread = y - x*beta - alpha
  adf = sm.tsa.stattools.adfuller(spread, autolag='AIC')
  return adf[1]

for i in range(consolidated.shape[1]):
  # print("here")
  if consolidated.iloc[:,i].isnull().values.any():
    print (consolidated.columns[i])

c1 = non_self_dup['Stock 1'][:200]
c2 = non_self_dup['Stock 2'][:200]

p_values = []

c_chart = pd.DataFrame()
c_chart['Stock 1'] = c1
c_chart['Stock 2'] = c2

for i in range(c1.shape[0]):
  column_1 = consolidated[c1.iloc[i]]
  column_2 = consolidated[c2.iloc[i]]
  coint_val = cointegration(column_1,column_2)
  p_values.append(coint_val)

c_chart['p_value'] = p_values
c_chart = c_chart.reset_index()
c_chart = c_chart.sort_values('p_value')
c_chart = c_chart[c_chart['p_value'] <= .05]
print(c_chart)

compare_cols = ['Stock 1','Stock 2']
mask = pd.Series(list(zip(*[non_self[c] for c in compare_cols]))).isin(list(zip(*[c_chart[c] for c in compare_cols])))

mask_list = mask.tolist()
corr_coint = non_self[mask_list]

corr_coint = corr_coint[corr_coint.iloc[:,2] > .7]
print(corr_coint.tail())

#Filtering stocks based on Cointegration AD Fuller test and Kendall Correlation test

#Want to find stocks above 95% correlation that we can confidently say are cointegrated (p value <= .05)

stock1 = 'NEE'
stock2 = 'AEP'
s1_pos = f'Position {stock1}'
s2_pos = f'Position {stock2}'

standard_BB_train_df = run_strategy(consolidated, 10, 2.75, stock1, stock2, s1_pos, s2_pos) #20-day and 2 STD are the standard BB parameters used
standard_BB_train_df['P&L'].value_counts()
standard_BB_train_df['P&L'].sum()
# standard_BB_train_df['P&L'].value_counts()


#plot intermediate and final results of using these parameters on training data
fig = plt.figure(figsize=(24,16)) #overall plot size
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)

ax1.plot(standard_BB_train_df['hedge_ratio'])
ax1.set_title("Hedge Ratio")

ax2.plot(standard_BB_train_df['spread'])
ax2.set_title("Spread")

ax3.plot(standard_BB_train_df['spread'], label = 'Spread')
ax3.plot(standard_BB_train_df['rolling_spread'], label = 'Spread 20-Day SMA')
ax3.plot(standard_BB_train_df['upper_band'], label = 'Upper BB')
ax3.plot(standard_BB_train_df['lower_band'], label = 'Lower BB')
ax3.set_title("Bollinger Bands")
ax3.legend(loc="upper right")


ax4.plot(standard_BB_train_df['P&L'].cumsum()) #note: we use cumsum here to see *cumulative* profit
ax4.set_title("Cumulative P&L")

ax5.plot(standard_BB_train_df[f'Position {stock1}'], label = "GLD")
ax5.plot(standard_BB_train_df[f'Position {stock2}'], label = "GDXJ")
ax5.set_title("Positions")

plt.xlabel("Time")
plt.show()


last_fifty = standard_BB_train_df.iloc[:]
fig = plt.figure(figsize=(24,15))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(last_fifty['spread'], label = 'Spread')
ax1.plot(last_fifty['upper_band'], label = 'Upper BB')
ax1.plot(last_fifty['lower_band'], label = 'Lower BB')
ax1.set_title("Bollinger Bands")
ax1.legend(loc="upper right")

ax2.plot(last_fifty[s1_pos], label = stock1)
ax2.plot(last_fifty[s2_pos], label = stock2)
ax2.set_title("Positions")
ax2.legend(loc="upper right")

ax3.plot(standard_BB_train_df['P&L'].cumsum().iloc[-20] + last_fifty['P&L'].cumsum()) #note: we use cumsum here to see *cumulative* profit
ax3.set_title("Cumulative P&L")

plt.show()

