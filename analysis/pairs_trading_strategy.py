import pandas as pd
import matplotlib.pyplot as plt


def run_strategy(data, lookback, width, stock1, stock2, s1_pos, s2_pos):
  #calculating our 63-day hedge ratio lookback window like this makes our program more flexible and readable
  hr_lookback_months = 3
  monthly_trading_days = 21
  hr_lookback = monthly_trading_days * hr_lookback_months

  #these are our familiar hedge ratio and spread calculations
  df = data.copy()
  df['hedge_ratio'] = df[stock1].rolling(hr_lookback).corr(df[stock2]) * df[stock1].rolling(hr_lookback).std() / df[stock2].rolling(hr_lookback).std()
  df['spread'] = df[stock1] - df['hedge_ratio'] * df[stock2]

  #BB calculations
  df['rolling_spread'] = df['spread'].rolling(lookback).mean() #lookback-day SMA of spread
  df['rolling_spread_std'] = df['spread'].rolling(lookback).std() #lookback-day rolling STD of spread
  df['rolling_z_score'] = (df['spread'] - df['rolling_spread'])

  df['upper_band'] = df['rolling_spread'] + (width * df['rolling_spread_std']) #upper = SMA + width * STD
  df['lower_band'] = df['rolling_spread'] - (width * df['rolling_spread_std']) #lower = SMA - width * STD



  s1_pnl = f'P&L {stock1}'
  s2_pnl = f'P&L {stock2}'

  df[s1_pos] = 0
  df[s2_pos] = 0
  df[s1_pnl] = 0
  df[s2_pnl] = 0


  # S1 = df.loc[:,stock1]
  # S2 = df.loc[:,stock2]
  # ratios = df['hedge_ratio']

  money = 0
  countS1 = 0
  countS2 = 0
  signal = False
  trades = 0
  c1_pos = 0
  c2_pos = 0
  c_hedge = 0
  trade = False
  trades = 0
  p1 = 0
  p2 = 0


  for date, row in df.iterrows():
    # Sell short if the z-score is > 1
    trade = False
    if row['rolling_z_score'] > width:
      if not signal:
        c1_pos = -1
        c_hedge = row['hedge_ratio']
        c2_pos = -c1_pos * c_hedge
        df.loc[date,s1_pos] = c1_pos
        df.loc[date,s2_pos] = c2_pos
        p1 = row[stock1]
        p2 = row[stock2]
        signal = True
        trade = True
      else: #there was a signal (assuming same as before)
        df.loc[date,s1_pos] = c1_pos
        df.loc[date,s2_pos] = c2_pos
    # Buy long if the z-score is < 1
    elif row['rolling_z_score'] < -width:
      if not signal:
        c1_pos = 1
        c_hedge = row['hedge_ratio']
        c2_pos = -c1_pos * c_hedge
        df.loc[date,s1_pos] = c1_pos
        df.loc[date,s2_pos] = c2_pos
        p1 = row[stock1]
        p2 = row[stock2]
        signal = True
        trade = True
      else: #there was a signal (assuming same as before)
        df.loc[date,s1_pos] = c1_pos
        df.loc[date,s2_pos] = c2_pos
    # Clear positions if the z-score between -.5 and .5
    if abs(row['rolling_z_score']) < (width+1)/4 or abs(row['rolling_z_score']) > width * 1.5:
      if signal:
        c1_pos = -c1_pos
        c2_pos = -c2_pos
        df.loc[date,s1_pos] = 0
        df.loc[date,s2_pos] = 0
        signal = False
        trade = True

    if trade:
      trades +=1
      # row[s1_pnl] = c1_pos * row[stock1]
      # row[s2_pnl] = c2_pos * row[stock2]
      # c1_pnl = c1_pos * row[stock1]
      # c2_pnl = c1_pos * row[stock2]
      df.loc[date,s1_pnl] = c1_pos * (row[stock1] - p1)
      df.loc[date,s2_pnl] = c2_pos * (row[stock2] - p2)


#       print('Z-score: '+ str(df['rolling_z_score'][i]), countS1, countS2, S1[i] , S2[i])
  #How to determine prices and loses. PnL only occurs when trade is made, trade is made
  df['P&L'] = df[s1_pnl] + df[s2_pnl]
  # print (df[s1_pnl].value_counts())
  return df
