import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd
import numpy as np


def calculate_kendall_correlation(df_pivot):
    corr_df = df_pivot.corr(method='kendall')
    corr_df.reset_index()
    stacked = corr_df.rename_axis(None).rename_axis(None, axis=1)
    stacked = stacked.stack().reset_index()
    stacked = stacked.sort_values(0)
    non_self = stacked[stacked[0] != 1]
    non_self = non_self.rename(columns={'level_0': 'Stock 1', 'level_1': 'Stock 2', 0: "Kendall Corr"})
    non_self_dup = non_self[-1::-2]
    return non_self_dup


def cointegration(x, y):
    regr = linear_model.LinearRegression()
    x_constant = pd.concat([x, pd.Series([1]*len(x), index=x.index)], axis=1)
    regr.fit(x_constant, y)
    beta = regr.coef_[0]
    alpha = regr.intercept_
    spread = y - x*beta - alpha
    adf = sm.tsa.stattools.adfuller(spread, autolag='AIC')
    return adf[1]

def find_cointegrated_pairs(consolidated, non_self_dup):
    for i in range(consolidated.shape[1]):
        if consolidated.iloc[:, i].isnull().values.any():
            print(consolidated.columns[i])

    c1 = non_self_dup['Stock 1'][:200]
    c2 = non_self_dup['Stock 2'][:200]

    p_values = []

    c_chart = pd.DataFrame()
    c_chart['Stock 1'] = c1
    c_chart['Stock 2'] = c2

    for i in range(c1.shape[0]):
        column_1 = consolidated[c1.iloc[i]]
        column_2 = consolidated[c2.iloc[i]]
        coint_val = cointegration(column_1, column_2)
        p_values.append(coint_val)

    c_chart['p_value'] = p_values
    c_chart = c_chart.reset_index()
    c_chart = c_chart.sort_values('p_value')
    c_chart = c_chart[c_chart['p_value'] <= .05]
    print(c_chart)

    compare_cols = ['Stock 1', 'Stock 2']
    mask = pd.Series(list(zip(*[non_self_dup[c] for c in compare_cols]))).isin(list(zip(*[c_chart[c] for c in compare_cols])))

    mask_list = mask.tolist()
    corr_coint = non_self_dup[mask_list]

    corr_coint = corr_coint[corr_coint.iloc[:, 2] > .7]
    print(corr_coint.tail())
    return corr_coint
