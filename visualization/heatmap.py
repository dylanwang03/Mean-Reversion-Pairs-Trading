import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(corr_df):
    plt.figure(figsize=(26, 16))
    sns.heatmap(corr_df, annot=True, cmap="magma")
    plt.show()
