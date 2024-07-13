import matplotlib.pyplot as plt

def plot_pairs_trading_results(df, stock1, stock2):
    fig, axs = plt.subplots(3, 1, figsize=(24, 15))
    axs[0].plot(df['spread'], label='Spread')
    axs[0].plot(df['upper_band'], label='Upper BB')
    axs[0].plot(df['lower_band'], label='Lower BB')
    axs[0].set_title("Bollinger Bands")
    axs[0].legend(loc="upper right")

    axs[1].plot(df[f'Position {stock1}'], label=stock1)
    axs[1].plot(df[f'Position {stock2}'], label=stock2)
    axs[1].set_title("Positions")
    axs[1].legend(loc="upper right")

    axs[2].plot(df['P&L'].cumsum())
    axs[2].set_title("Cumulative P&L")

    plt.show()
