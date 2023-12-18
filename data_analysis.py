import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants
import graphs
import utils
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

from data_loader import Loader
from preprocess import Preprocess


def weather_features():
    fig, axes = plt.subplots(nrows=len(constants.WEATHER_COLUMNS), figsize=(8, 7))

    for i, col in enumerate(constants.WEATHER_COLUMNS):
        axes[i].hist(data[col], bins=50, edgecolor='k', alpha=0.4)
        axes[i].set_ylabel(col, fontsize=9)

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.25)
    fig.align_ylabels()


def hourly_distribution():
    fig_box, axes_box = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(10, 8))
    fig_lines, axes_lines = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True,  figsize=(10, 8))

    for i, dma in enumerate(constants.DMA_NAMES):
        temp = data[[dma]]
        pivot_temp = temp.pivot_table(values=dma, index=temp.index.date, columns=temp.index.hour)
        axes_box[i].boxplot(pivot_temp.values, positions=range(24))
        axes_box[i].set_ylabel(dma[:5])

        axes_lines[i].plot(pivot_temp.values.T, c='C0', alpha=0.2)
        axes_lines[i].set_ylabel(dma[:5])
        axes_lines[i].grid()

    fig_box.align_ylabels()
    fig_box.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    fig_box.text(0.5, 0.04, 'Hour of the day', ha='center')

    fig_lines.align_ylabels()
    fig_lines.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.9, hspace=0.1)
    fig_lines.text(0.5, 0.02, 'Hour of the day', ha='center')


def correlation_analysis(data):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
    axes = axes.ravel()

    for i, col in enumerate(constants.DMA_NAMES):
        axes[i].scatter(data[col], data[col].shift(), s=15, alpha=0.3)
        axes[i].grid()
        axes[i].set_title(col[:5])

    fig.suptitle(f"(t) - (t-1) correlation")
    fig.text(0.5, 0.02, 'Value at time t', ha='center')
    fig.text(0.04, 0.5, 'Value at time t-1', va='center', rotation='vertical')
    fig.subplots_adjust(bottom=0.12, top=0.88, left=0.1, right=0.95, wspace=0.35, hspace=0.35)

    # plot correlation between DMAs
    fig, ax = plt.subplots()
    corr = data[constants.DMA_NAMES].corr()
    im = ax.imshow(corr, cmap='Blues')

    n = len(data.columns)
    for i in range(n):
        for j in range(n):
            label = corr.iloc[i, j]
            ax.text(i, j, f"{label:.2f}", color='k', ha='center', va='center', fontsize=8)

    ax.set_xticks(np.arange(-.5, len(constants.DMA_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(constants.DMA_NAMES), 1), minor=True)
    ax.grid(which='minor', color='k')

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar(im)
    plt.tight_layout()


def portion_of_total():
    fig, axes = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(10, 8))
    fig_pivot, axes_pivot = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(10, 8))

    data['total'] = data[constants.DMA_NAMES].sum(axis=1)

    for i, dma in enumerate(constants.DMA_NAMES):
        axes[i].plot(data[dma] / data['total'])
        axes[i].set_ylabel(dma[:5])
        axes[i].grid()

        temp = pd.DataFrame(data[dma] / data['total'], columns=[dma])
        pivot_temp = temp.pivot_table(values=dma, index=temp.index.date, columns=temp.index.hour)
        axes_pivot[i].plot(pivot_temp.values.T, c='C0', alpha=0.2)
        axes_pivot[i].set_ylabel(dma[:5])
        axes_pivot[i].grid()

    fig.align_ylabels()
    fig.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    fig.suptitle('Portion from total')

    fig_pivot.align_ylabels()
    fig_pivot.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.9, hspace=0.1)
    fig_pivot.text(0.5, 0.02, 'Hour of the day', ha='center')
    fig_pivot.suptitle('Hourly portion from total')


def specific_demand():
    specific_data = data[constants.DMA_NAMES].div(constants.DMA_DATA['n'])
    fig, axes = plt.subplots(nrows=len(constants.DMA_NAMES), sharex=True, figsize=(10, 8))

    for i, dma in enumerate(constants.DMA_NAMES):
        temp = specific_data[[dma]]
        pivot_temp = temp.pivot_table(values=dma, index=temp.index.date, columns=temp.index.hour)

        axes[i].plot(pivot_temp.values.T, c='C0', alpha=0.2)
        axes[i].set_ylabel(dma[:5])
        axes[i].grid()

    fig.align_ylabels()
    fig.subplots_adjust(bottom=0.07, top=0.95, left=0.1, right=0.9, hspace=0.1)
    fig.text(0.5, 0.02, 'Hour of the day', ha='center')
    fig.suptitle('Specific demand by hour of the day')


def cluster_dmas():
    df = data.T.groupby(constants.DMA_DATA['type_code']).sum().T
    df = df.rename(columns=dict(zip(constants.DMA_DATA['type_code'], constants.DMA_DATA['type'])))
    df = df.loc[~(df == 0).all(axis=1)]

    daily = df.resample('1D').sum()
    daily = daily.loc[~(daily == 0).all(axis=1)]

    graphs.plot_time_series(data=df, columns=df.columns)
    graphs.plot_time_series(data=daily, columns=daily.columns)


def covid_data():
    covid = pd.read_csv("resources/covid.csv", index_col=0)
    covid.index = pd.to_datetime(covid.index, format="%d/%m/%Y", utc=True)
    covid.index = covid.index.tz_convert(constants.TZ).normalize()

    covid = covid.pivot_table(index=covid.index, columns='region_name', values='total_cases')
    covid = covid.sum(axis=1)
    covid.name = 'covid'

    covid.loc[(covid < 0) | (covid > covid.mean() + 3 * covid.std())] = covid.mean()
    return covid


def moving_stat(data, window_size):
    fig, axes = plt.subplots(nrows=10, ncols=2, sharex=True, figsize=(8, 6))

    for i, col in enumerate(constants.DMA_NAMES):
        rolling_windows = data[col].rolling(window=window_size, min_periods=1)
        axes[i, 0].plot(data.index, rolling_windows.mean().shift(window_size).values)
        axes[i, 1].plot(data.index, rolling_windows.std().shift(window_size).values)
        axes[i, 0].grid()
        axes[i, 1].grid()

        axes[i, 0].set_ylabel(col[:5])

    axes[0, 0].set_title("Moving Average")
    axes[0, 1].set_title("Moving STD")

    fig.subplots_adjust(bottom=0.06, top=0.95, left=0.1, right=0.9, hspace=0.25)
    fig.align_ylabels()
    plt.gcf().autofmt_xdate()


def stationary_test(data):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 6))
    axes = axes.ravel()

    for i, col in enumerate(constants.DMA_NAMES):
        data_diff = data[col].dropna()
        plot_acf(data_diff, lags=24, ax=axes[i])
        axes[i].set_title(f"{col[:5]}")

        adftest = adfuller(data[col].dropna(), autolag='AIC', regression='ct')
        print("ADF Test Results")
        print("Null Hypothesis: The series has a unit root (non-stationary)")
        print("ADF-Statistic:", adftest[0])
        print("P-Value:", adftest[1])
        print("Number of lags:", adftest[2])
        print("Number of observations:", adftest[3])
        print("Critical Values:", adftest[4])
        print("Note: If P-Value is smaller than 0.05, we reject the null hypothesis and the series is stationary")

    fig.subplots_adjust(bottom=0.12, top=0.92, left=0.1, right=0.9, hspace=0.15)
    fig.text(0.5, 0.02, 'Lags', ha='center')


if __name__ == "__main__":
    # loader = Loader()
    # preprocess = Preprocess(loader.inflow, loader.weather, cyclic_time_features=False, n_neighbors=3)
    # preprocess.export("resources/preprocessed_not_cyclic.csv")
    # data = preprocess.data

    data = utils.import_preprocessed("resources/preprocessed_not_cyclic.csv")

    # correlation_analysis(data.loc[constants.DATES_OF_LATEST_WEEK['start_train']:
    #                               constants.DATES_OF_LATEST_WEEK['end_test'],
    #                      constants.DMA_NAMES])
    # weather_features()
    # hourly_distribution()
    # portion_of_total()
    # specific_demand()
    # cluster_dmas()
    # moving_stat(data, window_size=24)
    #

    stationary_test(data)

    plt.show()
