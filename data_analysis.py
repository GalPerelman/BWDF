import pandas as pd
import matplotlib.pyplot as plt
import constants
import utils


def weather_features():
    fig, axes = plt.subplots(nrows=len(constants.WEATHER_COLUMNS), figsize=(8, 7))

    for i, col in enumerate(constants.WEATHER_COLUMNS):
        axes[i].hist(data[col], bins=50, edgecolor='k', alpha=0.4)
        axes[i].set_ylabel(col, fontsize=9)

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.25)
    fig.align_ylabels()


def hourly_distribution():
    for i, dma in enumerate(constants.DMA_NAMES):
        data_hourly = data.pivot_table(index=data.index, columns=data['hour'], values=dma, aggfunc='mean')
        data_hourly.boxplot()


def correlation_analysis(data):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 5))
    axes = axes.ravel()

    for i, col in enumerate(constants.DMA_NAMES):
        axes[i].scatter(data[col], data[col].shift(), s=15, alpha=0.3)
        axes[i].grid()

    plt.tight_layout()


if __name__ == "__main__":
    data = utils.import_preprocessed("resources/preprocessed_data.csv")
    weather_features()

    plt.show()