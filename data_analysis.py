import pandas as pd
import matplotlib.pyplot as plt
import constants
import utils

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
    fig_box, axes_box = plt.subplots(nrows=len(constants.DMA_NAMES), figsize=(10, 8))
    fig_lines, axes_lines = plt.subplots(nrows=len(constants.DMA_NAMES), figsize=(10, 8))

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
    fig_lines.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    fig_lines.text(0.5, 0.04, 'Hour of the day', ha='center')


def correlation_analysis(data):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 5))
    axes = axes.ravel()

    for i, col in enumerate(constants.DMA_NAMES):
        axes[i].scatter(data[col], data[col].shift(), s=15, alpha=0.3)
        axes[i].grid()

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

    fig_pivot.align_ylabels()
    fig_pivot.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)


if __name__ == "__main__":
    # loader = Loader()
    # preprocess = Preprocess(loader.inflow, loader.weather, cyclic_time_features=False, n_neighbors=3)
    # preprocess.export("resources/preprocessed_not_cyclic.csv")
    # data = preprocess.data

    data = utils.import_preprocessed("resources/preprocessed_not_cyclic.csv")

    # weather_features()
    # hourly_distribution()
    portion_of_total()

    plt.show()