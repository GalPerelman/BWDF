import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches

import constants
import evaluation


def draw_test_periods(ax, y_min, y_max):
    for test_name, test_dates in constants.TEST_TIMES.items():
        rect = patches.Rectangle((test_dates['start'], y_min),
                                 pd.Timestamp(test_dates['end']) - pd.Timestamp(test_dates['start']),
                                 y_max - y_min,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    return ax


def plot_raw_data(data, columns, downscale=0, shade_missing: bool = False, axes=None, linestyle=None):
    if downscale > 0:
        data = data.iloc[::downscale]

    if axes is None:
        fig, axes = plt.subplots(nrows=len(data.columns), sharex=True, figsize=(12, 8))

    for i, col in enumerate(columns):
        axes[i].plot(data[col], linestyle=linestyle, label='Raw')
        axes[i].text(0.03, 0.8, f"{col[:-6]}", transform=axes[i].transAxes, ha='center', va='center')

        y_min, y_max = axes[i].get_ylim()

        if shade_missing:
            nan_positions = data[col].isna().values
            axes[i].fill_between(data.index, y_min, y_max, where=nan_positions, color='gray', alpha=0.3)

        axes[i] = draw_test_periods(axes[i], y_min, y_max)

    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    return axes


def visualize_data_completion(raw_data, completed_data, columns):
    axes = plot_raw_data(data=raw_data, columns=columns, downscale=0, shade_missing=True)

    for i, col in enumerate(columns):
        nan_idx = raw_data[raw_data[col].isna()].index

        gaps = np.where(np.diff(nan_idx) > pd.Timedelta('1H'))[0]
        start = 0

        for gap in gaps:
            axes[i].plot(completed_data.loc[nan_idx[start]:nan_idx[gap], col], 'navy', marker='o', markersize=2)
            start = gap + 1

        if start < len(nan_idx):
            axes[i].plot(completed_data.loc[nan_idx[start]:nan_idx[-1], col], 'navy', marker='o', markersize=2)

        line1 = Line2D([0], [0], label='Raw data', color='C0')
        line2 = Line2D([0], [0], label='Imputed data', color='navy')
        patch = patches.Patch(edgecolor='r', facecolor='none', label='Test period')

        axes[-1].legend(handles=[line1, line2, patch])


def plot_test(observed, predicted, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    mae = evaluation.mean_abs_error(observed, predicted)

    ax.plot(observed.index, observed)
    ax.plot(predicted.index, predicted)
    ax.text(0.05, 0.92, f"MAE={mae:.3f}", transform=ax.transAxes, ha='left', va='center')
    ax.set_ylabel("Net Inflow L/s")
    ax.grid()
    return ax


def plot_weather(weather_data: pd.DataFrame, shade_missing=False, axes=None):
    if axes is None:
        fig, axes = plt.subplots(nrows=len(weather_data.columns), sharex=True, figsize=(10, 7))

    for i, col in enumerate(weather_data.columns):
        axes[i].plot(weather_data[col])
        axes[i].set_ylabel(col)

        y_min, y_max = axes[i].get_ylim()

        if shade_missing:
            nan_positions = weather_data[col].isna().values
            axes[i].fill_between(weather_data.index, y_min, y_max, where=nan_positions, color='gray', alpha=0.3)

        axes[i] = draw_test_periods(axes[i], y_min, y_max)
        axes[i].grid()

    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.2)
    return axes


def plot_pareto():
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 6))
    axes = axes.ravel()

    for i, dma in enumerate(constants.DMA_NAMES):
        rf = pd.read_csv(f"grid_search_output/short_{dma[:5]}_rf.csv")
        xgb = pd.read_csv(f"grid_search_output/short_{dma[:5]}_xgb.csv")

        axes[i].scatter(rf['mean_test_MAE'], rf['mean_test_MAX_E'], label='RF', s=10, alpha=0.5)
        axes[i].scatter(xgb['mean_test_MAE'], xgb['mean_test_MAX_E'], label='XGB', s=10, alpha=0.5)
        axes[i].grid()

        # print long term best params
        rf = pd.read_csv(f"grid_search_output/long_{dma[:5]}_rf.csv")
        xgb = pd.read_csv(f"grid_search_output/long_{dma[:5]}_xgb.csv")
        print(dma, 'RF:', rf['mean_test_MAE'].max(), rf.loc[rf['rank_test_MAE'].idxmin()]['params'])
        print(dma, 'XGB:', xgb['mean_test_MAE'].max(), xgb.loc[xgb['rank_test_MAE'].idxmin()]['params'])
        print('=========================================================================================')

    fig.delaxes(axes[10])
    fig.delaxes(axes[11])

    handles, labels = axes[0].get_legend_handles_labels()
    plt.legend(handles, labels)

    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.3, wspace=0.3)
    plt.show()