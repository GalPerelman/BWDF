import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches

import constants


def plot_dmas(data: pd.DataFrame, downscale=0, shade_missing: bool = False, axes=None, linestyle=None):
    if downscale > 0:
        data = data.iloc[::downscale]

    if axes is None:
        fig, axes = plt.subplots(nrows=len(data.columns), sharex=True, figsize=(12, 8))

    for i, col in enumerate(constants.DMA_NAMES):
        axes[i].plot(data[col], linestyle=linestyle, label='Raw')
        axes[i].text(0.03, 0.8, f"{col[:-6]}", transform=axes[i].transAxes, ha='center', va='center')

        y_min, y_max = axes[i].get_ylim()

        if shade_missing:
            nan_positions = data[col].isna().values
            axes[i].fill_between(data.index, y_min, y_max, where=nan_positions, color='gray', alpha=0.3)

        for test_name, test_dates in constants.TEST_TIMES.items():
            rect = patches.Rectangle((test_dates['start'], y_min),
                                     pd.Timestamp(test_dates['end']) - pd.Timestamp(test_dates['start']),
                                     y_max - y_min,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            axes[i].add_patch(rect)

    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    return axes


def visualize_data_completion(raw_data, completed_data):
    axes = plot_dmas(data=raw_data, downscale=0, shade_missing=True)

    for i, col in enumerate(constants.DMA_NAMES):
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