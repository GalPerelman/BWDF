import pandas as pd
import matplotlib.pyplot as plt

import constants


def plot_dmas(data: pd.DataFrame, downscale=0, shade_missing: bool = False, axes=None, linestyle=None):
    if downscale > 0:
        data = data.iloc[::downscale]

    if axes is None:
        fig, axes = plt.subplots(nrows=len(data.columns), sharex=True, figsize=(12, 8))

    for i, col in enumerate(constants.DMA_NAMES):
        axes[i].plot(data[col], linestyle=linestyle)

        if shade_missing:
            nan_positions = data[col].isna().values
            y_min, y_max = axes[i].get_ylim()
            axes[i].fill_between(data.index, y_min, y_max, where=nan_positions, color='gray', alpha=0.3,)

    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    return axes