import math
import ast
import glob
import os

import pandas as pd
import numpy as np
import warnings
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as patches
from matplotlib import colors as mcolors
import textwrap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import constants
import evaluation
import utils

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

MODELS_COLORS = {"xgb": "#0077b8", "prophet": "#aa341c", "lstm": "#4ab85e", "multi": "#faa80f", "rf": "#fb8500",
                 "patch": "#94D2BD", "sarima": "#001219"}


def draw_test_periods(ax, y_min, y_max):
    for test_name, test_dates in constants.TEST_TIMES.items():
        rect = patches.Rectangle((test_dates['start_test'], y_min),
                                 pd.Timestamp(test_dates['end_test']) - pd.Timestamp(test_dates['start_test']),
                                 y_max - y_min,
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)

    return ax


def plot_time_series(data, columns, downscale=0, shade_missing: bool = False, test_periods: bool = False, fig=None,
                     linestyle=None):
    if downscale > 0:
        data = data.iloc[::downscale]

    if fig is None:
        fig, axes = plt.subplots(nrows=len(columns), sharex=True, figsize=(12, 8))
    else:
        axes = fig.axes

    for i, col in enumerate(columns):
        axes[i].plot(data[col], linestyle=linestyle, label='Raw')
        if len(f"{col}") > 15:
            axes[i].set_ylabel(textwrap.fill(f"{col}", 15))
        axes[i].grid(True)

        y_min, y_max = axes[i].get_ylim()

        if shade_missing:
            nan_positions = data[col].isna().values
            axes[i].fill_between(data.index, y_min, y_max, where=nan_positions, color='gray', alpha=0.3)

        if test_periods:
            axes[i] = draw_test_periods(axes[i], y_min, y_max)

    plt.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    fig.align_ylabels()
    return fig


def visualize_data_completion(raw_data, completed_data, columns):
    axes = plot_time_series(data=raw_data, columns=columns, downscale=0, shade_missing=True, test_periods=True)

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


def plot_test(observed, predicted, ylabel='', ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    mae = evaluation.mean_abs_error(observed, predicted)
    mape = evaluation.mean_abs_percentage_error(observed, predicted)
    i1, i2, i3 = evaluation.one_week_score(observed, predicted)
    ax.plot(observed.index, observed)
    ax.plot(predicted.index, predicted)
    anchored_text = AnchoredText(f"I1={i1:.3f}\nI2={i2:.3f}\nI3={i3:.3f}\nMAE={mae:.3f}\nMAPE={mape:.2%}",
                                 loc='upper right', prop=dict(fontsize=7))
    ax.add_artist(anchored_text)
    ax.set_ylabel(f"{ylabel}")
    ax.grid()
    return ax


def feature_importance(model, x_train: pd.DataFrame):
    max_bars_in_subplot = 28
    importances = model.feature_importances_

    # sort features by their importance
    indices = np.argsort(importances)[::-1]
    names = [x_train.columns[i] for i in indices]
    values = importances[indices]

    nrows = max(1, int(np.ceil(x_train.shape[1] / max_bars_in_subplot)))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharey=True, figsize=(8, 7))
    if nrows == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        start_idx = i * max_bars_in_subplot
        end_idx = min(start_idx + max_bars_in_subplot, len(values))
        ax.bar(range(start_idx, end_idx, 1), values[start_idx: end_idx])
        ax.set_xticks(range(start_idx, end_idx, 1))
        ax.set_xticklabels(names[start_idx: end_idx], rotation=45, ha="right", va="top", fontsize=8)

    plt.subplots_adjust(bottom=0.2, top=0.96, left=0.1, right=0.9, hspace=1.2)


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
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 7))
    axes = axes.ravel()

    for i, dma in enumerate(constants.DMA_NAMES):
        rf = pd.read_csv(f"grid_search_output/short_{dma[:5]}_rf.csv") * -1
        xgb = pd.read_csv(f"grid_search_output/short_{dma[:5]}_xgb.csv") * -1
        prophet = pd.read_csv(f"grid_search_output/short_{dma[:5]}_prophet.csv") * -1
        lstm = pd.read_csv(f"grid_search_output/short_{dma[:5]}_lstm.csv") * -1
        mulit = pd.read_csv(f"grid_search_output/short_{dma[:5]}_multiseries.csv")

        axes[i].scatter(rf['mean_test_MAE'], rf['mean_test_MAX_E'], label='RF', s=10, alpha=0.8)
        axes[i].scatter(xgb['mean_test_MAE'], xgb['mean_test_MAX_E'], label='XGB', s=10, alpha=0.8)
        axes[i].scatter(prophet['mean_test_MAE'], prophet['mean_test_MAX_E'], label='Prophet', s=10, alpha=0.8)
        axes[i].scatter(lstm['mean_test_MAE'], lstm['mean_test_MAX_E'], label='LSTM', s=10, alpha=0.8)
        axes[i].scatter(mulit['mean_absolute_error'], mulit['max_abs_error'], label='MULTI', s=10, alpha=0.8)
        axes[i].set_axisbelow(True)
        axes[i].grid(zorder=0)
        axes[i].set_title(f"{dma[:5]}", fontsize=10)
        axes[i].set_xlim(0, 5)
        axes[i].set_ylim(0, 12)

        # print long term best params
        rf = pd.read_csv(f"grid_search_output/long_{dma[:5]}_rf.csv")
        xgb = pd.read_csv(f"grid_search_output/long_{dma[:5]}_xgb.csv")
        prophet = pd.read_csv(f"grid_search_output/long_{dma[:5]}_prophet.csv")
        lstm = pd.read_csv(f"grid_search_output/long_{dma[:5]}_lstm.csv")
        mulit = pd.read_csv(f"grid_search_output/long_{dma[:5]}_multiseries.csv")
        print(dma, 'RF:', rf['mean_test_MAE'].max(), rf.loc[rf['rank_test_MAE'].idxmin()]['params'])
        print(dma, 'XGB:', xgb['mean_test_MAE'].max(), xgb.loc[xgb['rank_test_MAE'].idxmin()]['params'])
        print(dma, 'Prophet:', prophet['mean_test_MAE'].max(), prophet.loc[prophet['rank_test_MAE'].idxmin()]['params'])
        print(dma, 'LSTM:', lstm['mean_test_MAE'].max(), lstm.loc[lstm['rank_test_MAE'].idxmin()]['params'])
        print(dma, 'MULTI:', mulit['mean_absolute_error'].min(), mulit.loc[mulit['max_abs_error'].idxmax()]['params'])
        print('=========================================================================================')

    fig.delaxes(axes[10])
    fig.delaxes(axes[11])
    fig.text(0.5, 0.03, 'Mean Abs Error', ha='center')
    fig.text(0.04, 0.5, 'Max Abs Error', va='center', rotation='vertical')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.5, 0.12), bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.1, top=0.95, left=0.1, right=0.9, hspace=0.4, wspace=0.3)
    plt.show()


def visualize_nans(df):
    """
    Plot the df as matrix with white square where data exist and black square where Nan
    Used mostly for debugging
    """
    nan_mask = df.isna()
    fig = plt.figure(figsize=(15, 8))

    plt.imshow(nan_mask, aspect='auto', cmap='gray_r', interpolation='none')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, rotation=90)

    plt.colorbar(label='NaN Values\nBlack: NaN\nWhite: Not NaN')
    plt.subplots_adjust(bottom=0.35, top=0.95, left=0.1, right=0.95)
    return fig


def select_models(df, subplots_col, hover_cols, colors_col, markers_col, x_col, y_col, file_name):
    df[x_col] = df[x_col].astype(float).abs()
    df[y_col] = df[y_col].astype(float).abs()

    markers = ["circle", "square", "diamond", "x", "triangle-up"]
    columns = ['i1', 'i2', 'i3', 'mape', 'model_name', 'short_model_name', 'dates_idx', 'start_test', 'horizon',
               'cols_to_move_stat', 'window_size', 'cols_to_decompose', 'norm', 'clusters_idx']

    _hover_cols = columns + hover_cols
    hover_cols = list(set(_hover_cols) & set(df.columns))

    index_map = {value: index for index, value in enumerate(_hover_cols)}
    hover_cols = sorted(hover_cols, key=lambda x: index_map[x])

    if subplots_col is not None:
        n_subplots = df[subplots_col].nunique()
        nrows = math.ceil(n_subplots / 3)
        ncols = 3
    else:
        n_subplots = 1
        nrows = 1
        ncols = 1

    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=constants.DMA_NAMES,
                        vertical_spacing=0.07, horizontal_spacing=0.04)

    for i in range(n_subplots):
        if n_subplots > 1:
            subplots_title = df[subplots_col].unique()[i]
            temp = df.loc[df[subplots_col] == subplots_title]
            temp['colors'] = temp[colors_col].map(MODELS_COLORS)
        else:
            subplots_title = ''
            temp = df.copy()
            temp['colors'] = "#1f91ad"

        markers_map = {}
        if markers_col:
            for _, element in enumerate(df[markers_col].unique()):
                markers_map[element] = markers[_]

            temp['markers'] = temp[markers_col].map(markers_map)
        else:
            temp['markers'] = "circle"

        row, col = divmod(i, 3)
        row += 1
        col += 1

        hover_text = [
            "".join(f"{col}: {temp.iloc[j][col]}<br>" for col in hover_cols if not pd.isnull(temp.iloc[j][col]))
            for j in range(len(temp))]
        temp = temp.dropna(axis=1, how='all')
        fig.add_trace(go.Scatter(x=temp[x_col], y=temp[y_col], mode='markers', name=subplots_title, showlegend=False,
                                 hoverinfo='text',
                                 text=hover_text,
                                 marker=dict(color=temp['colors'], symbol=temp['markers'], line_color="black",
                                             line_width=0.5)),
                      row=row, col=col)

        if temp[x_col].max() > 50:
            fig.update_xaxes(range=[0, 10], row=row, col=col)
        if temp[y_col].max() > 50:
            fig.update_yaxes(range=[0, 10], row=row, col=col)

        row = row,
        col = col + 1

    fig.show()
    if file_name:
        fig.write_html(f"{file_name}.html")


def collect_gridsearch(dir_path, dmas, horizon, model):
    df = pd.DataFrame()
    for fname in glob.glob(dir_path + "/*.csv"):
        spitedname = os.path.basename(fname).split('_')
        _horizon, _dma, _model = spitedname[0], spitedname[1], spitedname[2][:-4]

        if _horizon == horizon and _dma + ' (L/s)' in dmas and _model == model:
            temp = pd.read_csv(fname, index_col=0)
            temp['dma'] = _dma
            temp['model_name'] = _model
            df = pd.concat([df, temp])

    df.reset_index(inplace=True)
    return df


def analyze_hyperparameters(dir_path, dma_idx, horizon, dates_idx, models):
    df = utils.collect_experiments(dir_path, p=10, dmas=[dma_idx], horizon=horizon, dates_idx=dates_idx, models=models)
    df = df.drop(['level_0', 'index'], axis=1)
    df = df.dropna(how='all', axis=1)

    param_cols = [col for col in df.columns if col.startswith("param_") and
                  col not in ['param_bootstrap', 'param_min_sample_leaf', 'param_min_sample_split']]
    n_params = len(param_cols)

    colors = ["#f8da76", "#34b680", "#105f90", "#5f097a", "#c7f9cc"]
    markers = ["o", "*", "s", "d", "^", "+"]
    norm_methods = list(df['norm'].unique())

    fig, axes = plt.subplots(nrows=2, ncols=n_params, figsize=(14, 7))
    for i, col in enumerate(param_cols):
        param_values = list(df[col].dropna().unique())
        temp_color_map = {param_values[_]: colors[_] for _ in range(len(param_values))}

        for val, color in temp_color_map.items():
            for j, norm in enumerate(norm_methods):
                temp = df.loc[(df[col] == val) & (df['norm'] == norm)]
                label = f"{col[6:]}={val} | {norm}"
                axes[0, i].scatter(temp['i1'], temp['i2'], c=color, marker=markers[j], alpha=0.5, label=label, zorder=3)

        axes[0, i].grid(zorder=0)
        axes[0, i].set_xlabel('i1')
        axes[0, i].set_ylabel('i2')
        axes[1, i].axis('off')
        handles, labels = axes[0, i].get_legend_handles_labels()
        axes[1, i].legend(handles, labels, fontsize=7)
        axes[0, i].set_title(col[6:])

    fig.suptitle(f"Model: {models} | Dates idx: {dates_idx}")
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.06, right=0.96, wspace=0.28)

    lag_cols = [col for col in df.columns if col.startswith("lags_") or col.startswith("cols_to")]
    n = len(lag_cols)
    fig, axes = plt.subplots(nrows=2, ncols=n, figsize=(14, 7))
    for i, col in enumerate(lag_cols):
        param_values = list(df[col].unique())
        temp_color_map = {param_values[_]: colors[_] for _ in range(len(param_values))}

        for val, color in temp_color_map.items():
            for j, norm in enumerate(norm_methods):
                temp = df.loc[(df[col] == val) & (df['norm'] == norm)]
                label = f"{col}={val}\n{norm}"
                axes[0, i].scatter(temp['i1'], temp['i2'], c=color, marker=markers[j], alpha=0.5, label=label, zorder=3)

        axes[0, i].grid(zorder=0)
        axes[1, i].axis('off')
        handles, labels = axes[0, i].get_legend_handles_labels()
        axes[1, i].legend(handles, labels, fontsize=7)
        axes[0, i].set_title(col[6:])

    fig.suptitle(f"Model: {models} | Dates idx: {dates_idx}")
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.06, right=0.96, wspace=0.28)


def plot_all_experiments(p):
    df = utils.collect_experiments("exp_output/v3", p=p, dmas=[_ for _ in range(10)], horizon='short',
                                   dates_idx=[0, 2, 3, 4],
                                   models=['multi', 'xgb', 'lstm', 'prophet', 'patch', 'sarima'])

    select_models(df=df,
                  subplots_col='dma',
                  hover_cols=[_ for _ in df.columns if (_.startswith('param_')) or (_.startswith('lags_'))],
                  colors_col='model_name',
                  markers_col='start_test',
                  x_col='i1',
                  y_col='i2',
                  file_name='short')

    df = utils.collect_experiments("exp_output/v3", p=p, dmas=[_ for _ in range(10)], horizon='long',
                                   dates_idx=[0, 2, 3, 4],
                                   models=['multi', 'xgb', 'lstm', 'prophet', 'patch', 'sarima'])

    select_models(df=df,
                  subplots_col='dma',
                  hover_cols=[_ for _ in df.columns if (_.startswith('param_')) or (_.startswith('lags_'))],
                  colors_col='model_name',
                  markers_col='start_test',
                  x_col='i3',
                  y_col='mape',
                  file_name='long')


def analyze_cv(cv_path, ranking_cols):
    df = pd.read_csv(cv_path)
    # df = df[df[ranking_cols[0]] <= 100]
    # df = df[df[ranking_cols[0]] <= df['i1'].mean() + df['i1'].std() * 5]

    fig, axes = plt.subplots(nrows=len(ranking_cols), figsize=(10, 3*len(ranking_cols)))
    axes = np.atleast_2d(axes).ravel()

    category_color_map = {str(category): MODELS_COLORS[label] for
                          category, label in zip(df['model_idx'], df['model_name'])}

    for i, col in enumerate(ranking_cols):
        sns.boxplot(x='model_idx', y=col, data=df, ax=axes[i], palette=category_color_map, showfliers=False)

        means = df.groupby('model_idx')[col].mean().reset_index()
        axes[i].scatter(x=means['model_idx'], y=means[col], color='black', marker='o', s=10, zorder=10)

        axes[i].grid()
        axes[i].set_ylim(axes[i].get_ylim()[0], axes[i].get_ylim()[1] + 1)

        categories = df['model_idx'].unique()
        for j, category in enumerate(categories):
            label = df[df['model_idx'] == category]['model_name'].iloc[0]
            axes[i].text(j, axes[i].get_ylim()[1] * 0.8, label, horizontalalignment='center', rotation=45)

    plt.subplots_adjust(bottom=0.1, top=0.93, left=0.08, right=0.95, hspace=0.28)
    fig.suptitle(cv_path.split('-')[1].split('_')[0] + ' - ' + cv_path.split('-')[1].split('_')[1].split('.')[0])


def analyze_all_cv():
    for dma in constants.DMA_NAMES:
        analyze_cv(f"experiments_analysis/cv/cv_dma-{dma[:5]}_short.csv", ranking_cols=['i1', 'i2'])
        analyze_cv(f"experiments_analysis/cv/cv_dma-{dma[:5]}_long.csv", ranking_cols=['i3'])


if __name__ == "__main__":
    # plot_pareto()
    # analyze_hyperparameters("exp_output/v3", dma_idx=0, horizon="short", dates_idx=[0, 1, 2, 3],
    #                         models=['multi', 'xgb', 'lstm', 'prophet'], p=5)
    # analyze_hyperparameters("exp_output/v3", dma_idx=0, horizon="short", dates_idx=[0, 3, 4], models=['multi'])
    # analyze_hyperparameters("exp_output/v3", dma_idx=0, horizon="short", dates_idx=[3], models=['multi'])
    # analyze_hyperparameters("exp_output/v3", dma_idx=0, horizon="short", dates_idx=[4], models=['multi'])
    # plot_all_experiments(p=0.2)
    analyze_all_cv()

    plt.show()
