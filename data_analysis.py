import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

from data_loader import Loader
from preprocess import Preprocess
import constants
import graphs
import utils


def weather_features():
    fig, axes = plt.subplots(nrows=len(constants.WEATHER_COLUMNS), figsize=(8, 7))

    for i, col in enumerate(constants.WEATHER_COLUMNS):
        axes[i].hist(data[col], bins=50, edgecolor='k', alpha=0.4)
        axes[i].set_ylabel(col, fontsize=9)

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.25)
    fig.align_ylabels()


def hourly_distribution(data, columns):
    fig_box, axes_box = plt.subplots(nrows=len(columns), sharex=True, figsize=(10, 8))
    fig_lines, axes_lines = plt.subplots(nrows=len(columns), sharex=True,  figsize=(10, 8))
    fig_hist, axes_hist = plt.subplots(nrows=len(columns), sharex=True,  figsize=(10, 8))

    for i, col in enumerate(data[columns].columns):
        temp = data[[col]]
        pivot_temp = temp.pivot_table(values=col, index=temp.index.date, columns=temp.index.hour)
        axes_box[i].boxplot(pivot_temp.values, positions=range(24))
        axes_box[i].set_ylabel(textwrap.fill(f"{col}", 15))

        axes_lines[i].plot(pivot_temp.values.T, c='C0', alpha=0.2)
        axes_lines[i].set_ylabel(textwrap.fill(f"{col}", 15))
        axes_lines[i].grid()

        axes_hist[i].hist(data[col], bins=25)
        axes_hist[i].set_ylabel(textwrap.fill(f"{col}", 15))

    fig_box.align_ylabels()
    fig_box.subplots_adjust(bottom=0.05, top=0.95, left=0.1, right=0.9, hspace=0.1)
    fig_box.text(0.5, 0.04, 'Hour of the day', ha='center')

    fig_lines.align_ylabels()
    fig_lines.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.9, hspace=0.1)
    fig_lines.text(0.5, 0.02, 'Hour of the day', ha='center')

    fig_hist.align_ylabels()
    fig_hist.subplots_adjust(bottom=0.1, top=0.98, left=0.1, right=0.9, hspace=0.1)


def correlation_analysis(data, norm_method=''):
    data = data.loc[(data.index >= constants.DATES_OF_LATEST_WEEK['start_train'])
                    & (data.index < constants.DATES_OF_LATEST_WEEK['end_test']), constants.DMA_NAMES]

    if norm_method:
        data, scalers = Preprocess.fit_transform(data, columns=constants.DMA_NAMES, method=norm_method)

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
    axes = axes.ravel()

    for i, col in enumerate(constants.DMA_NAMES):
        # axes[i].scatter(data[col], data[col].shift(), s=15, alpha=0.3, zorder=5)
        axes[i].plot(data[col])
        axes[i].grid(zorder=0)
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

        adfstat, pval, usedlag, nobs, critvalues, icbest = adfuller(data[col].dropna(), autolag='AIC', regression='ct')
        print("ADF Test Results")
        print("Null Hypothesis: The series has a unit root (non-stationary)")
        print("ADF-Statistic:", adfstat[0])
        print("P-Value:", pval[1])
        print("Number of lags:", usedlag[2])
        print("Number of observations:", nobs[3])
        print("Critical Values:", critvalues[4])
        print("Note: If P-Value is smaller than 0.05, we reject the null hypothesis and the series is stationary")

    fig.subplots_adjust(bottom=0.12, top=0.92, left=0.1, right=0.9, hspace=0.15)
    fig.text(0.5, 0.02, 'Lags', ha='center')


def outliers_analysis(data, method, param, window_size, stuck_threshold):
    """
    Use before configuring forecast to identify preprocess parameters
    """
    numeric_columns = data.select_dtypes(include=np.number).columns
    n_cols = len(numeric_columns)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 1.4 * n_cols), sharex='col',
                             gridspec_kw={'width_ratios': [4, 1]})

    for i, column in enumerate(numeric_columns):
        if method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=0.05)  # Adjust contamination as needed
            data[column + '_outlier'] = iso_forest.fit_predict(data[[column]])
            colors = np.where(data[column + '_outlier'] == -1, 'red', 'C0')
            axes[i, 0].scatter(data.index, data[column], c=colors, s=15, alpha=0.6, zorder=5)

        elif method == 'z_score':
            z_scores = stats.zscore(data[column].dropna())
            outliers = z_scores > param  # You can adjust this threshold
            axes[i, 0].scatter(data[column].index, data[column], c='C0', s=15, alpha=0.6, zorder=5)
            axes[i, 0].scatter(data[column].index[outliers], data[column][outliers], c='red', s=15, alpha=0.6, zorder=5)

        elif method =='iqr':
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - param * iqr
            upper_bound = q3 + param * iqr
            outliers = (data[column] < lower_bound) | (data[column] > upper_bound)

            axes[i, 0].scatter(data[column].index, data[column], c='C0', s=15, alpha=0.6, zorder=5)
            axes[i, 0].scatter(data[column].index[outliers], data[column][outliers], c='red', s=15, alpha=0.6, zorder=5)

        elif method == "rolling_iqr":
            rolling_q1 = data[column].rolling(window=window_size, min_periods=1).quantile(0.25)
            rolling_q3 = data[column].rolling(window=window_size, min_periods=1).quantile(0.75)

            # Calculate the rolling IQR
            rolling_iqr = rolling_q3 - rolling_q1
            data['rolling_q1'] = rolling_q1
            data['rolling_q3'] = rolling_q3
            data['rolling_iqr'] = rolling_iqr

            # Calculate lower and upper bounds for outlier detection
            data['lower_bound'] = data['rolling_q1'] - param * data['rolling_iqr']
            data['upper_bound'] = data['rolling_q3'] + param * data['rolling_iqr']

            # Identify outliers
            outliers = (data[column] < data['lower_bound']) | (data[column] > data['upper_bound'])

            # outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            axes[i, 0].scatter(data[column].index, data[column], c='C0', s=15, alpha=0.6, zorder=5)
            axes[i, 0].scatter(data[column].index[outliers], data[column][outliers], c='red', s=15, alpha=0.6, zorder=5)

        # for all methods identify stuck data streams
        diff = data[column].diff().ne(0)
        groups = diff.cumsum()
        group_sizes = data.groupby(groups)[column].transform('size')
        outliers = group_sizes >= stuck_threshold
        axes[i, 0].scatter(data[column].index[outliers], data[column][outliers], c='purple', s=15, alpha=0.6, zorder=5)

        axes[i, 1].hist(data[column], bins=20, edgecolor='k', linewidth=0.1, alpha=0.8, zorder=5)
        axes[i, 0].set_ylabel(column)
        axes[i, 0].grid(zorder=0)
        axes[i, 1].grid(zorder=0)

    fig.suptitle(f"outliers according to {param} STDs from mean")
    fig.subplots_adjust(bottom=0.08, top=0.92, left=0.1, right=0.95, hspace=0.2)
    fig.align_ylabels()


def knn_outlier_detection(df, n_neighbors=5, outlier_fraction=0.05):
    df = df.dropna(axis=0)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(df)
    distances, indices = nbrs.kneighbors(df)

    outlier_scores = distances.mean(axis=1)
    threshold = np.percentile(outlier_scores, 100 * (1 - outlier_fraction))
    outliers = outlier_scores > threshold

    fig, axes = plt.subplots(len(df.columns), 1, figsize=(10, 1.8 * len(df.columns)), sharex=True)
    for i, column in enumerate(df.columns):
        axes[i].scatter(df[column].index, df[column], c='C0', s=15, alpha=0.6, zorder=5)
        axes[i].scatter(df.index[outliers], df[column][outliers], c='red', s=15, alpha=0.6, zorder=5)
        axes[i].grid()

    fig.legend()
    plt.tight_layout()
    plt.show()
    return outliers


def check_scalers(scaler):
    """
    Plot the original data, normalized data, and the restored data to check that scalers are working as expected
    """
    fig, axes = plt.subplots(nrows=2, ncols=len(constants.WEATHER_COLUMNS), sharex=True, figsize=(8, 6))
    for i, col in enumerate(constants.WEATHER_COLUMNS):
        axes[0, i].plot(data[col])

    data[constants.WEATHER_COLUMNS] = scaler.fit_transform(data[constants.WEATHER_COLUMNS])
    for i, col in enumerate(constants.WEATHER_COLUMNS):
        axes[1, i].plot(data[col])
        axes[1, i].grid()

    data[constants.WEATHER_COLUMNS] = scaler.inverse_transform(data[constants.WEATHER_COLUMNS])
    for i, col in enumerate(constants.WEATHER_COLUMNS):
        axes[0, i].plot(data[col], linestyle='--')
        axes[0, i].grid()
    plt.tight_layout()


if __name__ == "__main__":
    loader = Loader()
    # preprocess = Preprocess(loader.inflow, loader.weather, cyclic_time_features=False, n_neighbors=3)
    # preprocess.export("resources/preprocessed_not_cyclic.csv")
    # data = preprocess.data
    data = utils.import_preprocessed("resources/preprocessed_not_cyclic.csv")

    # hourly_distribution(data, columns=constants.DMA_NAMES)
    # correlation_analysis(data)
    # correlation_analysis(data, norm_method='diff')
    # weather_features()
    # hourly_distribution()
    # portion_of_total()
    # specific_demand()
    # cluster_dmas()
    # moving_stat(data, window_size=24)
    # stationary_test(data)

    outliers_analysis(data=loader.inflow, method='rolling_iqr', param=3, window_size=168*4, stuck_threshold=5)
    outliers_analysis(data=loader.weather, method='iqr', param=5, window_size=None, stuck_threshold=24)
    # knn_outlier_detection(data[constants.DMA_NAMES])
    # graphs.plot_time_series(loader.inflow, columns=constants.DMA_NAMES, shade_missing=True)
    # graphs.plot_time_series(loader.weather, columns=constants.WEATHER_COLUMNS, shade_missing=True)
    # check_scalers(scaler=preprocess.FixedWindowScaler())
    plt.show()
