import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2
from keras.preprocessing.sequence import TimeseriesGenerator

import constants
import utils
import graphs
from preprocess import Preprocess
from data_loader import Loader


class LSTMForecaster:
    def __init__(self, data, start_train, start_valid, start_test, end_test, y_label, look_back):
        self.data = data
        self.start_train = start_train
        self.start_valid = start_valid
        self.start_test = start_test
        self.end_test = end_test
        self.y_label = y_label
        self.look_back = look_back

        self.train_data = None
        self.valid_data = None
        self.scalers = None

        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

        self.model = None

        self.preprocess()
        self.train_gen, self.valid_gen = self.build_supervised()
        self.build_model()

    def preprocess(self):
        self.data = utils.drop_other_dmas(self.data, self.y_label)

        train = self.data.loc[(self.data.index >= self.start_train) & (self.data.index < self.start_valid)]
        valid = self.data.loc[(self.data.index >= self.start_valid) & (self.data.index < self.start_test)]

        standardize_columns = constants.WEATHER_COLUMNS + [self.y_label]
        self.train_data, self.scalers = Preprocess.fit_transform(train, columns=standardize_columns, method='standard')
        self.valid_data = Preprocess.transform(valid, columns=standardize_columns, scalers=self.scalers)

        self.x_train = train.loc[:, [col for col in train.columns if col != self.y_label]]
        self.y_train = train.loc[:, self.y_label]
        self.x_valid = valid.loc[:, [col for col in valid.columns if col != self.y_label]]
        self.y_valid = valid.loc[:, self.y_label]

    def build_supervised(self):
        train_generator = TimeseriesGenerator(self.x_train, self.y_train, length=self.look_back, batch_size=12)
        valid_generator = TimeseriesGenerator(self.x_valid, self.y_valid, length=self.look_back, batch_size=12)
        return train_generator, valid_generator

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.look_back, self.x_train.shape[1]), return_sequences=True))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self):
        history = self.model.fit(self.train_gen, epochs=12, batch_size=48, validation_data=self.valid_gen, shuffle=False)
        return history

    def vaildation_predict(self):
        pred = self.model.predict(self.valid_gen)
        pred = self.scalers[self.y_label].inverse_transform(pred)
        true_values = self.scalers[self.y_label].inverse_transform(self.y_valid.values.reshape(-1, 1)).flatten()
        results = pd.DataFrame({'true': true_values[self.look_back:]}, index=self.y_valid.iloc[self.look_back:].index)
        results['pred'] = pred.flatten()
        return results

    def future_predict_preprocess(self, n_periods):
        # generate data set that includes train and validation
        final_data = self.data.loc[self.data.index < self.start_test]

        # normalize the final data
        standardize_columns = constants.WEATHER_COLUMNS + [self.y_label]
        final_data = Preprocess.transform(final_data, columns=standardize_columns, scalers=self.scalers)

        # split to features and target
        final_x = final_data.loc[:, [col for col in final_data.columns if col != self.y_label]]
        final_y = final_data.loc[:, self.y_label]
        final_train_x = []
        final_train_y = []

        for i in range(self.look_back, len(final_data) - n_periods + 1):
            final_train_x.append(final_x.iloc[i - self.look_back:i])
            final_train_y.append(final_y.iloc[i + n_periods - 1:i + n_periods])

        final_train_x, final_train_y = np.array(final_train_x), np.array(final_train_y)

        # prepare the future periods exogenous data
        future_exog = self.data.loc[(self.data.index >= self.start_test) & (self.data.index < self.end_test)]
        future_exog = future_exog.drop(self.y_label, axis=1)
        future_exog = future_exog.values

        return final_train_x, final_train_y, future_exog

    def future_predict(self):
        n_periods = utils.num_hours_between_timestamps(self.start_test, self.end_test)
        final_train_x, final_train_y, future_exog = self.future_predict_preprocess(n_periods)
        input_sequence = final_train_x[-1].reshape(1, final_train_x.shape[1], final_train_x.shape[2])

        forecast = []
        for i in range(n_periods):
            # Predict the next value using the current input sequence
            predicted_value = self.model.predict(input_sequence)[0, 0]

            # Append the predicted value to the forecast list
            forecast.append(predicted_value)

            # Roll the input sequence one step backward to remove the oldest time-step
            input_sequence = np.roll(input_sequence, shift=-1, axis=1)

            # Construct a new timestep data: take the first 15 exogenous features and append the predicted value
            new_timestep = np.append(future_exog[i, :-1], predicted_value)

            # Insert this new timestep at the end of the input sequence
            input_sequence[0, -1] = new_timestep

        forecast = np.array(forecast).reshape(-1, 1)
        forecast = self.scalers[self.y_label].inverse_transform(forecast).flatten()
        forecast_period_dates = pd.date_range(start=self.start_test,
                                              end=self.end_test - datetime.timedelta(hours=1), freq='1H')
        forecast = pd.DataFrame({'forecast': forecast}, index=forecast_period_dates)
        return forecast

    def one_step_future_predict(self):
        n_periods = utils.num_hours_between_timestamps(self.start_test, self.end_test)
        final_data = self.data.loc[self.data.index < self.start_test]

        # normalize the final data
        standardize_columns = constants.WEATHER_COLUMNS + [self.y_label]
        final_data = Preprocess.transform(final_data, columns=standardize_columns, scalers=self.scalers)
        final_x = final_data.loc[:, [col for col in final_data.columns if col != self.y_label]]
        print(final_x.columns)
        # prepare the future periods exogenous data
        future_exog = self.data.loc[(self.data.index >= self.start_test) & (self.data.index < self.end_test)]
        print(future_exog.columns)
        future_exog = Preprocess.transform(future_exog, columns=standardize_columns, scalers=self.scalers)
        future_exog = future_exog.drop(self.y_label, axis=1)
        future_exog = future_exog.values

        initial_sequence_x = final_x.values[-self.look_back:]

        forecast = []
        current_sequence_x = initial_sequence_x.copy()

        for i in range(n_periods):
            # Predict the next step using only the feature sequence
            prediction = self.model.predict(current_sequence_x[np.newaxis, :, :])[0, 0]
            forecast.append(prediction)

            # Update the feature sequence:
            # 1. Remove the oldest data point
            # 2. Append the latest feature data from future_exog
            current_sequence_x = np.vstack((current_sequence_x[1:], future_exog[i].reshape(1, -1)))

        forecast = np.array(forecast).reshape(-1, 1)
        forecast = self.scalers[self.y_label].inverse_transform(forecast).flatten()
        forecast_period_dates = pd.date_range(start=self.start_test,
                                              end=self.end_test - datetime.timedelta(hours=1), freq='1H')
        forecast = pd.DataFrame({'forecast': forecast}, index=forecast_period_dates)
        return forecast


if __name__ == "__main__":
    # usage example
    loader = Loader()
    preprocess = Preprocess(loader.inflow, loader.weather, n_neighbors=3)
    data = preprocess.data
    y_label = constants.DMA_NAMES[0]

    start_train = constants.TZ.localize(datetime.datetime(2022, 1, 1, 0, 0))
    start_valid = constants.TZ.localize(datetime.datetime(2022, 7, 11, 0, 0))
    start_test = constants.TZ.localize(datetime.datetime(2022, 7, 18, 0, 0))
    end_test = start_test + datetime.timedelta(hours=168)

    lstm = LSTMForecaster(data=data, start_train=start_train, start_valid=start_valid, start_test=start_test,
                          end_test=end_test, y_label=y_label, look_back=6)

    lstm.fit()
    valid_pred = lstm.vaildation_predict()
    ax = graphs.plot_test(valid_pred['true'], valid_pred['pred'])

    test_pred = lstm.one_step_future_predict()

    # plot test pred on the same plot as train and valid
    ax.plot(test_pred)

    # plot test true and test pred
    test_true = data.loc[(data.index >= start_test) & (data.index < end_test), y_label]
    ax = graphs.plot_test(test_true, pd.DataFrame(test_pred, index=test_pred.index))
    plt.show()