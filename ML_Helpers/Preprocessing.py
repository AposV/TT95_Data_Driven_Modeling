import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from ML_Helpers.Definitions import Model_Types
from ML_Helpers.ScalingHelpers import *


class Data_Preprocessor:
    model_type = None
    timesteps = None

    def __init__(self, model_type):
        self.model_type = model_type
        self.remove_gaps = False

    def create_ts_batches(self, series, timesteps, remove_gaps=True):
        x = []
        for i in range(len(series) - timesteps):
            batch = series[i:i + timesteps]

            if remove_gaps:
                dt_diffs = batch.index.to_series().diff()
                if any(d >= pd.Timedelta(days=1) for d in dt_diffs[1:]):
                    continue

            x.append(batch)
        return np.array(x)

    def set_timesteps(self, timesteps):
        self.timesteps = timesteps

    def preprocess_inputs_outputs(self, x, y):
        x_pp = self.preprocess_inputs(x)
        y_pp = self.preprocess_outputs(y)
        return x_pp, y_pp

    def preprocess_inputs(self, x, remove_gaps=False):
        if self.model_type == Model_Types.LSTM or self.model_type == Model_Types.CNN:
            batches = self.create_ts_batches(x, self.timesteps, remove_gaps)
            return batches

    def preprocess_outputs(self, y, remove_gaps=False):
        if self.model_type == Model_Types.LSTM:
            y_array = []
            if remove_gaps:
                for i in range(len(y) - self.timesteps):
                    batch = y[i:i + self.timesteps]
                    dt_diffs = batch.index.to_series().diff()
                    if any(d >= pd.Timedelta(days=1) for d in dt_diffs[1:]):
                        continue
                    y_array.append(y.iloc[i + self.timesteps])

            else:
                y_array = y[self.timesteps:]

            return np.array(y_array)

    def set_remove_gaps(self, remove_gaps):
        self.remove_gaps = remove_gaps


class Dataset:
    x_orig = None
    y_orig = None
    x_preproc = None
    y_preproc = None
    x_train = None
    y_train = None
    x_scaler = None
    y_scaler = None
    x_test = None
    y_test = None
    x_val = None
    y_val = None

    data_preprocessor = None
    are_scaled = False

    def __init__(self, x_columns=None, y_columns=None, dataframe=None, data_preprocessor=None):
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.dataframe = dataframe
        self.data_preprocessor = data_preprocessor

    def set_data_preprocessor(self, data_preprocessor):
        self.data_preprocessor = data_preprocessor

    def select_x_and_y_columns(self, x_cols, y_cols):
        combined_cols = set(x_cols + y_cols)
        combined_cols_list = list(combined_cols)
        subDf = self.dataframe.loc[:, combined_cols_list]
        subDf = subDf.dropna()
        self.x_orig = subDf.loc[:, x_cols].values
        self.y_orig = subDf.loc[:, y_cols].values

    def fit_scalers(self, scaling_type="standard"):
        if scaling_type == "standard":
            self.x_scaler = StandardScaler3D().fit(self.x_orig)
            self.y_scaler = StandardScaler3D().fit(self.y_orig)
        if scaling_type == "minmax":
            self.x_scaler = MinMaxScaler3D().fit(self.x_orig)
            self.y_scaler = MinMaxScaler3D().fit(self.y_orig)

    def preprocess_dataset(self, remove_gaps=False):
        if self.data_preprocessor is None:
            print("No preprocessor set for dataset")
            return
        self.x_preproc = self.data_preprocessor.preprocess_inputs(self.x_orig, remove_gaps)
        self.y_preproc = self.data_preprocessor.preprocess_outputs(self.y_orig, remove_gaps)

    def split_data(self, test_size=0.2, validation_size=0, random_state=0, shuffle=False):
        if self.x_orig is None or self.y_orig is None:
            print("Error empty dataset")
            return

        x = self.x_orig
        y = self.y_orig

        if (self.x_preproc is not None) and (self.y_preproc is not None):
            x = self.x_preproc
            y = self.y_preproc

        self.x_train, \
        self.x_test, \
        self.y_train, \
        self.y_test = train_test_split(x,
                                       y,
                                       test_size=test_size,
                                       random_state=random_state, shuffle=shuffle)

    def get_train_data(self, scaled=False, preprocessed=True, shuffled=False):
        x_train, y_train = self.x_train, self.y_train

        if scaled:
            if x_train.ndim == 3:
                x_train = self.x_scaler.transform3D(self.x_train)
            else:
                x_train = self.x_scaler.transform3D(self.x_train)
            y_train = self.y_scaler.transform(self.y_train)

        return x_train, y_train

    def get_test_data(self, scaled=False):
        x_test, y_test = self.x_test, self.y_test

        if scaled:
            if x_test.ndim == 3:
                x_test = self.x_scaler.transform3D(self.x_test)
            else:
                x_test = self.x_scaler.transform(self.x_test)
            y_test = self.y_scaler.transform(self.y_test)

        return x_test, y_test

    def get_orig_data(self, scaled=False, preprocessed=False):
        x_orig, y_orig = self.x_orig, self.y_orig

        if preprocessed is not None:
            x_orig = self.x_preproc
            y_orig = self.y_preproc

        if scaled:
            if x_orig.ndim == 3:
                x_orig = self.x_scaler.transform3D(x_orig)
            else:
                x_orig = self.x_scaler.transform(x_orig)
            y_orig = self.y_scaler.transform(y_orig)

        return x_orig, y_orig

    def inverse_transform_X(self, x):
        if self.x_scaler is None:
            print("X scaller not fitted")
            return None
        return self.x_scaler.inverse_transform(x)

    def inverse_transform_Y(self, y):
        if self.y_scaler is None:
            print("Y scaller not fitted")
            return None
        return self.y_scaler.inverse_transform(y)

    def inverse_transform(self, x, y):
        return self.inverse_transform_X(x), self.inverse_transform_Y(y)
