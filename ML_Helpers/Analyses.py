from keras.layers import GlobalAveragePooling1D, Dropout

from ML_Helpers import Modeling, Definitions
from ML_Helpers.Preprocessing import Dataset, Data_Preprocessor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf

import os
import matplotlib.pyplot as plt
import pandas as pd


class Analysis():

    training_r2 = []
    testing_r2 = []
    master_path = ""
    path = ""
    name = ""
    description = ""
    parameter = ""

    def __init__(self, name="Test_Analysis", path="", description="",
                 parameter=""):
        self.master_path = path
        self.name = name
        self.description = description
        self.parameter = parameter

        self.path = self.master_path + "/" + self.name
        self.plots_path = self.path + "/plots"

        self.create_analysis_folders()

    def run(self):
        pass

    def import_data(self, data_path):
        pass

    def summary(self):
        print("Training R^2: " + self.training_r2.__str__())
        print("Testing R^2: " + self.testing_r2.__str__())

    def create_analysis_folders(self):
        if not os.path.exists(self.master_path):
            os.makedirs(self.master_path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

    def export_r2_plots(self):
        pass

    def plot_r2_curves(self):
        if (len(self.testing_r2) == 0) or (len(self.training_r2)== 0):
            return None

        fig, ax = plt.subplots()

        ax.plot(self.testing_r2, label="Testing R^2")
        ax.plot(self.training_r2, label="Training R^2")

        ax.set_title(self.description)

        ax.set_xlabel(self.parameter)
        ax.set_ylabel("R^2 Score")

        ax.legend()
        fig.savefig(self.plots_path+"r2_score.png")

    def clear_scores(self):
        self.training_r2 = []
        self.testing_r2 = []


class Analysis1(Analysis):

    def __init__(self, **kwargs):
        super(Analysis1, self).__init__(**kwargs)
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")
        # self.dataset_stats(dataset.dataframe)

    def run(self, save_graphs=False):
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")
        #self.dataset_stats(dataset)
        x_cols = [
            "air_temp_ext1",
            "air_temp_ib_facade",
            "air_temp_ib_roof",
        ]
        y_cols = [
            "apperture_ext1"
        ]

        dataset.select_x_and_y_columns(x_cols, y_cols)
        dataset.fit_scalers(scaling_type="minmax")

        data_preprocessor = Data_Preprocessor(Modeling.Model_Types.LSTM)
        dataset.set_data_preprocessor(data_preprocessor)

        lookback_range = range(7, 8)

        for i in lookback_range:
            data_preprocessor.set_timesteps(i*24)
            dataset.preprocess_dataset(remove_gaps=True)
            dataset.split_data(0.3, shuffle=True)

            cnn_model = Sequential()
            cnn = Modeling.LSTM_ModelWrapper(name="Test CNN Model", model=cnn_model, time_window=i * 24, dataset=dataset)
            cnn_model.add(Conv1D(filters=64, kernel_size=3,
                                 activation='relu', input_shape=cnn.get_input_shape()))
            cnn_model.add(MaxPooling1D(pool_size=3))
            cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            cnn_model.add(GlobalAveragePooling1D())
            cnn_model.add(Dropout(0.5))
            cnn_model.add(Dense(units=1, activation='sigmoid'))

            cnn_model.compile(optimizer='adam', loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])

            cnn.fit(epochs=3)
            self.training_r2.append(cnn.get_training_r_squared())
            self.testing_r2.append(cnn.get_test_r_squared())
            if save_graphs:
                cnn.plot_train_test_predictions(self.plots_path, "lbd" + str(i))
                cnn.plot_all_dataset_prediction(self.plots_path, "lbd" + str(i))
                cnn.plot_training_curves(self.plots_path, "lbd" + str(i))

            cnn.plot_all_dataset_prediction()
            plt.show()


        return self.training_r2, self.testing_r2

    def import_data(self, data_path):
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index("datetime", inplace=True)
        data['air_temp_diff'] = data['air_temp_ext1'].diff()
        data['app_diff'] = data['apperture_ext1'].diff()
        dataset = Dataset(dataframe=data)
        return dataset

    def dataset_stats(self, dataset):
        count = dataset.shape[0]
        avail_data = []
        lb_days = range(1,21)

        for lb in lb_days:
            avail_data.append(count/(lb*24))

        plt.plot(lb_days, avail_data)
        plt.xlabel("Lookback days")
        plt.ylabel("# of data")
        plt.show()


class Analysis2(Analysis):

    def __init__(self, **kwargs):
        super(Analysis2, self).__init__(**kwargs)
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")
        # self.dataset_stats(dataset.dataframe)

    def run(self, save_graphs=False):
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")

        x_cols = [
            "air_temp_ext1",
            "air_temp_ib_facade",
            "air_temp_ib_roof",
            "r_humid_ib_facade",
            "r_humid_ib_roof"
        ]
        y_cols = [
            "apperture_ext1"
        ]

        dataset.select_x_and_y_columns(x_cols, y_cols)
        dataset.fit_scalers(scaling_type="minmax")

        data_preprocessor = Data_Preprocessor(Modeling.Model_Types.LSTM)
        dataset.set_data_preprocessor(data_preprocessor)

        lookback_range = range(1, 21)

        for i in lookback_range:
            data_preprocessor.set_timesteps(i*24)
            dataset.preprocess_dataset()
            dataset.split_data(0.3, shuffle=True)

            cnn_model = Sequential()
            cnn = Modeling.LSTM_ModelWrapper(name="Test CNN Model", model=cnn_model, time_window=i * 24, dataset=dataset)
            cnn_model.add(Conv1D(filters=64, kernel_size=3,
                                 activation='relu', input_shape=cnn.get_input_shape()))
            cnn_model.add(MaxPooling1D(pool_size=3))
            cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            cnn_model.add(GlobalAveragePooling1D())
            cnn_model.add(Dropout(0.5))
            cnn_model.add(Dense(units=1, activation='sigmoid'))

            cnn_model.compile(optimizer='adam', loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])

            cnn.fit(epochs=50)
            self.training_r2.append(cnn.get_training_r_squared())
            self.testing_r2.append(cnn.get_test_r_squared())
            if save_graphs:
                cnn.plot_train_test_predictions(self.plots_path, "lbd" + str(i))
                cnn.plot_all_dataset_prediction(self.plots_path, "lbd" + str(i))
                cnn.plot_training_curves(self.plots_path, "lbd" + str(i))


        return self.training_r2, self.testing_r2

    def import_data(self, data_path):
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index("datetime", inplace=True)
        data['air_temp_diff'] = data['air_temp_ext1'].diff()
        data['app_diff'] = data['apperture_ext1'].diff()
        dataset = Dataset(dataframe=data)
        return dataset


class Analysis3(Analysis):

    def __init__(self, **kwargs):
        super(Analysis3, self).__init__(**kwargs)
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")
        # self.dataset_stats(dataset.dataframe)

    def run(self, save_graphs=False):
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")

        x_cols = [
            "r_humid_ib_facade",
            "r_humid_ib_roof"
        ]
        y_cols = [
            "apperture_ext1"
        ]

        dataset.select_x_and_y_columns(x_cols, y_cols)
        dataset.fit_scalers(scaling_type="minmax")

        data_preprocessor = Data_Preprocessor(Modeling.Model_Types.LSTM)
        dataset.set_data_preprocessor(data_preprocessor)

        lookback_range = range(1, 21)

        for i in lookback_range:
            data_preprocessor.set_timesteps(i * 24)
            dataset.preprocess_dataset()
            dataset.split_data(0.3, shuffle=True)

            cnn_model = Sequential()
            cnn = Modeling.LSTM_ModelWrapper(name="Test CNN Model", model=cnn_model, time_window=i * 24,
                                             dataset=dataset)
            cnn_model.add(Conv1D(filters=64, kernel_size=3,
                                 activation='relu', input_shape=cnn.get_input_shape()))
            cnn_model.add(MaxPooling1D(pool_size=3))
            cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            cnn_model.add(GlobalAveragePooling1D())
            cnn_model.add(Dropout(0.5))
            cnn_model.add(Dense(units=1, activation='sigmoid'))

            cnn_model.compile(optimizer='adam', loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])

            cnn.fit(epochs=50)
            self.training_r2.append(cnn.get_training_r_squared())
            self.testing_r2.append(cnn.get_test_r_squared())
            if save_graphs:
                cnn.plot_train_test_predictions(self.plots_path, "lbd" + str(i))
                cnn.plot_all_dataset_prediction(self.plots_path, "lbd" + str(i))
                cnn.plot_training_curves(self.plots_path, "lbd" + str(i))

        return self.training_r2, self.testing_r2

    def import_data(self, data_path):
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index("datetime", inplace=True)
        data['air_temp_diff'] = data['air_temp_ext1'].diff()
        data['app_diff'] = data['apperture_ext1'].diff()
        dataset = Dataset(dataframe=data)
        return dataset


class Analysis4(Analysis):

    def __init__(self, **kwargs):
        super(Analysis4, self).__init__(**kwargs)
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")
        # self.dataset_stats(dataset.dataframe)

    def run(self, save_graphs=False):
        dataset = self.import_data("Prepared Datasets/Joined_Dataset.csv")

        x_cols = [
            "r_humid_ib_facade",
            "r_humid_ib_roof"
        ]
        y_cols = [
            "apperture_ext1"
        ]

        dataset.select_x_and_y_columns(x_cols, y_cols)
        dataset.fit_scalers(scaling_type="minmax")

        data_preprocessor = Data_Preprocessor(Modeling.Model_Types.LSTM)
        dataset.set_data_preprocessor(data_preprocessor)

        lookback_range = range(1, 21)

        for i in lookback_range:
            data_preprocessor.set_timesteps(i * 24)
            dataset.preprocess_dataset()
            dataset.split_data(0.3, shuffle=True)

            cnn_model = Sequential()
            cnn = Modeling.LSTM_ModelWrapper(name="Test CNN Model", model=cnn_model, time_window=i * 24,
                                             dataset=dataset)
            cnn_model.add(Conv1D(filters=64, kernel_size=3,
                                 activation='relu', input_shape=cnn.get_input_shape()))
            cnn_model.add(MaxPooling1D(pool_size=3))
            cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            cnn_model.add(GlobalAveragePooling1D())
            cnn_model.add(Dropout(0.5))
            cnn_model.add(Dense(units=1, activation='sigmoid'))

            cnn_model.compile(optimizer='adam', loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])

            cnn.fit(epochs=50)
            self.training_r2.append(cnn.get_training_r_squared())
            self.testing_r2.append(cnn.get_test_r_squared())
            if save_graphs:
                cnn.plot_train_test_predictions(self.plots_path, "lbd" + str(i))
                cnn.plot_all_dataset_prediction(self.plots_path, "lbd" + str(i))
                cnn.plot_training_curves(self.plots_path, "lbd" + str(i))

        return self.training_r2, self.testing_r2

    def import_data(self, data_path):
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index("datetime", inplace=True)
        data['air_temp_diff'] = data['air_temp_ext1'].diff()
        data['app_diff'] = data['apperture_ext1'].diff()
        dataset = Dataset(dataframe=data)
        return dataset