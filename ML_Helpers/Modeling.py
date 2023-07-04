from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import json

import os

from ML_Helpers.Preprocessing import Data_Preprocessor
from ML_Helpers.Preprocessing import Dataset
from ML_Helpers.Definitions import Model_Types


class Model_Wrapper:
    """
    Abstract Class defining a model
    Subclass it for each model type you want to use
    """

    model = None
    name = None
    dataset = None
    history = None
    data_preprocessor = None

    def __init__(self, model=None, name=None, dataset=None):
        self.model = model
        self.name = name
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_model(self, model):
        self.model = model

    def plot_training_curves(self, parent_path="", img_name="img"):

        path = parent_path + "/training_curves/"
        if not os.path.exists(path):
            os.makedirs(path)

        fig, ax = plt.subplots()

        plots = []

        plots.append(ax.plot(self.history.history['loss'], label="Train. Loss"))
        plots.append(ax.plot(self.history.history['val_loss'], label="Val. Loss"))
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax2 = ax.twinx()
        plots.append(ax2.plot(self.history.history['mean_absolute_error'], linestyle="dashed", label="Train. MAE"))
        plots.append(ax2.plot(self.history.history['val_mean_absolute_error'], linestyle="dashed", label="Val. MAE"))
        ax2.set_ylabel("MAE")

        lns = ax.plot()
        for p in plots:
            lns = lns + p

        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)
        fig.savefig(path + img_name + ".png")

    def plot_all_dataset_prediction(self, parent_path="", img_name="img"):
        path = parent_path + "/all_dataset_predictions/"

        if not os.path.exists(path):
            os.makedirs(path)

        fig, ax = plt.subplots()
        x_data, y_data = self.dataset.get_orig_data(scaled=True)

        y_data = self.dataset.inverse_transform_Y(y_data)

        y_pred = self.model.predict(x_data)
        y_pred = self.dataset.inverse_transform_Y(y_pred)

        ax.scatter(range(len(y_data)), y_data, label="Actual")
        ax.scatter(range(len(y_pred)), y_pred, label="Predicted", linestyle="dashed")

        ax.set_ylabel("Displacement")
        ax.set_title("Model Prediction on Entire Dataset")
        ax.legend()
        fig.savefig(path + img_name + ".png")

    def plot_train_test_predictions(self, parent_path="", img_name="img"):

        path = parent_path + "/train_test_predictions/"
        if not os.path.exists(path):
            os.makedirs(path)
        fig, ax = plt.subplots()

        x_train, y_train = self.dataset.get_train_data(scaled=True)
        x_test, y_test = self.dataset.get_test_data(scaled=True)

        y_test_pred = self.model.predict(x_test)
        y_train_pred = self.model.predict(x_train)

        y_test_pred = self.dataset.inverse_transform_Y(y_test_pred)
        y_train_pred = self.dataset.inverse_transform_Y(y_train_pred)

        y_actual_test = self.dataset.inverse_transform_Y(y_test)
        y_actual_train = self.dataset.inverse_transform_Y(y_train)

        train_len = len(y_train_pred)
        test_len = len(y_test_pred)
        total_len = train_len + test_len

        ax.plot(range(0, train_len), y_train_pred, label="Predicted - Training")
        ax.plot(range(train_len, total_len), y_test_pred, label="Predicted - Test")
        ax.plot(range(0, train_len), y_actual_train, label="Ground Truth - Training")
        ax.plot(range(train_len, total_len), y_actual_test, label="Ground Truth - Test")

        ax.set_title(f"Ground Truth vs Model Predictions - Model: {self.name}")
        fig.savefig(path+img_name+".png")

    def get_training_r_squared(self):
        x_train, y_train = self.dataset.get_train_data(scaled=True)

        y_train_pred = self.predict(x_train)
        y_train_pred = self.dataset.inverse_transform_Y(y_train_pred)

        y_actual_train = self.dataset.inverse_transform_Y(y_train)

        return r2_score(y_train_pred, y_actual_train)

    def get_test_r_squared(self):
        x_test, y_test = self.dataset.get_test_data(scaled=True)

        y_test_pred = self.predict(x_test)
        y_test_pred = self.dataset.inverse_transform_Y(y_test_pred)

        y_actual_test = self.dataset.inverse_transform_Y(y_test)


        return r2_score(y_test_pred, y_actual_test)


class CNN_ModelWrapper(Model_Wrapper):

    timesteps = None

    def __init__(self, model=None, name=None, dataset=None, time_window=1):
        super(CNN_ModelWrapper, self).__init__(model, name)
        self.timesteps = time_window
        self.data_preprocessor = Data_Preprocessor(Model_Types.CNN)
        self.data_preprocessor.set_timesteps(time_window)
        self.dataset = dataset

        if self.dataset is not None:
            self.dataset.set_data_preprocessor(self.data_preprocessor)

    def set_time_window(self, time_window):
        self.timesteps = time_window
        self.data_preprocessor.set_timesteps(time_window)

    def read_model_from_dict(self, model_dict):
        pass

    def compile(self, optimizer='adam', loss="mean_squared_error"):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, epochs=None):
        x_train, y_train = self.dataset.get_train_data(scaled=True)
        self.model.fit(x_train, y_train, epochs=epochs)

    def predict(self, X, preprocess=False):
        x = X
        if preprocess: x = self.data_preprocessor.preprocess_inputs(x)
        return self.model.predict(x)

    def get_input_shape(self):
        x_train, y_train = self.dataset.get_train_data()
        return x_train.shape[1], x_train.shape[2]


class LSTM_ModelWrapper(Model_Wrapper):

    timesteps = None

    def __init__(self, model=None, name=None, dataset=None, time_window=1):
        super(LSTM_ModelWrapper, self).__init__(model, name)
        self.timesteps = time_window
        self.dataset = dataset

    def compile(self, optimizer='adam', loss="mean_squared_error"):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, epochs=10):
        x_train, y_train = self.dataset.get_train_data(scaled=True)
        self.history = self.model.fit(x_train, y_train, validation_split=0.3, epochs=epochs, batch_size=10)

    def predict(self, X, preprocess=False):
        x = X
        if preprocess: x = self.data_preprocessor.preprocess_inputs(x)
        return self.model.predict(x)

    def get_input_shape(self):
        x_train, y_train = self.dataset.get_train_data()
        return x_train.shape[1], x_train.shape[2]


def read_json(filename):
    with open(filename) as json_file:
        return json.load(json_file)


