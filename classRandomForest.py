import numpy as np
import pandas as pd
import shutil
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import RepeatedKFold, cross_val_score
from pathlib import Path
from django.core.files.storage import FileSystemStorage

NUMBER_COLUMNS_TEST = 42

NUMBER_COLUMNS_TRAIN = 43

INPUT_FILE_FORMAT = '.csv'



def check_dataset(file, check_train=True):
    if Path(file).suffix == INPUT_FILE_FORMAT:
        dataset = pd.read_csv(file)
        if check_train:
            if len(dataset.columns) == NUMBER_COLUMNS_TRAIN:
                return True
            else:
                return False
        else:
            if len(dataset.columns) == NUMBER_COLUMNS_TEST:
                return True
            else:
                return False


def dataset_split_train(INPUT_PATH_TRAIN='user_data/train.csv'):
    data_train = pd.read_csv(INPUT_PATH_TRAIN)

    y_train = data_train.revenue
    y_train = y_train.astype(int)
    x_train = data_train.drop(columns=['revenue', 'Id', 'City'])

    x_train = x_train.replace({'City Group': {'Other': 0, 'Big Cities': 1}})
    x_train = x_train.replace({'Type': {'FC': 0, 'IL': 1, 'DT': 2, 'MB': 3}})
    x_train['Open Date'] = pd.to_datetime(x_train['Open Date'], format='%m/%d/%Y').astype(np.int64)

    return [x_train, y_train]


def dataset_split_test(INPUT_PATH_TEST='prediction/input/test.csv'):
    data_test = pd.read_csv(INPUT_PATH_TEST)

    x_test = data_test.drop(columns=['Id', 'City'])
    x_test = x_test.replace({'City Group': {'Other': 0, 'Big Cities': 1}})
    x_test = x_test.replace({'Type': {'FC': 0, 'IL': 1, 'DT': 2, 'MB': 3}})
    x_test['Open Date'] = pd.to_datetime(x_test['Open Date'], format='%m/%d/%Y').astype(np.int64)
    return x_test


class Parameters:
    RANDOM_STATE_MODEL = 1
    TREES_COUNT = 50
    N_SPLITS = 10
    N_REPEATS = 3
    RANDOM_STATE_METRIC = 1


class MyRandomForest:

    def __init__(self, param=Parameters()):
        self.scores = None
        self.model = None
        self.param = param
        self.RANDOM_STATE_METRIC = self.param.RANDOM_STATE_METRIC
        self.N_REPEATS = self.param.N_REPEATS
        self.N_SPLITS = self.param.N_SPLITS
        self.RANDOM_STATE_MODEL = self.param.RANDOM_STATE_MODEL
        self.TREES_COUNT = self.param.TREES_COUNT

    def fit_model(self, input_path_train):
        dataset = dataset_split_train(input_path_train)
        x_train = dataset[0]
        y_train = dataset[1]
        model = RandomForestRegressor(n_estimators=self.TREES_COUNT,
                                      random_state=self.RANDOM_STATE_MODEL)
        model.fit(x_train, y_train)
        self.model = model
        cv = RepeatedKFold(n_splits=self.N_SPLITS, n_repeats=self.N_REPEATS,
                           random_state=self.RANDOM_STATE_METRIC)
        scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        self.scores = abs(scores)

    def get_metric(self):
        if self.scores is not None:
            return 'Mean MAE: %.3f (%.3f)' % (self.scores.mean(), self.scores.std())
        return None

    def predict_data(self, input_path_test):
        dataset = dataset_split_test(input_path_test)
        x_test = dataset
        if self.model is not None:
            prediction = self.model.predict(x_test)
            return prediction
        return None

    @staticmethod
    def save_model(model):
        joblib.dump(model, "./model_random_forest.joblib")
