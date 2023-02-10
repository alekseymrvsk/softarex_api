import json
from fastapi import FastAPI
from fastapi import UploadFile
import shutil
from pathlib import Path
from classRandomForest import MyRandomForest
import pandas as pd

app = FastAPI()

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


@app.post("/predict")
def predict_data(INPUT_TRAIN: UploadFile, INPUT_TEST: UploadFile):
    if not INPUT_TRAIN.filename.lower().endswith('.csv'):
        return 404, "Please upload csv  file."

    filepath = "user_data/train.csv"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(INPUT_TRAIN.file, buffer)

    if not check_dataset(buffer, True):  # check train dataset
        return 404, "Please check your file"

    model = MyRandomForest()
    model.fit_model(filepath)

    filepath = "user_data/test.csv"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(INPUT_TEST.file, buffer)

    if not check_dataset(buffer, False):  # check test dataset
        return 404, "Please check your file"

    predict = model.predict_data(filepath)

    metric = model.get_metric()

    context = {'metric': metric, 'predict': predict}

    return context
