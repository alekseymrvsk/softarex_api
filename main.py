import json
from fastapi import FastAPI
from fastapi import UploadFile
import shutil
from pathlib import Path
from classRandomForest import MyRandomForest
import pandas as pd
from fastapi.responses import FileResponse

app = FastAPI()

NUMBER_COLUMNS_TEST = 42

NUMBER_COLUMNS_TRAIN = 43

INPUT_FILE_FORMAT = '.csv'


def check_dataset(file_path, check_train=True):
    dataset = pd.read_csv(filepath_or_buffer=file_path)
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

    if not check_dataset(filepath, True):  # check train dataset
        return 404, "Please check your file"

    model = MyRandomForest()
    model.fit_model(filepath)

    filepath = "user_data/test.csv"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(INPUT_TEST.file, buffer)

    if not check_dataset(filepath, False):  # check test dataset
        return 404, "Please check your file"

    predict = model.predict_data(filepath)

    metric = model.get_metric()
    df = pd.DataFrame(predict)
    df.columns = ["Prediction"]
    df.to_csv("output_data/prediction.csv", index_label="Id")

    return FileResponse("output_data/prediction.csv", filename="prediction", media_type='text/csv')
