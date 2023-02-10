import json
from fastapi import FastAPI
from fastapi import UploadFile
import shutil
from pathlib import Path
from classRandomForest import MyRandomForest


app = FastAPI()


@app.post("/predict")
def predict_data(INPUT_TRAIN: UploadFile, INPUT_TEST: UploadFile):
    if not INPUT_TRAIN.filename.lower().endswith('.csv'):
        return 404, "Please upload csv  file."

    filepath = "user_data/train.csv"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(INPUT_TRAIN.file, buffer)

    model = MyRandomForest()
    model.fit_model(filepath)

    filepath = "user_data/test.csv"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(INPUT_TEST.file, buffer)

    predict = model.predict_data(filepath)

    metric = model.get_metric()
    context = {'metric': metric, 'predict': predict}

    return context
