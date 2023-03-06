import asyncio
import functools
from fastapi import FastAPI
from fastapi import UploadFile, File
import shutil
from app.classRandomForest import MyRandomForest
import pandas as pd
from fastapi.responses import FileResponse
from typing import Optional
from concurrent.futures import ProcessPoolExecutor

app = FastAPI()

NUMBER_COLUMNS_TEST = 42

NUMBER_COLUMNS_TRAIN = 43


# For checking train dataset set check_train True, for checking test dataset - False
def check_dataset(filepath, check_train=True):
    dataset = pd.read_csv(filepath_or_buffer=filepath)
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


def predict_data(data):
    # def predict_data(filepath_test: str, filepath_train: str = None):
    model = MyRandomForest()
    model.fit_model(data['filepath_train'])

    predict = model.predict_data(data['filepath_test'])

    df = pd.DataFrame(predict)
    df.columns = ["Prediction"]
    df.to_csv("app/output_data/prediction.csv", index_label="Id")


@app.get('/get_data')
async def run(INPUT_TEST: UploadFile, INPUT_TRAIN: Optional[UploadFile] = File(None)):
    if not INPUT_TEST.filename.lower().endswith('.csv'):
        return 404, "Please upload csv  file."

    if INPUT_TRAIN is None:
        filepath_train = "app/input/train.csv"
    else:
        if not INPUT_TRAIN.filename.lower().endswith('.csv'):
            return 404, "Please upload csv  file."

        filepath_train = "app/user_data/train.csv"

        with open(filepath_train, "wb") as buffer:
            shutil.copyfileobj(INPUT_TRAIN.file, buffer)

            if not check_dataset(filepath_train, True):  # check train dataset
                return 404, "Please check your file"

    filepath_test = "app/user_data/test.csv"
    with open(filepath_test, "wb") as buffer:
        shutil.copyfileobj(INPUT_TEST.file, buffer)

    if not check_dataset(filepath_test, False):  # check test dataset
        return 404, "Please check your file"

    with ProcessPoolExecutor() as executor:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor,
                                   functools.partial(predict_data, data={"filepath_train": filepath_train,
                                                                         "filepath_test": filepath_test}))

        return FileResponse("app/output_data/prediction.csv", filename="prediction", media_type='text/csv')
