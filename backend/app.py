from fastapi import FastAPI, UploadFile, File
import pandas as pd
from detect import detect

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    results = detect("CIC-IDS2017", df)
    return results[["prediction"]].to_dict(orient="records")


# TODO Step 4 for Harsh!