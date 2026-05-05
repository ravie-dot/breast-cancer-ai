from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from PIL import Image
import numpy as np

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Breast Cancer AI Backend Running"}

# -------------------------
# IMAGE UPLOAD API
# -------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Uploaded successfully", "file": file.filename}

# -------------------------
# AI PREDICTION (DUMMY FOR NOW)
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img_size = os.path.getsize(file_path)

    if img_size % 3 == 0:
        prediction = "Benign (Non-Cancer)"
        risk = "Low Risk"
        confidence = 0.72
        advice = "Regular checkups recommended."
    else:
        prediction = "Malignant (Cancer)"
        risk = "High Risk"
        confidence = 0.86
        advice = "Immediate medical consultation required."

    return JSONResponse({
        "prediction": prediction,
        "risk_level": risk,
        "confidence": confidence,
        "advice": advice
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)