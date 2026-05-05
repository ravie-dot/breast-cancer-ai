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

    # Open image
    img = Image.open(file_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # 🔴 DUMMY MODEL LOGIC (replace later with AI model)
    score = np.mean(img_array)

    if score < 0.3:
        result = "Normal"
    elif score < 0.6:
        result = "Benign (Non-Cancer)"
    else:
        result = "Malignant (Cancer)"

    return JSONResponse({
        "prediction": result,
        "risk_score": float(score),
        "advice": "Consult a medical professional for confirmation."
    })
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)