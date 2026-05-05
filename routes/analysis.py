from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import uuid
import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.predictor import analyze

router = APIRouter(prefix="/api", tags=["analysis"])

# In-memory history store (replace with DB in production)
analysis_history = []

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".bmp", ".tiff", ".tif"}


@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    patient: Optional[str] = Form(None)
):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file bytes
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Parse patient data
    patient_data = {}
    if patient:
        try:
            patient_data = json.loads(patient)
        except json.JSONDecodeError:
            patient_data = {}

    # Run analysis
    try:
        result = analyze(file_bytes, file.filename, patient_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # Build response
    analysis_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.datetime.utcnow().isoformat()

    record = {
        "id": analysis_id,
        "timestamp": timestamp,
        "filename": file.filename,
        "file_type": "DICOM" if ext == ".dcm" else "Image",
        "patient": patient_data,
        **result
    }

    # Save to history
    analysis_history.insert(0, {
        "id": analysis_id,
        "timestamp": timestamp,
        "filename": file.filename,
        "classification": result["classification"],
        "confidence": result["confidence"],
        "risk_level": result["risk_level"],
        "patient_name": patient_data.get("name", "Anonymous"),
        "patient_age": patient_data.get("age", "N/A"),
    })

    return record


@router.get("/analyze/{analysis_id}")
async def get_analysis(analysis_id: str):
    for item in analysis_history:
        if item["id"] == analysis_id:
            return item
    raise HTTPException(status_code=404, detail="Analysis not found")
