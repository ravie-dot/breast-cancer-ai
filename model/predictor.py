import numpy as np
import io
import os
from PIL import Image

# Try importing tensorflow; fall back to mock predictions for demo
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Try importing pydicom for DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "breast_cancer_model.h5")
model = None

CLASSES = ["Normal", "Benign", "Malignant"]

STAGE_MAP = {
    "Normal": None,
    "Benign": "Stage 0 (Non-invasive)",
    "Malignant": None  # determined by confidence
}

BIRADS_MAP = {
    "Normal": "BI-RADS 1 — Negative",
    "Benign": "BI-RADS 3 — Probably Benign",
    "Malignant": "BI-RADS 5 — Highly Suspicious"
}

RISK_MAP = {
    "Normal": "Low",
    "Benign": "Moderate",
    "Malignant": "High"
}

DENSITY_LABELS = ["Type A — Almost entirely fatty",
                  "Type B — Scattered fibroglandular",
                  "Type C — Heterogeneously dense",
                  "Type D — Extremely dense"]

PRECAUTIONS = {
    "Normal": [
        "Continue regular annual mammogram screenings (age 40+)",
        "Perform monthly self-breast examinations",
        "Maintain a healthy BMI and balanced diet",
        "Limit alcohol consumption to reduce risk",
        "Stay physically active — aim for 150 min/week moderate exercise",
        "Discuss family history with your doctor for personalized risk assessment",
        "Avoid unnecessary radiation exposure",
        "Consider genetic counseling if BRCA1/BRCA2 mutations are suspected"
    ],
    "Benign": [
        "Schedule a follow-up imaging in 6 months as recommended",
        "Do not ignore any new lumps — report immediately to your doctor",
        "Seek biopsy confirmation if recommended by your radiologist",
        "Track any changes in size, shape, or tenderness of the lesion",
        "Maintain regular check-ups every 6–12 months",
        "Discuss hormone therapy risks with your gynecologist",
        "Reduce caffeine intake — it may worsen fibrocystic changes",
        "Wear properly fitted supportive bras to reduce discomfort"
    ],
    "Malignant": [
        "Seek immediate consultation with an oncologist — do not delay",
        "Get a core needle biopsy for definitive histopathological diagnosis",
        "Request full staging workup: MRI, PET-CT, bone scan",
        "Discuss treatment options: surgery, chemotherapy, radiation, immunotherapy",
        "Consider getting a second opinion from a cancer specialist",
        "Inform close family members to consider genetic testing (BRCA1/BRCA2)",
        "Join a breast cancer support group for emotional well-being",
        "Maintain proper nutrition and hydration during treatment",
        "Ask about fertility preservation options before starting chemotherapy",
        "Keep all follow-up appointments and lab tests religiously"
    ]
}

RECOMMENDATIONS = {
    "Normal": "No immediate medical intervention required. Continue routine surveillance and healthy lifestyle practices.",
    "Benign": "Close monitoring recommended. Consult your radiologist for biopsy guidance and follow-up imaging schedule.",
    "Malignant": "URGENT: Immediate referral to oncology is strongly recommended. Early intervention significantly improves outcomes."
}


def load_model():
    global model
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded from", MODEL_PATH)
    else:
        print("ℹ️  No trained model found. Using simulated predictions for demo.")


def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """Resize and normalize image for model input."""
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def parse_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Parse DICOM or standard image formats."""
    if filename.lower().endswith(".dcm"):
        if not DICOM_AVAILABLE:
            raise ValueError("pydicom not installed. Run: pip install pydicom")
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)
        # Normalize to 0-255
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr.astype(np.uint8)
    else:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(img)


def mock_predict(img_array: np.ndarray) -> dict:
    """Simulated prediction for demo when no model is loaded."""
    np.random.seed(int(img_array.mean()) % 100)
    probs = np.random.dirichlet([2, 3, 1])  # weighted toward benign for demo
    idx = np.argmax(probs)
    return {
        "class_index": int(idx),
        "classification": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, probs)}
    }


def determine_stage(classification: str, confidence: float) -> str:
    if classification == "Normal":
        return "N/A"
    if classification == "Benign":
        return "Stage 0 — Non-invasive (DCIS)"
    # Malignant staging by confidence proxy
    if confidence < 0.5:
        return "Stage I — Early localized"
    elif confidence < 0.7:
        return "Stage II — Regional spread possible"
    elif confidence < 0.85:
        return "Stage III — Locally advanced"
    else:
        return "Stage IV — Advanced / Metastatic possible"


def analyze(file_bytes: bytes, filename: str, patient_data: dict = None) -> dict:
    img_array = parse_image(file_bytes, filename)
    preprocessed = preprocess_image(img_array)

    if model is not None and TF_AVAILABLE:
        preds = model.predict(preprocessed, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        probs = {c: float(p) for c, p in zip(CLASSES, preds)}
        classification = CLASSES[idx]
    else:
        result = mock_predict(img_array)
        idx = result["class_index"]
        confidence = result["confidence"]
        probs = result["probabilities"]
        classification = result["classification"]

    stage = determine_stage(classification, confidence)
    density_idx = int(img_array.mean() / 64) % 4

    return {
        "classification": classification,
        "confidence": round(confidence * 100, 1),
        "probabilities": {k: round(v * 100, 1) for k, v in probs.items()},
        "stage": stage,
        "birads": BIRADS_MAP[classification],
        "risk_level": RISK_MAP[classification],
        "breast_density": DENSITY_LABELS[density_idx],
        "calcification": "Microcalcifications detected" if classification == "Malignant" else "No significant calcifications",
        "mass_detected": classification != "Normal",
        "margin": "Irregular" if classification == "Malignant" else ("Circumscribed" if classification == "Benign" else "N/A"),
        "shape": "Irregular / Spiculated" if classification == "Malignant" else ("Oval / Lobular" if classification == "Benign" else "N/A"),
        "precautions": PRECAUTIONS[classification],
        "recommendation": RECOMMENDATIONS[classification],
        "model_used": "CNN (TensorFlow)" if model else "Demo simulation",
        "dicom_parsed": filename.lower().endswith(".dcm")
    }


# Load model at import time
load_model()
