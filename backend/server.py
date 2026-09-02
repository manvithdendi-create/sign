import os
import random
import math
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from backend.model import sign_model
from backend.translator import translator_engine, ASL_DICTIONARY

app = FastAPI(title="Sign Language AI API", version="1.0.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to train initial ML model
@app.on_event("startup")
def startup_event():
    sign_model.train_initial_model()

# Pydantic Schemas
class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float

class PredictRequest(BaseModel):
    landmarks: List[LandmarkPoint]

class CustomTrainRequest(BaseModel):
    label: str
    samples: List[List[LandmarkPoint]]

def validate_landmarks(landmarks: List[LandmarkPoint]) -> None:
    if len(landmarks) != 21:
        raise HTTPException(status_code=400, detail="Exactly 21 hand keypoints are required.")

    if any(not all(math.isfinite(value) for value in (point.x, point.y, point.z)) for point in landmarks):
        raise HTTPException(status_code=400, detail="Landmark coordinates must be finite numbers.")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_trained": sign_model.is_trained,
        "classes_count": len(sign_model.classes),
        "available_classes": sign_model.classes
    }

@app.post("/api/predict")
def predict_sign(req: PredictRequest):
    validate_landmarks(req.landmarks)
    
    # Convert Pydantic points to dict list
    lm_dicts = [{'x': p.x, 'y': p.y, 'z': p.z} for p in req.landmarks]
    
    result = sign_model.predict(lm_dicts)
    trans_res = translator_engine.process_prediction(result["prediction"], result["confidence"])
    
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "top_candidates": result["top_candidates"],
        "translator_status": trans_res["status"],
        "sentence_buffer": trans_res["current_word"]
    }

@app.get("/api/dictionary")
def get_dictionary():
    return {"dictionary": ASL_DICTIONARY}

@app.get("/api/quiz/question")
def get_quiz_question():
    """Generates a random quiz challenge for the user."""
    keys = list(ASL_DICTIONARY.keys())
    target_key = random.choice(keys)
    info = ASL_DICTIONARY[target_key]
    
    return {
        "target_sign": target_key,
        "name": info["name"],
        "category": info["category"],
        "tips": info["tips"],
        "description": info["description"]
    }

@app.post("/api/train/custom")
def train_custom_sign(req: CustomTrainRequest):
    label = req.label.strip().upper()
    if not label or not req.samples:
        raise HTTPException(status_code=400, detail="Label and landmark samples required.")
    if len(label) > 40 or not re.fullmatch(r"[A-Z0-9_ -]+", label):
        raise HTTPException(status_code=400, detail="Label may contain only letters, numbers, spaces, hyphens, and underscores.")
    
    formatted_samples = []
    for sample in req.samples:
        validate_landmarks(sample)
        formatted_samples.append([{'x': p.x, 'y': p.y, 'z': p.z} for p in sample])
    
    success, msg = sign_model.add_custom_gesture(label, formatted_samples)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
        
    return {"status": "success", "message": msg, "classes": sign_model.classes}

@app.post("/api/translator/clear")
def clear_translator():
    text = translator_engine.clear()
    return {"sentence_buffer": text}

@app.post("/api/translator/reset")
def reset_translator_stability():
    translator_engine.reset_stability()
    return {"status": "reset"}

@app.post("/api/translator/space")
def space_translator():
    text = translator_engine.add_space()
    return {"sentence_buffer": text}

@app.post("/api/translator/backspace")
def backspace_translator():
    text = translator_engine.backspace()
    return {"sentence_buffer": text}

# Mount static frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def read_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Sign Language AI API is running. Frontend index.html not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
