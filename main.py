"""
Plant Leaf Disease Classification API
FastAPI service for predicting plant diseases and quality
"""

import os
import io
import json
import base64
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Parámetros del modelo
IMG_SIZE = (200, 300)  # (height, width)
MODEL_PATH = "/app/model/plant_classifier_f1_95.18_20251122_055734.keras"
GEOGRAPHY_CSV_PATH = "/app/data/plant_geography.csv"

# Cloud Storage - Cloud Run puede acceder sin hacer archivos públicos
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "datasets_bucket_01")
GCS_MODEL_BLOB = os.getenv("GCS_MODEL_BLOB", "models/trained/plant_classifier_f1_95.18_20251122_055734.keras")
GCS_GEOGRAPHY_BLOB = os.getenv("GCS_GEOGRAPHY_BLOB", "results/plant_geography.csv")

# Opción alternativa: Google Drive file IDs
MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID", "1Mru306ChgsV9My3DbWUiRbPWUf3TS-ew")
GEOGRAPHY_GDRIVE_ID = os.getenv("GEOGRAPHY_GDRIVE_ID", "1GnF0qT9I8FpTsrQR9cvBx7ypr8bTy3yQ")

# ============================================================================
# INICIALIZACIÓN DE FASTAPI
# ============================================================================

app = FastAPI(
    title="Plant Disease Classification API",
    description="AI-powered plant leaf disease detection for global export certification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_class: str = Field(..., description="Clase predicha")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")
    top_predictions: List[Dict[str, float]] = Field(..., description="Top 3 predicciones")
    plant_info: Optional[Dict] = Field(None, description="Información geográfica y botánica de la planta")
    reference_image_base64: Optional[str] = Field(None, description="Imagen procesada en base64")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    geography_data_loaded: bool
    version: str
    download_source: Optional[str] = None

# ============================================================================
# ESTADO GLOBAL
# ============================================================================

class ModelState:
    """Global state for model and data"""
    model: Optional[keras.Model] = None
    classes: Optional[List[str]] = None
    geography_df: Optional[pd.DataFrame] = None
    download_source: str = "not_loaded"
    
model_state = ModelState()

# ============================================================================
# FUNCIONES AUXILIARES - DESCARGA
# ============================================================================

def download_from_gcs(bucket_name: str, blob_name: str, destination: Path) -> bool:
    """Download from Cloud Storage (autenticado)"""
    try:
        logger.info(f"Downloading {blob_name} from GCS")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
        
        logger.info(f"GCS download success")
        return True
    except Exception as e:
        logger.error(f"GCS error: {str(e)}")
        return False

def download_from_google_drive(file_id: str, destination: Path) -> bool:
    """Download from Google Drive"""
    try:
        import gdown
        logger.info(f"Downloading from Google Drive: {file_id}")
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(destination), quiet=False)
        
        logger.info(f"✅ Google Drive download success")
        return True
    except Exception as e:
        logger.error(f"Google Drive error: {str(e)}")
        return False

# ============================================================================
# CARGA DE MODELO
# ============================================================================

def load_model_and_data():
    """Load model and geography data at startup"""
    try:
        # ===== CARGAR MODELO =====
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            logger.info("Model not found, downloading...")
            success = download_from_gcs(GCS_BUCKET_NAME, GCS_MODEL_BLOB, model_path)
            if success:
                model_state.download_source = "cloud_storage"
            else:
                success = download_from_google_drive(MODEL_GDRIVE_ID, model_path)
                if success:
                    model_state.download_source = "google_drive"
            
            if not success:
                raise Exception("Failed to download model")
        else:
            model_state.download_source = "local_cache"
        
        # Cargar modelo
        logger.info("Loading model...")
        model_state.model = keras.models.load_model(str(model_path))
        logger.info(f"Model loaded: {model_state.model.count_params():,} params")
        
        # Obtener número de clases del modelo
        num_classes = model_state.model.output_shape[-1]
        logger.info(f"Model expects {num_classes} classes")
        
        # ===== LAS 35 CLASES EXACTAS (del entrenamiento) =====
        model_state.classes = [
            'test-Alstonia Scholaris diseased (P2a)',
            'test-Alstonia Scholaris healthy (P2b)',
            'test-Arjun diseased (P1a)',
            'test-Arjun healthy (P1b)',
            'test-Bael diseased (P4b)',
            'test-Basil healthy (P8)',
            'test-Chinar diseased (P11b)',
            'test-Chinar healthy (P11a)',
            'test-Gauva diseased (P3b)',
            'test-Gauva healthy (P3a)',
            'test-Jamun diseased (P5b)',
            'test-Jamun healthy (P5a)',
            'test-Jatropha diseased (P6b)',
            'test-Jatropha healthy (P6a)',
            'test-Lemon diseased (P10b)',
            'test-Lemon healthy (P10a)',
            'test-Mango diseased (P0b)',
            'test-Mango healthy (P0a)',
            'test-Pomegranate diseased (P9b)',
            'test-Pomegranate healthy (P9a)',
            'test-Pongamia Pinnata diseased (P7b)',
            'test-Pongamia Pinnata healthy (P7a)',
            'train-Alstonia Scholaris diseased (P2a)',
            'train-Alstonia Scholaris healthy (P2b)',
            'train-Arjun diseased (P1a)',
            'train-Arjun healthy (P1b)',
            'train-Bael diseased (P4b)',
            'train-Basil healthy (P8)',
            'train-Chinar diseased (P11b)',
            'train-Chinar healthy (P11a)',
            'train-Gauva diseased (P3b)',
            'train-Gauva healthy (P3a)',
            'train-Jamun diseased (P5b)',
            'train-Jamun healthy (P5a)',
            'train-Jatropha diseased (P6b)'
        ]
        
        # VERIFICAR
        if len(model_state.classes) != num_classes:
            logger.error(f"Mismatch: model has {num_classes}, list has {len(model_state.classes)}")
            raise ValueError(f"Class count mismatch")
        
        logger.info(f"Classes configured: {len(model_state.classes)} classes")
        logger.info(f"First 3: {model_state.classes[:3]}")
        logger.info(f"Last 3: {model_state.classes[-3:]}")
        
        # ===== CARGAR CSV DE GEOGRAFÍA =====
        geography_path = Path(GEOGRAPHY_CSV_PATH)
        if not geography_path.exists():
            success = download_from_gcs(GCS_BUCKET_NAME, GCS_GEOGRAPHY_BLOB, geography_path)
            if not success:
                success = download_from_google_drive(GEOGRAPHY_GDRIVE_ID, geography_path)
        
        if geography_path.exists():
            model_state.geography_df = pd.read_csv(str(geography_path))
            logger.info(f"Geography loaded: {len(model_state.geography_df)} records")
        
        logger.info(f"✅ API ready")
        return True
        
    except Exception as e:
        logger.error(f"Load error: {str(e)}")
        return False

def extract_plant_name(class_label: str) -> str:
    """Extract plant name from class label"""
    # Remover prefijos: test-, train-, valid-
    label = class_label
    for prefix in ['test-', 'train-', 'valid-']:
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    
    # Remover healthy/diseased y código (P#)
    plant_name = label.split(' healthy')[0].split(' diseased')[0]
    return plant_name.strip()

COUNTRY_COORDINATES = {
    'India': {'lat': 20.5937, 'lon': 78.9629, 'capital': 'New Delhi'},
    'Iran': {'lat': 32.4279, 'lon': 53.6880, 'capital': 'Tehran'},
    'Greece': {'lat': 39.0742, 'lon': 21.8243, 'capital': 'Athens'},
    'Mexico': {'lat': 23.6345, 'lon': -102.5528, 'capital': 'Mexico City'},
    'Brazil': {'lat': -14.2350, 'lon': -51.9253, 'capital': 'Brasília'},
}

def get_plant_geography(plant_name: str) -> Optional[Dict]:
    """Get geographic info for plant"""
    if model_state.geography_df is None:
        return None
    
    try:
        plant_data = model_state.geography_df[
            model_state.geography_df['plant'].str.lower() == plant_name.lower()
        ]
        
        if not plant_data.empty:
            row = plant_data.iloc[0]
            country = row['origin_country']
            coordinates = COUNTRY_COORDINATES.get(country, {'lat': 0, 'lon': 0, 'capital': 'Unknown'})
            
            return {
                "plant_name": row['plant'],
                "common_name": row['common_name'],
                "scientific_name": row['scientific_name'],
                "origin_country": country,
                "origin_continent": row['origin_continent'],
                "crop_type": row['crop_type'],
                "coordinates": {
                    "latitude": coordinates['lat'],
                    "longitude": coordinates['lon'],
                    "capital": coordinates['capital']
                }
            }
    except Exception as e:
        logger.warning(f"Geography error for {plant_name}: {str(e)}")
    
    return None

def preprocess_image(image_bytes: bytes) -> tuple:
    """Preprocess image for prediction"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE[1], IMG_SIZE[0]))
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img_resized

def image_to_base64(image_array: np.ndarray) -> str:
    """Convert image to base64"""
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up...")
    success = load_model_and_data()
    if success:
        logger.info("Ready")
    else:
        logger.error("Failed to load")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Disease API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check"""
    return HealthResponse(
        status="healthy" if model_state.model else "unhealthy",
        model_loaded=model_state.model is not None,
        geography_data_loaded=model_state.geography_df is not None,
        version="1.0.0",
        download_source=model_state.download_source
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    include_reference_image: bool = True
):
    """Predict plant disease"""
    if model_state.model is None:
        raise HTTPException(503, "Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type")
    
    try:
        image_bytes = await file.read()
        img_array, img_resized = preprocess_image(image_bytes)
        
        predictions = model_state.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_class = model_state.classes[predicted_idx]
        
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [
            {"class": model_state.classes[i], "confidence": float(predictions[0][i])}
            for i in top3_idx
        ]
        
        plant_name = extract_plant_name(predicted_class)
        plant_info = get_plant_geography(plant_name)
        
        reference_image = image_to_base64(img_resized) if include_reference_image else None
        
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            top_predictions=top3_predictions,
            plant_info=plant_info,
            reference_image_base64=reference_image
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, str(e))

@app.get("/classes")
async def get_classes():
    """Get available classes"""
    if model_state.classes is None:
        raise HTTPException(503, "Model not loaded")
    
    plant_names = sorted(list(set([extract_plant_name(c) for c in model_state.classes])))
    
    return {
        "classes": model_state.classes,
        "total": len(model_state.classes),
        "plants": plant_names,
        "total_plants": len(plant_names)
    }

@app.get("/geography/{plant_name}")
async def get_plant_info(plant_name: str):
    """Get plant geographic info"""
    info = get_plant_geography(plant_name)
    if info is None:
        raise HTTPException(404, f"No info for: {plant_name}")
    return info

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")