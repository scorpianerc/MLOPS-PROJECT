"""
FastAPI Model Serving untuk IndoBERT Sentiment Analysis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from pathlib import Path
import logging
from datetime import datetime
import psycopg2
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
import os
from dotenv import load_dotenv
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.mlops.drift_detection import PredictionLogger, create_prediction_logs_table

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= Prometheus Metrics =============
PREDICTION_COUNTER = Counter(
    'sentiment_predictions_total',
    'Total number of predictions made',
    ['sentiment']
)

PREDICTION_LATENCY = Histogram(
    'sentiment_prediction_latency_seconds',
    'Prediction latency in seconds'
)

MODEL_CONFIDENCE = Gauge(
    'sentiment_prediction_confidence',
    'Average prediction confidence'
)

ERROR_COUNTER = Counter(
    'sentiment_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# ============= Pydantic Models =============
class PredictionRequest(BaseModel):
    """Request model untuk single prediction"""
    text: str = Field(..., min_length=1, max_length=5000, description="Review text")
    review_id: Optional[int] = Field(None, description="Optional review ID untuk logging")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Aplikasi ini sangat bagus dan mudah digunakan!",
                "review_id": 12345
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model untuk batch prediction"""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "Aplikasi bagus",
                    "Sangat buruk",
                    "Lumayan"
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response model untuk prediction"""
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    review_id: Optional[int] = None
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model untuk batch prediction"""
    predictions: List[PredictionResponse]
    total: int
    model_version: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    timestamp: str


# ============= FastAPI App =============
app = FastAPI(
    title="IndoBERT Sentiment Analysis API",
    description="REST API untuk sentiment analysis menggunakan IndoBERT",
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

# ============= Global Variables =============
MODEL = None
TOKENIZER = None
LABEL_MAP = None
REVERSE_LABEL_MAP = None
MODEL_VERSION = "1.0.0"
START_TIME = datetime.now()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Database config
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
    'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'password')
}

# Prediction logger
PREDICTION_LOGGER = None


# ============= Model Loading =============
def load_model():
    """Load IndoBERT model dan tokenizer"""
    global MODEL, TOKENIZER, LABEL_MAP, REVERSE_LABEL_MAP, PREDICTION_LOGGER
    
    try:
        model_path = Path("models/bert_model")
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return False
        
        # Load model
        MODEL = AutoModelForSequenceClassification.from_pretrained(model_path)
        MODEL.to(DEVICE)
        MODEL.eval()
        
        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        
        # Load label map
        with open(model_path / 'label_map.json') as f:
            LABEL_MAP = json.load(f)
        
        REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
        
        # Initialize prediction logger
        create_prediction_logs_table(DB_CONFIG)
        PREDICTION_LOGGER = PredictionLogger(DB_CONFIG)
        
        logger.info(f"✅ Model loaded successfully on {DEVICE}")
        logger.info(f"   Labels: {list(LABEL_MAP.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model saat startup"""
    logger.info("🚀 Starting FastAPI server...")
    
    # Skip model loading in CI/test environment
    if os.getenv('CI') == 'true' or os.getenv('SKIP_MODEL_LOADING') == 'true':
        logger.info("⏭️  Skipping model loading (CI/test mode)")
        return
    
    success = load_model()
    if not success:
        logger.warning("⚠️  Model not loaded - API will return errors")


# ============= Helper Functions =============
def predict_sentiment(text: str) -> Dict:
    """
    Predict sentiment untuk single text
    
    Returns:
        Dictionary dengan sentiment, confidence, dan probabilities
    """
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize
        encoding = TOKENIZER(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = MODEL(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()
        
        # Get sentiment label
        sentiment = REVERSE_LABEL_MAP[pred_id]
        confidence = probs[pred_id].item()
        
        # Get all probabilities
        probabilities = {
            REVERSE_LABEL_MAP[i]: float(probs[i].item())
            for i in range(len(probs))
        }
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probabilities
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        ERROR_COUNTER.labels(error_type='prediction_error').inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def log_prediction_async(
    review_id: Optional[int],
    text: str,
    sentiment: str,
    confidence: float
):
    """Log prediction secara asynchronous"""
    if PREDICTION_LOGGER:
        try:
            PREDICTION_LOGGER.log_prediction(
                review_id=review_id,
                text=text,
                predicted_sentiment=sentiment,
                confidence=confidence,
                model_version=MODEL_VERSION
            )
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")


# ============= API Endpoints =============
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint"""
    return {
        "message": "IndoBERT Sentiment Analysis API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        model_version=MODEL_VERSION,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict sentiment untuk single text
    
    - **text**: Review text (required)
    - **review_id**: Optional review ID untuk logging
    """
    with PREDICTION_LATENCY.time():
        # Predict
        result = predict_sentiment(request.text)
        
        # Update metrics
        PREDICTION_COUNTER.labels(sentiment=result['sentiment']).inc()
        MODEL_CONFIDENCE.set(result['confidence'])
        
        # Log prediction in background
        background_tasks.add_task(
            log_prediction_async,
            request.review_id,
            request.text,
            result['sentiment'],
            result['confidence']
        )
        
        return PredictionResponse(
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            review_id=request.review_id,
            model_version=MODEL_VERSION,
            timestamp=datetime.now().isoformat()
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict sentiment untuk multiple texts (batch)
    
    - **texts**: List of review texts (max 100)
    """
    predictions = []
    
    with PREDICTION_LATENCY.time():
        for text in request.texts:
            try:
                result = predict_sentiment(text)
                
                # Update metrics
                PREDICTION_COUNTER.labels(sentiment=result['sentiment']).inc()
                
                # Log prediction in background
                background_tasks.add_task(
                    log_prediction_async,
                    None,
                    text,
                    result['sentiment'],
                    result['confidence']
                )
                
                predictions.append(PredictionResponse(
                    sentiment=result['sentiment'],
                    confidence=result['confidence'],
                    probabilities=result['probabilities'],
                    model_version=MODEL_VERSION,
                    timestamp=datetime.now().isoformat()
                ))
                
            except Exception as e:
                logger.error(f"Error predicting text: {e}")
                ERROR_COUNTER.labels(error_type='batch_prediction_error').inc()
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        model_version=MODEL_VERSION
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    num_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    
    return {
        "model_version": MODEL_VERSION,
        "model_type": "IndoBERT",
        "num_parameters": num_params,
        "trainable_parameters": trainable_params,
        "labels": list(LABEL_MAP.keys()),
        "device": str(DEVICE),
        "loaded_at": START_TIME.isoformat()
    }


@app.get("/model/reload", tags=["Model"])
async def reload_model():
    """Reload model (hot reload)"""
    try:
        success = load_model()
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """Get API statistics"""
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600,
        "model_version": MODEL_VERSION,
        "device": str(DEVICE),
        "status": "operational" if MODEL is not None else "degraded"
    }


# ============= Run Server =============
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
