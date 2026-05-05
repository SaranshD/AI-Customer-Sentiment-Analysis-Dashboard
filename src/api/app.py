from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import time

from config.settings import MODEL_NAME, DB_PATH
from src.db import Database
from src.models import SentimentClassifier

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing customer review sentiments",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(DB_PATH)
clf = SentimentClassifier(MODEL_NAME)


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Review text to analyze")
    store_result: bool = Field(default=True, description="Whether to store the result in database")


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of review texts to analyze")
    store_results: bool = Field(default=True, description="Whether to store results in database")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    stored: bool = False


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_processing_time_ms: float
    average_processing_time_ms: float


class StatisticsResponse(BaseModel):
    total_inferences: int
    sentiment_distribution: Dict[str, int]
    average_confidence: float
    average_processing_time_ms: float


@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze",
            "batch_analyze": "/batch",
            "statistics": "/statistics",
            "recent_results": "/results",
            "health": "/health"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": clf is not None}


@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    start_time = time.time()
    
    try:
        result = clf.predict(request.text, return_probs=True)
        probs = result.get('probabilities', {})
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        stored = False
        if request.store_result:
            try:
                db.add_result(
                    text=request.text,
                    sentiment=result['sentiment'],
                    confidence=result['confidence'],
                    positive_prob=probs.get('Positive', 0),
                    negative_prob=probs.get('Negative', 0),
                    neutral_prob=probs.get('Neutral', 0),
                    processing_time_ms=processing_time_ms
                )
                stored = True
            except Exception as e:
                print(f"Database storage error: {e}")
        
        return SentimentResponse(
            text=request.text,
            sentiment=result['sentiment'],
            confidence=result['confidence'],
            probabilities=probs,
            processing_time_ms=processing_time_ms,
            stored=stored
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchSentimentResponse)
def batch_analyze(request: BatchSentimentRequest):
    start_time = time.time()
    
    try:
        raw_results = clf.predict(request.texts, return_probs=True)
        
        if not isinstance(raw_results, list):
            raw_results = [raw_results]
        
        results = []
        for i, result in enumerate(raw_results):
            probs = result.get('probabilities', {})
            
            stored = False
            if request.store_results:
                try:
                    db.add_result(
                        text=request.texts[i],
                        sentiment=result['sentiment'],
                        confidence=result['confidence'],
                        positive_prob=probs.get('Positive', 0),
                        negative_prob=probs.get('Negative', 0),
                        neutral_prob=probs.get('Neutral', 0),
                        processing_time_ms=0
                    )
                    stored = True
                except Exception as e:
                    print(f"Database storage error: {e}")
            
            results.append(SentimentResponse(
                text=request.texts[i],
                sentiment=result['sentiment'],
                confidence=result['confidence'],
                probabilities=probs,
                processing_time_ms=0,
                stored=stored
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchSentimentResponse(
            results=results,
            total_processing_time_ms=total_time,
            average_processing_time_ms=total_time / len(results) if results else 0
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics", response_model=StatisticsResponse)
def get_statistics():
    try:
        stats = db.get_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
def get_recent_results(limit: int = 50):
    try:
        results = db.get_all_results(limit=limit)
        return {"results": [r.to_dict() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/results")
def clear_results():
    try:
        db.clear_all()
        return {"message": "All results cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
