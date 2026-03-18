"""
RAPTOR Research Assistant - Main FastAPI Application

Combines all API routers and provides the main application entry point.
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.retrieve import router as retrieve_router
from app.api.train import router as train_router
from app.api.chat import router as chat_router
from app.api.feedback import router as feedback_router
from app.api.eval import router as eval_router

# Create FastAPI app
app = FastAPI(
    title="RAPTOR Research Assistant",
    description="Hierarchical Retrieval-Augmented Generation for Research Papers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(retrieve_router)
app.include_router(train_router)
app.include_router(chat_router)
app.include_router(feedback_router)
app.include_router(eval_router)

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "RAPTOR Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "features": [
            "Hierarchical paper retrieval (RAPTOR trees)",
            "Paper-specific learning and debate",
            "Continuous model fine-tuning",
            "Interactive chat with citations",
            "Feedback collection and preference learning"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RAPTOR Research Assistant"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)