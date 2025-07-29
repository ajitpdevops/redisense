"""
Main FastAPI application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.routes import devices, energy, search, anomalies
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="Redisense - Smart Energy Monitoring",
    description="AI-powered energy usage monitoring and anomaly detection using Redis 8",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(devices.router, prefix="/api/v1/devices", tags=["devices"])
app.include_router(energy.router, prefix="/api/v1/energy", tags=["energy"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(anomalies.router, prefix="/api/v1/anomalies", tags=["anomalies"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Redisense API",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "redis_configured": bool(settings.REDIS_HOST),
        "debug_mode": settings.DEBUG
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
