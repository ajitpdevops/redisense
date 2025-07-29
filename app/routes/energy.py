"""
Energy data API routes
"""
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import EnergyReading
from app.services.redis_service import redis_service
from app.services.ai_service import ai_service

router = APIRouter()

@router.post("/", response_model=EnergyReading)
async def create_energy_reading(reading: EnergyReading):
    """Create a new energy reading"""
    # Check if device exists
    device = redis_service.get_device(reading.device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # Get recent readings for anomaly detection
    recent_readings = redis_service.get_device_readings(reading.device_id, limit=50)

    if len(recent_readings) >= 5:  # Need enough history for anomaly detection
        values = [r.energy_kwh for r in recent_readings]
        is_anomaly, score = ai_service.detect_anomaly_simple(reading.energy_kwh, values)

        reading.is_anomaly = bool(is_anomaly)
        reading.anomaly_score = float(score)

    # Store the reading
    success = redis_service.store_energy_reading(reading)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store energy reading")

    return reading

@router.get("/{device_id}", response_model=List[EnergyReading])
async def get_energy_readings(
    device_id: str,
    limit: int = Query(default=100, le=1000),
    hours: Optional[int] = Query(default=None, description="Get readings from last N hours")
):
    """Get energy readings for a device"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    start_time = None
    if hours:
        start_time = datetime.utcnow() - timedelta(hours=hours)

    readings = redis_service.get_device_readings(
        device_id,
        limit=limit,
        start_time=start_time
    )

    return readings

@router.get("/{device_id}/latest", response_model=Optional[EnergyReading])
async def get_latest_reading(device_id: str):
    """Get the latest energy reading for a device"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    readings = redis_service.get_device_readings(device_id, limit=1)

    return readings[0] if readings else None

@router.get("/{device_id}/anomalies", response_model=List[EnergyReading])
async def get_anomalous_readings(
    device_id: str,
    limit: int = Query(default=50, le=500)
):
    """Get anomalous energy readings for a device"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # Get recent readings and filter for anomalies
    readings = redis_service.get_device_readings(device_id, limit=limit*2)
    anomalous_readings = [r for r in readings if r.is_anomaly]

    return anomalous_readings[:limit]
