"""
Device management API routes
"""
from typing import List
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import Device, DeviceStats
from app.services.redis_service import redis_service
from app.services.ai_service import ai_service

router = APIRouter()

@router.post("/", response_model=Device)
async def create_device(device: Device):
    """Create a new device"""
    # Check if device already exists
    existing = redis_service.get_device(device.id)
    if existing:
        raise HTTPException(status_code=400, detail="Device already exists")

    success = redis_service.store_device(device)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to store device")

    return device

@router.get("/", response_model=List[Device])
async def get_devices():
    """Get all devices"""
    devices = redis_service.get_all_devices()
    return devices

@router.get("/{device_id}", response_model=Device)
async def get_device(device_id: str):
    """Get device by ID"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device

@router.get("/{device_id}/stats", response_model=DeviceStats)
async def get_device_stats(device_id: str):
    """Get device statistics"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # Get recent readings
    readings = redis_service.get_device_readings(device_id, limit=100)

    if not readings:
        return DeviceStats(
            device_id=device_id,
            total_readings=0,
            average_consumption=0.0,
            anomaly_count=0,
            last_reading=None,
            consumption_pattern="unknown"
        )

    # Calculate statistics
    values = [r.energy_kwh for r in readings]
    pattern_analysis = ai_service.analyze_device_pattern(values)
    anomaly_count = sum(1 for r in readings if r.is_anomaly)

    return DeviceStats(
        device_id=device_id,
        total_readings=len(readings),
        average_consumption=pattern_analysis.get("mean", 0.0),
        anomaly_count=anomaly_count,
        last_reading=readings[0] if readings else None,
        consumption_pattern=pattern_analysis.get("pattern", "unknown")
    )

@router.delete("/{device_id}")
async def delete_device(device_id: str):
    """Delete a device"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    # Note: In a real implementation, you'd also delete all associated readings
    # For now, just remove from device index
    try:
        redis_service.redis_client.srem("devices:index", device_id)
        redis_service.redis_client.delete(f"device:{device_id}")
        return {"message": "Device deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete device: {e}")
