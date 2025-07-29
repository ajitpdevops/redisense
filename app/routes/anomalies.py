"""
Anomaly detection and alerts API routes
"""
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from app.models.schemas import AnomalyAlert, EnergyReading
from app.services.redis_service import redis_service
from app.services.ai_service import ai_service

router = APIRouter()

@router.get("/", response_model=List[AnomalyAlert])
async def get_anomalies(
    device_id: Optional[str] = Query(default=None),
    hours: Optional[int] = Query(default=24, description="Get anomalies from last N hours"),
    severity: Optional[str] = Query(default=None, description="Filter by severity level")
):
    """Get anomaly alerts"""
    try:
        # For now, generate alerts from anomalous readings
        # In a real implementation, these would be stored separately
        alerts = []

        if device_id:
            device_ids = [device_id]
        else:
            devices = redis_service.get_all_devices()
            device_ids = [d.id for d in devices]

        start_time = datetime.utcnow() - timedelta(hours=hours) if hours else None

        for dev_id in device_ids:
            readings = redis_service.get_device_readings(dev_id, limit=100, start_time=start_time)
            anomalous_readings = [r for r in readings if r.is_anomaly]

            for reading in anomalous_readings:
                # Determine severity based on anomaly score
                if reading.anomaly_score > 4.0:
                    alert_severity = "high"
                elif reading.anomaly_score > 3.0:
                    alert_severity = "medium"
                else:
                    alert_severity = "low"

                if severity and alert_severity != severity:
                    continue

                alert = AnomalyAlert(
                    device_id=dev_id,
                    timestamp=reading.timestamp,
                    energy_kwh=reading.energy_kwh,
                    anomaly_score=reading.anomaly_score,
                    message=f"Anomalous energy consumption detected: {reading.energy_kwh:.2f} kWh (score: {reading.anomaly_score:.2f})",
                    severity=alert_severity
                )
                alerts.append(alert)

        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)

        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {e}")

@router.get("/{device_id}/analyze")
async def analyze_device_anomalies(device_id: str):
    """Analyze anomaly patterns for a specific device"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    try:
        # Get recent readings
        readings = redis_service.get_device_readings(device_id, limit=200)

        if not readings:
            return {"message": "No readings available for analysis"}

        # Analyze patterns
        values = [r.energy_kwh for r in readings]
        pattern_analysis = ai_service.analyze_device_pattern(values)

        # Generate insights
        device_data = device.model_dump()
        insights = ai_service.generate_device_insights(device_data, pattern_analysis)

        # Count anomalies by time period
        anomaly_count_24h = sum(1 for r in readings[:24] if r.is_anomaly)
        anomaly_count_7d = sum(1 for r in readings[:168] if r.is_anomaly)  # 24*7

        return {
            "device_id": device_id,
            "analysis_period": f"Last {len(readings)} readings",
            "pattern_analysis": pattern_analysis,
            "anomaly_stats": {
                "last_24h": anomaly_count_24h,
                "last_7d": anomaly_count_7d,
                "total_analyzed": len(readings)
            },
            "insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@router.post("/{device_id}/reanalyze")
async def reanalyze_device_readings(device_id: str):
    """Re-run anomaly detection on historical readings"""
    device = redis_service.get_device(device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    try:
        # Get all readings for the device
        readings = redis_service.get_device_readings(device_id, limit=1000)

        if len(readings) < 5:
            return {"message": "Insufficient data for reanalysis"}

        # Re-analyze each reading
        updated_count = 0
        for i, reading in enumerate(readings):
            # Get context window (previous readings)
            if i < len(readings) - 5:
                context_readings = readings[i+1:i+21]  # Next 20 readings as history
                values = [r.energy_kwh for r in context_readings]

                is_anomaly, score = ai_service.detect_anomaly_simple(reading.energy_kwh, values)

                # Update if changed
                if bool(is_anomaly) != reading.is_anomaly or abs(float(score) - reading.anomaly_score) > 0.1:
                    reading.is_anomaly = bool(is_anomaly)
                    reading.anomaly_score = float(score)

                    # Store updated reading
                    redis_service.store_energy_reading(reading)
                    updated_count += 1

        return {
            "message": "Reanalysis completed",
            "total_readings": len(readings),
            "updated_readings": updated_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reanalysis failed: {e}")
