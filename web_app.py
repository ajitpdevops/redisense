#!/usr/bin/env python3
"""
FastAPI Web Application for Redisense Energy Monitoring
A modern, intuitive alternative to the Streamlit dashboard
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from config.settings import Settings
from app.services.redis_service import RedisService
from app.services.ai_service import AIService

# Custom JSON encoder for datetime objects
def serialize_datetime(obj):
    """JSON serializer function that handles datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        # For custom objects, convert to dict and recursively serialize
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, 'value'):  # Handle enum values
                result[key] = value.value
            else:
                result[key] = value
        return result
    elif hasattr(obj, 'value'):  # Handle enum values directly
        return obj.value
    return obj

def safe_serialize_device(device):
    """Safely serialize a device object for JSON response"""
    try:
        if hasattr(device, 'model_dump'):
            device_dict = device.model_dump()
        else:
            device_dict = device.__dict__.copy()

        # Convert datetime objects and enums to serializable formats
        for key, value in device_dict.items():
            if isinstance(value, datetime):
                device_dict[key] = value.isoformat()
            elif hasattr(value, 'value'):  # Handle enum values
                device_dict[key] = value.value

        return device_dict
    except Exception as e:
        # Fallback to basic dict conversion
        return {
            'device_id': getattr(device, 'device_id', 'unknown'),
            'name': getattr(device, 'name', 'Unknown Device'),
            'device_type': getattr(device, 'device_type', {}).get('value', 'unknown') if hasattr(getattr(device, 'device_type', {}), 'value') else str(getattr(device, 'device_type', 'unknown')),
            'location': getattr(device, 'location', 'Unknown'),
            'manufacturer': getattr(device, 'manufacturer', 'Unknown'),
            'model': getattr(device, 'model', 'Unknown'),
            'status': getattr(device, 'status', {}).get('value', 'unknown') if hasattr(getattr(device, 'status', {}), 'value') else str(getattr(device, 'status', 'unknown'))
        }

# Initialize FastAPI app
app = FastAPI(
    title="Redisense Energy Monitoring",
    description="Modern energy monitoring dashboard powered by Redis + AI",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="web/templates")

# Initialize services with error handling
settings = Settings()
redis_service = None
ai_service = None

try:
    redis_service = RedisService(settings)
    print("✅ Redis service initialized")
except Exception as e:
    print(f"⚠️ Redis service failed: {e}")

try:
    ai_service = AIService()
    print("✅ AI service initialized")
except Exception as e:
    print(f"⚠️ AI service failed: {e}")

def calculate_metrics(devices: List, readings: List) -> Dict:
    """Calculate dashboard metrics from devices and readings"""
    try:
        total_devices = len(devices)
        active_devices = len([d for d in devices if d.status == "active"])
        offline_devices = total_devices - active_devices

        # Calculate power metrics from readings
        if readings:
            current_power = sum(r.power_kw for r in readings[-10:] if r.power_kw) / min(10, len(readings))
            avg_consumption = sum(r.power_kw for r in readings if r.power_kw) / len([r for r in readings if r.power_kw]) if readings else 0
            total_energy_24h = sum(r.energy_kwh for r in readings if r.energy_kwh)
        else:
            current_power = 0
            avg_consumption = 0
            total_energy_24h = 0

        return {
            "total_devices": total_devices,
            "active_devices": active_devices,
            "offline_devices": offline_devices,
            "current_power": round(current_power, 2),
            "avg_consumption": round(avg_consumption, 2),
            "total_energy_24h": round(total_energy_24h, 2)
        }
    except Exception as e:
        # Return default metrics on error
        return {
            "total_devices": len(devices) if devices else 0,
            "active_devices": 0,
            "offline_devices": 0,
            "current_power": 0,
            "avg_consumption": 0,
            "total_energy_24h": 0
        }

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    try:
        # Load data with fallbacks
        devices = []
        metrics = {
            "total_devices": 0,
            "active_devices": 0,
            "offline_devices": 0,
            "current_power": 0,
            "avg_consumption": 0,
            "total_energy_24h": 0
        }

        if redis_service:
            devices = redis_service.get_all_devices()

            # Get recent readings
            all_readings = []
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)

            for device in devices[:10]:  # Limit to 10 devices for performance
                readings = redis_service.get_energy_readings(
                    device.device_id,
                    start_time,
                    end_time,
                    limit=50
                )
                all_readings.extend(readings)

            # Calculate metrics
            metrics = calculate_metrics(devices, all_readings)
        else:
            # Provide demo data when Redis is not available
            devices = []

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "devices": devices[:8],  # Show first 8 devices on main page
            "total_devices": len(devices),
            "metrics": metrics,
            "page_title": "Dashboard"
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "page_title": "Error"
        })

@app.get("/devices", response_class=HTMLResponse)
async def devices_page(request: Request):
    """Devices management page"""
    try:
        devices = []
        latest_readings = {}

        if redis_service:
            devices = redis_service.get_all_devices()

            # Get latest readings for each device
            for device in devices:
                readings = redis_service.get_energy_readings(
                    device.device_id,
                    datetime.utcnow() - timedelta(hours=1),
                    datetime.utcnow(),
                    limit=1
                )
                if readings:
                    latest_readings[device.device_id] = readings[0]

        return templates.TemplateResponse("devices.html", {
            "request": request,
            "devices": devices,
            "latest_readings": latest_readings,
            "page_title": "Devices"
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "page_title": "Error"
        })

@app.get("/device/{device_id}", response_class=HTMLResponse)
async def device_detail(request: Request, device_id: str):
    """Device detail page"""
    try:
        if not redis_service:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error": "Redis service not available",
                "page_title": "Service Error"
            })

        # Get device
        devices = redis_service.get_all_devices()
        device = next((d for d in devices if d.device_id == device_id), None)

        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # Get device readings
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        readings = redis_service.get_energy_readings(
            device_id,
            start_time,
            end_time,
            limit=100
        )

        # Calculate device-specific metrics
        device_metrics = calculate_device_metrics(device, readings)

        return templates.TemplateResponse("device_detail.html", {
            "request": request,
            "device": device,
            "readings": readings,
            "metrics": device_metrics,
            "page_title": f"Device - {getattr(device, 'name', device_id)}"
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "page_title": "Error"
        })

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: Optional[str] = Query(None)):
    """Semantic search page"""
    try:
        results = []
        if q and ai_service:
            results = ai_service.semantic_search(q, limit=10)
        elif q and not ai_service:
            # Fallback: basic text search
            results = []

        return templates.TemplateResponse("search.html", {
            "request": request,
            "query": q or "",
            "results": results,
            "page_title": "Device Search"
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "page_title": "Error"
        })

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics and trends page"""
    try:
        devices = []
        all_readings = []
        analytics_data = {'hourly': [], 'device_consumption': []}

        if redis_service:
            devices = redis_service.get_all_devices()

            # Get readings for analytics
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)

            for device in devices:
                readings = redis_service.get_energy_readings(
                    device.device_id,
                    start_time,
                    end_time,
                    limit=50
                )
                all_readings.extend(readings)

            # Process analytics data
            analytics_data = process_analytics_data(all_readings)

        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "analytics": analytics_data,
            "devices": devices,
            "page_title": "Analytics"
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "page_title": "Error"
        })

# API Endpoints for AJAX calls

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        if redis_service:
            devices = redis_service.get_all_devices()
            return {"status": "healthy", "devices_count": len(devices), "services": {"redis": True, "ai": ai_service is not None}}
        else:
            return {"status": "degraded", "devices_count": 0, "services": {"redis": False, "ai": ai_service is not None}}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "services": {"redis": False, "ai": False}}
        )

@app.get("/api/metrics")
async def get_metrics():
    """Get real-time metrics for dashboard updates"""
    try:
        if not redis_service:
            return {
                "total_devices": 0,
                "active_devices": 0,
                "offline_devices": 0,
                "current_power": 0,
                "avg_consumption": 0,
                "total_energy_24h": 0
            }

        devices = redis_service.get_all_devices()

        # Get recent readings
        all_readings = []
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        for device in devices[:10]:  # Limit for performance
            readings = redis_service.get_energy_readings(
                device.device_id,
                start_time,
                end_time,
                limit=5
            )
            all_readings.extend(readings)

        metrics = calculate_metrics(devices, all_readings)
        return metrics

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/search")
async def api_search(q: str = Query(...)):
    """API endpoint for semantic search"""
    try:
        if not ai_service:
            return {"results": [], "error": "AI service not available"}

        results = ai_service.semantic_search(q, limit=10)
        return {"results": results}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/device/{device_id}/readings")
async def get_device_readings(device_id: str, hours: int = Query(24, ge=1, le=168)):
    """Get device readings for charts"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        readings = redis_service.get_energy_readings(
            device_id,
            start_time,
            end_time,
            limit=200
        )

        # Format for charts
        chart_data = []
        for reading in readings:
            chart_data.append({
                'timestamp': reading.timestamp.isoformat(),
                'energy_kwh': reading.energy_kwh,
                'power_kw': reading.power_kw or reading.energy_kwh,
                'voltage': reading.voltage,
                'current': reading.current
            })

        return JSONResponse(content=chart_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def api_search(query: str = Form(...)):
    """API endpoint for semantic search"""
    try:
        results = ai_service.semantic_search(query, limit=10)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/devices/similar/{device_id}")
async def get_similar_devices(device_id: str, limit: int = Query(5, ge=1, le=10)):
    """Get devices similar to the specified device"""
    try:
        if not redis_service:
            return JSONResponse(content={"error": "Redis service not available"}, status_code=503)

        devices = redis_service.get_all_devices()
        device = next((d for d in devices if d.device_id == device_id), None)

        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        # If AI service is not available, return devices of same type as fallback
        if not ai_service:
            similar_devices = []
            device_type = getattr(device, 'device_type', None)
            device_location = getattr(device, 'location', None)

            for d in devices:
                if d.device_id != device_id:
                    score = 0.5  # Base similarity
                    if hasattr(d, 'device_type') and hasattr(device, 'device_type') and d.device_type == device.device_type:
                        score += 0.3
                    if hasattr(d, 'location') and hasattr(device, 'location') and d.location == device.location:
                        score += 0.2

                    similar_devices.append({
                        'device': safe_serialize_device(d),
                        'score': score,
                        'content': f"Similar device: {getattr(d, 'name', d.device_id)}"
                    })

            # Sort by score and return top results
            similar_devices.sort(key=lambda x: x['score'], reverse=True)
            return JSONResponse(content=similar_devices[:limit])

        # AI service is available - use semantic search
        device_data = device.model_dump() if hasattr(device, 'model_dump') else device.__dict__
        device_content = ai_service.create_device_embedding(device_data)

        # Search for similar devices
        similar_results = ai_service.semantic_search(device_content, limit + 1)

        # Filter out the original device
        similar_devices = []
        for result in similar_results:
            if result.get('device_id') != device_id:
                similar_device = next((d for d in devices if d.device_id == result.get('device_id')), None)
                if similar_device:
                    similar_devices.append({
                        'device': safe_serialize_device(similar_device),
                        'score': result.get('score', 0),
                        'content': result.get('content', '')
                    })

        return JSONResponse(content=similar_devices[:limit])

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error finding similar devices: {str(e)}"},
            status_code=500
        )

# Admin Routes

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Admin panel for data management"""
    try:
        # Get system status
        status = {
            "redis": redis_service is not None,
            "ai": ai_service is not None,
            "devices_count": 0,
            "readings_count": 0
        }

        if redis_service:
            devices = redis_service.get_all_devices()
            status["devices_count"] = len(devices)

            # Get data statistics
            stats = redis_service.get_data_statistics()
            status["readings_count"] = stats.get("estimated_total_readings", 0)

        return templates.TemplateResponse("admin.html", {
            "request": request,
            "status": status,
            "page_title": "Admin Panel"
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "page_title": "Error"
        })

@app.post("/admin/api/seed-data")
async def seed_historical_data(request: Request):
    """Seed historical data for testing"""
    try:
        if not redis_service:
            return JSONResponse(content={"success": False, "error": "Redis service not available"})

        data = await request.json()
        days = data.get("days", 7)
        interval_minutes = data.get("interval_minutes", 15)

        devices = redis_service.get_all_devices()
        if not devices:
            return JSONResponse(content={"success": False, "error": "No devices found"})

        # Generate historical readings
        readings_created = 0
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        current_time = start_time
        while current_time <= end_time:
            for device in devices:
                # Generate realistic energy reading
                base_power = getattr(device, 'power_rating', 1000) / 1000  # Convert to kW
                variation = (hash(str(current_time) + device.device_id) % 40 - 20) / 100  # ±20% variation
                power_kw = max(0.1, base_power * (1 + variation))
                energy_kwh = power_kw * (interval_minutes / 60)  # Energy = Power × Time

                # Create reading
                from app.models.schemas import EnergyReading
                reading = EnergyReading(
                    device_id=device.device_id,
                    timestamp=current_time,
                    energy_kwh=energy_kwh,
                    power_kw=power_kw,
                    voltage=230.0 + (hash(str(current_time)) % 20 - 10),  # 220-240V
                    current=power_kw * 1000 / 230,  # I = P/V
                    frequency=50.0
                )

                redis_service.store_energy_reading(reading)
                readings_created += 1

            current_time += timedelta(minutes=interval_minutes)

        return JSONResponse(content={
            "success": True,
            "readings_created": readings_created,
            "days": days,
            "devices": len(devices)
        })

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.post("/admin/api/stream-data")
async def stream_realtime_data():
    """Generate real-time data for all devices"""
    try:
        if not redis_service:
            return JSONResponse(content={"success": False, "error": "Redis service not available"})

        devices = redis_service.get_all_devices()
        if not devices:
            return JSONResponse(content={"success": False, "error": "No devices found"})

        readings_created = 0
        current_time = datetime.utcnow()

        for device in devices:
            # Generate realistic current reading
            base_power = getattr(device, 'power_rating', 1000) / 1000  # Convert to kW
            variation = (hash(str(current_time) + device.device_id) % 30 - 15) / 100  # ±15% variation
            power_kw = max(0.1, base_power * (1 + variation))
            energy_kwh = power_kw / 12  # 5-minute interval (1/12 of an hour)

            # Create reading
            from app.models.schemas import EnergyReading
            reading = EnergyReading(
                device_id=device.device_id,
                timestamp=current_time,
                energy_kwh=energy_kwh,
                power_kw=power_kw,
                voltage=230.0 + (hash(str(current_time)) % 20 - 10),
                current=power_kw * 1000 / 230,
                frequency=50.0
            )

            redis_service.store_energy_reading(reading)
            readings_created += 1

        return JSONResponse(content={
            "success": True,
            "readings_created": readings_created,
            "timestamp": current_time.isoformat()
        })

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

@app.post("/admin/api/clear-data")
async def clear_data(request: Request):
    """Clear system data"""
    try:
        if not redis_service:
            return JSONResponse(content={"success": False, "error": "Redis service not available"})

        data = await request.json()
        clear_type = data.get("type", "readings")

        if clear_type == "readings":
            # Clear only energy readings
            cleared_count = redis_service.clear_energy_readings()

            return JSONResponse(content={
                "success": True,
                "message": f"Cleared {cleared_count} energy reading sets",
                "cleared_readings": cleared_count
            })

        elif clear_type == "all":
            # Clear all data including devices
            clear_stats = redis_service.clear_all_data()

            return JSONResponse(content={
                "success": True,
                "message": f"Cleared all data: {clear_stats['devices']} devices, {clear_stats['readings']} reading sets",
                "cleared_devices": clear_stats['devices'],
                "cleared_readings": clear_stats['readings']
            })

        else:
            return JSONResponse(content={"success": False, "error": "Invalid clear type"})

    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)})

# Helper functions

def calculate_metrics(devices: List, readings: List) -> Dict:
    """Calculate dashboard metrics"""
    if not readings:
        return {
            'total_devices': len(devices),
            'current_power': 0,
            'avg_consumption': 0,
            'total_energy_24h': 0,
            'active_devices': 0,
            'offline_devices': 0
        }

    # Get latest readings per device
    latest_readings = {}
    for reading in readings:
        if reading.device_id not in latest_readings:
            latest_readings[reading.device_id] = reading
        elif reading.timestamp > latest_readings[reading.device_id].timestamp:
            latest_readings[reading.device_id] = reading

    current_power = sum(r.power_kw or r.energy_kwh for r in latest_readings.values())
    total_devices = len(devices)
    avg_consumption = current_power / total_devices if total_devices > 0 else 0

    # Calculate total energy over last 24 hours
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    total_energy_24h = sum(
        r.energy_kwh for r in readings
        if r.timestamp >= last_24h
    )

    # Calculate device status
    now = datetime.utcnow()
    active_devices = 0
    for device_id, reading in latest_readings.items():
        time_diff = now - reading.timestamp
        if time_diff.total_seconds() < 300:  # Less than 5 minutes
            active_devices += 1

    return {
        'total_devices': total_devices,
        'current_power': round(current_power, 2),
        'avg_consumption': round(avg_consumption, 2),
        'total_energy_24h': round(total_energy_24h, 2),
        'active_devices': active_devices,
        'offline_devices': total_devices - active_devices
    }

def calculate_device_metrics(device, readings: List) -> Dict:
    """Calculate metrics for a specific device"""
    try:
        if not readings:
            return {
                "current_power": 0,
                "avg_power": 0,
                "max_power": 0,
                "total_energy": 0,
                "readings_count": 0,
                "status": getattr(device, 'status', 'unknown')
            }

        # Calculate power metrics
        power_values = [r.power_kw for r in readings if r.power_kw is not None]
        current_power = readings[-1].power_kw if readings and readings[-1].power_kw else 0
        avg_power = sum(power_values) / len(power_values) if power_values else 0
        max_power = max(power_values) if power_values else 0
        total_energy = sum(r.energy_kwh for r in readings if r.energy_kwh)

        return {
            "current_power": round(current_power, 2),
            "avg_power": round(avg_power, 2),
            "max_power": round(max_power, 2),
            "total_energy": round(total_energy, 2),
            "readings_count": len(readings),
            "status": getattr(device, 'status', 'active')
        }
    except Exception:
        return {
            "current_power": 0,
            "avg_power": 0,
            "max_power": 0,
            "total_energy": 0,
            "readings_count": 0,
            "status": "unknown"
        }

def process_analytics_data(readings: List) -> Dict:
    """Process readings for analytics charts"""
    if not readings:
        return {'hourly': [], 'device_consumption': []}

    # Group by hour
    hourly_data = {}
    device_consumption = {}

    for reading in readings:
        hour_key = reading.timestamp.strftime('%H:00')
        if hour_key not in hourly_data:
            hourly_data[hour_key] = []
        hourly_data[hour_key].append(reading.power_kw or reading.energy_kwh)

        # Device consumption
        if reading.device_id not in device_consumption:
            device_consumption[reading.device_id] = 0
        device_consumption[reading.device_id] += reading.energy_kwh

    # Calculate hourly averages
    hourly_chart = []
    for hour in sorted(hourly_data.keys()):
        avg_power = sum(hourly_data[hour]) / len(hourly_data[hour])
        hourly_chart.append({'hour': hour, 'power': round(avg_power, 2)})

    # Top consuming devices
    device_chart = []
    for device_id, consumption in sorted(device_consumption.items(), key=lambda x: x[1], reverse=True)[:10]:
        device_chart.append({'device': device_id, 'consumption': round(consumption, 2)})

    return {
        'hourly': hourly_chart,
        'device_consumption': device_chart
    }

@app.get("/api/analytics/time-series")
async def get_time_series_data(hours: int = Query(24, ge=1, le=168)):
    """Get time series data for all devices"""
    try:
        if not redis_service:
            return JSONResponse(content={"error": "Redis service not available"}, status_code=503)

        devices = redis_service.get_all_devices()
        if not devices:
            return JSONResponse(content={"error": "No devices found"}, status_code=404)

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Get readings for all devices
        time_series_data = {}

        for device in devices:
            try:
                readings = redis_service.get_energy_readings(
                    device.device_id,
                    start_time,
                    end_time,
                    limit=min(1000, hours * 4)  # Limit points based on time range
                )

                # Convert readings to serializable format
                device_readings = []
                for reading in readings:
                    device_readings.append({
                        'timestamp': reading.timestamp.isoformat(),
                        'power_kw': reading.power_kw or reading.energy_kwh,  # Fallback to energy if power not available
                        'energy_kwh': reading.energy_kwh
                    })

                if device_readings:  # Only include devices with data
                    time_series_data[device.device_id] = device_readings

            except Exception as e:
                # Continue with other devices if one fails
                print(f"Error getting readings for device {device.device_id}: {e}")
                continue

        return JSONResponse(content=time_series_data)

    except Exception as e:
        return JSONResponse(
            content={"error": f"Error fetching time series data: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
