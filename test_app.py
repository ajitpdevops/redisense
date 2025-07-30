#!/usr/bin/env python3
"""
Minimal FastAPI app for testing page loading
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Redisense Test")

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="web/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Test dashboard page"""
    try:
        # Mock data for testing
        mock_metrics = {
            "total_devices": 5,
            "active_devices": 4,
            "offline_devices": 1,
            "current_power": 15.7,
            "avg_consumption": 12.3,
            "total_energy_24h": 342.8
        }

        mock_devices = [
            {"device_id": "HVAC-001", "name": "Main HVAC System", "status": "active"},
            {"device_id": "LIGHT-002", "name": "LED Lighting Panel", "status": "active"},
            {"device_id": "SERVER-003", "name": "Data Center Server", "status": "active"},
        ]

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "devices": mock_devices,
            "total_devices": 5,
            "metrics": mock_metrics,
            "page_title": "Dashboard"
        })
    except Exception as e:
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/devices", response_class=HTMLResponse)
async def devices_page(request: Request):
    """Test devices page"""
    try:
        mock_devices = [
            {"device_id": "HVAC-001", "name": "Main HVAC System", "status": "active"},
            {"device_id": "LIGHT-002", "name": "LED Lighting Panel", "status": "active"},
            {"device_id": "SERVER-003", "name": "Data Center Server", "status": "offline"},
        ]

        return templates.TemplateResponse("devices.html", {
            "request": request,
            "devices": mock_devices,
            "latest_readings": {},
            "page_title": "Devices"
        })
    except Exception as e:
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Test analytics page"""
    try:
        mock_analytics = {
            'hourly': [
                {'hour': '00:00', 'power': 12.5},
                {'hour': '06:00', 'power': 15.8},
                {'hour': '12:00', 'power': 18.2},
                {'hour': '18:00', 'power': 16.1}
            ],
            'device_consumption': [
                {'device': 'HVAC-001', 'consumption': 125.3},
                {'device': 'LIGHT-002', 'consumption': 45.7},
                {'device': 'SERVER-003', 'consumption': 89.2}
            ]
        }

        return templates.TemplateResponse("analytics.html", {
            "request": request,
            "analytics": mock_analytics,
            "devices": [],
            "page_title": "Analytics"
        })
    except Exception as e:
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: str = ""):
    """Test search page"""
    try:
        mock_results = []
        if q:
            mock_results = [
                {"device_id": "HVAC-001", "content": "HVAC air conditioning system", "score": 0.95},
                {"device_id": "LIGHT-002", "content": "LED lighting control panel", "score": 0.87}
            ]

        return templates.TemplateResponse("search.html", {
            "request": request,
            "query": q,
            "results": mock_results,
            "page_title": "Search"
        })
    except Exception as e:
        return HTMLResponse(f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    print("🧪 Starting Test FastAPI Server...")
    print("📍 URL: http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
