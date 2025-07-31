# Redisense Web Interface

A modern, intuitive web interface for the Redisense energy monitoring system, built with FastAPI and Jinja2 templates.

## 🌟 Features

- **Modern UI**: Clean, responsive design with Bootstrap 5 and custom CSS
- **Real-time Monitoring**: Live metrics with auto-refresh capabilities
- **Device Management**: Comprehensive device overview and management
- **Semantic Search**: AI-powered device search using Redis vector search
- **Analytics Dashboard**: Interactive charts and trend analysis
- **Mobile-Friendly**: Responsive design that works on all devices

## 🚀 Quick Start

### Option 1: Using the startup script

```bash
./start_web.sh
```

### Option 2: Direct command

```bash
uv run uvicorn web_app:app --host 0.0.0.0 --port 8080 --reload
```

### Option 3: Python script

```bash
uv run python web_app.py
```

## 📱 Accessing the Interface

Once started, open your browser and navigate to:

- **Local**: http://localhost:8080
- **Network**: http://0.0.0.0:8080

## 🎨 Interface Overview

### Dashboard (`/`)

- Real-time energy metrics
- System status overview
- Quick device summary
- Auto-refresh functionality

### Devices (`/devices`)

- Complete device listing
- Device status indicators
- Quick actions (view, edit, delete)
- Add new devices

### Device Details (`/devices/{device_id}`)

- Detailed device information
- Historical data charts
- Real-time readings
- Similar device recommendations

### Analytics (`/analytics`)

- Power consumption trends
- Device performance analytics
- Historical charts
- Energy usage patterns

### Search (`/search`)

- Semantic device search
- AI-powered query understanding
- Relevance scoring
- Quick device access

## 🛠️ API Endpoints

### Web Pages

- `GET /` - Main dashboard
- `GET /devices` - Device listing
- `GET /devices/{device_id}` - Device details
- `GET /analytics` - Analytics page
- `GET /search` - Search interface

### API Endpoints

- `GET /api/metrics` - Real-time metrics
- `GET /api/devices/{device_id}/readings` - Device readings
- `GET /api/search` - Semantic search
- `GET /api/health` - System health check
- `DELETE /api/devices/{device_id}` - Delete device

## 🎯 Key Features

### High Performance

- Fast page loads
- Efficient data handling
- Optimized memory usage
- Smart caching

### Enhanced User Experience

- Clean, intuitive interface
- Mobile-responsive design
- Fast navigation
- Professional design

### Flexible Architecture

- Custom styling and themes
- Modular component system
- Interactive elements
- Smooth animations

### Production Ready

- Robust error handling
- Proper HTTP status codes
- Security best practices
- Scalable architecture

## 🎨 Customization

### Styling

- Modify `web/static/css/dashboard.css` for custom styles
- Bootstrap 5 classes available
- CSS custom properties for theming
- Dark mode support

### Templates

- Jinja2 templates in `web/templates/`
- Extend `base.html` for consistent layout
- Custom blocks for page-specific content

### JavaScript

- Interactive features in `web/static/js/dashboard.js`
- Chart.js for data visualization
- Alpine.js for reactive components
- Bootstrap JS components

## 📦 Dependencies

The FastAPI interface uses:

- **FastAPI**: Modern Python web framework
- **Jinja2**: Template engine
- **Bootstrap 5**: CSS framework
- **Chart.js**: Chart library
- **Alpine.js**: Lightweight reactive framework
- **Bootstrap Icons**: Icon library

## 🔧 Development

### File Structure

```
web/
├── static/
│   ├── css/
│   │   └── dashboard.css
│   └── js/
│       └── dashboard.js
└── templates/
    ├── base.html
    ├── dashboard.html
    ├── devices.html
    ├── device_detail.html
    ├── analytics.html
    ├── search.html
    └── error.html
```

### Adding New Pages

1. Create template in `web/templates/`
2. Add route in `web_app.py`
3. Update navigation in `base.html`
4. Add any specific CSS/JS

### Environment Variables

- Same Redis and AI service configuration as main app
- No additional environment variables needed

## 🚀 Deployment

The FastAPI app is ready for production deployment with:

- Gunicorn or Uvicorn workers
- Nginx reverse proxy
- Docker containerization
- Cloud platform deployment

## � Summary

The FastAPI interface provides a professional, performant, and highly customizable web application designed for production use with an excellent user experience.
