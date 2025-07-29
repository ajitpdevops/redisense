# Redisense - Smart Energy Monitoring System

A POC application demonstrating Redis 8 AI features for real-time energy usage monitoring and anomaly detection. Built with FastAPI, Redis, and AI-powered analytics.

## 🚀 Features

- **Real-time Energy Monitoring**: Track energy consumption across multiple devices
- **AI-Powered Anomaly Detection**: Detect unusual energy patterns using statistical and ML methods
- **Semantic Search**: Search device data using natural language queries
- **Pattern Analysis**: Identify consumption trends (stable, increasing, decreasing, high variation)
- **RESTful API**: Complete REST API for device management and data analysis
- **Test-Driven Development**: Comprehensive test suite with >80% coverage

## 📋 Prerequisites

- Python 3.10+
- Redis 6.0+ (Redis Cloud supported)
- uv package manager

## 🛠 Installation

1. **Clone and navigate to the project**:

   ```bash
   cd /workspaces/intellistream/redisense
   ```

2. **Install dependencies with uv**:

   ```bash
   uv sync --extra test
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your Redis connection details
   ```

## 🏃 Quick Start

### 1. Test the Setup

```bash
# Test dependencies and connections
uv run python cli.py test-connection

# Run the test suite
uv run pytest -v
```

### 2. Load Demo Data

```bash
# Generate sample devices and energy readings
uv run python cli.py seed-data --device-count 5 --days 7
```

### 3. Start the API Server

```bash
# Start the FastAPI development server
uv run python -m app.main
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### 4. Explore the CLI

```bash
# Check system status
uv run python cli.py status

# Analyze a specific device
uv run python cli.py analyze device-001 --hours 48

# Perform semantic search
uv run python cli.py search "high energy consumption"
```

## 🧪 Testing

### Run All Tests

```bash
uv run pytest
```

### Run with Coverage

```bash
uv run pytest --cov=app --cov=config --cov-report=html
```

### Run Specific Test Categories

```bash
# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# Specific test file
uv run pytest tests/unit/test_ai_service.py -v
```

## 📖 API Documentation

### Core Endpoints

#### Devices

- `POST /api/v1/devices/` - Create a new device
- `GET /api/v1/devices/` - List all devices
- `GET /api/v1/devices/{id}` - Get device details
- `GET /api/v1/devices/{id}/stats` - Get device statistics

#### Energy Data

- `POST /api/v1/energy/` - Submit energy reading
- `GET /api/v1/energy/{device_id}` - Get device readings
- `GET /api/v1/energy/{device_id}/latest` - Get latest reading
- `GET /api/v1/energy/{device_id}/anomalies` - Get anomalous readings

#### Anomaly Detection

- `GET /api/v1/anomalies/` - List anomaly alerts
- `GET /api/v1/anomalies/{device_id}/analyze` - Analyze device patterns
- `POST /api/v1/anomalies/{device_id}/reanalyze` - Re-run anomaly detection

#### Semantic Search

- `POST /api/v1/search/` - Perform semantic search
- `POST /api/v1/search/index` - Index content for search

### Example API Usage

```python
import httpx

# Create a device
device_data = {
    "id": "hvac-001",
    "name": "Main HVAC Unit",
    "device_type": "HVAC",
    "location": "Building A"
}
response = httpx.post("http://localhost:8000/api/v1/devices/", json=device_data)

# Submit energy reading
reading_data = {
    "device_id": "hvac-001",
    "timestamp": "2024-01-15T10:00:00",
    "energy_kwh": 5.2
}
response = httpx.post("http://localhost:8000/api/v1/energy/", json=reading_data)

# Search for devices
search_data = {
    "query": "HVAC systems with high energy usage",
    "limit": 5
}
response = httpx.post("http://localhost:8000/api/v1/search/", json=search_data)
```

## 🏗 Architecture

### Project Structure

```
redisense/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── routes/              # API endpoints
│   │   ├── devices.py
│   │   ├── energy.py
│   │   ├── search.py
│   │   └── anomalies.py
│   └── services/            # Business logic
│       ├── ai_service.py    # AI/ML operations
│       └── redis_service.py # Redis operations
├── config/
│   └── settings.py          # Configuration management
├── data/
│   └── generator.py         # Test data generation
├── tests/
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Test fixtures
├── cli.py                  # Command-line interface
└── pyproject.toml          # Dependencies and build config
```

### Key Components

- **AI Service**: Handles anomaly detection, pattern analysis, and embedding generation
- **Redis Service**: Manages data storage, time series, and vector search
- **Data Generator**: Creates realistic test data for development and testing
- **CLI Tools**: Management commands for data loading and analysis

## ⚙️ Configuration

Environment variables (`.env` file):

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_USERNAME=
REDIS_PASSWORD=

# Application Settings
DEBUG=True
LOG_LEVEL=INFO

# AI/ML Settings
ANOMALY_THRESHOLD=2.0
EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L6-v2
VECTOR_DIMENSION=384

# Data Generation
DEVICE_COUNT=5
DATA_GENERATION_INTERVAL=60
```

## 🔬 AI Features

### Anomaly Detection

- **Statistical Method**: Z-score based detection with configurable threshold
- **ML Method**: Isolation Forest for complex pattern detection
- **Real-time Processing**: Anomalies detected as data arrives

### Pattern Analysis

- **Trend Detection**: Identifies increasing, decreasing, or stable patterns
- **Variability Assessment**: Classifies consumption variability levels
- **Statistical Insights**: Mean, standard deviation, and distribution analysis

### Semantic Search

- **Embedding Generation**: Uses sentence-transformers for text embeddings
- **Vector Similarity**: Cosine similarity for semantic matching
- **Metadata Support**: Rich metadata for context-aware search

## 🧪 Development

### Adding New Tests

```python
# Unit test example (tests/unit/test_new_feature.py)
def test_new_feature():
    """Test description"""
    # Arrange
    # Act
    # Assert
    pass

# Integration test example (tests/integration/test_new_api.py)
def test_new_endpoint(client):
    """Test API endpoint"""
    response = client.get("/api/v1/new-endpoint")
    assert response.status_code == 200
```

### Running in Development Mode

```bash
# Start with auto-reload
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run with debug logging
DEBUG=True uv run python -m app.main
```

### Docker Development (Optional)

```bash
# Build and run with docker-compose
cd /workspaces/intellistream/deploy
docker-compose -f docker-compose.dev.yml up redisense-api
```

## 📊 Performance & Monitoring

### Metrics Available

- Device count and status
- Reading ingestion rate
- Anomaly detection accuracy
- Search index size and performance
- Redis memory usage

### Monitoring Commands

```bash
# System status
uv run python cli.py status

# Device analysis
uv run python cli.py analyze <device-id>

# Clear test data
uv run python cli.py clear-data
```

## 🤝 Contributing

1. **Setup Development Environment**:

   ```bash
   uv sync --extra test --extra dev
   ```

2. **Run Tests Before Committing**:

   ```bash
   uv run pytest
   uv run black app/ tests/ config/
   uv run flake8 app/ tests/ config/
   ```

3. **Follow TDD Principles**:
   - Write tests first
   - Keep test coverage high
   - Use descriptive test names

## 📚 Learning Resources

- [Redis Documentation](https://redis.io/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Sentence Transformers](https://www.sbert.net/)

## 🐛 Troubleshooting

### Common Issues

1. **Redis Connection Failed**:

   ```bash
   # Check Redis configuration
   uv run python cli.py test-connection

   # Verify Redis is running
   redis-cli ping
   ```

2. **Import Errors**:

   ```bash
   # Reinstall dependencies
   uv sync --extra test
   ```

3. **Test Failures**:

   ```bash
   # Run specific failing test with verbose output
   uv run pytest tests/unit/test_specific.py::test_function -v -s
   ```

4. **Performance Issues**:

   ```bash
   # Check Redis memory usage
   uv run python cli.py status

   # Clear test data if needed
   uv run python cli.py clear-data
   ```

## 📄 License

This project is a POC for demonstration purposes. See the main repository for licensing information.

---

**Built with ❤️ for Redis 8 AI Features Testing**
