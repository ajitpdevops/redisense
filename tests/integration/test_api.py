"""
Integration tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock

from app.main import app
from app.models.schemas import Device, EnergyReading


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_redis_service():
    """Mock Redis service"""
    with patch('app.routes.devices.redis_service') as mock:
        yield mock


@pytest.fixture
def mock_ai_service():
    """Mock AI service"""
    with patch('app.routes.devices.ai_service') as mock:
        yield mock


class TestDeviceAPI:
    """Test device management endpoints"""

    def test_create_device(self, client, mock_redis_service):
        """Test device creation endpoint"""
        mock_redis_service.get_device.return_value = None  # Device doesn't exist
        mock_redis_service.store_device.return_value = True

        device_data = {
            "id": "test-device-1",
            "name": "Test Device",
            "device_type": "HVAC",
            "location": "Building A",
            "status": "active"
        }

        response = client.post("/api/v1/devices/", json=device_data)

        assert response.status_code == 200
        assert response.json()["id"] == "test-device-1"
        mock_redis_service.store_device.assert_called_once()

    def test_create_device_already_exists(self, client, mock_redis_service):
        """Test creating a device that already exists"""
        existing_device = Device(
            id="test-device-1",
            name="Existing Device",
            device_type="HVAC",
            location="Building A"
        )
        mock_redis_service.get_device.return_value = existing_device

        device_data = {
            "id": "test-device-1",
            "name": "Test Device",
            "device_type": "HVAC",
            "location": "Building A"
        }

        response = client.post("/api/v1/devices/", json=device_data)

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_get_device(self, client, mock_redis_service):
        """Test getting a device by ID"""
        device = Device(
            id="test-device-1",
            name="Test Device",
            device_type="HVAC",
            location="Building A"
        )
        mock_redis_service.get_device.return_value = device

        response = client.get("/api/v1/devices/test-device-1")

        assert response.status_code == 200
        assert response.json()["id"] == "test-device-1"

    def test_get_device_not_found(self, client, mock_redis_service):
        """Test getting a non-existent device"""
        mock_redis_service.get_device.return_value = None

        response = client.get("/api/v1/devices/non-existent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_all_devices(self, client, mock_redis_service):
        """Test getting all devices"""
        devices = [
            Device(id="device-1", name="Device 1", device_type="HVAC", location="A"),
            Device(id="device-2", name="Device 2", device_type="Server", location="B")
        ]
        mock_redis_service.get_all_devices.return_value = devices

        response = client.get("/api/v1/devices/")

        assert response.status_code == 200
        assert len(response.json()) == 2
        assert response.json()[0]["id"] == "device-1"


class TestEnergyAPI:
    """Test energy data endpoints"""

    def test_create_energy_reading(self, client, mock_redis_service, mock_ai_service):
        """Test creating an energy reading"""
        device = Device(id="test-device", name="Test", device_type="HVAC", location="A")
        mock_redis_service.get_device.return_value = device
        mock_redis_service.get_device_readings.return_value = []  # No history
        mock_redis_service.store_energy_reading.return_value = True

        reading_data = {
            "device_id": "test-device",
            "timestamp": datetime.utcnow().isoformat(),
            "energy_kwh": 5.5
        }

        response = client.post("/api/v1/energy/", json=reading_data)

        assert response.status_code == 200
        assert response.json()["device_id"] == "test-device"
        assert response.json()["energy_kwh"] == 5.5

    def test_create_energy_reading_with_anomaly_detection(self, client, mock_redis_service, mock_ai_service):
        """Test energy reading with anomaly detection"""
        device = Device(id="test-device", name="Test", device_type="HVAC", location="A")
        mock_redis_service.get_device.return_value = device

        # Mock historical readings
        historical_readings = [
            EnergyReading(device_id="test-device", timestamp=datetime.utcnow(), energy_kwh=5.0)
            for _ in range(10)
        ]
        mock_redis_service.get_device_readings.return_value = historical_readings
        mock_redis_service.store_energy_reading.return_value = True

        # Mock AI service to detect anomaly
        mock_ai_service.detect_anomaly_simple.return_value = (True, 3.5)

        reading_data = {
            "device_id": "test-device",
            "timestamp": datetime.utcnow().isoformat(),
            "energy_kwh": 15.0  # Anomalous value
        }

        response = client.post("/api/v1/energy/", json=reading_data)

        assert response.status_code == 200
        result = response.json()
        assert result["is_anomaly"] is True
        assert result["anomaly_score"] == 3.5

    def test_get_energy_readings(self, client, mock_redis_service):
        """Test getting energy readings for a device"""
        device = Device(id="test-device", name="Test", device_type="HVAC", location="A")
        mock_redis_service.get_device.return_value = device

        readings = [
            EnergyReading(device_id="test-device", timestamp=datetime.utcnow(), energy_kwh=5.0),
            EnergyReading(device_id="test-device", timestamp=datetime.utcnow(), energy_kwh=5.2)
        ]
        mock_redis_service.get_device_readings.return_value = readings

        response = client.get("/api/v1/energy/test-device")

        assert response.status_code == 200
        assert len(response.json()) == 2


class TestSearchAPI:
    """Test semantic search endpoints"""

    def test_semantic_search(self, client):
        """Test semantic search endpoint"""
        with patch('app.routes.search.ai_service') as mock_ai, \
             patch('app.routes.search.redis_service') as mock_redis:

            mock_ai.generate_embedding.return_value = [0.1, 0.2, 0.3]
            mock_redis.semantic_search.return_value = []

            search_data = {
                "query": "high energy consumption",
                "limit": 5
            }

            response = client.post("/api/v1/search/", json=search_data)

            assert response.status_code == 200
            assert isinstance(response.json(), list)

    def test_index_content(self, client):
        """Test content indexing endpoint"""
        with patch('app.routes.search.ai_service') as mock_ai, \
             patch('app.routes.search.redis_service') as mock_redis:

            mock_ai.generate_embedding.return_value = [0.1, 0.2, 0.3]
            mock_redis.store_embedding.return_value = True

            response = client.post(
                "/api/v1/search/index",
                params={
                    "content": "This device shows high energy usage",
                    "device_id": "test-device"
                }
            )

            assert response.status_code == 200
            assert "indexed successfully" in response.json()["message"]


class TestAnomaliesAPI:
    """Test anomaly detection endpoints"""

    def test_get_anomalies(self, client, mock_redis_service):
        """Test getting anomaly alerts"""
        devices = [Device(id="test-device", name="Test", device_type="HVAC", location="A")]
        mock_redis_service.get_all_devices.return_value = devices

        anomalous_reading = EnergyReading(
            device_id="test-device",
            timestamp=datetime.utcnow(),
            energy_kwh=15.0,
            is_anomaly=True,
            anomaly_score=3.5
        )
        mock_redis_service.get_device_readings.return_value = [anomalous_reading]

        response = client.get("/api/v1/anomalies/")

        assert response.status_code == 200
        alerts = response.json()
        assert len(alerts) == 1
        assert alerts[0]["device_id"] == "test-device"
        assert alerts[0]["severity"] == "medium"  # Score 3.5 = medium

    def test_analyze_device_anomalies(self, client, mock_redis_service, mock_ai_service):
        """Test device anomaly analysis"""
        device = Device(id="test-device", name="Test", device_type="HVAC", location="A")
        mock_redis_service.get_device.return_value = device

        readings = [
            EnergyReading(device_id="test-device", timestamp=datetime.utcnow(), energy_kwh=5.0),
            EnergyReading(device_id="test-device", timestamp=datetime.utcnow(), energy_kwh=15.0, is_anomaly=True)
        ]
        mock_redis_service.get_device_readings.return_value = readings

        mock_ai_service.analyze_device_pattern.return_value = {
            "mean": 10.0,
            "pattern": "increasing"
        }
        mock_ai_service.generate_device_insights.return_value = "Device shows increasing trend"

        response = client.get("/api/v1/anomalies/test-device/analyze")

        assert response.status_code == 200
        result = response.json()
        assert result["device_id"] == "test-device"
        assert "pattern_analysis" in result
        assert "anomaly_stats" in result


class TestHealthEndpoints:
    """Test health and status endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["message"] == "Redisense API"
        assert response.json()["status"] == "running"

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        assert "status" in response.json()
        assert "redis_configured" in response.json()
