"""
Test configuration and fixtures
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from datetime import datetime, timedelta
import fakeredis
from unittest.mock import Mock, AsyncMock

from app.models.schemas import Device, EnergyReading, DeviceType, DeviceStatus


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def sample_device() -> Device:
    """Sample device for testing"""
    return Device(
        device_id="test_device_001",
        device_type=DeviceType.HVAC,
        location="Building A",
        install_date=datetime.now() - timedelta(days=30),
        status=DeviceStatus.NORMAL,
        metadata={
            "manufacturer": "TestCorp",
            "model": "Model-123",
            "capacity": "5 kW"
        }
    )


@pytest.fixture
def sample_devices() -> list[Device]:
    """Multiple sample devices for testing"""
    devices = []
    device_types = [DeviceType.HVAC, DeviceType.LIGHTING, DeviceType.SERVER]
    locations = ["Building A", "Building B", "Building C"]

    for i, (device_type, location) in enumerate(zip(device_types, locations)):
        device = Device(
            device_id=f"test_device_{i:03d}",
            device_type=device_type,
            location=location,
            install_date=datetime.now() - timedelta(days=30 + i),
            status=DeviceStatus.NORMAL,
            metadata={
                "manufacturer": f"TestCorp{i}",
                "model": f"Model-{100 + i}",
                "capacity": f"{5 + i} kW"
            }
        )
        devices.append(device)

    return devices


@pytest.fixture
def sample_energy_reading(sample_device: Device) -> EnergyReading:
    """Sample energy reading for testing"""
    return EnergyReading(
        device_id=sample_device.device_id,
        timestamp=datetime.now(),
        energy_kwh=5.2,
        is_anomaly=False
    )


@pytest.fixture
def sample_energy_readings(sample_device: Device) -> list[EnergyReading]:
    """Multiple sample energy readings for testing"""
    readings = []
    base_time = datetime.now() - timedelta(hours=24)

    for i in range(24):  # 24 hourly readings
        timestamp = base_time + timedelta(hours=i)
        # Normal consumption with some variation
        energy_kwh = 3.0 + (i % 6) * 0.5 + (0.1 * (i % 3))

        reading = EnergyReading(
            device_id=sample_device.device_id,
            timestamp=timestamp,
            energy_kwh=energy_kwh,
            is_anomaly=False
        )
        readings.append(reading)

    return readings


@pytest.fixture
def anomalous_energy_reading(sample_device: Device) -> EnergyReading:
    """Anomalous energy reading for testing"""
    return EnergyReading(
        device_id=sample_device.device_id,
        timestamp=datetime.now(),
        energy_kwh=15.0,  # Unusually high
        is_anomaly=True,
        anomaly_score=2.5
    )


@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for testing embeddings"""
    mock = Mock()
    mock.encode.return_value = [0.1] * 384  # Mock 384-dim embedding
    return mock


@pytest.fixture
def mock_ai_service():
    """Mock AI service for testing"""
    mock = Mock()
    mock.generate_embedding.return_value = [0.1] * 384
    mock.detect_anomaly_simple.return_value = (False, 0.5)
    mock.detect_anomaly_ml.return_value = ([False], [0.5])
    mock.analyze_device_pattern.return_value = {
        "mean": 5.0,
        "std": 1.0,
        "anomaly_count": 0,
        "pattern": "stable"
    }
    return mock


@pytest.fixture
def mock_redis_service():
    """Mock Redis service for testing"""
    mock = Mock()
    mock.store_device.return_value = True
    mock.get_device.return_value = None
    mock.add_energy_reading.return_value = True
    mock.get_energy_readings.return_value = []
    mock.search_devices.return_value = []
    mock.get_recent_anomalies.return_value = []
    mock.store_embedding.return_value = True
    mock.semantic_search.return_value = []
    return mock


class TestDataFactory:
    """Factory for creating test data"""

    @staticmethod
    def create_device(**kwargs) -> Device:
        """Create a device with optional overrides"""
        defaults = {
            "device_id": "factory_device_001",
            "device_type": DeviceType.HVAC,
            "location": "Test Building",
            "install_date": datetime.now(),
            "status": DeviceStatus.NORMAL,
            "metadata": {"test": True}
        }
        defaults.update(kwargs)
        return Device(**defaults)

    @staticmethod
    def create_energy_reading(**kwargs) -> EnergyReading:
        """Create an energy reading with optional overrides"""
        defaults = {
            "device_id": "factory_device_001",
            "timestamp": datetime.now(),
            "energy_kwh": 5.0,
            "is_anomaly": False
        }
        defaults.update(kwargs)
        return EnergyReading(**defaults)

    @staticmethod
    def create_energy_series(device_id: str, hours: int = 24,
                           base_energy: float = 5.0) -> list[EnergyReading]:
        """Create a series of energy readings"""
        readings = []
        base_time = datetime.now() - timedelta(hours=hours)

        for i in range(hours):
            timestamp = base_time + timedelta(hours=i)
            # Add some realistic variation
            energy_kwh = base_energy + (i % 6) * 0.3 + (0.1 * (i % 3))

            reading = EnergyReading(
                device_id=device_id,
                timestamp=timestamp,
                energy_kwh=energy_kwh,
                is_anomaly=False
            )
            readings.append(reading)

        return readings


@pytest.fixture
def test_data_factory():
    """Provide test data factory"""
    return TestDataFactory()
