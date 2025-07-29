"""
Integration tests for Redis service
"""
import os
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from app.services.redis_service import RedisService
from app.models.schemas import Device, EnergyReading
from config.settings import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    with patch.dict(os.environ, {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_USERNAME": "",
        "REDIS_PASSWORD": "",
        "DEBUG": "True"
    }, clear=True):
        yield Settings()


@pytest.fixture
def redis_service(mock_settings):
    """Redis service with mocked connection"""
    with patch('redis.Redis') as mock_redis_class:
        mock_redis = mock_redis_class.return_value
        mock_redis.ping.return_value = True

        service = RedisService(mock_settings)
        service.redis_client = mock_redis

        yield service


class TestRedisService:
    """Test Redis service operations"""

    def test_store_and_retrieve_device(self, redis_service):
        """Test storing and retrieving a device"""
        device = Device(
            id="test-device-1",
            name="Test Device",
            device_type="HVAC",
            location="Building A"
        )

        # Mock Redis operations
        redis_service.redis_client.hset.return_value = True
        redis_service.redis_client.sadd.return_value = True
        redis_service.redis_client.hgetall.return_value = {
            "id": "test-device-1",
            "name": "Test Device",
            "device_type": "HVAC",
            "location": "Building A",
            "status": "active",
            "metadata": "{}"
        }

        # Store device
        result = redis_service.store_device(device)
        assert result is True

        # Retrieve device
        retrieved = redis_service.get_device("test-device-1")
        assert retrieved.id == "test-device-1"
        assert retrieved.name == "Test Device"

    def test_get_nonexistent_device(self, redis_service):
        """Test retrieving a non-existent device"""
        redis_service.redis_client.hgetall.return_value = {}

        result = redis_service.get_device("nonexistent")
        assert result is None

    def test_get_all_devices(self, redis_service):
        """Test getting all devices"""
        # Mock device IDs in index
        redis_service.redis_client.smembers.return_value = {"device-1", "device-2"}

        # Mock device data
        def mock_hgetall(key):
            if key == "device:device-1":
                return {
                    "id": "device-1",
                    "name": "Device 1",
                    "device_type": "HVAC",
                    "location": "Building A",
                    "status": "active",
                    "metadata": "{}"
                }
            elif key == "device:device-2":
                return {
                    "id": "device-2",
                    "name": "Device 2",
                    "device_type": "Server",
                    "location": "Building B",
                    "status": "active",
                    "metadata": "{}"
                }
            return {}

        redis_service.redis_client.hgetall.side_effect = mock_hgetall

        devices = redis_service.get_all_devices()
        assert len(devices) == 2
        assert devices[0].id in ["device-1", "device-2"]

    def test_store_energy_reading(self, redis_service):
        """Test storing an energy reading"""
        reading = EnergyReading(
            device_id="test-device",
            timestamp=datetime.utcnow(),
            energy_kwh=5.5
        )

        redis_service.redis_client.hset.return_value = True
        redis_service.redis_client.zadd.return_value = True
        redis_service.redis_client.zremrangebyrank.return_value = 0

        result = redis_service.store_energy_reading(reading)
        assert result is True

        # Verify Redis calls
        redis_service.redis_client.hset.assert_called()
        redis_service.redis_client.zadd.assert_called()

    def test_get_device_readings(self, redis_service):
        """Test retrieving device readings"""
        # Mock reading keys from sorted set
        timestamp = datetime.utcnow()
        reading_key = f"energy:test-device:{timestamp.isoformat()}"

        redis_service.redis_client.zrevrangebyscore.return_value = [reading_key]
        redis_service.redis_client.hgetall.return_value = {
            "device_id": "test-device",
            "timestamp": timestamp.isoformat(),
            "energy_kwh": "5.5",
            "is_anomaly": "False",
            "anomaly_score": "0.0"
        }

        readings = redis_service.get_device_readings("test-device", limit=10)

        assert len(readings) == 1
        assert readings[0].device_id == "test-device"
        assert readings[0].energy_kwh == 5.5

    def test_store_and_search_embedding(self, redis_service):
        """Test storing and searching embeddings"""
        content = "High energy consumption detected"
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        metadata = {"device_id": "test-device", "type": "alert"}

        redis_service.redis_client.hset.return_value = True
        redis_service.redis_client.sadd.return_value = True

        # Store embedding
        result = redis_service.store_embedding(content, embedding, metadata)
        assert result is True

        # Mock search
        doc_id = f"doc:{hash(content)}"
        redis_service.redis_client.smembers.return_value = {doc_id}
        redis_service.redis_client.hgetall.return_value = {
            "content": content,
            "embedding": "[0.1, 0.2, 0.3, 0.4]",
            "metadata": '{"device_id": "test-device", "type": "alert"}'
        }

        # Search
        query_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        results = redis_service.semantic_search(query_embedding, limit=5)

        assert len(results) == 1
        assert results[0].content == content
        assert results[0].device_id == "test-device"
        assert results[0].score > 0.9  # Should be high similarity

    def test_get_stats(self, redis_service):
        """Test getting Redis statistics"""
        redis_service.redis_client.info.return_value = {
            "redis_version": "7.0.0",
            "used_memory_human": "1.5M",
            "total_commands_processed": 1000,
            "connected_clients": 5
        }
        redis_service.redis_client.scard.return_value = 10

        stats = redis_service.get_stats()

        assert stats["redis_version"] == "7.0.0"
        assert stats["device_count"] == 10
        assert stats["embedding_count"] == 10

    def test_connection_error_handling(self, mock_settings):
        """Test Redis connection error handling"""
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = mock_redis_class.return_value
            mock_redis.ping.side_effect = ConnectionError("Connection failed")

            with pytest.raises(ConnectionError):
                RedisService(mock_settings)

    def test_store_device_error_handling(self, redis_service):
        """Test device storage error handling"""
        device = Device(
            id="test-device",
            name="Test Device",
            device_type="HVAC",
            location="Building A"
        )

        redis_service.redis_client.hset.side_effect = Exception("Redis error")

        result = redis_service.store_device(device)
        assert result is False

    def test_get_device_error_handling(self, redis_service):
        """Test device retrieval error handling"""
        redis_service.redis_client.hgetall.side_effect = Exception("Redis error")

        result = redis_service.get_device("test-device")
        assert result is None
