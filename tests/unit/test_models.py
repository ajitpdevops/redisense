"""
Unit tests for Pydantic models
"""
import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from app.models.schemas import (
    Device, EnergyReading, DeviceType, DeviceStatus,
    AnomalyAlert, SemanticSearchQuery, DeviceStats
)


class TestDevice:
    """Test Device model"""

    def test_create_valid_device(self, sample_device):
        """Test creating a valid device"""
        assert sample_device.device_id == "test_device_001"
        assert sample_device.device_type == DeviceType.HVAC
        assert sample_device.location == "Building A"
        assert sample_device.status == DeviceStatus.NORMAL
        assert isinstance(sample_device.install_date, datetime)
        assert isinstance(sample_device.metadata, dict)

    def test_device_id_required(self):
        """Test that device_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Device(
                device_type=DeviceType.HVAC,
                location="Building A",
                install_date=datetime.now()
            )
        assert "device_id" in str(exc_info.value)

    def test_device_type_validation(self):
        """Test device type validation"""
        with pytest.raises(ValidationError):
            Device(
                device_id="test_001",
                device_type="INVALID_TYPE",
                location="Building A",
                install_date=datetime.now()
            )

    def test_device_status_default(self):
        """Test default device status"""
        device = Device(
            device_id="test_001",
            device_type=DeviceType.HVAC,
            location="Building A",
            install_date=datetime.now()
        )
        assert device.status == DeviceStatus.NORMAL

    def test_metadata_default(self):
        """Test default metadata is empty dict"""
        device = Device(
            device_id="test_001",
            device_type=DeviceType.HVAC,
            location="Building A",
            install_date=datetime.now()
        )
        assert device.metadata == {}

    def test_device_serialization(self, sample_device):
        """Test device can be serialized to dict"""
        device_dict = sample_device.model_dump()
        assert device_dict["device_id"] == "test_device_001"
        assert device_dict["device_type"] == "HVAC"
        assert device_dict["status"] == "normal"

    def test_device_json_serialization(self, sample_device):
        """Test device can be serialized to JSON"""
        json_str = sample_device.model_dump_json()
        assert '"device_id":"test_device_001"' in json_str
        assert '"device_type":"HVAC"' in json_str


class TestEnergyReading:
    """Test EnergyReading model"""

    def test_create_valid_reading(self, sample_energy_reading):
        """Test creating a valid energy reading"""
        assert sample_energy_reading.device_id == "test_device_001"
        assert sample_energy_reading.energy_kwh == 5.2
        assert isinstance(sample_energy_reading.timestamp, datetime)
        assert sample_energy_reading.is_anomaly is False
        assert sample_energy_reading.anomaly_score is None

    def test_energy_kwh_positive(self):
        """Test energy_kwh must be positive"""
        with pytest.raises(ValidationError) as exc_info:
            EnergyReading(
                device_id="test_001",
                energy_kwh=-1.0
            )
        assert "greater than 0" in str(exc_info.value)

    def test_energy_kwh_zero_invalid(self):
        """Test energy_kwh cannot be zero"""
        with pytest.raises(ValidationError):
            EnergyReading(
                device_id="test_001",
                energy_kwh=0.0
            )

    def test_timestamp_default(self):
        """Test timestamp defaults to current time"""
        before = datetime.now()
        reading = EnergyReading(
            device_id="test_001",
            energy_kwh=5.0
        )
        after = datetime.now()

        assert before <= reading.timestamp <= after

    def test_is_anomaly_default(self):
        """Test is_anomaly defaults to False"""
        reading = EnergyReading(
            device_id="test_001",
            energy_kwh=5.0
        )
        assert reading.is_anomaly is False

    def test_anomaly_score_optional(self):
        """Test anomaly_score is optional"""
        reading = EnergyReading(
            device_id="test_001",
            energy_kwh=5.0,
            anomaly_score=2.5
        )
        assert reading.anomaly_score == 2.5

    def test_reading_with_anomaly(self, anomalous_energy_reading):
        """Test creating anomalous reading"""
        assert anomalous_energy_reading.is_anomaly is True
        assert anomalous_energy_reading.anomaly_score == 2.5
        assert anomalous_energy_reading.energy_kwh == 15.0


class TestAnomalyAlert:
    """Test AnomalyAlert model"""

    def test_create_anomaly_alert(self):
        """Test creating an anomaly alert"""
        alert = AnomalyAlert(
            device_id="test_001",
            timestamp=datetime.now(),
            energy_kwh=15.0,
            anomaly_score=3.0,
            message="High energy consumption detected"
        )
        assert alert.device_id == "test_001"
        assert alert.energy_kwh == 15.0
        assert alert.anomaly_score == 3.0
        assert alert.severity == "medium"  # default

    def test_custom_severity(self):
        """Test custom severity level"""
        alert = AnomalyAlert(
            device_id="test_001",
            timestamp=datetime.now(),
            energy_kwh=20.0,
            anomaly_score=4.0,
            message="Critical energy spike",
            severity="high"
        )
        assert alert.severity == "high"


class TestSemanticSearchQuery:
    """Test SemanticSearchQuery model"""

    def test_create_search_query(self):
        """Test creating a search query"""
        query = SemanticSearchQuery(
            query="HVAC energy consumption anomaly",
            limit=5
        )
        assert query.query == "HVAC energy consumption anomaly"
        assert query.limit == 5

    def test_default_limit(self):
        """Test default limit is 10"""
        query = SemanticSearchQuery(query="test query")
        assert query.limit == 10

    def test_empty_query_invalid(self):
        """Test empty query is invalid"""
        with pytest.raises(ValidationError):
            SemanticSearchQuery(query="")


class TestDeviceStats:
    """Test DeviceStats model"""

    def test_create_device_stats(self):
        """Test creating device statistics"""
        stats = DeviceStats(
            device_id="test_001",
            total_readings=100,
            avg_energy=5.5,
            anomaly_count=3,
            last_reading=datetime.now(),
            status=DeviceStatus.NORMAL
        )
        assert stats.device_id == "test_001"
        assert stats.total_readings == 100
        assert stats.avg_energy == 5.5
        assert stats.anomaly_count == 3
        assert isinstance(stats.last_reading, datetime)
        assert stats.status == DeviceStatus.NORMAL

    def test_no_last_reading(self):
        """Test stats with no last reading"""
        stats = DeviceStats(
            device_id="test_001",
            total_readings=0,
            avg_energy=0.0,
            anomaly_count=0,
            last_reading=None,
            status=DeviceStatus.NORMAL
        )
        assert stats.last_reading is None
        assert stats.total_readings == 0


class TestEnumValidation:
    """Test enum validation"""

    def test_device_type_values(self):
        """Test all device type values are valid"""
        valid_types = ["HVAC", "Lighting", "Server", "Cooling", "Heating"]
        for device_type in valid_types:
            device = Device(
                device_id="test_001",
                device_type=device_type,
                location="Building A",
                install_date=datetime.now()
            )
            assert device.device_type == device_type

    def test_device_status_values(self):
        """Test all device status values are valid"""
        valid_statuses = ["normal", "anomaly", "maintenance"]
        for status in valid_statuses:
            device = Device(
                device_id="test_001",
                device_type=DeviceType.HVAC,
                location="Building A",
                install_date=datetime.now(),
                status=status
            )
            assert device.status == status
