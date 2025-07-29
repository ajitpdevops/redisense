from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class DeviceType(str, Enum):
    HVAC = "HVAC"
    LIGHTING = "Lighting"
    SERVER = "Server"
    COOLING = "Cooling"
    HEATING = "Heating"

class DeviceStatus(str, Enum):
    NORMAL = "normal"
    ANOMALY = "anomaly"
    MAINTENANCE = "maintenance"

class Device(BaseModel):
    device_id: str = Field(..., description="Unique device identifier")
    name: str = Field(..., description="Human-readable device name")
    device_type: DeviceType = Field(..., description="Type of device")
    location: str = Field(..., description="Device location (Building A/B/C)")
    description: Optional[str] = Field(default="", description="Detailed device description")
    manufacturer: Optional[str] = Field(default="", description="Device manufacturer")
    model: Optional[str] = Field(default="", description="Device model")
    power_rating: Optional[int] = Field(default=None, description="Power rating in watts")
    install_date: datetime = Field(default_factory=datetime.utcnow, description="Installation date")
    status: DeviceStatus = Field(default=DeviceStatus.NORMAL, description="Current device status")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional device metadata")

    # Add alias for backwards compatibility
    @property
    def id(self) -> str:
        return self.device_id

class EnergyReading(BaseModel):
    device_id: str = Field(..., description="Device identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Reading timestamp")
    energy_kwh: float = Field(..., gt=0, description="Energy consumption in kWh")
    power_kw: Optional[float] = Field(default=None, description="Current power draw in kW")
    voltage: Optional[float] = Field(default=None, description="Voltage measurement")
    current: Optional[float] = Field(default=None, description="Current measurement in Amperes")
    power_factor: Optional[float] = Field(default=None, description="Power factor")
    is_anomaly: bool = Field(default=False, description="Whether this reading is anomalous")
    anomaly_score: Optional[float] = Field(default=None, description="Anomaly score if applicable")

class AnomalyAlert(BaseModel):
    device_id: str
    timestamp: datetime
    energy_kwh: float
    anomaly_score: float
    message: str
    severity: str = Field(default="medium")

class SemanticSearchQuery(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    limit: int = Field(default=10, description="Maximum number of results")

class SemanticSearchResult(BaseModel):
    device_id: str
    content: str
    score: float
    metadata: dict

class DeviceStats(BaseModel):
    device_id: str
    total_readings: int
    avg_energy: float
    anomaly_count: int
    last_reading: Optional[datetime]
    status: DeviceStatus
