"""
Data generator for testing and demo purposes
"""
import random
import time
import math
from datetime import datetime, timedelta
from typing import List
from faker import Faker

from app.models.schemas import Device, EnergyReading, DeviceType, DeviceStatus
from config.settings import Settings

fake = Faker()

class DataGenerator:
    """Generate realistic test data for devices and energy readings"""

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()

    def generate_devices(self, count: int = None) -> List[Device]:
        """Generate a list of devices with realistic energy equipment names"""
        count = count or self.settings.DEVICE_COUNT

        device_types = [DeviceType.HVAC, DeviceType.SERVER, DeviceType.LIGHTING, DeviceType.COOLING, DeviceType.HEATING]
        locations = ["Building A - Floor 1", "Building A - Floor 2", "Building B - Main Floor",
                    "Building B - Basement", "Data Center - Rack Row 1", "Data Center - Rack Row 2",
                    "Warehouse - Loading Bay", "Warehouse - Storage Area", "Factory Floor - Assembly Line",
                    "Factory Floor - Quality Control", "Office - East Wing", "Office - West Wing"]

        statuses = [DeviceStatus.NORMAL, DeviceStatus.MAINTENANCE, DeviceStatus.NORMAL, DeviceStatus.NORMAL]  # Weight towards normal

        # Realistic manufacturers by device type
        manufacturers = {
            DeviceType.HVAC: ["Carrier", "Trane", "Lennox", "Daikin", "Johnson Controls", "Honeywell"],
            DeviceType.SERVER: ["Dell", "HP Enterprise", "IBM", "Cisco", "Supermicro", "Lenovo"],
            DeviceType.LIGHTING: ["Philips", "GE Lighting", "Osram", "Cree", "Acuity Brands", "Eaton"],
            DeviceType.COOLING: ["Emerson", "Schneider Electric", "Vertiv", "Stulz", "Rittal", "CoolIT"],
            DeviceType.HEATING: ["Rheem", "Bradford White", "AO Smith", "Rinnai", "Noritz", "Bosch"]
        }

        # Device naming patterns by type
        device_names = {
            DeviceType.HVAC: ["Rooftop Unit", "Air Handler", "Heat Pump", "Chiller", "Boiler", "VAV Box"],
            DeviceType.SERVER: ["Web Server", "Database Server", "Application Server", "File Server", "Mail Server", "Backup Server"],
            DeviceType.LIGHTING: ["LED Panel", "Smart Bulb Array", "Emergency Lighting", "Parking Lot Lights", "Office Lighting", "Hallway Lights"],
            DeviceType.COOLING: ["Precision AC", "CRAC Unit", "Cooling Tower", "Chilled Water Pump", "Condenser Unit", "Evaporator"],
            DeviceType.HEATING: ["Baseboard Heater", "Radiant Floor", "Heat Exchanger", "Furnace", "Electric Heater", "Infrared Heater"]
        }

        devices = []
        for i in range(count):
            device_type = random.choice(device_types)
            manufacturer = random.choice(manufacturers[device_type])
            device_name_type = random.choice(device_names[device_type])
            location = random.choice(locations)
            power_rating = random.randint(500, 15000)  # 0.5kW to 15kW

            # Create realistic device name
            zone_number = random.randint(1, 20)
            device_name = f"{device_name_type} - Zone {zone_number:02d}"

            # Create realistic model numbers
            model_prefix = manufacturer[:3].upper()
            model_number = f"{model_prefix}-{random.randint(1000, 9999)}{random.choice(['A', 'B', 'C', 'X', 'P'])}"

            # Create device description
            descriptions = {
                DeviceType.HVAC: f"Climate control system serving {location.split(' - ')[1] if ' - ' in location else location}. Maintains temperature and air quality.",
                DeviceType.SERVER: f"Enterprise server handling {random.choice(['web traffic', 'database operations', 'file storage', 'email services', 'backup operations'])} for the organization.",
                DeviceType.LIGHTING: f"Energy-efficient LED lighting system for {location.split(' - ')[1] if ' - ' in location else location}. Includes smart controls and dimming.",
                DeviceType.COOLING: f"Precision cooling equipment maintaining optimal temperature for {location.split(' - ')[1] if ' - ' in location else location}.",
                DeviceType.HEATING: f"Heating system providing thermal comfort for {location.split(' - ')[1] if ' - ' in location else location}."
            }

            device = Device(
                device_id=f"device-{i+1:03d}",
                name=device_name,
                device_type=device_type,
                location=location,
                description=descriptions[device_type],
                manufacturer=manufacturer,
                model=model_number,
                power_rating=power_rating,
                status=random.choice(statuses),
                metadata={
                    "installation_date": fake.date_between(start_date="-2y", end_date="today").isoformat(),
                    "warranty_expires": fake.date_between(start_date="today", end_date="+2y").isoformat(),
                    "maintenance_schedule": random.choice(["Monthly", "Quarterly", "Semi-Annual", "Annual"]),
                    "energy_efficiency_rating": random.choice(["A+", "A", "B+", "B", "C+"]),
                    "operating_hours": random.randint(2000, 8000),
                    "last_maintenance": fake.date_between(start_date="-6m", end_date="today").isoformat()
                }
            )
            devices.append(device)

        return devices

    def generate_energy_reading(
        self,
        device_id: str,
        base_consumption: float = None,
        timestamp: datetime = None,
        introduce_anomaly: bool = False
    ) -> EnergyReading:
        """Generate a single energy reading"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        if base_consumption is None:
            base_consumption = random.uniform(
                self.settings.ENERGY_MIN,
                self.settings.ENERGY_MAX
            )

        # Add some natural variation
        variation = random.uniform(-0.3, 0.3)
        energy_kwh = max(0.1, base_consumption + variation)

        # Introduce anomaly if requested
        if introduce_anomaly:
            energy_kwh *= self.settings.ANOMALY_SPIKE_MULTIPLIER

        return EnergyReading(
            device_id=device_id,
            timestamp=timestamp,
            energy_kwh=round(energy_kwh, 2)
        )

    def generate_time_series(
        self,
        device_id: str,
        hours: int = 24,
        interval_minutes: int = 60,
        base_consumption: float = None,
        anomaly_probability: float = 0.05
    ) -> List[EnergyReading]:
        """Generate a time series of energy readings"""
        readings = []

        if base_consumption is None:
            base_consumption = random.uniform(
                self.settings.ENERGY_MIN,
                self.settings.ENERGY_MAX
            )

        start_time = datetime.utcnow() - timedelta(hours=hours)
        interval = timedelta(minutes=interval_minutes)

        current_time = start_time
        end_time = datetime.utcnow()

        while current_time <= end_time:
            # Random chance for anomaly
            introduce_anomaly = random.random() < anomaly_probability

            reading = self.generate_energy_reading(
                device_id=device_id,
                base_consumption=base_consumption,
                timestamp=current_time,
                introduce_anomaly=introduce_anomaly
            )

            readings.append(reading)
            current_time += interval

        return readings

    def generate_pattern_data(
        self,
        device_id: str,
        pattern_type: str = "stable",
        hours: int = 24,
        interval_minutes: int = 60
    ) -> List[EnergyReading]:
        """Generate data with specific patterns"""
        readings = []
        start_time = datetime.utcnow() - timedelta(hours=hours)
        interval = timedelta(minutes=interval_minutes)

        current_time = start_time
        end_time = datetime.utcnow()

        base_value = random.uniform(self.settings.ENERGY_MIN, self.settings.ENERGY_MAX)
        step_count = 0

        while current_time <= end_time:
            if pattern_type == "stable":
                energy_kwh = base_value + random.uniform(-0.2, 0.2)
            elif pattern_type == "increasing":
                energy_kwh = base_value + (step_count * 0.1) + random.uniform(-0.1, 0.1)
            elif pattern_type == "decreasing":
                energy_kwh = base_value - (step_count * 0.1) + random.uniform(-0.1, 0.1)
            elif pattern_type == "high_variation":
                energy_kwh = base_value + random.uniform(-2.0, 2.0)
            else:
                energy_kwh = base_value + random.uniform(-0.5, 0.5)

            energy_kwh = max(0.1, energy_kwh)  # Ensure positive values

            reading = EnergyReading(
                device_id=device_id,
                timestamp=current_time,
                energy_kwh=round(energy_kwh, 2)
            )

            readings.append(reading)
            current_time += interval
            step_count += 1

        return readings

    def generate_historical_readings(
        self,
        device_id: str,
        start_time: datetime,
        end_time: datetime,
        readings_per_hour: int = 4
    ) -> List[EnergyReading]:
        """Generate historical energy readings for a specific time period"""
        readings = []

        # Calculate interval based on readings per hour
        interval_minutes = 60 / readings_per_hour
        interval = timedelta(minutes=interval_minutes)

        # Generate base consumption for this device
        base_consumption = random.uniform(self.settings.ENERGY_MIN, self.settings.ENERGY_MAX)

        current_time = start_time
        while current_time <= end_time:
            # Add some natural variation and drift over time
            time_factor = (current_time - start_time).total_seconds() / (24 * 3600)  # days since start
            seasonal_variation = 0.1 * math.sin(time_factor * 2 * math.pi / 7)  # Weekly pattern
            daily_variation = 0.05 * math.sin((current_time.hour / 24) * 2 * math.pi)  # Daily pattern
            random_variation = random.uniform(-0.2, 0.2)

            energy_kwh = base_consumption + seasonal_variation + daily_variation + random_variation
            energy_kwh = max(0.1, energy_kwh)  # Ensure positive values

            # Small chance for anomaly
            if random.random() < 0.02:  # 2% chance
                energy_kwh *= self.settings.ANOMALY_SPIKE_MULTIPLIER

            reading = EnergyReading(
                device_id=device_id,
                timestamp=current_time,
                energy_kwh=round(energy_kwh, 2)
            )

            readings.append(reading)
            current_time += interval

        return readings

    def generate_demo_dataset(self) -> tuple[List[Device], List[EnergyReading]]:
        """Generate a complete demo dataset"""
        # Generate devices
        devices = self.generate_devices()

        # Generate readings for each device
        all_readings = []

        for device in devices:
            # Generate 7 days of hourly data
            readings = self.generate_time_series(
                device_id=device.device_id,
                hours=24 * 7,  # 7 days
                interval_minutes=60,  # hourly
                anomaly_probability=0.03  # 3% chance of anomaly
            )
            all_readings.extend(readings)

        return devices, all_readings

# Global data generator instance
data_generator = DataGenerator()
