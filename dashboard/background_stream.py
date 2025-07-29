#!/usr/bin/env python3
"""
Background data streaming script for the dashboard
"""
import time
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings
from app.services.redis_service import RedisService
from data.generator import DataGenerator
from app.models.schemas import EnergyReading
import numpy as np

def start_background_streaming():
    """Start background data streaming for dashboard demo"""
    print("🔄 Starting background data streaming for dashboard...")
    print("📡 Streaming interval: 15 seconds")
    print("🏠 Devices: 5")
    print("Press Ctrl+C to stop\n")

    settings = Settings()
    redis_service = RedisService(settings)
    data_generator = DataGenerator(settings)

    # Get existing devices or create them
    devices = []
    for i in range(5):
        device_id = f"device-{i+1:03d}"
        device = redis_service.get_device(device_id)
        if device:
            devices.append(device)
        else:
            # Create new device
            new_devices = data_generator.generate_devices(1)
            device = new_devices[0]
            device.device_id = device_id
            redis_service.store_device(device)
            devices.append(device)
            print(f"✅ Created device: {device_id}")

    # Initialize base consumption
    device_base_consumption = {}
    for device in devices:
        device_base_consumption[device.device_id] = np.random.uniform(100, 1000)

    reading_count = 0
    start_time = datetime.utcnow()

    try:
        while True:
            batch_start = time.time()

            # Generate readings for all devices
            for device in devices:
                try:
                    base = device_base_consumption[device.device_id]
                    drift = np.random.normal(0, base * 0.1)  # 10% drift
                    current_consumption = max(50, base + drift)

                    # Update base consumption slightly
                    device_base_consumption[device.device_id] = max(50, base + np.random.normal(0, base * 0.01))

                    # Create energy reading
                    reading = EnergyReading(
                        device_id=device.device_id,
                        timestamp=datetime.utcnow(),
                        energy_kwh=current_consumption / 1000,
                        power_kw=current_consumption / 1000,
                        voltage=np.random.uniform(220, 240),
                        current=current_consumption / 230,
                        power_factor=np.random.uniform(0.85, 0.95)
                    )

                    redis_service.store_energy_reading(reading)
                    reading_count += 1

                except Exception as e:
                    print(f"⚠️ Error generating reading for {device.device_id}: {e}")

            # Display progress
            elapsed = datetime.utcnow() - start_time
            print(f"📊 Batch {reading_count // len(devices):>4} | "
                  f"Total readings: {reading_count:>6} | "
                  f"Runtime: {str(elapsed).split('.')[0]}")

            # Wait for next interval
            sleep_time = max(0, 15 - (time.time() - batch_start))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        elapsed = datetime.utcnow() - start_time
        print(f"\n🛑 Streaming stopped")
        print(f"📈 Total readings: {reading_count}")
        print(f"⏱️ Runtime: {str(elapsed).split('.')[0]}")

if __name__ == "__main__":
    start_background_streaming()
