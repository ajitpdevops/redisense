#!/usr/bin/env python3
"""
CLI for Redisense - Smart Energy Monitoring System
Provides commands for data seeding, testing, and system management.
"""

import asyncio
import click
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List

from config.settings import Settings
from app.services.redis_service import RedisService
from app.services.ai_service import AIService
from data.generator import DataGenerator
from app.models.schemas import Device, EnergyReading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Redisense CLI - Smart Energy Monitoring System"""
    pass

@cli.command()
@click.option('--device-count', default=10, help='Number of devices to generate')
@click.option('--days', default=7, help='Number of days of historical data to generate')
@click.option('--readings-per-hour', default=4, help='Number of readings per hour per device')
def seed_data(device_count: int, days: int, readings_per_hour: int):
    """Generate devices and load historical timeseries data into Redis."""
    settings = Settings()
    redis_service = RedisService(settings)
    data_generator = DataGenerator(settings)

    click.echo(f"🌱 Seeding {device_count} devices with {days} days of historical data...")

    try:
        # Generate devices
        devices = data_generator.generate_devices(device_count)
        click.echo(f"✅ Generated {len(devices)} devices")

        # Store devices in Redis
        for device in devices:
            redis_service.store_device(device)
        click.echo(f"✅ Stored {len(devices)} devices in Redis")

        # Generate historical energy readings
        total_readings = 0
        start_time = datetime.utcnow() - timedelta(days=days)

        for device in devices:
            readings = data_generator.generate_historical_readings(
                device_id=device.device_id,
                start_time=start_time,
                end_time=datetime.utcnow(),
                readings_per_hour=readings_per_hour
            )

            # Store readings in Redis
            for reading in readings:
                redis_service.store_energy_reading(reading)

            total_readings += len(readings)
            click.echo(f"   📊 Generated {len(readings)} readings for {device.device_id}")

        click.echo(f"✅ Successfully seeded {total_readings} total energy readings")
        click.echo(f"🎉 Historical data seeding complete!")

    except Exception as e:
        click.echo(f"❌ Error during data seeding: {e}")
        raise

@cli.command()
@click.option('--device-count', default=5, help='Number of devices to stream data for')
@click.option('--interval', default=15, help='Interval between readings in seconds')
@click.option('--drift-factor', default=0.1, help='Random variation factor (0.1 = 10% drift). Controls how much energy consumption fluctuates naturally.')
def stream_data(device_count: int, interval: int, drift_factor: float):
    """Continuously stream real-time energy readings from devices to Redis."""
    settings = Settings()
    redis_service = RedisService(settings)
    data_generator = DataGenerator(settings)

    click.echo(f"🔄 Starting real-time data streaming for {device_count} devices...")
    click.echo(f"📡 Streaming interval: {interval} seconds")
    click.echo(f"🎛️ Drift factor: {drift_factor}")
    click.echo("Press Ctrl+C to stop streaming\n")

    try:
        # Get or generate devices
        devices = []
        for i in range(device_count):
            device_id = f"device-{i+1:03d}"
            # Try to get existing device or create new one
            try:
                device = redis_service.get_device(device_id)
                if device:
                    devices.append(device)
                else:
                    # Generate new device
                    new_devices = data_generator.generate_devices(1)
                    device = new_devices[0]
                    device.device_id = device_id
                    redis_service.store_device(device)
                    devices.append(device)
                    click.echo(f"✅ Created new device: {device_id}")
            except Exception as e:
                click.echo(f"⚠️ Error with device {device_id}: {e}")
                continue

        if not devices:
            click.echo("❌ No devices available for streaming")
            return

        click.echo(f"✅ Ready to stream data for {len(devices)} devices\n")

        # Initialize base consumption for each device
        device_base_consumption = {}
        for device in devices:
            device_base_consumption[device.device_id] = np.random.uniform(100, 1000)  # Base consumption in watts

        reading_count = 0
        start_time = datetime.utcnow()
        last_minute_display = start_time

        while True:
            batch_start = time.time()

            # Generate and store readings for all devices
            for device in devices:
                try:
                    # Apply random drift to base consumption
                    base = device_base_consumption[device.device_id]
                    drift = np.random.normal(0, base * drift_factor)
                    current_consumption = max(0, base + drift)

                    # Update base consumption slightly for next iteration
                    device_base_consumption[device.device_id] = max(50, base + np.random.normal(0, base * 0.01))

                    # Create energy reading
                    reading = EnergyReading(
                        device_id=device.device_id,
                        timestamp=datetime.utcnow(),
                        energy_kwh=current_consumption / 1000,  # Convert to kWh
                        power_kw=current_consumption / 1000,     # Current power draw
                        voltage=np.random.uniform(220, 240),     # Voltage variation
                        current=current_consumption / 230,       # Current based on power/voltage
                        power_factor=np.random.uniform(0.85, 0.95)
                    )

                    # Store in Redis
                    redis_service.store_energy_reading(reading)
                    reading_count += 1

                except Exception as e:
                    click.echo(f"⚠️ Error generating reading for {device.device_id}: {e}")

            # Check if a minute has passed for progress indicator
            current_time = datetime.utcnow()
            if (current_time - last_minute_display).total_seconds() >= 60:
                elapsed_minutes = (current_time - start_time).total_seconds() / 60
                rate_per_minute = reading_count / elapsed_minutes if elapsed_minutes > 0 else 0
                click.echo(f"⏱️  {elapsed_minutes:.0f} min | "
                          f"📊 {reading_count:>6} readings | "
                          f"📈 {rate_per_minute:.1f} readings/min")
                last_minute_display = current_time

            # Display batch progress (less frequent than minute indicator)
            if reading_count % (len(devices) * 4) == 0:  # Every 4 batches
                elapsed_time = current_time - start_time
                click.echo(f"📊 Batch {reading_count // len(devices):>4} | "
                          f"Total readings: {reading_count:>6} | "
                          f"Runtime: {str(elapsed_time).split('.')[0]}")

            # Wait for next interval
            batch_duration = time.time() - batch_start
            sleep_time = max(0, interval - batch_duration)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        elapsed_time = datetime.utcnow() - start_time
        click.echo(f"\n🛑 Streaming stopped by user")
        click.echo(f"📈 Total readings generated: {reading_count}")
        click.echo(f"⏱️ Total runtime: {str(elapsed_time).split('.')[0]}")
        click.echo(f"📊 Average rate: {reading_count / elapsed_time.total_seconds():.2f} readings/second")
    except Exception as e:
        click.echo(f"❌ Error during streaming: {e}")
        raise

@cli.command()
def test_connection():
    """Test Redis connection and basic functionality."""
    click.echo("🔍 Testing Redis connection...")

    try:
        settings = Settings()
        redis_service = RedisService(settings)

        # Test basic Redis operations
        test_key = "redisense:test:connection"
        test_value = f"test-{datetime.utcnow().isoformat()}"

        redis_service.redis_client.set(test_key, test_value)
        retrieved_value = redis_service.redis_client.get(test_key)
        redis_service.redis_client.delete(test_key)

        if retrieved_value == test_value:
            click.echo("✅ Redis connection successful!")
            click.echo(f"🔗 Connected to: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        else:
            click.echo("❌ Redis connection test failed")

    except Exception as e:
        click.echo(f"❌ Redis connection failed: {e}")
        raise

@cli.command()
def status():
    """Check system status and data counts."""
    click.echo("📊 System Status\n" + "="*50)

    try:
        settings = Settings()
        redis_service = RedisService(settings)

        # Count devices
        device_keys = redis_service.redis_client.keys("device:*")
        device_count = len(device_keys)

        # Count energy readings
        reading_keys = redis_service.redis_client.keys("reading:*")
        reading_count = len(reading_keys)

        # Count timeseries data
        ts_keys = redis_service.redis_client.keys("energy:*")
        ts_count = len(ts_keys)

        click.echo(f"🏠 Devices: {device_count}")
        click.echo(f"⚡ Energy readings: {reading_count}")
        click.echo(f"📈 Timeseries keys: {ts_count}")
        click.echo(f"🔗 Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")

        # Memory usage
        memory_info = redis_service.redis_client.info('memory')
        used_memory = memory_info.get('used_memory_human', 'Unknown')
        click.echo(f"💾 Memory usage: {used_memory}")

        click.echo("\n✅ System operational")

    except Exception as e:
        click.echo(f"❌ Error checking status: {e}")
        raise

@cli.command()
@click.argument('device_id')
@click.option('--hours', default=24, help='Number of hours to analyze')
def analyze(device_id: str, hours: int):
    """Analyze energy patterns for a specific device."""
    click.echo(f"🔍 Analyzing device: {device_id} (last {hours} hours)")

    try:
        settings = Settings()
        redis_service = RedisService(settings)
        ai_service = AIService()

        # Get device info
        device_data = redis_service.get_device(device_id)
        if not device_data:
            click.echo(f"❌ Device {device_id} not found")
            return

        # Get recent readings
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        readings = redis_service.get_energy_readings(device_id, start_time, end_time)

        if not readings:
            click.echo(f"❌ No readings found for {device_id}")
            return

        click.echo(f"📊 Found {len(readings)} readings")

        # Perform analysis
        analysis = ai_service.analyze_consumption_pattern(device_id, hours * 60)  # Convert to minutes

        click.echo(f"\n📈 Analysis Results:")
        click.echo(f"   Pattern: {analysis.get('pattern', 'Unknown')}")
        click.echo(f"   Trend: {analysis.get('trend', 'Unknown')}")
        click.echo(f"   Average consumption: {analysis.get('avg_consumption', 0):.2f} kWh")

        # Check for anomalies
        anomalies = ai_service.detect_anomalies(device_id, hours * 60)
        if anomalies:
            click.echo(f"⚠️ Found {len(anomalies)} anomalies")
            for anomaly in anomalies[:3]:  # Show first 3
                click.echo(f"   - {anomaly.get('timestamp', 'Unknown')}: {anomaly.get('description', 'Unknown')}")
        else:
            click.echo("✅ No anomalies detected")

    except Exception as e:
        click.echo(f"❌ Error during analysis: {e}")
        raise

@cli.command()
@click.argument('query')
@click.option('--limit', default=5, help='Number of results to return')
def search(query: str, limit: int):
    """Perform semantic search on device data."""
    click.echo(f"🔍 Searching for: '{query}'")

    try:
        settings = Settings()
        redis_service = RedisService(settings)
        ai_service = AIService()

        results = ai_service.semantic_search(query, limit)

        if results:
            click.echo(f"📋 Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. Device: {result.device_id}")
                click.echo(f"   Score: {result.score:.3f}")
                click.echo(f"   Content: {result.content[:100]}...")
                click.echo()
        else:
            click.echo("❌ No results found")

    except Exception as e:
        click.echo(f"❌ Error during search: {e}")
        raise

@cli.command()
def clear_data():
    """Clear all data from Redis (use with caution)."""
    click.echo("⚠️ This will delete ALL data from Redis!")
    if click.confirm("Are you sure you want to continue?"):
        try:
            settings = Settings()
            redis_service = RedisService(settings)

            # Clear all keys with our prefixes
            prefixes = ["device:", "reading:", "energy:", "vector:", "anomaly:"]
            total_deleted = 0

            for prefix in prefixes:
                keys = redis_service.redis_client.keys(f"{prefix}*")
                if keys:
                    deleted = redis_service.redis_client.delete(*keys)
                    total_deleted += deleted
                    click.echo(f"🗑️ Deleted {deleted} keys with prefix '{prefix}'")

            click.echo(f"✅ Cleared {total_deleted} total keys from Redis")

        except Exception as e:
            click.echo(f"❌ Error clearing data: {e}")
            raise
    else:
        click.echo("❌ Operation cancelled")

@cli.command()
def dashboard():
    """Launch the Streamlit dashboard."""
    click.echo("🚀 Starting Redisense Dashboard...")
    click.echo("📊 Opening browser to view the dashboard...")

    import subprocess
    import os

    try:
        # Change to the correct directory
        dashboard_dir = os.path.join(os.path.dirname(__file__), 'dashboard')
        app_path = os.path.join(dashboard_dir, 'streamlit_app.py')

        click.echo(f"📂 Dashboard location: {app_path}")
        click.echo("🌐 Dashboard will be available at: http://localhost:8501")
        click.echo("🔄 The dashboard auto-refreshes every 30 seconds")
        click.echo("\n💡 Tip: Start streaming data in another terminal with:")
        click.echo("   uv run python cli.py stream-data --device-count 5 --interval 10")

        # Launch Streamlit
        subprocess.run([
            "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ])

    except Exception as e:
        click.echo(f"❌ Error launching dashboard: {e}")
        click.echo("📋 Try running manually:")
        click.echo("   cd dashboard && streamlit run streamlit_app.py")

@cli.command()
@click.argument('device_id')
@click.option('--name', help='Update device name')
@click.option('--description', help='Update device description')
@click.option('--location', help='Update device location')
@click.option('--status', type=click.Choice(['normal', 'maintenance', 'anomaly']), help='Update device status')
def update_device(device_id: str, name: str, description: str, location: str, status: str):
    """Update device profile information."""
    settings = Settings()
    redis_service = RedisService(settings)

    try:
        # Get existing device
        device = redis_service.get_device(device_id)
        if not device:
            click.echo(f"❌ Device {device_id} not found")
            return

        click.echo(f"🔧 Updating device: {device.name}")

        # Update fields if provided
        updated = False
        if name:
            device.name = name
            updated = True
            click.echo(f"   ✅ Name updated to: {name}")

        if description:
            device.description = description
            updated = True
            click.echo(f"   ✅ Description updated")

        if location:
            device.location = location
            updated = True
            click.echo(f"   ✅ Location updated to: {location}")

        if status:
            from app.models.schemas import DeviceStatus
            device.status = DeviceStatus(status)
            updated = True
            click.echo(f"   ✅ Status updated to: {status}")

        if updated:
            # Store updated device
            success = redis_service.store_device(device)
            if success:
                click.echo(f"✅ Device {device_id} updated successfully")
            else:
                click.echo(f"❌ Failed to update device {device_id}")
        else:
            click.echo("⚠️ No updates specified. Use --help to see available options.")

    except Exception as e:
        click.echo(f"❌ Error updating device: {e}")
        raise

@cli.command()
def list_devices():
    """List all devices with their profiles."""
    settings = Settings()
    redis_service = RedisService(settings)

    try:
        devices = redis_service.get_all_devices()

        if not devices:
            click.echo("❌ No devices found")
            return

        click.echo(f"📋 Found {len(devices)} devices:\n")

        for device in devices:
            click.echo(f"🏠 {device.name}")
            click.echo(f"   ID: {device.device_id}")
            click.echo(f"   Type: {device.device_type.value}")
            click.echo(f"   Manufacturer: {device.manufacturer}")
            click.echo(f"   Model: {device.model}")
            click.echo(f"   Location: {device.location}")
            click.echo(f"   Power Rating: {device.power_rating}W")
            click.echo(f"   Status: {device.status.value.title()}")
            if device.description:
                click.echo(f"   Description: {device.description}")
            click.echo()

    except Exception as e:
        click.echo(f"❌ Error listing devices: {e}")
        raise


if __name__ == '__main__':
    cli()
