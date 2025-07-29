#!/usr/bin/env python3
"""
Simple test script to verify the Redisense setup
"""
import sys
import traceback
from datetime import datetime

def test_imports():
    """Test that all modules can be imported"""
    try:
        from config.settings import Settings
        print("✓ Settings import successful")

        from app.models.schemas import Device, EnergyReading
        print("✓ Models import successful")

        from app.services.ai_service import ai_service
        print("✓ AI service import successful")

        # Test basic AI functionality
        embedding = ai_service.generate_embedding("test message")
        print(f"✓ AI embedding generation works (dim: {len(embedding)})")

        # Test anomaly detection
        values = [5.0, 5.2, 4.8, 5.1, 4.9]
        is_anomaly, score = ai_service.detect_anomaly_simple(15.0, values)
        print(f"✓ Anomaly detection works (anomaly: {is_anomaly}, score: {score:.2f})")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_models():
    """Test Pydantic models"""
    try:
        from app.models.schemas import Device, EnergyReading

        # Test device creation
        device = Device(
            device_id="test-device",
            device_type="HVAC",
            location="Building A"
        )
        print(f"✓ Device model works: {device.device_id}")

        # Test energy reading
        reading = EnergyReading(
            device_id="test-device",
            timestamp=datetime.utcnow(),
            energy_kwh=5.5
        )
        print(f"✓ Energy reading model works: {reading.energy_kwh} kWh")

        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    try:
        from config.settings import Settings
        settings = Settings()

        print(f"✓ Configuration loaded")
        print(f"  Redis Host: {settings.REDIS_HOST}")
        print(f"  Redis Port: {settings.REDIS_PORT}")
        print(f"  Debug Mode: {settings.DEBUG}")
        print(f"  Embedding Model: {settings.EMBEDDING_MODEL}")

        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Redisense Setup Verification ===\n")

    tests = [
        ("Imports", test_imports),
        ("Models", test_models),
        ("Configuration", test_configuration)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        success = test_func()
        results.append(success)
        print(f"{test_name}: {'PASS' if success else 'FAIL'}\n")

    # Summary
    passed = sum(results)
    total = len(results)

    print("=== Summary ===")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! Redisense is ready.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
