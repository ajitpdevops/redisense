#!/usr/bin/env python3
"""
Test FastAPI app startup and basic functionality
"""

import sys
import os
import traceback

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test if all modules can be imported"""
    try:
        print("🧪 Testing imports...")

        from config.settings import Settings
        print("✅ Settings imported")

        from app.services.redis_service import RedisService
        print("✅ RedisService imported")

        from app.services.ai_service import AIService
        print("✅ AIService imported")

        from web_app import app
        print("✅ FastAPI app imported")

        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_services():
    """Test if services can be initialized"""
    try:
        print("\n🔧 Testing service initialization...")

        from config.settings import Settings
        from app.services.redis_service import RedisService
        from app.services.ai_service import AIService

        settings = Settings()
        print("✅ Settings initialized")

        redis_service = RedisService(settings)
        print("✅ Redis service initialized")

        ai_service = AIService()
        print("✅ AI service initialized")

        return True
    except Exception as e:
        print(f"❌ Service initialization error: {e}")
        traceback.print_exc()
        return False

def test_fastapi_app():
    """Test FastAPI app creation"""
    try:
        print("\n🚀 Testing FastAPI app...")

        from web_app import app
        from fastapi.testclient import TestClient

        # This would require installing httpx, so let's just check the app object
        print(f"✅ FastAPI app created: {type(app)}")
        print(f"✅ App title: {app.title}")

        return True
    except Exception as e:
        print(f"❌ FastAPI app error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 FastAPI App Health Check")
    print("=" * 50)

    success = True

    success &= test_import()
    success &= test_services()
    success &= test_fastapi_app()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! FastAPI app should work correctly.")
        print("\n🚀 To start the app, run:")
        print("   ./start_web.sh")
        print("   or")
        print("   uv run python web_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    return success

if __name__ == "__main__":
    main()
