#!/usr/bin/env python3
"""
Simple FastAPI server startup - bypassing heavy AI initialization
"""

import sys
import os
sys.path.append('.')

print("🚀 Starting Redisense FastAPI Server...")
print("📍 URL: http://localhost:8080")
print("⚡ Fast loading without heavy AI initialization")

try:
    from web_app import app
    import uvicorn

    print("✅ FastAPI app loaded successfully")
    print("🌐 Starting server on port 8080...")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=False,  # Disable reload for faster startup
        log_level="warning"  # Reduce log noise
    )

except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
