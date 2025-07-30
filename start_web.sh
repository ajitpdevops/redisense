#!/bin/bash
# Simple startup script for the FastAPI web app

echo "🚀 Starting Redisense FastAPI Web Application..."
echo "📍 Navigate to: http://localhost:8080"
echo "🔄 Auto-reload enabled for development"
echo ""

cd /workspaces/intellistream/redisense

# Start the FastAPI application
uv run python -c "
import uvicorn
import sys
import os

# Add current directory to path
sys.path.append('.')

print('Starting FastAPI server...')
try:
    uvicorn.run(
        'web_app:app',
        host='0.0.0.0',
        port=8080,
        reload=True,
        log_level='info'
    )
except Exception as e:
    print(f'Error starting server: {e}')
    import traceback
    traceback.print_exc()
"
