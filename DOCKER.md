# 🐳 Docker Setup for Redisense

This guide explains how to run Redisense using Docker with local Redis or Redis Cloud.

## 🚀 Quick Start

### 1. Prerequisites

- Docker Desktop or Docker Engine
- Docker Compose

### 2. Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` to configure Redis:

```bash
# For local Redis (default)
USE_REDIS_CLOUD=false
REDIS_LOCAL_HOST=redis  # Use 'redis' for Docker, 'localhost' for local

# For Redis Cloud
USE_REDIS_CLOUD=true
REDIS_CLOUD_HOST=your-redis-cloud-host.com
REDIS_CLOUD_PORT=6380
REDIS_CLOUD_USERNAME=your-username
REDIS_CLOUD_PASSWORD=your-password
```

### 3. Start Services

#### Option A: Using the Management Script (Recommended)

```bash
# Start all services
./docker.sh start

# Check status
./docker.sh status

# View logs
./docker.sh logs

# Seed test data
./docker.sh seed

# Stop services
./docker.sh stop
```

#### Option B: Using Docker Compose Directly

```bash
# Start with regular Redis
docker-compose up -d

# OR start with Redis Stack (includes RedisSearch, RedisJSON, etc.)
docker-compose -f docker-compose.redis-stack.yml up -d
```

## 📊 Service URLs

Once started, access the services:

- **🌐 Redisense Web App**: http://localhost:8080
- **🔍 RedisInsight**: http://localhost:5540 (Redis GUI)
- **📊 Redis**: localhost:6379

## 🛠 Docker Management Commands

The `docker.sh` script provides easy management:

```bash
./docker.sh build      # Build the application
./docker.sh start      # Start all services
./docker.sh stop       # Stop all services
./docker.sh restart    # Restart all services
./docker.sh logs       # View all logs
./docker.sh logs app   # View app logs only
./docker.sh status     # Check service status
./docker.sh seed       # Seed test data
./docker.sh clean      # Remove everything (destructive!)
./docker.sh help       # Show help
```

## 🔧 Configuration Options

### Redis Options

1. **Local Redis** (`USE_REDIS_CLOUD=false`):

   - Uses Redis container
   - Data persisted in Docker volume
   - No external dependencies

2. **Redis Stack** (Alternative):

   - Includes RedisSearch, RedisJSON, RedisTimeSeries
   - Use: `docker-compose -f docker-compose.redis-stack.yml up -d`

3. **Redis Cloud** (`USE_REDIS_CLOUD=true`):
   - Uses external Redis Cloud service
   - Configure cloud credentials in `.env`

### Environment Variables

Key environment variables in `.env`:

```bash
# Redis Toggle
USE_REDIS_CLOUD=false

# Redis Cloud (when USE_REDIS_CLOUD=true)
REDIS_CLOUD_HOST=your-host.com
REDIS_CLOUD_PORT=6380
REDIS_CLOUD_USERNAME=username
REDIS_CLOUD_PASSWORD=password

# Local Redis (when USE_REDIS_CLOUD=false)
REDIS_LOCAL_HOST=redis
REDIS_LOCAL_PORT=6379
REDIS_LOCAL_PASSWORD=

# Application
DEBUG=true
WEB_HOST=0.0.0.0
WEB_PORT=8080

# AI/ML
ANOMALY_THRESHOLD=2.0
EMBEDDING_MODEL=sentence-transformers/paraphrase-MiniLM-L6-v2
```

## 📁 Docker Files

- `Dockerfile` - Python application container
- `docker-compose.yml` - Standard Redis setup
- `docker-compose.redis-stack.yml` - Redis Stack setup
- `redis.conf` - Redis configuration
- `docker.sh` - Management script
- `.dockerignore` - Build optimization

## 🔍 Troubleshooting

### Common Issues

1. **Port conflicts**:

   ```bash
   # Check what's using the ports
   lsof -i :8080
   lsof -i :6379
   lsof -i :5540
   ```

2. **Permission errors**:

   ```bash
   # Fix Docker permissions
   sudo chown -R $USER:$USER .
   ```

3. **Build issues**:

   ```bash
   # Clean rebuild
   ./docker.sh clean
   ./docker.sh build
   ./docker.sh start
   ```

4. **Service not starting**:

   ```bash
   # Check logs
   ./docker.sh logs

   # Check specific service
   ./docker.sh logs app
   ./docker.sh logs redis
   ```

### Health Checks

Services include health checks:

```bash
# Check container health
docker ps

# Service should show "healthy" status
```

## 🧪 Development

### Local Development with Docker Redis

For development, you can run the Python app locally but use Docker Redis:

```bash
# Start only Redis and RedisInsight
docker-compose up redis redisinsight -d

# Set local Redis host
echo "REDIS_LOCAL_HOST=localhost" >> .env

# Run app locally
./start_web.sh
```

### Debugging

```bash
# Enter the app container
docker-compose exec app bash

# Run CLI commands in container
docker-compose exec app uv run python cli.py status

# View live logs
docker-compose logs -f app
```

## 📈 Production Deployment

For production:

1. Use Redis Cloud or dedicated Redis instance
2. Set `USE_REDIS_CLOUD=true`
3. Configure proper secrets management
4. Use production-grade Docker orchestration (Kubernetes, Docker Swarm)
5. Set up monitoring and logging

Example production `.env`:

```bash
USE_REDIS_CLOUD=true
REDIS_CLOUD_HOST=prod-redis.cloud.com
REDIS_CLOUD_PORT=6380
REDIS_CLOUD_USERNAME=prod-user
REDIS_CLOUD_PASSWORD=secure-password
DEBUG=false
LOG_LEVEL=INFO
```
