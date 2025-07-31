#!/bin/bash

# Redisense Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Build the application
build() {
    print_header "Building Redisense Application"
    check_docker

    print_status "Building Docker image..."
    docker compose build app

    print_status "✅ Build completed successfully!"
}

# Start all services
start() {
    print_header "Starting Redisense Services"
    check_docker

    print_status "Starting Redis, RedisInsight, and Redisense App..."
    docker compose up -d

    print_status "Waiting for services to be ready..."
    sleep 10

    print_status "✅ Services started successfully!"
    print_status "🌐 Web App: http://localhost:8080"
    print_status "🔍 RedisInsight: http://localhost:5540"
    print_status "📊 Redis: localhost:6379"
}

# Stop all services
stop() {
    print_header "Stopping Redisense Services"
    check_docker

    print_status "Stopping all services..."
    docker compose down

    print_status "✅ Services stopped successfully!"
}

# Restart all services
restart() {
    print_header "Restarting Redisense Services"
    stop
    start
}

# View logs
logs() {
    check_docker

    if [ -z "$1" ]; then
        print_status "Showing logs for all services..."
        docker compose logs -f
    else
        print_status "Showing logs for service: $1"
        docker compose logs -f "$1"
    fi
}

# Check service status
status() {
    print_header "Redisense Services Status"
    check_docker

    docker compose ps

    echo ""
    print_status "Service URLs:"
    echo "  🌐 Web App: http://localhost:8080"
    echo "  🔍 RedisInsight: http://localhost:5540"
    echo "  📊 Redis: localhost:6379"
}

# Clean up (remove containers and volumes)
clean() {
    print_header "Cleaning Up Redisense"
    check_docker

    print_warning "This will remove all containers and data volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Stopping and removing containers..."
        docker compose down -v --remove-orphans

        print_status "Removing images..."
        docker compose down --rmi all

        print_status "✅ Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Seed data
seed() {
    print_header "Seeding Test Data"
    check_docker

    print_status "Seeding device and energy data..."
    docker compose exec app uv run python cli.py seed-data --device-count 5 --days 7

    print_status "✅ Data seeded successfully!"
}

# Show help
help() {
    echo "Redisense Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build      Build the application Docker image"
    echo "  start      Start all services (Redis, RedisInsight, App)"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  logs       View logs (optionally specify service: app, redis, redisinsight)"
    echo "  status     Show service status and URLs"
    echo "  seed       Seed test data"
    echo "  clean      Remove all containers and volumes"
    echo "  help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start all services"
    echo "  $0 logs app           # View app logs only"
    echo "  $0 status             # Check service status"
}

# Main script logic
case "$1" in
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs "$2"
        ;;
    status)
        status
        ;;
    seed)
        seed
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        help
        exit 1
        ;;
esac
