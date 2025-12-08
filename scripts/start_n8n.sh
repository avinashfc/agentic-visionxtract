#!/bin/bash

# Script to start n8n for ADK Face Extraction

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of scripts/)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

echo "🚀 Starting n8n self-hosted..."
echo "📁 Working directory: $PROJECT_ROOT"

# Check if Docker is running, try to start if not
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running."
    echo ""
    
    # Try to start Docker Desktop on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🔄 Attempting to start Docker Desktop..."
        if [ -f "$SCRIPT_DIR/start_docker.sh" ]; then
            "$SCRIPT_DIR/start_docker.sh"
            if docker info > /dev/null 2>&1; then
                echo "✅ Docker started successfully! Continuing with n8n setup..."
            else
                echo "⚠️  Docker did not start. Please start it manually and run this script again."
                exit 1
            fi
        elif [ -d "/Applications/Docker.app" ]; then
            open -a Docker
            echo "⏳ Waiting for Docker to start (this may take 30-60 seconds)..."
            echo "   Please wait for the Docker icon to appear in your menu bar."
            echo "   Then run this script again."
            exit 1
        else
            echo "⚠️  Docker Desktop not found. Please install Docker Desktop from:"
            echo "   https://www.docker.com/products/docker-desktop"
            exit 1
        fi
    else
        echo "📋 To start Docker:"
        echo "   On Linux: sudo systemctl start docker"
        echo "   Then run this script again"
        exit 1
    fi
fi

# Create n8n data directory if it doesn't exist
mkdir -p n8n_data

# Start n8n
echo "📦 Starting n8n container..."
docker-compose -f docker-compose.n8n.yml up -d

# Wait for n8n to be ready
echo "⏳ Waiting for n8n to start..."
sleep 5

# Check if n8n is running
if docker ps | grep -q n8n-face-extraction; then
    echo "✅ n8n is running!"
    echo ""
    echo "🌐 Access n8n at: http://localhost:5678"
    echo "👤 Username: admin"
    echo "🔑 Password: changeme"
    echo ""
    echo "📝 To view logs: docker-compose -f docker-compose.n8n.yml logs -f"
    echo "🛑 To stop: docker-compose -f docker-compose.n8n.yml down"
else
    echo "❌ Failed to start n8n. Check logs:"
    docker-compose -f docker-compose.n8n.yml logs
    exit 1
fi

