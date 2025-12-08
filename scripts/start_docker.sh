#!/bin/bash

# Script to start Docker Desktop on macOS

if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is for macOS only."
    echo "   On Linux, start Docker with: sudo systemctl start docker"
    exit 1
fi

echo "🐳 Checking Docker status..."

# Check if Docker is already running
if docker info > /dev/null 2>&1; then
    echo "✅ Docker is already running!"
    exit 0
fi

# Check if Docker Desktop is installed
if [ ! -d "/Applications/Docker.app" ]; then
    echo "❌ Docker Desktop is not installed."
    echo ""
    echo "📥 Please install Docker Desktop from:"
    echo "   https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "🚀 Starting Docker Desktop..."
open -a Docker

echo "⏳ Waiting for Docker to start..."
echo "   This may take 30-60 seconds..."

# Wait for Docker to be ready (max 2 minutes)
timeout=120
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if docker info > /dev/null 2>&1; then
        echo "✅ Docker is now running!"
        exit 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    if [ $((elapsed % 10)) -eq 0 ]; then
        echo "   Still waiting... ($elapsed seconds)"
    fi
done

echo "⚠️  Docker did not start within 2 minutes."
echo "   Please check Docker Desktop manually and ensure it's running."
exit 1

