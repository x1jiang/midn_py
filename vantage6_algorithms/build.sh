#!/bin/bash
# Build script for all vantage6 algorithms
# Uses a unified Dockerfile with build arguments
# Must be run from the parent directory (midn_py/)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building Vantage6 Algorithms..."
echo "================================"
echo "Build context: $(pwd)"
echo "Using unified Dockerfile with build arguments"
echo ""

# Build SIMI using unified Dockerfile
echo "Building SIMI..."
docker build \
    --build-arg ALGORITHM=SIMI \
    -f Dockerfile \
    -t simi-algorithm:latest \
    .

if [ $? -eq 0 ]; then
    echo "✓ SIMI build successful"
else
    echo "✗ SIMI build failed"
    exit 1
fi

# Build SIMICE using unified Dockerfile
echo ""
echo "Building SIMICE..."
docker build \
    --build-arg ALGORITHM=SIMICE \
    -f Dockerfile \
    -t simice-algorithm:latest \
    .

if [ $? -eq 0 ]; then
    echo "✓ SIMICE build successful"
else
    echo "✗ SIMICE build failed"
    exit 1
fi

echo ""
echo "================================"
echo "Build complete!"
echo ""
echo "Images created:"
docker images | grep -E "(simi-algorithm|simice-algorithm)" || echo "  - simi-algorithm:latest"
echo "  - simice-algorithm:latest"
echo ""
echo "To push to registry:"
echo "  docker tag simi-algorithm:latest your-registry.com/simi-algorithm:latest"
echo "  docker push your-registry.com/simi-algorithm:latest"
echo ""
echo "To build individual algorithms:"
echo "  docker build --build-arg ALGORITHM=SIMI -t simi-algorithm:latest -f Dockerfile ."
echo "  docker build --build-arg ALGORITHM=SIMICE -t simice-algorithm:latest -f Dockerfile ."

