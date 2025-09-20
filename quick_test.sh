#!/bin/bash

# Quick test launcher for the Image Matchmaking API

echo "🚀 Image Matchmaking API - Quick Test"
echo "====================================="

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Server not running. Starting API server..."
    echo "   Run: uvicorn image_matchmaking_api:app --reload"
    exit 1
fi

echo "✅ Server is running"
echo ""

# Run performance test
echo "🧪 Running performance tests..."
python3 test_performance.py

echo ""
echo "🔧 Quick API Tests:"
echo ""

# Test health endpoint
echo "1. Health check:"
curl -s http://localhost:8000/health | jq '.'
echo ""

# Test a simple search
echo "2. Simple search test:"
RESULT=$(curl -s -X POST "http://localhost:8000/api/v1/matchmaking/images/search" \
     -H "Content-Type: application/json" \
     -d '{
       "operation": "IMAGE_MATCHMAKING",
       "criteria": [{"field": "Place", "value": "Basel"}],
       "from_date": "1600",
       "until_date": "1610",
       "maxResults": 2,
       "validate_images": false
     }')

COUNT=$(echo $RESULT | jq -r '.count // 0')
echo "Found $COUNT images"

if [ "$COUNT" -gt 0 ]; then
    TITLE=$(echo $RESULT | jq -r '.images[0].metadata.title // "No title"')
    echo "First image: $TITLE"
else
    echo "No images found for this search"
fi

echo ""
echo "✅ All tests complete!"
echo ""
echo "📖 Next steps:"
echo "   • Check cache stats: curl http://localhost:8000/api/v1/cache/stats"
echo "   • View performance config: curl http://localhost:8000/api/v1/performance/config"
echo "   • Clear cache: curl -X POST http://localhost:8000/api/v1/cache/clear"