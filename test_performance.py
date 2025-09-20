#!/usr/bin/env python3
"""
Test script to demonstrate the performance improvements:
1. Caching layer
2. Concurrent processing  
3. Optional image validation
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_performance_improvements():
    """Test the new performance features"""
    
    print("🚀 Testing Performance Improvements")
    print("=" * 50)
    
    # Test data - using real place names that exist in e-rara
    test_request = {
        "operation": "IMAGE_MATCHMAKING",
        "criteria": [
            {
                "field": "Place",
                "value": "Basel"  # Real place name, not "Basel*" to ensure results
            }
        ],
        "from_date": "1600",
        "until_date": "1610",  # Narrower date range for faster testing
        "maxResults": 3  # Fewer results for faster testing
    }
    
    # Test 1: With image validation (slower)
    print("\n1. Testing WITH image validation...")
    test_request["validate_images"] = True
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/v1/matchmaking/images/search", 
                           json=test_request, timeout=60)
    duration_with_validation = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        count = result.get('count', 0)
        print(f"   ✅ Found {count} images in {duration_with_validation:.2f}s")
        if count == 0:
            print("   ⚠️  No results found - this might be normal for this search criteria")
            print(f"   🔍 Search was: Place='{test_request['criteria'][0]['value']}', dates {test_request['from_date']}-{test_request['until_date']}")
    else:
        print(f"   ❌ Error: {response.status_code} - {response.text}")
        return
    
    # Test 2: Without image validation (faster)
    print("\n2. Testing WITHOUT image validation...")
    test_request["validate_images"] = False
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/v1/matchmaking/images/search", 
                           json=test_request, timeout=60)
    duration_without_validation = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        count = result.get('count', 0)
        print(f"   ✅ Found {count} images in {duration_without_validation:.2f}s")
        if count > 0 and duration_with_validation > 0:
            improvement = ((duration_with_validation - duration_without_validation) / duration_with_validation) * 100
            print(f"   📈 Speed improvement: {improvement:.1f}% faster")
        elif count == 0:
            print("   ⚠️  No results found for comparison")
    else:
        print(f"   ❌ Error: {response.status_code} - {response.text}")
        return
    
    # Test 3: Cache performance (should be much faster on second call)
    print("\n3. Testing cache performance (second call)...")
    test_request["validate_images"] = True  # Back to validation
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/api/v1/matchmaking/images/search", 
                           json=test_request, timeout=60)
    duration_cached = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Found {result.get('count', 0)} images in {duration_cached:.2f}s")
        cache_improvement = ((duration_with_validation - duration_cached) / duration_with_validation) * 100
        print(f"   🚀 Cache improvement: {cache_improvement:.1f}% faster than first call")
    
    # Test 4: Check cache statistics
    print("\n4. Checking cache statistics...")
    cache_response = requests.get(f"{BASE_URL}/api/v1/cache/stats")
    if cache_response.status_code == 200:
        cache_stats = cache_response.json()
        manifest_stats = cache_stats["cache_stats"]["manifest_cache"]
        validation_stats = cache_stats["cache_stats"]["image_validation_cache"]
        
        print(f"   📊 Manifest cache: {manifest_stats['hits']} hits, {manifest_stats['misses']} misses")
        print(f"   📊 Validation cache: {validation_stats['hits']} hits, {validation_stats['misses']} misses")
        
        if manifest_stats['hits'] > 0:
            hit_rate = (manifest_stats['hits'] / (manifest_stats['hits'] + manifest_stats['misses'])) * 100
            print(f"   🎯 Cache hit rate: {hit_rate:.1f}%")
    
    # Test 5: Check performance config
    print("\n5. Performance configuration...")
    config_response = requests.get(f"{BASE_URL}/api/v1/performance/config")
    if config_response.status_code == 200:
        config = config_response.json()
        print(f"   ⚙️  Concurrent workers: {config['concurrency']['max_workers']}")
        print(f"   ⚙️  Cache sizes: {config['caching']['manifest_cache_size']} manifests, {config['caching']['image_validation_cache_size']} validations")
        print(f"   ✅ Features enabled: {', '.join([k for k, v in config['features'].items() if v])}")
    
    print(f"\n🎉 Performance testing complete!")
    print(f"💡 Key improvements:")
    print(f"   • Manifest caching reduces repeated API calls")
    print(f"   • Concurrent processing speeds up multi-record requests")
    print(f"   • Optional image validation can provide {improvement:.0f}% speed boost")

if __name__ == "__main__":
    try:
        test_performance_improvements()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("   Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")