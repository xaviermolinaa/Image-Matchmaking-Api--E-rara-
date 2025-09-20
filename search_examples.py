#!/usr/bin/env python3
"""
Examples of different search scenarios for the E-rara Image Matchmaking API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_search_scenario(name, request_data):
    """Test a specific search scenario"""
    print(f"\n🔍 {name}")
    print("-" * 50)
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/matchmaking/images/search", 
                               json=request_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            count = result.get('count', 0)
            print(f"✅ Found {count} images")
            
            if count > 0:
                # Show first result
                first_image = result['images'][0]
                metadata = first_image.get('metadata', {})
                print(f"📖 First result: {metadata.get('title', 'No title')}")
                print(f"📅 Date: {metadata.get('date', 'Unknown')}")
                print(f"📍 Place: {metadata.get('place', 'Unknown')}")
                if 'thumbnail_url' in first_image:
                    print(f"🖼️  Thumbnail: {first_image['thumbnail_url']}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    print("🚀 E-rara API Search Examples")
    print("=" * 50)
    
    # Example 1: Place search
    test_search_scenario("Place Search - Basel", {
        "operation": "IMAGE_MATCHMAKING",
        "criteria": [{"field": "Place", "value": "Basel"}],
        "from_date": "1600",
        "until_date": "1620",
        "maxResults": 3,
        "validate_images": False  # For speed
    })
    
    # Example 2: Author search
    test_search_scenario("Author Search - Erasmus", {
        "operation": "IMAGE_MATCHMAKING", 
        "criteria": [{"field": "Author", "value": "Erasmus"}],
        "from_date": "1500",
        "until_date": "1600",
        "maxResults": 2,
        "validate_images": False
    })
    
    # Example 3: Publisher search
    test_search_scenario("Publisher Search - Froben", {
        "operation": "IMAGE_MATCHMAKING",
        "criteria": [{"field": "Publisher", "value": "Froben"}],
        "from_date": "1500",
        "until_date": "1550",
        "maxResults": 2,
        "validate_images": False
    })
    
    # Example 4: Multiple criteria
    test_search_scenario("Multiple Criteria - Basel + 16th Century", {
        "operation": "IMAGE_MATCHMAKING",
        "criteria": [
            {"field": "Place", "value": "Basel"},
            {"field": "Author", "value": "Erasmus"}
        ],
        "from_date": "1500",
        "until_date": "1600", 
        "maxResults": 2,
        "validate_images": False
    })
    
    # Example 5: High performance mode
    test_search_scenario("High Performance Mode - No Validation, More Workers", {
        "operation": "IMAGE_MATCHMAKING",
        "criteria": [{"field": "Place", "value": "Basel"}],
        "from_date": "1600",
        "until_date": "1650",
        "maxResults": 5,
        "validate_images": False,  # Skip validation for speed
        "max_workers": 6,          # More concurrent workers
        "avoid_covers": True       # Skip book covers
    })
    
    print(f"\n📊 Cache Statistics:")
    try:
        cache_response = requests.get(f"{BASE_URL}/api/v1/cache/stats")
        if cache_response.status_code == 200:
            cache_data = cache_response.json()
            manifest_stats = cache_data["cache_stats"]["manifest_cache"]
            print(f"   🗄️  Manifest cache: {manifest_stats['hits']} hits, {manifest_stats['misses']} misses")
            
            if manifest_stats['hits'] > 0:
                hit_rate = (manifest_stats['hits'] / (manifest_stats['hits'] + manifest_stats['misses'])) * 100
                print(f"   🎯 Hit rate: {hit_rate:.1f}%")
    except Exception as e:
        print(f"   ❌ Could not get cache stats: {e}")
    
    print(f"\n✅ Examples complete!")
    print(f"💡 Try different search terms like:")
    print(f"   • Places: Zürich, Bern, Genève, Strasbourg")
    print(f"   • Authors: Luther, Calvin, Zwingli") 
    print(f"   • Publishers: Petri, Apiarius, Cratander")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("   Make sure the server is running on http://localhost:8000")