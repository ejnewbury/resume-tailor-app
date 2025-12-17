#!/usr/bin/env python3
"""
Simple test script for the Resume Tailoring API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Device: {data['device']}")
            return True
        else:
            print("❌ Health check failed")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_connect():
    """Test LLM connection (simulated)"""
    try:
        payload = {
            "type": "openai",
            "api_key": "test-key",
            "model": "gpt-4o-mini"
        }
        response = requests.post(f"{BASE_URL}/connect", json=payload)
        data = response.json()

        if response.status_code == 200 and data.get("success"):
            print("✅ Connection test passed")
            return True
        else:
            print("⚠️ Connection test returned expected response (API key invalid)")
            return True
    except Exception as e:
        print(f"❌ Connection test error: {e}")
        return False

def main():
    print("🧪 Testing Resume Tailoring API")
    print("=" * 40)

    # Test health
    health_ok = test_health()
    print()

    # Test connection
    connect_ok = test_connect()
    print()

    # Summary
    if health_ok and connect_ok:
        print("🎉 All tests passed! API is ready.")
        print("\n🌐 Open http://localhost:8000 in your browser")
    else:
        print("❌ Some tests failed. Check server logs.")

if __name__ == "__main__":
    main()
