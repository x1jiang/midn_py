#!/usr/bin/env python3
"""
Create SIMI job and test protocol
"""

import requests
import json
import asyncio
import websockets
from datetime import datetime

CENTRAL_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/1"
TOKEN = "remote_1_token_2024"

def create_simi_job():
    """Create a SIMI job via the API"""
    
    # First, let's check if the job creation endpoint exists
    # Based on the web interface, it might be a POST to create jobs
    
    job_data = {
        "algorithm": "SIMI",
        "method": "gaussian",
        "participants": [1, 2]  # Site IDs that should participate
    }
    
    try:
        # Try to create a job via API
        response = requests.post(f"{CENTRAL_BASE_URL}/api/jobs/", json=job_data)
        if response.status_code == 200 or response.status_code == 201:
            job_info = response.json()
            print(f"✅ Created SIMI job: {job_info}")
            return job_info.get("id") or job_info.get("job_id")
        else:
            print(f"❌ Failed to create job: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating job: {e}")
        return None

async def test_simi_connection():
    """Test SIMI connection with proper authentication"""
    
    print(f"🧪 Testing SIMI Connection with Authentication")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Connect with token as query parameter
        url_with_token = f"{WS_URL}?token={TOKEN}"
        print(f"🔌 Connecting to: {url_with_token}")
        
        async with websockets.connect(url_with_token) as websocket:
            print("✅ Connected successfully!")
            
            # Step 1: Send CONNECT message
            print("\n📤 Step 1: Sending CONNECT message...")
            connect_message = {
                "type": "CONNECT",
                "token": TOKEN,
                "site_id": 1
            }
            await websocket.send(json.dumps(connect_message))
            print(f"   Sent: {connect_message}")
            
            # Wait for response
            print("⏳ Waiting for central response...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                print(f"📥 Received: {response_data}")
                
                if response_data.get("type") == "ERROR":
                    error_code = response_data.get("error_code")
                    print(f"ℹ️  Error response: {error_code}")
                    if error_code == "NO_JOBS_AVAILABLE":
                        print("💡 Try creating a SIMI job first")
                    return False
                
                # Step 2: Send SITE_READY if connection confirmed
                if response_data.get("type") == "connection_confirmed":
                    print("\n📤 Step 2: Sending SITE_READY message...")
                    ready_message = {
                        "type": "SITE_READY",
                        "site_id": 1
                    }
                    await websocket.send(json.dumps(ready_message))
                    print(f"   Sent: {ready_message}")
                    
                    # Wait for START_COMPUTATION
                    print("⏳ Waiting for START_COMPUTATION...")
                    computation_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    computation_data = json.loads(computation_response)
                    print(f"📥 Received: {computation_data}")
                    
                    if computation_data.get("type") == "START_COMPUTATION":
                        print("✅ Protocol flow working correctly!")
                        
                        # Wait for METHOD message
                        print("⏳ Waiting for METHOD message...")
                        method_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        method_data = json.loads(method_response)
                        print(f"📥 Received: {method_data}")
                        
                        if method_data.get("type") == "method":
                            print("🎉 SIMI service is properly using standardized protocol!")
                            print(f"📋 Method: {method_data.get('method')}")
                            return True
                        else:
                            print(f"❌ Expected METHOD message, got: {method_data.get('type')}")
                    else:
                        print(f"❌ Expected START_COMPUTATION, got: {computation_data.get('type')}")
                
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for response")
                return False
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 SIMI Protocol Test")
    print("=" * 60)
    
    # First try to create a job
    print("📋 Step 1: Creating SIMI job...")
    job_id = create_simi_job()
    
    if job_id:
        print(f"✅ Job created with ID: {job_id}")
        print("\n📋 Step 2: Testing protocol...")
        success = await test_simi_connection()
        
        if success:
            print("\n🎉 All tests passed! SIMI protocol is working correctly.")
        else:
            print("\n❌ Protocol test failed.")
    else:
        print("\n📋 Skipping protocol test due to job creation failure")
        print("💡 You can manually create a SIMI job via the web interface and then test the connection")
        
        # Still test connection to see the error
        print("\n📋 Testing connection anyway...")
        await test_simi_connection()

if __name__ == "__main__":
    asyncio.run(main())
