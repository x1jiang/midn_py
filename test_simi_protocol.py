#!/usr/bin/env python3
"""
Test SIMI Protocol Implementation
Tests that SIMI service properly follows the standardized job protocol.
"""

import asyncio
import json
import websockets
import requests
import time
from datetime import datetime

# Static JWT tokens used in the system
TOKENS = {
    "863a2efd": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI4NjNhMmVmZCIsImV4cCI6MTc1NzY4Nzg4NH0.Uxv-62JIpgr95a9WKL0TU3wxvce_Em42zjUU7qWizfs",
    "224bdbc5": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyMjRiZGJjNSIsImV4cCI6MTc1NzY4Nzg3OX0.vdW5NDgBBKTvRk945Ixg19P34X6D_-AQtBNcEsdycHs"
}

# Use site 863a2efd by default
SITE_ID = "863a2efd"
TOKEN = TOKENS[SITE_ID]
CENTRAL_URL = f"ws://localhost:8000/ws/{SITE_ID}?token={TOKEN}"

async def check_for_running_job():
    """Check if there's a running job on the central server"""
    try:
        response = requests.get("http://localhost:8000/test/job-check", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('running_job_id')
        return None
    except Exception as e:
        print(f"❌ Error checking for running jobs: {e}")
        return None

async def test_simi_protocol():
    """Test SIMI service with standardized protocol flow"""
    
    print(f"🧪 Testing SIMI Protocol Implementation")
    print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🔗 Using site ID: {SITE_ID}")
    print("=" * 60)
    
    try:
        print("🔌 Connecting to central server...")
        print(f"📡 Connection URL: {CENTRAL_URL[:80]}...")
        
        async with websockets.connect(CENTRAL_URL) as websocket:
            print("✅ Connected successfully!")
            
            # Step 1: Send CONNECT message
            print("\n📤 Step 1: Sending CONNECT message...")
            connect_message = {
                "type": "connect",
                "job_id": 1,  # Assuming job 1 is running
                "site_id": SITE_ID,
                "is_binary": True  # For logistic regression
            }
            await websocket.send(json.dumps(connect_message))
            print(f"   Sent: {connect_message}")
            
            # Wait for connection confirmation
            print("⏳ Waiting for connection confirmation...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                print(f"📥 Received: {response_data}")
                
                if response_data.get("type") == "error":
                    error_code = response_data.get("code")
                    error_message = response_data.get("message", "Unknown error")
                    print(f"❌ Error: {error_message} (Code: {error_code})")
                    return
                
                # Step 2: Wait for method message or start_computation
                if response_data.get("type") == "connection_confirmed":
                    print("✅ Connection confirmed!")
                    
                    print("⏳ Waiting for next message (method or start_computation)...")
                    next_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    next_data = json.loads(next_response)
                    print(f"📥 Received: {next_data}")
                    
                    if next_data.get("type") == "method":
                        method = next_data.get("method")
                        print(f"✅ Method received: {method}")
                        
                        # Step 3: Send SITE_READY
                        print("\n📤 Step 3: Sending SITE_READY message...")
                        ready_message = {
                            "type": "site_ready",
                            "job_id": 1,
                            "site_id": SITE_ID,
                            "status": "ready"
                        }
                        await websocket.send(json.dumps(ready_message))
                        print(f"   Sent: {ready_message}")
                        
                        # Step 4: Wait for START_COMPUTATION
                        print("⏳ Waiting for START_COMPUTATION...")
                        computation_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        computation_data = json.loads(computation_response)
                        print(f"📥 Received: {computation_data}")
                        
                        if computation_data.get("type") == "start_computation":
                            print("✅ START_COMPUTATION received!")
                            
                            # Step 5: Wait for algorithm method message
                            print("⏳ Waiting for algorithm method message...")
                            alg_method_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                            alg_method_data = json.loads(alg_method_response)
                            print(f"📥 Received: {alg_method_data}")
                            
                            if alg_method_data.get("type") == "method":
                                print("✅ Algorithm method message received!")
                                print("🎉 SIMI protocol flow working correctly!")
                                print("🔄 Full standardized protocol sequence completed")
                                return True
                            else:
                                print(f"❌ Expected algorithm method, got: {alg_method_data.get('type')}")
                        else:
                            print(f"❌ Expected START_COMPUTATION, got: {computation_data.get('type')}")
                            
                    elif next_data.get("type") == "start_computation":
                        # Job is already active - we're joining mid-stream
                        print("✅ START_COMPUTATION received (joining active job)!")
                        
                        # Wait for method message that should follow
                        print("⏳ Waiting for method message...")
                        method_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        method_data = json.loads(method_response)
                        print(f"📥 Received: {method_data}")
                        
                        if method_data.get("type") == "method":
                            method = method_data.get("method")
                            print(f"✅ Method received: {method}")
                            print("🎉 SIMI protocol reconnection working correctly!")
                            print("🔄 Successfully joined active job and received method")
                            return True
                        else:
                            print(f"❌ Expected method message after start_computation, got: {method_data.get('type')}")
                    else:
                        print(f"❌ Expected method or start_computation, got: {next_data.get('type')}")
                else:
                    print(f"❌ Expected connection_confirmed, got: {response_data.get('type')}")
                
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for response")
                
    except websockets.exceptions.InvalidStatusCode as e:
        if "403" in str(e):
            print("❌ Authentication failed - JWT token may be invalid or expired")
            print("💡 Check if the tokens are still valid in the central server")
        else:
            print(f"❌ Server returned status code: {e}")
    except ConnectionRefusedError:
        print("❌ Connection refused - make sure central server is running")
        print("💡 Start server with: uvicorn central.app.main:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    return False

async def main():
    """Main test function"""
    # First check if there's a running job
    print(f"🔍 Checking for running jobs...")
    running_job_id = await check_for_running_job()
    
    if running_job_id:
        print(f"✅ Found running job {running_job_id}")
        await test_simi_protocol()
    else:
        print("⚠️  No running jobs found")
        print("\n📋 To test the SIMI protocol:")
        print("1. Start central server: uvicorn central.app.main:app --reload --host 0.0.0.0 --port 8000")
        print("2. Create a SIMI job through the web interface at http://127.0.0.1:8000")
        print("3. Run this test to see if the protocol works")
        print("\n🔑 Available static tokens:")
        for site_id, token in TOKENS.items():
            print(f"   Site {site_id}: {token[:50]}...")

if __name__ == "__main__":
    asyncio.run(main())
