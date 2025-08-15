#!/usr/bin/env python3
"""
Test script to demonstrate the improved communication flow.
This script shows what happens when a remote site tries to connect
but no jobs are available.
"""

import asyncio
import json
import websockets
from datetime import datetime


async def test_no_jobs_available():
    """Test what happens when connecting but no jobs are running."""
    
    print("🧪 Test: Connecting to central server with no jobs available")
    print("=" * 60)
    
    # Configuration - update these to match your setup
    CENTRAL_URL = "ws://localhost:8000"  # Update port if needed
    SITE_ID = "test_site_001"
    TOKEN = "your_jwt_token_here"  # You'll need a valid JWT token
    
    uri = f"{CENTRAL_URL}/ws/{SITE_ID}?token={TOKEN}"
    
    try:
        print(f"🔌 Attempting connection to: {uri}")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # Try to connect
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established!")
            
            # Send connect message (SIMICE example)
            connect_message = {
                "type": "connect",
                "job_id": 999,  # Non-existent job ID
                "iteration_before_first_imputation": 10,
                "iteration_between_imputations": 10
            }
            
            print(f"📤 Sending connect message: {connect_message}")
            await websocket.send(json.dumps(connect_message))
            
            # Wait for response
            print("⏳ Waiting for response from central server...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                response_data = json.loads(response)
                
                print(f"📨 Received response: {response_data}")
                
                # Check response type
                if response_data.get("type") == "error":
                    error_code = response_data.get("code")
                    error_message = response_data.get("message")
                    
                    print(f"❌ Error received - Code: {error_code}")
                    print(f"💬 Error message: {error_message}")
                    
                    if error_code == "NO_JOBS_AVAILABLE":
                        print("✅ SUCCESS: Central correctly responded 'no jobs available'")
                        print("🕐 Remote should wait 15 seconds before retry")
                        
                    elif error_code == "JOB_NOT_FOUND":
                        print("✅ SUCCESS: Central correctly responded 'job not found'")
                        print("🕐 Remote should wait 2 minutes before retry")
                        
                    else:
                        print(f"ℹ️  Received other error code: {error_code}")
                        
                else:
                    print(f"⚠️  Unexpected response type: {response_data.get('type')}")
                    
            except asyncio.TimeoutError:
                print("⏰ Timeout waiting for response - this is the OLD behavior!")
                print("❌ FAIL: Central should have sent an error response")
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"🔌 Connection closed by server: {e}")
        if e.code == 4009:
            print("✅ SUCCESS: Server rejected connection (job already running)")
        else:
            print(f"ℹ️  Connection closed with code: {e.code}")
            
    except Exception as e:
        print(f"💥 Connection failed: {e}")
        print("ℹ️  This could be expected if central server is not running")


async def test_connect_retry_pattern():
    """Test the retry pattern when no jobs are available."""
    
    print("\n🧪 Test: Retry pattern simulation")
    print("=" * 60)
    
    retry_delay = 15  # 15 seconds for NO_JOBS_AVAILABLE
    completion_wait = 120  # 2 minutes for job conflicts
    
    print("📋 Simulated retry delays:")
    print(f"   NO_JOBS_AVAILABLE: {retry_delay} seconds")
    print(f"   JOB_NOT_FOUND: {completion_wait} seconds")
    print(f"   UNAUTHORIZED_SITE: {completion_wait} seconds")
    print(f"   Other errors: {completion_wait} seconds")
    
    print("\n🔄 Simulated connection attempts:")
    
    for attempt in range(3):
        print(f"\n🔌 Attempt {attempt + 1}: {datetime.now().strftime('%H:%M:%S')}")
        print("   Connecting...")
        
        # Simulate different responses
        if attempt == 0:
            print("   ❌ Response: NO_JOBS_AVAILABLE")
            print(f"   ⏳ Waiting {retry_delay} seconds...")
            # In real code: await asyncio.sleep(retry_delay)
            
        elif attempt == 1:
            print("   ❌ Response: JOB_NOT_FOUND")  
            print(f"   ⏳ Waiting {completion_wait} seconds...")
            # In real code: await asyncio.sleep(completion_wait)
            
        else:
            print("   ✅ Response: connection_confirmed")
            print("   🎉 Job started successfully!")
            break


def print_communication_flow():
    """Print the expected communication flow."""
    
    print("\n📋 Expected Communication Flow")
    print("=" * 60)
    
    flows = [
        {
            "scenario": "No Jobs Available",
            "steps": [
                "1. Remote connects to central",
                "2. Remote sends connect message", 
                "3. Central checks: no jobs running",
                "4. Central sends ERROR with NO_JOBS_AVAILABLE",
                "5. Remote waits 15 seconds",
                "6. Remote tries again"
            ]
        },
        {
            "scenario": "Job Not Found", 
            "steps": [
                "1. Remote connects to central",
                "2. Remote sends connect with job_id=999",
                "3. Central checks: job 999 doesn't exist", 
                "4. Central sends ERROR with JOB_NOT_FOUND",
                "5. Remote waits 2 minutes",
                "6. Remote tries again"
            ]
        },
        {
            "scenario": "Job Already Running",
            "steps": [
                "1. Remote tries to connect to central",
                "2. Central checks: job already running",
                "3. Central closes connection with code 4009",
                "4. Remote waits 2 minutes", 
                "5. Remote tries again"
            ]
        }
    ]
    
    for flow in flows:
        print(f"\n🎯 {flow['scenario']}:")
        for step in flow["steps"]:
            print(f"   {step}")


if __name__ == "__main__":
    print("🚀 Communication Flow Test Suite")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print expected behavior
    print_communication_flow()
    
    # Show retry pattern
    asyncio.run(test_connect_retry_pattern())
    
    print("\n" + "=" * 60)
    print("🔧 To test with actual server:")
    print("1. Start central server: python -m central.app.main")
    print("2. Update CENTRAL_URL and TOKEN in this script")
    print("3. Run: python test_communication_flow.py")
    print("4. Check that central responds with proper error messages")
    
    # Uncomment to test with real server
    # asyncio.run(test_no_jobs_available())
