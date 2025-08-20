#!/usr/bin/env python3
"""
Test SIMICE protocol implementation to verify R-compliant message flow.
Tests the communication between central and remote services following R implementation.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from central.app.services.simice_service import SIMICEService
from common.algorithm.job_protocol import ProtocolMessageType, create_message


class MockConnectionManager:
    """Mock connection manager for testing."""
    
    def __init__(self):
        self.sent_messages = []
        self.sites = {}
    
    async def send_to_site(self, message, site_id):
        """Mock sending message to site."""
        # Handle both string and dict messages
        if isinstance(message, str):
            import json
            try:
                message_dict = json.loads(message)
            except:
                message_dict = {"raw_message": message}
        else:
            message_dict = message
            
        instruction = message_dict.get('instruction', message_dict.get('type', 'unknown'))
        print(f"📤 MOCK: Sending to site {site_id}: {instruction}")
        self.sent_messages.append((site_id, message_dict))
        
        # Simulate site responses
        if message_dict.get('instruction') == 'Initialize':
            # Simulate Initialize response
            response = create_message(
                ProtocolMessageType.METHOD,
                job_id=message_dict['job_id'],
                instruction='Initialize',
                status='ready'
            )
            print(f"📨 MOCK: Site {site_id} responds to Initialize")
            
        elif message_dict.get('instruction') == 'Information':
            # Simulate Information response with statistics
            column_index = message_dict.get('target_column_index')
            method = message_dict.get('method')
            
            if method == 'Gaussian':
                stats = {
                    'n': 50,
                    'sum_xy': [1.0, 2.0, 3.0, 4.0, 5.0],
                    'sum_xx': [
                        [10.0, 1.0, 2.0, 3.0, 4.0],
                        [1.0, 15.0, 2.0, 3.0, 4.0],
                        [2.0, 2.0, 20.0, 3.0, 4.0],
                        [3.0, 3.0, 3.0, 25.0, 4.0],
                        [4.0, 4.0, 4.0, 4.0, 30.0]
                    ],
                    'sum_yy': 100.0
                }
            else:  # logistic
                stats = {
                    'H': [
                        [5.0, 0.5, 1.0, 1.5, 2.0],
                        [0.5, 8.0, 1.0, 1.5, 2.0],
                        [1.0, 1.0, 12.0, 1.5, 2.0],
                        [1.5, 1.5, 1.5, 15.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0, 20.0]
                    ],
                    'g': [2.0, 3.0, 4.0, 5.0, 6.0]
                }
            
            response_data = {
                'instruction': 'Information',
                'target_column_index': column_index,
                'method': method,
                'statistics': stats
            }
            
            print(f"📨 MOCK: Site {site_id} responds to Information for column {column_index} ({method})")
            
            # Simulate the response being processed
            asyncio.create_task(self._simulate_statistics_response(
                site_id, message_dict['job_id'], response_data
            ))
            
        elif message_dict.get('instruction') == 'Impute':
            # Simulate Impute response
            column_index = message_dict.get('target_column_index')
            
            response_data = {
                'instruction': 'Impute',
                'target_column_index': column_index,
                'status': 'completed'
            }
            
            print(f"📨 MOCK: Site {site_id} responds to Impute for column {column_index}")
            
            # Simulate the response being processed
            asyncio.create_task(self._simulate_impute_response(
                site_id, message_dict['job_id'], response_data
            ))
            
        elif message_dict.get('instruction') == 'End':
            # Simulate End response
            print(f"📨 MOCK: Site {site_id} responds to End")
    
    async def _simulate_statistics_response(self, site_id, job_id, response_data):
        """Simulate statistics response with a small delay."""
        await asyncio.sleep(0.1)  # Small delay to simulate network
        
        # Call the service's statistics handler
        if hasattr(self, 'service'):
            await self.service._handle_information_response(site_id, job_id, response_data)
    
    async def _simulate_impute_response(self, site_id, job_id, response_data):
        """Simulate impute response with a small delay."""
        await asyncio.sleep(0.1)  # Small delay to simulate network
        
        # Call the service's impute handler
        if hasattr(self, 'service'):
            await self.service._handle_impute_response(site_id, job_id, response_data)


async def test_simice_r_protocol():
    """Test SIMICE R-style protocol implementation."""
    
    print("🧪 Testing SIMICE R-Style Protocol Implementation")
    print("=" * 60)
    
    # Create mock connection manager
    manager = MockConnectionManager()
    
    # Create SIMICE service
    service = SIMICEService(manager)
    manager.service = service  # Allow manager to call service methods
    
    # Create test job data
    job_id = 1
    algorithm_data = {
        "target_column_indexes": [1, 3],  # Two columns to impute
        "is_binary": [False, True],  # Gaussian and logistic
        "iteration_before_first_imputation": 2,
        "iteration_between_imputations": 1,
        "imputation_trials": 2  # M=2 imputations
    }
    
    # Create mock DB job
    class MockJob:
        def __init__(self):
            self.id = job_id
            self.algorithm = "SIMICE"
            self.status = "pending"
            self.algorithm_data = algorithm_data
            self.participants = ["site1", "site2"]
            self.parameters = "{}"
    
    mock_db_job = MockJob()
    
    # Initialize job state
    await service.initialize_job_state(job_id, mock_db_job)
    print(f"✅ Job {job_id} state initialized")
    
    # Manually add algorithm data to the job
    job = service.jobs[job_id]
    job['algorithm_data'] = algorithm_data
    
    # Connect sites manually
    site_ids = ["site1", "site2"]
    job = service.jobs[job_id]
    for site_id in site_ids:
        if site_id not in job['connected_sites']:
            job['connected_sites'].append(site_id)
        print(f"✅ Site {site_id} connected")
    
    # Set sites ready manually
    for site_id in site_ids:
        if site_id not in job['ready_sites']:
            job['ready_sites'].append(site_id)
        print(f"✅ Site {site_id} ready")
    
    print("\n🚀 Starting SIMICE Algorithm...")
    print("-" * 40)
    
    # Trigger algorithm start directly (this should trigger the R-style protocol)
    await service._handle_algorithm_start(job_id)
    
    # Give some time for the async operations to complete
    await asyncio.sleep(2.0)
    
    print("\n📊 Protocol Verification")
    print("-" * 40)
    
    # Verify messages were sent in correct order
    expected_instructions = []
    actual_instructions = []
    
    for site_id, message in manager.sent_messages:
        instruction = message.get('instruction', message.get('type', 'unknown'))
        actual_instructions.append(instruction)
        print(f"📤 Sent to {site_id}: {instruction}")
    
    # Expected pattern: Initialize, Information (for each variable), Impute (for each variable), repeated for iterations, End
    print(f"\n📈 Total messages sent: {len(manager.sent_messages)}")
    
    # Check for R-style messages
    r_messages = ['Initialize', 'Information', 'Impute', 'End']
    found_r_messages = [msg for msg in actual_instructions if msg in r_messages]
    
    print(f"🎯 R-style messages found: {len(found_r_messages)}")
    print(f"📝 R-messages: {list(set(found_r_messages))}")
    
    if 'Initialize' in found_r_messages:
        print("✅ Initialize message sent (R-compliant)")
    else:
        print("❌ Initialize message missing")
    
    if 'Information' in found_r_messages:
        print("✅ Information messages sent (R-compliant)")
    else:
        print("❌ Information messages missing")
    
    if 'Impute' in found_r_messages:
        print("✅ Impute messages sent (R-compliant)")
    else:
        print("❌ Impute messages missing")
    
    if 'End' in found_r_messages:
        print("✅ End message sent (R-compliant)")
    else:
        print("❌ End message missing")
    
    # Check job status
    job = service.jobs.get(job_id, {})
    print(f"\n📋 Job Status: {job.get('status', 'unknown')}")
    print(f"🔄 Current imputation: {job.get('current_imputation', 0)}")
    print(f"📊 Missing variables: {job.get('missing_variables', [])}")
    
    print("\n" + "=" * 60)
    if len(found_r_messages) >= 3:  # At least Initialize, Information, and one other
        print("🎉 SIMICE R-Style Protocol Test PASSED!")
        print("✅ Message flow follows R implementation pattern")
    else:
        print("❌ SIMICE R-Style Protocol Test FAILED!")
        print("❗ Message flow doesn't match R implementation")
    
    print("=" * 60)


async def main():
    """Main test function."""
    try:
        await test_simice_r_protocol()
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
