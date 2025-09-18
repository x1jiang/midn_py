# Federated Learning Job Management Protocol

This document describes the standardized job management protocol used across all federated learning algorithms in the PYMIDN project.

> **IMPORTANT UPDATE**: The protocol implementation has been consolidated into `common.algorithm.job_protocol.py`. All references to the original `protocol.py` module have been removed and repl### Proposed FederatedJobProtocolClient Implementation

```python
"""
Federated Job Protocol Client for remote site.
Provides standardized job management communication protocol for remote federated learning clients.
"""

import asyncio
import json
from typing import Dict, Any, Optional

from common.algorithm.job_protocol import create_message, parse_message, ProtocolMessageType standardized implementation.

## Overview

The federated learning system consists of a central server and multiple remote sites. The central server coordinates the execution of jobs, while remote sites contribute data and computation power while keeping their data private. This protocol standardizes the communication between central and remote sites across different algorithms.

## Job States

### Central Site Job States
- **Waiting**: Job created but waiting for participants to connect
- **Active**: Job running with connected participants
- **Completed**: Job finished successfully
- **Failed**: Job failed or terminated abnormally

### Remote Site States
- **Disconnected**: Not connected to central
- **Connected**: Connected but not yet ready for computation
- **Ready**: Ready to participate in computation
- **Computing**: Actively participating in computation
- **Completed**: Local processing completed

## Message Protocol

### 1. Connection Establishment

#### Remote to Central
```json
{
  "type": "connect",
  "job_id": 123,
  "site_id": "site1"
}
```

#### Central to Remote on Successful Connection
```json
{
  "type": "connection_confirmed",
  "job_id": 123,
  "algorithm": "ALGORITHM_NAME",
  "status": "waiting",
  "method": "METHOD_NAME"
}
```

#### Central to Remote on Failed Connection
```json
{
  "type": "error",
  "code": "ERROR_CODE",
  "message": "Error message"
}
```
Error codes include: `NO_JOBS_AVAILABLE`, `MISSING_JOB_ID`, `JOB_NOT_FOUND`, `UNAUTHORIZED_SITE`

### 2. Site Ready Notification

#### Remote to Central
```json
{
  "type": "site_ready",
  "job_id": 123,
  "site_id": "site1",
  "status": "ready"
}
```

#### Central to All Sites (When All Ready)
```json
{
  "type": "start_computation",
  "job_id": 123
}
```

### 3. Algorithm-Specific Communication

After connection establishment, algorithm-specific messages are exchanged (data, statistics, models, etc.). These messages differ by algorithm but should always include:

- `type`: Message type
- `job_id`: Job identifier

### 4. Job Completion

#### Central to All Remote Sites
```json
{
  "type": "job_completed",
  "job_id": 123,
  "status": "completed",
  "message": "Job completed successfully",
  "result_path": "path/to/results"
}
```

#### Remote to Central (Acknowledgement)
```json
{
  "type": "completion_ack",
  "job_id": 123,
  "site_id": "site1"
}
```

## Connection Management Rules

### Remote Sites
1. Establish connection to central with `connect` message
2. Wait for confirmation before sending data
3. Send `site_ready` when ready to compute
4. Wait for `start_computation` before starting algorithm
5. Acknowledge job completion with `completion_ack`
6. Disconnect after completion acknowledgment

### Central Site
1. Accept connections from authorized sites
2. Track connected sites per job
3. Send `start_computation` when all sites are ready
4. Send `job_completed` to all sites when job finishes
5. Track completion acknowledgments
6. Clean up job data after all sites acknowledge completion

## Error Handling

### Connection Errors
- Central responds with appropriate error message
- Remote sites should implement exponential backoff for reconnection attempts

### Computation Errors
- Sites report errors to central
- Central decides whether to continue with remaining sites or fail job

### Timeout Handling
- Central implements timeouts for expected responses
- If a site doesn't respond within timeout, mark as disconnected
- Continue job if possible with remaining sites

## Implementation Details

The `FederatedJobProtocolService` base class implements the standardized job management protocol. Algorithm-specific services should inherit from this base class and override the following methods:

1. `_handle_algorithm_message`: Handle algorithm-specific messages
2. `_handle_algorithm_connection`: Handle algorithm-specific connection logic
3. `get_algorithm_name`: Return the name of the algorithm
4. `start_job`: Implement the algorithm-specific job execution

The base class provides standard implementations for:
1. `_handle_connect`: Handle initial connection from remote sites
2. `_handle_site_ready`: Process site ready notifications
3. `_handle_completion_ack`: Handle completion acknowledgments
4. `notify_job_completed`: Send job completion notifications
5. `wait_for_participants`: Utility to wait for all sites to connect
6. `initialize_job_state`: Set up common job state data

Algorithm services should focus on implementing their specific data exchange protocols while leveraging the standardized job management functionality.

## Base Class Structure

### Central Service

The `FederatedJobProtocolService` base class provides the following architecture:

```
FederatedJobProtocolService
├── Common Job Management
│   ├── Connection handling
│   ├── Site readiness tracking
│   ├── Job state management
│   └── Completion notification
├── Base Methods (to override)
│   ├── _handle_algorithm_message
│   ├── _handle_algorithm_connection
│   ├── get_algorithm_name
│   └── start_job
└── Utility Methods
    ├── initialize_job_state
    ├── wait_for_participants
    ├── add_status_message
    ├── notify_job_completed
    └── send_to_all_sites
```

### Remote Client

The `BaseAlgorithmClient` base class provides the following structure:

```
BaseAlgorithmClient
├── Common Job Management
│   ├── Connection establishment
│   ├── Message handling
│   ├── Status reporting
│   └── Completion acknowledgment
├── Base Methods (to override)
│   ├── run_algorithm
│   ├── handle_message
│   └── process_algorithm_specific_messages
└── Utility Methods
    ├── create_connection_client
    ├── send_algorithm_data
    └── manage_reconnection
```

Algorithm-specific services (like SIMI and SIMICE) inherit from these base classes, which ensures consistent job management across all algorithms while allowing for flexibility in the specific data exchange protocols.

## Protocol Compatibility

The standardized job management protocol ensures that:

1. All algorithms follow the same pattern for connection establishment
2. Site readiness is tracked consistently
3. Job status transitions happen in a predictable manner
4. Completion notifications are handled uniformly

This allows for:
- Easier debugging of job management issues
- Consistency in user experience across different algorithms
- Simpler development of new federated learning algorithms
- Centralized error handling and logging

Remote sites follow the same connection protocol regardless of which algorithm they are executing, improving interoperability and reducing code duplication.

## Remote Client Protocol Implementation

The remote client side of the protocol implements the following standard flow:

### Connection Phase
1. Connect to the central server with job ID and site ID
2. Handle connection confirmations and errors (retry as needed)
3. Notify readiness for computation
4. Wait for computation start signal

### Computation Phase
1. Process algorithm-specific instructions from central
2. Send algorithm-specific data in appropriate format
3. Continue with iterations as directed by central server
4. Process job completion notification

### Reconnection Handling
1. Implement exponential backoff for connection failures
2. Handle unexpected disconnections gracefully
3. Respect job completion signals and avoid unnecessary reconnection attempts
4. Maintain job state between reconnections when appropriate

Each algorithm client extends the base client and implements the algorithm-specific data processing while inheriting the standard job management protocol.

### Client Protocol Implementation

The `FederatedJobProtocolClient` now provides a standardized implementation of the protocol. Algorithm-specific clients simply inherit from this base class and override specific methods:

1. **Connection Establishment**:
   ```python
   # Using the Protocol utility class for standard messages
   connect_message = Protocol.create_connect_message(job_id, site_id)
   
   # Add any algorithm-specific parameters
   if extra_params:
       connect_message.update(extra_params)
   
   # Send using the connection client
   await client.send_message(websocket, connect_message)
   ```

2. **Site Ready Notification**:
   ```python
   # Using the Protocol utility class
   site_ready_message = Protocol.create_site_ready_message(job_id, site_id)
   await client.send_message(websocket, site_ready_message)
   ```

3. **Handle Algorithm-Specific Computation**:
   ```python
   # Override this method in algorithm-specific subclasses
   async def _handle_algorithm_computation(self, client, websocket, data, target_column, job_id, initial_data):
       await client.send_status("Starting computation")
       
       # Algorithm-specific processing
       # Process messages until job completion
       while True:
           message = await client.receive_message(websocket)
           if not message:
               break
               
           message_type = message.get("type")
           
           if message_type == ProtocolMessageType.JOB_COMPLETED.value:
               # Handle job completion
               await self._handle_job_completed(client, websocket, message)
               break
               
           # Algorithm-specific message handling
   ```

4. **Processing Algorithm Messages**:
   ```python
   # Override this method to handle algorithm-specific messages
   async def _process_algorithm_message(self, client, websocket, message):
       message_type = message.get("type")
       
       # Algorithm-specific message handling
       # Return True to continue processing, False to exit processing loop
       return True
   ```

5. **Completion Acknowledgment**:
   ```python
   # Using the Protocol utility class
   ack_message = Protocol.create_completion_ack_message(job_id, site_id)
   await client.send_message(websocket, ack_message)
   ```

### Algorithm-Specific Implementation Example

The SIMI client implementation inherits from `FederatedJobProtocolClient` and overrides the necessary methods:

```python
class SIMIClient(FederatedJobProtocolClient):
    def __init__(self, algorithm_class):
        super().__init__(algorithm_class)
        self.method = "gaussian"  # Default method
    
    async def _process_method_instruction(self, client, method):
        # Update the method based on the instruction from central
        self.method = method.lower()
        await client.send_status(f"Using SIMI method: {self.method}")
        
        # Make sure the algorithm instance has the correct method
        if hasattr(self.algorithm_instance, 'method'):
            self.algorithm_instance.method = self.method
    
    async def _handle_algorithm_computation(self, client, websocket, data, target_column, job_id, initial_data):
        # Different handling based on the method
        if self.method == "gaussian":
            # For Gaussian, send the statistics
            ls_stats = await self.algorithm_instance.process_message("method", {"method": self.method})
            await client.send_message(websocket, ProtocolMessageType.DATA, job_id=job_id, **ls_stats)
            
        else:  # Logistic method
            # First send initial sample size
            await client.send_message(websocket, ProtocolMessageType.DATA, job_id=job_id, **initial_data)
    
    async def _process_algorithm_message(self, client, websocket, message):
        message_type = message.get("type")
        
        if message_type == "mode":
            mode = message.get("mode", 0)
            
            # Process the message with the algorithm
            results = await self.algorithm_instance.process_message("mode", message)
            
            # Send results based on the mode
            # Mode-specific handling for SIMI algorithm
            
            return True  # Continue processing
            
        return True  # Continue processing
```
   ```

This standardized approach ensures that all algorithm-specific clients follow the same protocol, with common code centralized in the base class and only algorithm-specific behavior implemented in subclasses.

## Recommended Improvements

To fully standardize the job management protocol across both central and remote components, the following improvements are recommended:

1. Create a dedicated `FederatedJobProtocolClient` class that mirrors the functionality of `FederatedJobProtocolService`
2. Refactor existing algorithm clients to inherit from this new base class
3. Ensure consistent handling of connection, readiness, and completion acknowledgments
4. Standardize error handling and reconnection logic

### Proposed FederatedJobProtocolClient Implementation

```python
"""
Federated Job Protocol Client for remote site.
Provides standardized job management communication protocol for remote federated learning clients.
"""

import asyncio
import json
from typing import Dict, Any, Optional

from common.algorithm.job_protocol import create_message, parse_message, ProtocolMessageType, Protocol
from .base_algorithm_client import BaseAlgorithmClient
from ..websockets.connection_client import ConnectionClient


class FederatedJobProtocolClient(BaseAlgorithmClient):
    """
    Standardized client for job management protocol in federated learning algorithms.
    Handles job connections, status transitions, and completion notifications.
    Algorithm-specific data exchange is delegated to subclasses.
    """
    
    async def run_algorithm(self, data, target_column, job_id, site_id, 
                           central_url, token, extra_params=None, status_callback=None):
        """
        Run the algorithm with standardized protocol handling.
        """
        # Create connection client
        client = ConnectionClient(central_url, site_id, token, status_callback)
        
        # Main connection loop
        while not client.is_job_stopped():
            # Try to connect
            success, websocket = await client.connect()
            if not success:
                await self._handle_connection_failure(client)
                continue
                
            try:
                # Connection established, send connect message
                await self._handle_connection(client, websocket, job_id, extra_params)
                
                # Handle algorithm-specific computation
                await self._handle_algorithm_computation(client, websocket, data, target_column, job_id)
                
                # Handle job completion
                await self._handle_job_completion(client)
                
            except Exception as e:
                await self._handle_error(client, e)
    
    async def _handle_connection(self, client, websocket, job_id, extra_params):
        """Handle initial connection to central"""
        # Standard protocol implementation
        pass
        
    async def _handle_site_ready(self, client, websocket, job_id):
        """Send site_ready message when ready for computation"""
        # Standard protocol implementation
        pass
    
    async def _handle_completion_ack(self, client, websocket, job_id):
        """Handle job completion acknowledgment"""
        # Standard protocol implementation
        pass
        
    async def _handle_algorithm_computation(self, client, websocket, data, target_column, job_id):
        """
        Handle algorithm-specific computation.
        To be overridden by subclasses.
        """
        pass
```

This standard client implementation would complete the standardization of the protocol on both sides of the communication.

## Conclusion

The standardized federated learning protocol establishes a clear separation between:

1. **Job Management**: Connection handling, status tracking, and completion notification (handled by the base classes)
2. **Algorithm-Specific Logic**: Data exchange formats, computational methods, and algorithm parameters (implemented by subclasses)

This separation of concerns:
- Increases code maintainability
- Ensures consistency across algorithms
- Reduces duplication of common functionality
- Provides a solid foundation for developing new federated learning algorithms
- Standardizes behavior across both central and remote components

When implementing a new federated learning algorithm, developers should focus on the algorithm-specific data exchange while inheriting the standardized job management functionality from the appropriate base classes:
- `FederatedJobProtocolService` for central services
- `BaseAlgorithmClient` (and eventually `FederatedJobProtocolClient`) for remote clients

## Implementation Plan

To ensure that remote clients follow the same job management protocol as central services:

1. **Create common protocol utilities**:
   - Created `/common/algorithm/job_protocol.py` with shared protocol definitions
   - Implemented `Protocol` class with standardized message creation methods
   - Defined enums for job states, message types, and error codes

2. **Create the `FederatedJobProtocolClient` class**:
   - Implemented standardized connection handling
   - Added site readiness notification
   - Implemented job completion acknowledgment
   - Added reconnection logic with exponential backoff
   - Created hooks for algorithm-specific processing

3. **Update existing clients**:
   - Created updated `SIMIClient` to inherit from `FederatedJobProtocolClient`
   - Maintained backward compatibility with legacy messages
   - Moved common code to base class
   - Kept algorithm-specific handling in subclasses

4. **Test protocol compatibility**:
   - Ensure all clients follow the same connection protocol
   - Verify proper handling of site readiness
   - Test completion acknowledgments
   - Validate reconnection behavior

Following this implementation plan ensures that both central services and remote clients adhere to the standardized job management protocol, improving robustness and maintainability of the federated learning system.

## Protocol Consolidation

As of August 2025, all protocol functionality has been consolidated into the `job_protocol.py` module. The consolidation included:

1. Creating a unified `ProtocolMessageType` enum that incorporates all message types
2. Moving `create_message` and `parse_message` functions to `job_protocol.py`
3. Creating the `Protocol` utility class for standardized message creation
4. Updating all services and clients to use the consolidated protocol

This consolidation has eliminated code duplication, improved consistency across the codebase, and made future protocol changes easier to implement and maintain.
