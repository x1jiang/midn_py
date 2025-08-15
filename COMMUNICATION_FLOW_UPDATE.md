# Communication Flow Updates

## Overview
Updated the communication flow between central and remote sites to meet the following requirements:

1. **Only central starts jobs**: Central server controls job initiation via websocket connections
2. **15-second retry delay**: Remote sites wait 15 seconds between connection attempts when no job is running
3. **2-minute completion wait**: After job completion or connection loss, remote sites wait 2 minutes before reconnecting
4. **Job conflict detection**: Central server rejects new connections when a job is already running

## Key Changes Made

### 1. Central Server Changes (`central/app/`)

#### WebSocket Endpoint (`main.py`)
- Added job conflict detection before accepting websocket connections
- Rejects connections with code 4009 when another job is running
- Checks `JobStatusTracker` for running jobs before accepting new connections

#### Base Algorithm Service (`services/base_algorithm_service.py`)
- Added job conflict handling in message processing
- New method structure: `handle_site_message()` calls `_handle_site_message_impl()`
- Sends error responses when job conflicts are detected
- Updated SIMI and SIMICE services to use new method signature

### 2. Remote Client Changes (`remote/app/`)

#### Connection Client (`websockets/connection_client.py` & `communication/connection_client.py`)
- **Retry delay increased**: From 5 seconds to 15 seconds
- **New completion wait time**: 120 seconds (2 minutes) after job completion
- **Connection state tracking**: Added `is_connection_established` and `job_completed` flags
- **Enhanced error handling**: Better detection of job conflicts and server busy states
- **New methods**:
  - `mark_job_completed()`: Mark job as finished
  - `reset_for_new_job()`: Reset state for new connection attempts

#### SIMI Client (`services/simi_client.py`)
- **Improved connection logic**: Proper handling of job conflicts and busy server states
- **Better job completion detection**: Waits for proper completion signals
- **Enhanced retry pattern**:
  - 15s retry when connection fails (no job running)
  - 2min wait after job completion before next attempt
  - 2min wait when connection lost during job execution

#### SIMICE Client (`services/simice_client.py`)
- **Similar improvements** to SIMI client
- **Protocol completion tracking**: Returns boolean from `run_simice_protocol()` to indicate success
- **Proper job lifecycle management**: Complete wait cycle after job finishes

## Communication Flow Patterns

### Pattern 1: Successful Job Execution
1. Remote attempts connection → Central accepts (no running jobs)
2. Job executes normally → Completes successfully  
3. Remote waits 2 minutes → Attempts new connection

### Pattern 2: Job Already Running
1. Remote attempts connection → Central rejects (job conflict)
2. Remote waits 2 minutes → Attempts new connection
3. Repeat until central is free

### Pattern 3: Connection Lost During Job
1. Job running normally → Connection drops unexpectedly
2. Remote detects lost connection → Waits 2 minutes (central may still be finishing)
3. Remote attempts reconnection → Central state determines response

### Pattern 4: No Jobs Available
1. Remote attempts connection → Central accepts but no job to start
2. Remote waits 15 seconds → Attempts reconnection
3. Repeat until job becomes available

## Configuration Parameters

### Remote Client Settings
- `retry_delay`: 15 seconds (connection retry when no job)
- `completion_wait_time`: 120 seconds (wait after job completion)  
- `connect_timeout`: 5 seconds (connection establishment timeout)
- `message_timeout`: 15 seconds (individual message timeout)

### Central Server Settings
- Job conflict detection via `JobStatusTracker.get_first_running_job_id()`
- WebSocket close code 4009 for job conflicts
- Error message responses for busy server state

## Testing Recommendations

### Test Scenario 1: Normal Operation
1. Start central server
2. Create and start a job via GUI
3. Connect remote site → Should connect and execute job
4. Verify 2-minute wait after completion

### Test Scenario 2: Concurrent Job Prevention
1. Start job A from central
2. Attempt remote connection for job B → Should be rejected
3. Wait for job A completion
4. Retry connection → Should succeed

### Test Scenario 3: Connection Recovery
1. Start job execution
2. Simulate network interruption
3. Verify remote waits 2 minutes before retry
4. Verify proper reconnection behavior

## Benefits

1. **Prevents job conflicts**: Only one job can run at a time
2. **Reduces server load**: Longer retry delays prevent connection spam
3. **Better fault tolerance**: Proper handling of connection losses
4. **Clear job lifecycle**: Well-defined start, execution, and completion phases
5. **Resource efficiency**: Appropriate wait times balance responsiveness with resource usage
