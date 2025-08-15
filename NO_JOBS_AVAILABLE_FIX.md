# Fix for "No Jobs Available" Connection Issue

## Problem Description

When a remote site connected to the central server but no jobs were available, the central server would log:

```
📋 SIMICE: Available jobs: []
```

But then the connection would hang indefinitely. The remote site would never receive a response and would be stuck waiting.

## Root Cause

The central server's `_handle_connect` methods in both SIMI and SIMICE services had a logic gap:

1. **SIMICE Service**: When `self.job_data.keys()` was empty (no jobs), the method would just print the empty list but not send any response to the remote site
2. **SIMI Service**: When `job_id not in self.jobs`, the method would simply `return` without sending any response

This left remote sites hanging indefinitely, waiting for a response that never came.

## Solution Implemented

### 1. Enhanced Error Handling in Central Services

Both SIMI and SIMICE services now properly handle all connection scenarios:

```python
# Check if there are no jobs available at all  
if not self.jobs:  # or self.job_data for SIMICE
    error_message = create_message(
        MessageType.ERROR,
        message="No jobs are currently running. Please try again later.",
        code="NO_JOBS_AVAILABLE"
    )
    await self.manager.send_to_site(error_message, site_id)
    return

# Check if specific job doesn't exist
if job_id not in self.jobs:  # or self.job_data for SIMICE
    error_message = create_message(
        MessageType.ERROR,
        message=f"Job {job_id} is not currently running or does not exist",
        code="JOB_NOT_FOUND"
    )
    await self.manager.send_to_site(error_message, site_id)
    return
```

### 2. Specific Error Codes

The solution introduces specific error codes for different scenarios:

- **`NO_JOBS_AVAILABLE`**: No jobs are running at all
- **`JOB_NOT_FOUND`**: Specific job ID doesn't exist  
- **`MISSING_JOB_ID`**: No job_id provided in request
- **`UNAUTHORIZED_SITE`**: Site not authorized for the job

### 3. Smart Retry Logic in Remote Clients

Remote clients now handle different error codes with appropriate retry delays:

```python
if error_code == "NO_JOBS_AVAILABLE":
    # No jobs available - use normal retry delay (15 seconds)
    await asyncio.sleep(client.retry_delay)  # 15 seconds
elif error_code in ["JOB_NOT_FOUND", "MISSING_JOB_ID", "UNAUTHORIZED_SITE"]:
    # Configuration errors - wait longer (2 minutes)
    await asyncio.sleep(client.completion_wait_time)  # 120 seconds
else:
    # Default case - assume job conflict (2 minutes)
    await asyncio.sleep(client.completion_wait_time)  # 120 seconds
```

## Expected Behavior Now

### Scenario 1: No Jobs Available
```
📋 SIMICE: Available jobs: []
❌ SIMICE: No jobs available for site 224bdbc5
📤 SIMICE: Sent 'no jobs available' error to site 224bdbc5
```

Remote receives error and waits 15 seconds before retry.

### Scenario 2: Job Not Found  
```
📋 SIMICE: Available jobs: [123, 456]
❌ SIMICE: Job 999 not found for site 224bdbc5
📤 SIMICE: Sent 'job not found' error to site 224bdbc5
```

Remote receives error and waits 2 minutes before retry.

### Scenario 3: Successful Connection
```
📋 SIMICE: Available jobs: [123]
✅ SIMICE: Found job 123 for site 224bdbc5
📤 SIMICE: Sent connection confirmation to site 224bdbc5
```

Remote proceeds with job execution.

## Benefits

1. **No More Hanging Connections**: Remote sites always receive a response
2. **Appropriate Retry Delays**: Different wait times based on error type
3. **Better Debugging**: Clear error messages and codes in logs
4. **Resource Efficiency**: Avoids connection spam with smart retry logic
5. **Clear Communication**: Both sides understand what's happening

## Testing

The fix can be tested by:

1. Starting central server with no jobs
2. Attempting remote connection
3. Verifying error response is received
4. Confirming 15-second retry delay for `NO_JOBS_AVAILABLE`

This resolves the hanging connection issue you encountered and provides a much more robust communication flow.
