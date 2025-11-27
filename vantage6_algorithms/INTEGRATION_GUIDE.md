# Vantage6 Integration Guide

Complete guide for integrating and deploying the MIDN algorithms with vantage6.

## ✅ Containers Built Successfully

Both algorithm containers have been built and are ready for deployment:

- `simi-algorithm:latest` - Single Imputation algorithm
- `simice-algorithm:latest` - Multiple Column Imputation algorithm

## Quick Start

### 1. Verify Containers

```bash
# List images
docker images | grep -E "(simi|simice)-algorithm"

# Test SIMI container
docker run --rm simi-algorithm:latest python -c "from algorithm import master_simi; print('OK')"

# Test SIMICE container
docker run --rm simice-algorithm:latest python -c "from algorithm import master_simice; print('OK')"
```

### 2. Push to Container Registry (Optional)

If using a remote registry:

```bash
# Tag images
docker tag simi-algorithm:latest your-registry.com/simi-algorithm:v1.0.0
docker tag simice-algorithm:latest your-registry.com/simice-algorithm:v1.0.0

# Push images
docker push your-registry.com/simi-algorithm:v1.0.0
docker push your-registry.com/simice-algorithm:v1.0.0
```

### 3. Register with Vantage6 Server

```python
from vantage6.client import UserClient

# Connect to your vantage6 server
client = UserClient(
    server_url="https://your-vantage6-server.com",
    api_key="your-api-key"
)

# Register SIMI algorithm
simi_algorithm = client.algorithm.create(
    name="simi",
    image="simi-algorithm:latest",  # or registry path
    description="Single Imputation for Missing Data",
    version="1.0.0",
    collaboration_id=your_collaboration_id  # Required
)

print(f"SIMI algorithm registered: {simi_algorithm['id']}")

# Register SIMICE algorithm
simice_algorithm = client.algorithm.create(
    name="simice",
    image="simice-algorithm:latest",  # or registry path
    description="Single Imputation for Multiple Columns",
    version="1.0.0",
    collaboration_id=your_collaboration_id  # Required
)

print(f"SIMICE algorithm registered: {simice_algorithm['id']}")
```

### 4. Create and Run Tasks

#### SIMI Task Example

```python
# Create a SIMI task
task = client.task.create(
    name="SIMI Imputation - Column 2",
    image="simi-algorithm:latest",
    input_={
        'target_column_index': 2,  # 1-based index
        'is_binary': False,        # False for continuous, True for binary
        'imputation_trials': 10    # Number of imputation datasets to generate
    },
    organizations=[org1_id, org2_id, org3_id],  # List of organization IDs
    database="your_database_name"  # Database name at each node
)

print(f"Task created: {task['id']}")

# Monitor task status
import time
while True:
    result = client.task.get(task['id'])
    status = result.get('status')
    print(f"Status: {status}")
    
    if status in ['completed', 'failed']:
        break
    
    time.sleep(5)

# Get results
if result.get('status') == 'completed':
    output = result.get('result', {})
    imputed_datasets = output.get('imputed_datasets', [])
    print(f"Generated {len(imputed_datasets)} imputed datasets")
```

#### SIMICE Task Example

```python
# Create a SIMICE task
task = client.task.create(
    name="SIMICE Imputation - Multiple Columns",
    image="simice-algorithm:latest",
    input_={
        'target_column_indexes': [2, 4, 5],  # 1-based indices
        'is_binary_list': [False, True, False],  # One flag per column
        'imputation_trials': 5,
        'iteration_before_first_imputation': 10,
        'iteration_between_imputations': 5
    },
    organizations=[org1_id, org2_id],
    database="your_database_name"
)

print(f"Task created: {task['id']}")
```

## Algorithm Parameters

### SIMI Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `target_column_index` | int | 1-based index of column to impute | Yes |
| `is_binary` | bool | True if target is binary (0/1), False for continuous | Yes |
| `imputation_trials` | int | Number of imputed datasets to generate | Yes |

### SIMICE Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `target_column_indexes` | list[int] | 1-based indices of columns to impute | Yes |
| `is_binary_list` | list[bool] | Binary flags for each column (same length as indexes) | Yes |
| `imputation_trials` | int | Number of imputed datasets to generate | Yes |
| `iteration_before_first_imputation` | int | Iterations before first imputation | Yes |
| `iteration_between_imputations` | int | Iterations between imputations | Yes |

## Container Structure

### SIMI Container

```
/app/
├── algorithm.py      # Main algorithm logic
├── wrapper.py        # Vantage6 entry point
├── Core/            # Shared utilities
│   ├── LS.py
│   ├── Logit.py
│   └── ...
└── requirements.txt
```

### SIMICE Container

```
/app/
├── algorithm.py      # Main algorithm logic
├── wrapper.py        # Vantage6 entry point
├── Core/            # Shared utilities
│   ├── LS.py
│   ├── Logit.py
│   └── ...
└── requirements.txt
```

## How It Works

### Execution Flow

1. **Task Creation**: User creates a task via vantage6 API
2. **Container Launch**: Vantage6 launches algorithm containers at each node
3. **Master Execution**: Central node executes `master_*()` function
4. **RPC Calls**: Master function calls remote functions via `client.create_new_task()`
5. **Remote Execution**: Each node executes `RPC_*()` functions on local data
6. **Result Aggregation**: Master function aggregates results from all nodes
7. **Imputation**: Master function generates imputed datasets
8. **Return Results**: Results returned via vantage6 API

### Communication Pattern

```
Central Node (Master)
    ↓ create_new_task()
Remote Node 1 → Statistics → Aggregate
Remote Node 2 → Statistics → Aggregate
Remote Node 3 → Statistics → Aggregate
    ↓
Final Results
```

## Troubleshooting

### Container Build Issues

**Problem**: Build fails with "file not found"
- **Solution**: Ensure you're running `build.sh` from the correct directory
- **Check**: Verify `Core/` and algorithm directories exist

**Problem**: Import errors in container
- **Solution**: Check that all dependencies are in `requirements.txt`
- **Verify**: Run `docker run --rm <image> pip list`

### Runtime Issues

**Problem**: Task fails with "method not found"
- **Solution**: Verify wrapper.py is correctly routing method names
- **Check**: Ensure method names match between master and RPC calls

**Problem**: No results returned
- **Solution**: Check task logs: `client.task.get(task_id).get('log')`
- **Verify**: Ensure organizations have data accessible

**Problem**: Data access errors
- **Solution**: Verify database name matches at all nodes
- **Check**: Ensure nodes have proper permissions

## Testing Locally

Before deploying to production, test locally:

```python
# Use mock client (already implemented in test_local.py)
from test_local import MockVantage6Client, create_test_data

# Create test data
central_data = create_test_data(50, 4, 0.2)
remote_data = [create_test_data(60, 4, 0.0), create_test_data(70, 4, 0.0)]

# Create mock client
client = MockVantage6Client(remote_data_sets=remote_data)

# Test algorithm
from SIMI.algorithm import master_simi
result = master_simi(
    client,
    {
        'data': central_data,
        'target_column_index': 1,
        'is_binary': False,
        'imputation_trials': 3
    }
)

print(f"Generated {len(result['imputed_datasets'])} datasets")
```

## Next Steps

1. ✅ Containers built
2. ⏭ Push to registry (if using remote registry)
3. ⏭ Register with vantage6 server
4. ⏭ Create test tasks
5. ⏭ Monitor and verify results
6. ⏭ Deploy to production

## Support

For issues or questions:
- Check `DEPLOYMENT_CHECKLIST.md` for deployment steps
- Review `TEST_RESULTS.md` for test information
- See `MIGRATION_GUIDE.md` for technical details

## References

- [Vantage6 Documentation](https://docs.vantage6.ai/en/main/)
- [Algorithm Development Guide](https://docs.vantage6.ai/en/main/algorithm-development/)
- [Docker Documentation](https://docs.docker.com/)


