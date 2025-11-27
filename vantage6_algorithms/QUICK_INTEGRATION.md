# Quick Integration Guide

## ✅ Containers Built!

Both containers are ready:
- `simi-algorithm:latest`
- `simice-algorithm:latest`

## 3-Step Integration

### Step 1: Push to Registry (if needed)

```bash
docker tag simi-algorithm:latest your-registry.com/simi-algorithm:v1.0.0
docker push your-registry.com/simi-algorithm:v1.0.0

docker tag simice-algorithm:latest your-registry.com/simice-algorithm:v1.0.0
docker push your-registry.com/simice-algorithm:v1.0.0
```

### Step 2: Register with Vantage6

```python
from vantage6.client import UserClient

client = UserClient("https://your-server.com", "your-api-key")

# Register SIMI
client.algorithm.create(
    name="simi",
    image="simi-algorithm:latest",  # or registry path
    collaboration_id=your_collab_id
)

# Register SIMICE
client.algorithm.create(
    name="simice",
    image="simice-algorithm:latest",
    collaboration_id=your_collab_id
)
```

### Step 3: Run Tasks

```python
# SIMI Task
task = client.task.create(
    name="SIMI Test",
    image="simi-algorithm:latest",
    input_={
        'target_column_index': 2,
        'is_binary': False,
        'imputation_trials': 10
    },
    organizations=[org1, org2],
    database="your_db"
)

# SIMICE Task
task = client.task.create(
    name="SIMICE Test",
    image="simice-algorithm:latest",
    input_={
        'target_column_indexes': [2, 4],
        'is_binary_list': [False, True],
        'imputation_trials': 5,
        'iteration_before_first_imputation': 10,
        'iteration_between_imputations': 5
    },
    organizations=[org1, org2],
    database="your_db"
)
```

## That's It!

Your algorithms are now integrated with vantage6! 🎉

See `INTEGRATION_GUIDE.md` for detailed documentation.


