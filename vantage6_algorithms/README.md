# Vantage6 Algorithms - MIDN Federated Imputation

**Status**: ✅ Production Ready  
**Version**: 1.0.0

Federated learning algorithms for privacy-preserving missing data imputation using the vantage6 framework.

## 🚀 Quick Start

```bash
# Build containers
./build.sh

# Test locally
python test_local.py

# Deploy to vantage6 server
# See PLAYBOOK.md for complete instructions
```

## 📚 Documentation

- **[PLAYBOOK.md](PLAYBOOK.md)** - **START HERE** - Complete guide with setup, usage, examples
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Detailed integration steps
- **[QUICK_INTEGRATION.md](QUICK_INTEGRATION.md)** - Quick 3-step deployment

## 🎯 What's Included

### Algorithms

- **SIMI**: Single Imputation for Missing Data
  - Imputes one column at a time
  - Supports continuous and binary variables
  - Generates multiple imputation datasets

- **SIMICE**: Single Imputation for Multiple Columns
  - Imputes multiple columns simultaneously
  - Iterative refinement process
  - Handles correlated missing data

### Features

✅ Privacy-preserving (only aggregated statistics shared)  
✅ Dockerized (easy deployment)  
✅ Fully tested (comprehensive test suite)  
✅ Production ready (battle-tested)  
✅ Well documented (complete playbook)  

## 📁 Structure

```
vantage6_algorithms/
├── PLAYBOOK.md              # Complete guide - START HERE
├── Dockerfile               # Unified Dockerfile
├── build.sh                 # Build script
├── Core/                    # Shared utilities
├── SIMI/                    # SIMI algorithm
├── SIMICE/                  # SIMICE algorithm
└── test_*.py               # Test suites
```

## 🔧 Prerequisites

- Docker 20.10+
- Python 3.11+ (for local testing)
- Vantage6 server (for deployment)

## 📖 Usage

### Build Containers

```bash
./build.sh
```

### Test Locally

```bash
python test_local.py
```

### Deploy to Vantage6

See **[PLAYBOOK.md](PLAYBOOK.md)** for complete deployment instructions.

## 🧪 Testing

```bash
# Unit tests
python test_local.py

# Comprehensive tests
python test_comprehensive.py
```

## 📝 Example

```python
from vantage6.client import UserClient

client = UserClient("https://your-server.com", "api-key")

# Create SIMI task
task = client.task.create(
    name="Imputation Task",
    image="simi-algorithm:latest",
    input_={
        'target_column_index': 2,
        'is_binary': False,
        'imputation_trials': 10
    },
    organizations=[org1, org2],
    database="your_database"
)
```

## 🆘 Support

- **Complete Guide**: See [PLAYBOOK.md](PLAYBOOK.md)
- **Integration Help**: See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Quick Start**: See [QUICK_INTEGRATION.md](QUICK_INTEGRATION.md)

## 📄 License

[Add your license]

---

**For complete documentation, see [PLAYBOOK.md](PLAYBOOK.md)**
