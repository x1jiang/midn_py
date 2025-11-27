# MIDN - Federated Missing Data Imputation Platform

**Multi-Institutional Data Network (MIDN)** - A privacy-preserving federated learning platform for missing data imputation across healthcare institutions.

[![Status](https://img.shields.io/badge/status-active%20development-green)](https://github.com/x1jiang/midn_py)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Overview

MIDN enables multiple healthcare institutions to collaboratively impute missing data **without sharing raw patient data**. Only aggregated statistics are exchanged, ensuring patient privacy while leveraging the power of federated learning.

### Key Features

✅ **Privacy-Preserving**: Only aggregated statistics shared, never raw data  
✅ **Federated Learning**: Multiple institutions collaborate without data sharing  
✅ **Multiple Algorithms**: SIMI, SIMICE, AVGMMI, AVGMMICE, CSLMI, CSLMICE, HDMI, IMI, IMICE  
✅ **Production Ready**: FastAPI-based central/remote architecture  
✅ **Vantage6 Integration**: Algorithms adapted for vantage6 framework  

---

## 📋 Development Roadmap & Collaboration Plan

### Current Status: **Active Development** 🚧

We're working collaboratively to migrate and test the MIDN algorithms in the vantage6 framework. Here's our development workflow:

### Phase 1: GitHub Repository ✅ (Current)
- **Status**: ✅ Complete
- **Purpose**: Enable collaborative development
- **What's Done**: 
  - All algorithms migrated to vantage6 format
  - Comprehensive documentation (PLAYBOOK.md)
  - Test suites ready
  - Docker containers prepared

### Phase 2: Vantage6 Local Simulator ✅ (Complete)
- **Status**: ✅ Complete
- **Purpose**: Test algorithms locally using multiple files as "nodes"
- **Goal**: Verify code works correctly before Docker deployment
- **Collaboration**: 
  - **Ivan**: Wrapping code into vantage6 format (master/RPC functions) ✅
  - **Luyao**: Sparring partner for testing and validation ✅
- **Deliverables**:
  - ✅ Working local simulator setup (`vantage6_simulator_test/`)
  - ✅ Verified algorithm execution (SIMI and SIMICE both working)
  - ✅ Test results with dummy data
  - ✅ Mock vantage6 client for local testing

### Phase 3: Docker Image Deployment 📦 (Planned)
- **Status**: 📦 Planned
- **Purpose**: Create proper Docker images for vantage6
- **Goal**: Containerize algorithms for network deployment
- **Deliverables**:
  - Production-ready Docker images
  - Verified container builds
  - Integration tests

### Phase 4: Network Testing with Dummy Data 🌐 (Planned)
- **Status**: 🌐 Planned
- **Purpose**: Test in a real network environment
- **Goal**: Validate end-to-end federated learning workflow
- **Deliverables**:
  - Multi-node network setup
  - Dummy data test scenarios
  - Performance benchmarks

---

## 🏗️ Project Structure

```
midn_py/
├── README.md                    # This file - project overview
├── ARCHITECTURE.md              # System architecture documentation
├── DEMO.md                      # Local demo walkthrough
│
├── vantage6_algorithms/         # ⭐ Vantage6-compatible algorithms
│   ├── PLAYBOOK.md             # Complete guide (START HERE)
│   ├── GETTING_STARTED.md      # 5-minute quick start
│   ├── README.md               # Algorithm overview
│   ├── Dockerfile              # Unified Dockerfile
│   ├── build.sh                # Build script
│   ├── Core/                   # Shared utilities
│   ├── SIMI/                   # SIMI algorithm
│   ├── SIMICE/                 # SIMICE algorithm
│   └── test_*.py              # Test suites
│
├── MIDN_R_PY/                  # Original algorithm implementations
│   ├── SIMI/                   # SIMI (Python + R)
│   ├── SIMICE/                 # SIMICE (Python + R)
│   ├── AVGMMI/                 # AVGMMI algorithm
│   ├── AVGMMICE/               # AVGMMICE algorithm
│   ├── CSLMI/                  # CSLMI algorithm
│   ├── CSLMICE/                # CSLMICE algorithm
│   ├── HDMI/                   # HDMI algorithm
│   ├── IMI/                    # IMI algorithm
│   ├── IMICE/                  # IMICE algorithm
│   └── Core/                   # Core utilities
│
├── central/                    # Central FastAPI server
│   └── app/                    # Central application code
│
├── remote/                     # Remote FastAPI servers
│   └── app/                    # Remote application code
│
└── config/                     # Algorithm configuration schemas
```

---

## 🚀 Quick Start

### For Vantage6 Algorithms (Current Focus)

```bash
# Navigate to vantage6 algorithms
cd vantage6_algorithms

# Read the complete guide
cat PLAYBOOK.md

# Or quick start
cat GETTING_STARTED.md

# Build containers
./build.sh

# Test locally
python test_local.py
```

**📖 Full Documentation**: See [`vantage6_algorithms/PLAYBOOK.md`](vantage6_algorithms/PLAYBOOK.md)

### For Original FastAPI System

```bash
# Start central server
uvicorn central.app.main:app --host 0.0.0.0 --port 8000

# Start remote servers (in separate terminals)
uvicorn remote.app.main:app --host 0.0.0.0 --port 8001
uvicorn remote.app.main:app --host 0.0.0.0 --port 8002

# Access GUI
# Central: http://localhost:8000
# Remote 1: http://localhost:8001
# Remote 2: http://localhost:8002
```

**📖 Full Documentation**: See [`DEMO.md`](DEMO.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 📚 Documentation

### Vantage6 Integration (Current Focus)
- **[vantage6_algorithms/PLAYBOOK.md](vantage6_algorithms/PLAYBOOK.md)** - Complete guide with setup, usage, examples
- **[vantage6_algorithms/GETTING_STARTED.md](vantage6_algorithms/GETTING_STARTED.md)** - 5-minute quick start
- **[vantage6_algorithms/INTEGRATION_GUIDE.md](vantage6_algorithms/INTEGRATION_GUIDE.md)** - Detailed integration steps
- **[vantage6_algorithms/QUICK_INTEGRATION.md](vantage6_algorithms/QUICK_INTEGRATION.md)** - Quick deployment guide

### Original System
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and components
- **[DEMO.md](DEMO.md)** - Local demo walkthrough
- **[EXTENDING_ALGORITHMS.md](EXTENDING_ALGORITHMS.md)** - How to add new algorithms

---

## 🧪 Algorithms

### Currently Implemented for Vantage6

- **SIMI** (Single Imputation for Missing Data)
  - Imputes one column at a time
  - Supports continuous and binary variables
  - Generates multiple imputation datasets

- **SIMICE** (Single Imputation for Multiple Columns)
  - Imputes multiple columns simultaneously
  - Iterative refinement process
  - Handles correlated missing data

### Available in Original System

- **AVGMMI** / **AVGMMICE** - Average-based imputation
- **CSLMI** / **CSLMICE** - Conditional specification learning
- **HDMI** - High-dimensional imputation
- **IMI** / **IMICE** - Iterative imputation

---

## 🔧 Prerequisites

### For Vantage6 Development
- Docker 20.10+
- Python 3.11+
- Vantage6 server (for deployment)
- Vantage6 local simulator (for testing)

### For Original FastAPI System
- Python 3.11+
- FastAPI, uvicorn
- SQLAlchemy, SQLite
- See `requirements.txt` for full list

---

## 🧪 Testing

### Vantage6 Algorithms

```bash
cd vantage6_algorithms

# Unit tests with mock client
python test_local.py

# Comprehensive tests
python test_comprehensive.py

# Tests with real sample data
python test_with_real_data.py
```

### Local Simulator Testing

```bash
cd vantage6_simulator_test

# Setup test data (central + 2 remote nodes)
python3 simulator_setup.py

# Run simulator tests
python3 simulator_test.py
```

**Status**: ✅ Both SIMI and SIMICE algorithms tested and working in local simulator

### Original System

See [`DEMO.md`](DEMO.md) for testing instructions.

---

## 🤝 Contributing

### Development Workflow

1. **Fork & Clone**: Fork this repository and clone your fork
2. **Create Branch**: Create a feature branch for your work
3. **Develop**: Make your changes
4. **Test**: Run test suites to verify functionality
5. **Document**: Update relevant documentation
6. **Commit**: Commit with clear messages
7. **Push & PR**: Push to your fork and create a pull request

### Current Focus Areas

- ✅ Vantage6 algorithm migration (SIMI, SIMICE)
- 🔄 Vantage6 local simulator testing
- 📦 Docker image optimization
- 🌐 Network deployment testing
- 📝 Documentation improvements

---

## 📝 Next Steps for Team

### For Ivan (Vantage6 Integration)
1. Review `vantage6_algorithms/` structure
2. Test algorithms in vantage6 local simulator
3. Verify master/RPC function wrapping
4. Optimize Docker images
5. Coordinate with Luyao for testing

### For Luyao (Testing Partner)
1. Set up vantage6 local simulator
2. Test algorithms with dummy data
3. Validate results and provide feedback
4. Test edge cases and error handling
5. Document findings and issues

### For All Contributors
1. Review and test current implementations
2. Report issues and suggest improvements
3. Contribute to documentation
4. Share test results and feedback

---

## 📊 Status Dashboard

| Component | Status | Notes |
|-----------|--------|-------|
| SIMI Algorithm | ✅ Complete | Vantage6-ready, tested in simulator |
| SIMICE Algorithm | ✅ Complete | Vantage6-ready, tested in simulator |
| Docker Images | ✅ Complete | Unified Dockerfile |
| Test Suites | ✅ Complete | Comprehensive coverage |
| Documentation | ✅ Complete | PLAYBOOK.md + guides |
| Vantage6 Local Simulator | ✅ Complete | Mock client working, both algorithms tested |
| Network Deployment | 📦 Planned | Ready for full vantage6 testing |

---

## 🐛 Known Issues & Limitations

- Vantage6 local simulator testing in progress
- Some algorithms not yet migrated to vantage6 format
- Network deployment pending simulator validation

See [Issues](https://github.com/x1jiang/midn_py/issues) for current bug reports.

---

## 📄 License

[Add your license information here]

---

## 🙏 Acknowledgments

- Original MIDN algorithm developers
- Vantage6 framework team
- All contributors and testers

---

## 📞 Contact & Support

- **Repository**: https://github.com/x1jiang/midn_py
- **Issues**: https://github.com/x1jiang/midn_py/issues
- **Team**: Ivan (Vantage6 integration), Luyao (Testing), x1jiang (Maintainer)

---

## 🔗 Related Resources

- [Vantage6 Documentation](https://docs.vantage6.ai/)
- [Vantage6 Algorithm Development](https://docs.vantage6.ai/en/main/algorithm-development/)
- [Federated Learning Overview](https://en.wikipedia.org/wiki/Federated_learning)

---

**Last Updated**: 2025-11-20  
**Version**: 1.0.0  
**Status**: 🚧 Active Development

---

## 🎯 Quick Links

- **[Start Here](vantage6_algorithms/PLAYBOOK.md)** - Complete guide for vantage6 algorithms
- **[Quick Start](vantage6_algorithms/GETTING_STARTED.md)** - 5-minute setup guide
- **[Architecture](ARCHITECTURE.md)** - System architecture details
- **[Demo](DEMO.md)** - Local demo walkthrough

