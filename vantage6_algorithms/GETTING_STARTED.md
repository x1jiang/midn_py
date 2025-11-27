# Getting Started - 5 Minutes

## Step 1: Build Containers (2 minutes)

```bash
cd vantage6_algorithms
./build.sh
```

Wait for: `✓ SIMI build successful` and `✓ SIMICE build successful`

## Step 2: Test Locally (1 minute)

```bash
python test_local.py
```

Wait for: `🎉 All tests passed!`

## Step 3: Verify Containers (30 seconds)

```bash
docker images | grep -E "(simi|simice)-algorithm"
```

You should see both images listed.

## Step 4: Read Documentation (2 minutes)

- **[PLAYBOOK.md](PLAYBOOK.md)** - Complete guide with everything you need
- **[QUICK_INTEGRATION.md](QUICK_INTEGRATION.md)** - Quick deployment steps

## That's It! 🎉

Your algorithms are ready to deploy to vantage6!

### Next Steps

1. **For Complete Guide**: Read [PLAYBOOK.md](PLAYBOOK.md)
2. **For Quick Deployment**: See [QUICK_INTEGRATION.md](QUICK_INTEGRATION.md)
3. **For Integration Details**: See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)

### Quick Reference

```bash
# Build
./build.sh

# Test
python test_local.py

# Deploy (see QUICK_INTEGRATION.md)
# 1. Push to registry
# 2. Register with vantage6
# 3. Create tasks
```

---

**Questions?** See [PLAYBOOK.md](PLAYBOOK.md) for complete documentation.
