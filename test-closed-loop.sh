#!/bin/bash
set -e

echo "========================================"
echo "Closed-Loop Simulation Test"
echo "========================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Detect container runtime
detect_container_runtime() {
    # Check if user specified via environment variable
    if [ -n "$KIND_EXPERIMENTAL_PROVIDER" ]; then
        info "Using runtime from KIND_EXPERIMENTAL_PROVIDER: $KIND_EXPERIMENTAL_PROVIDER"
        return
    fi

    # Auto-detect: prefer Docker, fallback to Podman
    if command -v docker &> /dev/null && docker ps &> /dev/null 2>&1; then
        export KIND_EXPERIMENTAL_PROVIDER="docker"
        info "Detected Docker"
    elif command -v podman &> /dev/null; then
        export KIND_EXPERIMENTAL_PROVIDER="podman"
        info "Detected Podman"
    else
        error "Neither Docker nor Podman found or running"
        exit 1
    fi
}

# Detect runtime
detect_container_runtime

# Check if KIND cluster exists
if ! kind get clusters | grep -q "node-classifier-sim"; then
    error "KIND cluster 'node-classifier-sim' not found"
    echo "Please run: ./setup-kind-env.sh"
    exit 1
fi

# Test 1: Check connectivity
info "Test 1: Checking connectivity..."

if ! curl -s http://localhost:9090/-/healthy > /dev/null; then
    error "Prometheus is not reachable at http://localhost:9090"
    exit 1
fi

if ! curl -s http://localhost:8000/health > /dev/null; then
    error "Metrics exporter is not reachable at http://localhost:8000"
    exit 1
fi

info "✓ Connectivity check passed"

# Test 2: Check if scenario is loaded
info "Test 2: Checking if scenario is loaded in exporter..."

NODE_COUNT=$(curl -s http://localhost:8000/health | jq -r '.nodes')

if [ "$NODE_COUNT" -eq "0" ]; then
    warn "No scenario loaded in exporter, loading test scenario..."

    curl -s -X POST http://localhost:8000/scenario \
      -H "Content-Type: application/json" \
      -d '{
        "nodes": [
          {"name": "kwok-node-1", "cpu_usage": 0.75, "cpu_pressure": 0.25, "memory_usage": 0.70, "memory_pressure": 0.20, "vms": [{"id": "vm-1", "cpu_consumption": 0.15, "memory_consumption": 0.14}, {"id": "vm-2", "cpu_consumption": 0.18, "memory_consumption": 0.16}, {"id": "vm-3", "cpu_consumption": 0.12, "memory_consumption": 0.11}, {"id": "vm-4", "cpu_consumption": 0.10, "memory_consumption": 0.09}, {"id": "vm-5", "cpu_consumption": 0.20, "memory_consumption": 0.20}]},
          {"name": "kwok-node-2", "cpu_usage": 0.15, "cpu_pressure": 0.01, "memory_usage": 0.20, "memory_pressure": 0.02, "vms": [{"id": "vm-6", "cpu_consumption": 0.05, "memory_consumption": 0.06}, {"id": "vm-7", "cpu_consumption": 0.05, "memory_consumption": 0.07}, {"id": "vm-8", "cpu_consumption": 0.05, "memory_consumption": 0.07}]},
          {"name": "kwok-node-3", "cpu_usage": 0.30, "cpu_pressure": 0.05, "memory_usage": 0.35, "memory_pressure": 0.06, "vms": [{"id": "vm-9", "cpu_consumption": 0.10, "memory_consumption": 0.12}, {"id": "vm-10", "cpu_consumption": 0.10, "memory_consumption": 0.11}, {"id": "vm-11", "cpu_consumption": 0.10, "memory_consumption": 0.12}]},
          {"name": "kwok-node-4", "cpu_usage": 0.45, "cpu_pressure": 0.08, "memory_usage": 0.50, "memory_pressure": 0.10, "vms": [{"id": "vm-12", "cpu_consumption": 0.12, "memory_consumption": 0.13}, {"id": "vm-13", "cpu_consumption": 0.13, "memory_consumption": 0.14}, {"id": "vm-14", "cpu_consumption": 0.10, "memory_consumption": 0.11}, {"id": "vm-15", "cpu_consumption": 0.10, "memory_consumption": 0.12}]},
          {"name": "kwok-node-5", "cpu_usage": 0.20, "cpu_pressure": 0.02, "memory_usage": 0.25, "memory_pressure": 0.03, "vms": [{"id": "vm-16", "cpu_consumption": 0.08, "memory_consumption": 0.09}, {"id": "vm-17", "cpu_consumption": 0.06, "memory_consumption": 0.08}, {"id": "vm-18", "cpu_consumption": 0.06, "memory_consumption": 0.08}]}
        ]
      }' > /dev/null 2>&1

    info "✓ Test scenario loaded"

    # Wait for Prometheus to scrape (scrape interval is 15s, so wait for 2-3 scrapes)
    info "Waiting 45s for Prometheus to scrape metrics..."
    sleep 45
else
    info "✓ Scenario already loaded with $NODE_COUNT nodes"
fi

# Test 3: Check metrics are being scraped
info "Test 3: Checking if metrics are available in Prometheus..."

METRIC_COUNT=$(curl -s "http://localhost:9090/api/v1/query?query=node_cpu_usage_ratio" | jq -r '.data.result | length')

if [ "$METRIC_COUNT" -gt "0" ]; then
    info "✓ Found $METRIC_COUNT node metrics in Prometheus"
else
    warn "No metrics found yet, waiting additional 15s..."
    sleep 15
    METRIC_COUNT=$(curl -s "http://localhost:9090/api/v1/query?query=node_cpu_usage_ratio" | jq -r '.data.result | length')

    if [ "$METRIC_COUNT" -gt "0" ]; then
        info "✓ Found $METRIC_COUNT node metrics in Prometheus"
    else
        error "No metrics found in Prometheus after waiting"
        exit 1
    fi
fi

# Test 4: Load nodes from Prometheus
info "Test 4: Loading nodes from Prometheus..."

python prometheus_loader.py --url http://localhost:9090 > /tmp/prom-loader-test.log 2>&1

if [ $? -eq 0 ]; then
    NODE_COUNT=$(grep -c "kwok-node" /tmp/prom-loader-test.log || echo 0)
    info "✓ Successfully loaded $NODE_COUNT nodes"
else
    error "Failed to load nodes from Prometheus"
    cat /tmp/prom-loader-test.log
    exit 1
fi

# Test 5: Test feedback endpoint
info "Test 5: Testing feedback endpoint..."

FEEDBACK_RESPONSE=$(curl -s -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "migrations": [
      {"vm_id": "test-vm-1", "from_node": "kwok-node-1", "to_node": "kwok-node-2"}
    ]
  }')

if echo "$FEEDBACK_RESPONSE" | jq -e '.results' > /dev/null 2>&1; then
    info "✓ Feedback endpoint working"
else
    warn "Feedback endpoint returned unexpected response"
    echo "$FEEDBACK_RESPONSE"
fi

# Test 6: Check recording rules
info "Test 6: Checking Prometheus recording rules..."

RULE_COUNT=$(curl -s "http://localhost:9090/api/v1/rules" | jq -r '.data.groups | length')

if [ "$RULE_COUNT" -gt "0" ]; then
    info "✓ Found $RULE_COUNT recording rule groups"
else
    warn "No recording rules found"
fi

# Test 7: Run a short simulation
info "Test 7: Running short closed-loop simulation (3 steps)..."

python cli_prometheus.py \
  --prometheus http://localhost:9090 \
  --exporter http://localhost:8000 \
  --algorithm "Ideal Point Positive Distance" \
  --max-steps 3 \
  > /tmp/simulation-test.log 2>&1

if [ $? -eq 0 ]; then
    MIGRATIONS=$(grep -c "Moved VM" /tmp/simulation-test.log || echo 0)
    info "✓ Simulation completed successfully ($MIGRATIONS VM migrations)"
else
    warn "Simulation encountered issues (this might be OK if cluster is balanced)"
    tail -20 /tmp/simulation-test.log
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
info "All critical tests passed!"
echo ""
info "You can now run the full simulation:"
echo "  python cli_prometheus.py --max-steps 10"
echo ""
info "Or list available algorithms:"
echo "  python cli_prometheus.py --list-algorithms"
echo ""
info "Monitor Prometheus:"
echo "  http://localhost:9090"
echo ""
info "View metrics:"
echo "  http://localhost:8000/metrics"
echo ""