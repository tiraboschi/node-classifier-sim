# Quick Start Guide - Closed-Loop Simulation

## TL;DR

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up KIND cluster with Prometheus
./setup-kind-env.sh

# 3. Test the setup
./test-closed-loop.sh

# 4. Run closed-loop simulation
python cli_prometheus.py
```

## What You Get

A complete closed-loop testing environment with:
- **5 KWOK nodes** (fake Kubernetes nodes)
- **VirtualMachine CRD** (KubeVirt-like VM resources)
- **Prometheus** with recording rules
- **Synthetic metrics exporter** with dynamic feedback
- **Node classifier simulator** integrated with Prometheus

## Architecture Flow

```
1. Metrics Exporter generates synthetic node metrics
   ↓
2. Prometheus scrapes metrics every 15s
   ↓
3. Simulator queries Prometheus for current state
   ↓
4. Simulator classifies nodes and decides VM migrations
   ↓
5. Simulator sends migration feedback to exporter
   ↓
6. Exporter updates internal state and metrics
   ↓
7. Go to step 2 (closed loop!)
```

## Key Files

| File | Purpose |
|------|---------|
| `prometheus_exporter.py` | Flask app exposing metrics + feedback API |
| `prometheus_loader.py` | Queries Prometheus, returns Node objects |
| `prometheus_feedback.py` | Sends VM migrations to exporter |
| `cli_prometheus.py` | Closed-loop simulation orchestrator |
| `setup-kind-env.sh` | Automated KIND cluster setup |
| `test-closed-loop.sh` | Test suite |

## Usage Examples

### List Available Algorithms

```bash
python cli_prometheus.py --list-algorithms
```

### Run with Specific Algorithm

```bash
python cli_prometheus.py --algorithm "Ideal Point Positive Distance"
```

### Use Recording Rules

```bash
python cli_prometheus.py --recording-rules
```

### Run Long Simulation

```bash
python cli_prometheus.py --max-steps 20 --step-delay 10
```

### Load Custom Scenario

```bash
# Create scenario JSON
cat > my-scenario.json <<EOF
{
  "nodes": [
    {
      "name": "kwok-node-1",
      "cpu_usage": 0.9,
      "cpu_pressure": 0.4,
      "memory_usage": 0.8,
      "memory_pressure": 0.3,
      "vms": []
    }
  ]
}
EOF

# Load it
curl -X POST http://localhost:8000/scenario \
  -H "Content-Type: application/json" \
  -d @my-scenario.json

# Run simulation
python cli_prometheus.py
```

## Accessing Services

| Service | URL | Purpose |
|---------|-----|---------|
| Prometheus | http://localhost:9090 | Query metrics, view recording rules |
| Metrics Exporter | http://localhost:8000/metrics | View raw metrics |
| Exporter Health | http://localhost:8000/health | Check exporter status |
| Exporter State | http://localhost:8000/scenario | View current node state |

## Working with VirtualMachines

The VirtualMachine CRD is automatically installed by `setup-kind-env.sh`:

```bash
# List VMs
kubectl get vm

# Create example VMs
kubectl apply -f k8s/example-vms.yaml

# Describe a VM
kubectl describe vm vm-small-1

# View VM details
kubectl get vm -o wide

# Watch VM status changes
kubectl get vm -w
```

VMs show:
- **Allocated resources** (CPU cores, memory)
- **Utilization** (what VM is actually using)
- **Pod name** (virt-launcher pod executing the VM)
- **Node** (where scheduler placed it)

See [README_VM_CRD.md](README_VM_CRD.md) and [RESOURCE_MODEL.md](RESOURCE_MODEL.md) for details.

## Useful PromQL Queries

Visit http://localhost:9090/graph and try:

```promql
# Current CPU usage by node
descheduler:node:cpu_usage:ratio

# Nodes with high pressure
descheduler:node:cpu_pressure:psi > 0.5 or descheduler:node:memory_pressure:psi > 0.5

# Cluster average CPU
descheduler:cluster:cpu_usage:avg

# Algorithm scores (example: Ideal Point Positive Distance)
descheduler:node:ideal_point_positive_distance:score

# Linear Amplified IPPD (k=3.0)
descheduler:node:linear_amplified_ippd_k3:score
```

## Troubleshooting

### Metrics not showing up

```bash
# Check exporter is running
kubectl get pods -n monitoring

# Check exporter logs
kubectl logs -n monitoring -l app=metrics-exporter

# Check Prometheus targets
# Visit: http://localhost:9090/targets
```

### Simulation fails to connect

```bash
# Test connectivity
curl http://localhost:9090/-/healthy
curl http://localhost:8000/health

# Port forward manually if needed
kubectl port-forward -n monitoring svc/prometheus 9090:9090
kubectl port-forward -n monitoring svc/metrics-exporter 8000:8000
```

### Recording rules not working

```bash
# Check PrometheusRule
kubectl get prometheusrules -n monitoring

# View in Prometheus UI
# Visit: http://localhost:9090/rules
```

## Cleanup

```bash
# Delete KIND cluster
kind delete cluster --name node-classifier-sim
```

## Next Steps

1. **Benchmark algorithms**: Run different algorithms on the same scenario
2. **Add more nodes**: Edit `k8s/kwok-nodes.yaml`
3. **Custom metrics**: Modify `prometheus_exporter.py`
4. **Grafana dashboards**: Add Grafana to visualize results
5. **Real descheduler**: Integrate with actual K8s descheduler

## Full Documentation

- [README_PROMETHEUS.md](README_PROMETHEUS.md) - Complete Prometheus integration guide
- [PROMETHEUS_ALGORITHMS.md](PROMETHEUS_ALGORITHMS.md) - Algorithm implementation in PromQL
- [PODMAN_SETUP.md](PODMAN_SETUP.md) - Podman-specific setup
- [README.md](README.md) - Local simulator (GUI) documentation