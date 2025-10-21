# Node Classifier with Prometheus Integration

Complete closed-loop testing environment for node classification algorithms with real-time metric feedback.

## Features

- **Closed-Loop Simulation**: Decisions affect metrics, which affect next decisions
- **Prometheus Integration**: Query real metrics via PromQL
- **Recording Rules**: Pre-aggregated metrics for performance
- **Dynamic Feedback**: VM migrations update metrics in real-time
- **KWOK Nodes**: Fake Kubernetes nodes for realistic testing
- **Docker & Podman Support**: Works with both container runtimes

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment (auto-detects Docker/Podman)
./setup-kind-env.sh

# 3. Run simulation
python cli_prometheus.py
```

## Documentation

| File | Description |
|------|-------------|
| [README.md](README.md) | Local simulator (GUI and CLI) documentation |
| [QUICKSTART.md](QUICKSTART.md) | Quick reference for Prometheus integration |
| [PROMETHEUS_ALGORITHMS.md](PROMETHEUS_ALGORITHMS.md) | Algorithm mapping to PromQL recording rules |
| [PODMAN_SETUP.md](PODMAN_SETUP.md) | Podman-specific instructions |

## Container Runtime Support

### Docker (Default)
```bash
./setup-kind-env.sh  # Auto-detects Docker
```

### Podman (Alternative)
```bash
# Auto-detection
./setup-kind-env.sh  # Uses Podman if Docker not available

# Force Podman
export KIND_EXPERIMENTAL_PROVIDER=podman
./setup-kind-env.sh
```

See [PODMAN_SETUP.md](PODMAN_SETUP.md) for Podman-specific configuration.

## Architecture

```
Prometheus ← scrapes ← Metrics Exporter ← receives feedback ← Simulator
     ↓                        ↓                                    ↑
     └──────── queries ───────┴────────────────────────────────────┘
                        (closed loop)
```

## Components

### Python Scripts
- **`prometheus_exporter.py`** - Synthetic metrics server with feedback API
- **`prometheus_loader.py`** - Query Prometheus, return Node objects
- **`prometheus_feedback.py`** - Send VM migrations to exporter
- **`cli_prometheus.py`** - Closed-loop orchestrator

### Kubernetes Resources
- **`k8s/kind-config.yaml`** - KIND cluster configuration
- **`k8s/kwok-nodes.yaml`** - 5 fake Kubernetes nodes
- **`k8s/prometheus.yaml`** - Prometheus deployment
- **`k8s/prometheus-rules.yaml`** - Recording rules for all 19 algorithms
- **`k8s/metrics-exporter.yaml`** - Exporter deployment

### Scripts
- **`setup-kind-env.sh`** - Automated setup (Docker/Podman aware)
- **`test-closed-loop.sh`** - Test suite

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

### Run Extended Simulation
```bash
python cli_prometheus.py --max-steps 20 --step-delay 10
```

## Access Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| Prometheus | http://localhost:9090 | Metrics & PromQL |
| Metrics Exporter | http://localhost:8000 | Synthetic metrics |
| Health Check | http://localhost:8000/health | Status |

## Testing

```bash
# Run automated tests
./test-closed-loop.sh

# Manual tests
curl http://localhost:9090/-/healthy
curl http://localhost:8000/health
python prometheus_loader.py --url http://localhost:9090
```

## Customization

### Add More Nodes
Edit `k8s/kwok-nodes.yaml` and apply:
```bash
kubectl apply -f k8s/kwok-nodes.yaml
```

### Load Custom Scenario
```bash
curl -X POST http://localhost:8000/scenario \
  -H "Content-Type: application/json" \
  -d @my-scenario.json
```

### Modify Recording Rules
Edit `k8s/prometheus-rules.yaml` and reapply:
```bash
kubectl apply -f k8s/prometheus-rules.yaml
```

## Troubleshooting

### Check Container Runtime
```bash
# Docker
docker ps

# Podman
podman ps
podman machine list  # macOS/Windows
```

### Check Cluster Status
```bash
kubectl get nodes
kubectl get pods -n monitoring
kind get clusters
```

### View Logs
```bash
# Exporter logs
kubectl logs -n monitoring deployment/metrics-exporter -f

# Prometheus logs
kubectl logs -n monitoring prometheus-prometheus-0 -f
```

### Reset Environment
```bash
kind delete cluster --name node-classifier-sim
./setup-kind-env.sh
```

## Cleanup

```bash
# Delete cluster
kind delete cluster --name node-classifier-sim

# Clean containers (Docker)
docker system prune -a

# Clean containers (Podman)
podman system prune -a
```

## Requirements

- **Container Runtime**: Docker or Podman
- **KIND**: v0.30.0+ recommended (v0.20.0+ minimum for Podman support)
- **Kubernetes**: v1.34.0 (configured automatically by KIND)
- **kubectl**: Latest stable (compatible with K8s 1.34)
- **Python**: 3.11+
- **Helm**: Optional, recommended

## Next Steps

1. Read [PROMETHEUS_ALGORITHMS.md](PROMETHEUS_ALGORITHMS.md) to understand algorithm implementation
2. Configure Podman (if needed): [PODMAN_SETUP.md](PODMAN_SETUP.md)
3. Check [QUICKSTART.md](QUICKSTART.md) for quick reference
4. Benchmark algorithms against different scenarios
5. Integrate with real Kubernetes descheduler

## Support

- **Issues**: File issues specific to this integration
- **Documentation**: Check the docs listed above
- **Podman**: See [PODMAN_SETUP.md](PODMAN_SETUP.md)
- **KIND**: https://kind.sigs.k8s.io/
- **Prometheus**: https://prometheus.io/docs/