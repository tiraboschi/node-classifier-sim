# VM Resource Model: Allocation vs Utilization

## Overview

The simulator now uses a realistic resource model that separates **resource allocation** (what a VM has) from **resource utilization** (what a VM is actually using).

This matches how real VMs work:
- A VM might have **4 CPU cores and 8Gi memory** allocated (allocation)
- But only be using **60% of CPU and 75% of memory** at any given time (utilization)

## Resource Model

### VM Properties

Each VM has four key resource properties:

```python
@dataclass
class VM:
    # Resource allocation (what the VM has)
    cpu_cores: float = 1.0              # e.g., 2.0, 0.5, 4.0
    memory_bytes: int = 1073741824      # e.g., 4294967296 (4Gi)

    # Resource utilization (what the VM is using)
    cpu_utilization: float = 0.5        # 0.0-1.0+ (50% of allocated cores)
    memory_utilization: float = 0.8     # 0.0-1.0 (80% of allocated memory)
```

### Example

```python
vm = VM(
    id="my-vm",
    cpu_cores=4.0,           # VM has 4 CPU cores
    memory_bytes=8*1024**3,  # VM has 8Gi memory
    cpu_utilization=0.6,     # Using 60% = 2.4 cores actively
    memory_utilization=0.75  # Using 75% = 6Gi actively
)
```

## Pod Resource Requests

Virt-launcher pods request resources based on VM allocation:

- **CPU Request**: `vm.cpu_cores / 10` (10% overhead)
- **Memory Request**: `vm.memory_bytes` (1:1, no overhead)
- **No Limits**: Only requests are set, no limits

### Example

For a VM with 4 cores and 8Gi:
```yaml
resources:
  requests:
    cpu: "0.4"      # 4 / 10 = 0.4 cores
    memory: "8Gi"   # 8Gi (1:1)
  # No limits
```

**Why CPU/10?** This simulates virt-launcher overhead. The VM has 4 cores, but the pod only needs 0.4 cores to run the virtualization layer.

**Why Memory 1:1?** VM memory is directly mapped, so pod needs the full amount.

## Annotations

Pods track both allocation and utilization in annotations:

```yaml
annotations:
  # Allocation (what VM has)
  simulation.node-classifier.io/vm-cpu-cores: "4.0"
  simulation.node-classifier.io/vm-memory-bytes: "8589934592"  # 8Gi in bytes

  # Utilization (what VM is using - for simulation/metrics)
  simulation.node-classifier.io/vm-cpu-utilization: "0.6"
  simulation.node-classifier.io/vm-memory-utilization: "0.75"
```

**Why both?**
- **Allocation**: Used by Kubernetes scheduler for placement decisions
- **Utilization**: Used by Prometheus metrics to calculate actual load

## Metrics Calculation

Prometheus metrics aggregate **actual consumption** from pods:

```python
# For each pod on a node:
cpu_cores = float(annotations["vm-cpu-cores"])
cpu_util = float(annotations["vm-cpu-utilization"])

# Actual consumption = allocation * utilization
actual_cpu_used = cpu_cores * cpu_util  # e.g., 4.0 * 0.6 = 2.4 cores

# As ratio of node capacity (32 cores)
cpu_ratio = actual_cpu_used / 32.0  # e.g., 2.4 / 32 = 0.075 (7.5%)
```

### Node Capacity Assumptions

- **CPU**: 32 cores per node
- **Memory**: 128Gi per node

## VirtualMachine CR Structure

```yaml
apiVersion: simulation.node-classifier.io/v1alpha1
kind: VirtualMachine
metadata:
  name: my-vm
spec:
  resources:
    cpu: "4"        # 4 CPU cores allocated
    memory: "8Gi"   # 8Gi memory allocated
  utilization:
    cpu: "0.6"      # Using 60% of allocated CPU
    memory: "0.75"  # Using 75% of allocated memory
  running: true

status:
  phase: Running
  nodeName: kwok-node-1
  podName: virt-launcher-my-vm-xyz
  allocatedCpu: "4"
  allocatedMemory: "8Gi"
  cpuUtilization: "0.6"
  memoryUtilization: "0.75"
```

## Creating VMs

### Python

```python
from node import VM

# Method 1: Explicit values
vm = VM(
    id="web-server",
    cpu_cores=2.0,
    memory_bytes=4 * 1024**3,  # 4Gi
    cpu_utilization=0.5,
    memory_utilization=0.7
)

# Method 2: Using helper
vm = VM(
    id="database",
    cpu_cores=8.0,
    memory_bytes=16 * 1024**3,  # 16Gi
    cpu_utilization=0.8,
    memory_utilization=0.9
)

# Helper methods
print(f"Memory: {vm.memory_gi()} Gi")     # 16.0 Gi
print(f"Actual CPU used: {vm.cpu_cores * vm.cpu_utilization} cores")  # 6.4 cores
```

### YAML (VirtualMachine CR)

```yaml
apiVersion: simulation.node-classifier.io/v1alpha1
kind: VirtualMachine
metadata:
  name: database
spec:
  resources:
    cpu: "8"
    memory: "16Gi"
  utilization:
    cpu: "0.8"     # 80% utilization
    memory: "0.9"  # 90% utilization
```

## Scheduler Behavior

The Kubernetes scheduler sees **pod requests** (CPU/10, Memory 1:1):

```
VM: 4 cores, 8Gi
 ↓
Pod requests: 0.4 CPU, 8Gi
 ↓
Scheduler: "This node has 10 cores available, so 0.4 fits easily"
```

The scheduler does NOT see utilization - that's only for metrics!

## Complete Example

### 1. Create VM

```python
vm = VM(
    id="app-server",
    cpu_cores=4.0,        # Allocated 4 cores
    memory_bytes=8*1024**3,  # Allocated 8Gi
    cpu_utilization=0.7,  # Using 70% = 2.8 cores
    memory_utilization=0.6  # Using 60% = 4.8Gi
)
```

### 2. Pod Created

```yaml
# Pod spec (by pod_manager)
spec:
  containers:
  - name: compute
    resources:
      requests:
        cpu: "0.4"      # 4.0 / 10
        memory: "8192Mi"  # 8Gi
```

### 3. Pod Annotations

```yaml
annotations:
  simulation.node-classifier.io/vm-cpu-cores: "4.0"
  simulation.node-classifier.io/vm-memory-bytes: "8589934592"
  simulation.node-classifier.io/vm-cpu-utilization: "0.7"
  simulation.node-classifier.io/vm-memory-utilization: "0.6"
```

### 4. Scheduler Decision

```
Available resources on kwok-node-2:
  CPU: 31.6 cores available (out of 32)
  Memory: 120Gi available (out of 128Gi)

Pod requests 0.4 CPU and 8Gi
 ✓ Fits! Assigning to kwok-node-2
```

### 5. Metrics Calculation

```python
# Prometheus exporter reads pod annotation on kwok-node-2
actual_cpu = 4.0 * 0.7 = 2.8 cores
actual_mem = 8Gi * 0.6 = 4.8Gi

# Convert to node ratio
cpu_ratio = 2.8 / 32 = 0.0875 (8.75% of node)
mem_ratio = 4.8 / 128 = 0.0375 (3.75% of node)

# Export metrics
node_cpu_usage_ratio{node="kwok-node-2"} = 0.0875
node_memory_usage_ratio{node="kwok-node-2"} = 0.0375
```

## Migration Impact

When a VM migrates:

1. **Old pod deleted** (frees 0.4 CPU, 8Gi from scheduler's view)
2. **New pod created** (requests 0.4 CPU, 8Gi on new node)
3. **Scheduler assigns** based on available resources
4. **Metrics updated** on both nodes (old: -2.8 cores, new: +2.8 cores)

## Backward Compatibility

Old VM format is automatically converted:

```python
# Old format (still works)
{
    "id": "vm-1",
    "cpu_consumption": 0.05,     # 5% of node
    "memory_consumption": 0.08   # 8% of node
}

# Automatically converted to:
VM(
    id="vm-1",
    cpu_cores=2.0,          # Default
    memory_bytes=4*1024**3,  # Default 4Gi
    cpu_utilization=0.8,    # Scaled from 0.05 * 32 / 2.0
    memory_utilization=2.56  # Scaled from 0.08 * 128 / 4.0
)
```

## Key Differences from Old Model

| Aspect | Old Model | New Model |
|--------|-----------|-----------|
| CPU | Ratio of node (0.0-1.0) | Cores allocated (e.g., 2.0, 4.0) |
| Memory | Ratio of node (0.0-1.0) | Bytes allocated (e.g., 4Gi) |
| Utilization | Implicit in ratio | Explicit utilization field |
| Pod requests | Based on ratio * node size | CPU/10, Memory 1:1 |
| Annotations | Single consumption value | Separate allocation + utilization |
| Realism | Less realistic | Matches real VMs |

## Best Practices

### Realistic Values

```python
# ✓ Good: Realistic VM sizes
VM(cpu_cores=2.0, memory_bytes=4*1024**3)    # 2 cores, 4Gi
VM(cpu_cores=4.0, memory_bytes=8*1024**3)    # 4 cores, 8Gi
VM(cpu_cores=8.0, memory_bytes=16*1024**3)   # 8 cores, 16Gi

# ✓ Good: Varied utilization
cpu_utilization=0.5   # Idle VM
cpu_utilization=0.7   # Normal load
cpu_utilization=0.9   # High load
```

### Memory Specifications

```python
# All equivalent to 4Gi:
memory_bytes=4 * 1024**3           # 4294967296 bytes
memory_bytes=4 * 1024 * 1024 * 1024
# In YAML: "4Gi" or "4096Mi"
```

### CPU Specifications

```python
# Whole cores
cpu_cores=1.0, 2.0, 4.0, 8.0

# Fractional cores (for small VMs)
cpu_cores=0.5   # Half a core
cpu_cores=0.25  # Quarter core

# In YAML: "2" or "0.5"
```

## Troubleshooting

### Pod not being scheduled

Check if pod requests fit on any node:
```bash
kubectl describe pod <pod-name>
# Look for: "0/5 nodes are available: insufficient cpu/memory"
```

Remember: Pod requests CPU/10, not the full VM cores!

### Metrics seem wrong

Check pod annotations:
```bash
kubectl get pod <pod-name> -o yaml | grep -A 4 annotations
```

Should see all four annotation keys.

### Converting old scenarios

Use the backward compatibility in `VM.from_dict()` - it handles conversion automatically.

## Summary

- **Allocation**: What resources the VM has (cores, bytes)
- **Utilization**: What percentage the VM is using (0.0-1.0)
- **Pod Requests**: CPU/10, Memory 1:1 (for scheduler)
- **Annotations**: Track both allocation and utilization (for metrics)
- **Metrics**: Calculate actual consumption = allocation * utilization

This model is more realistic and allows proper separation between what VMs have versus what they're actually using!
