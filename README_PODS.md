# Pod-based VM Simulation with KWOK and Kubernetes Scheduler

This extension adds realistic Kubernetes pod simulation to the node classifier simulator. Each VM is represented by a `virt-launcher-<vm-name>-<random>` pod, with resource consumption tracked via pod annotations. **The Kubernetes scheduler decides pod placement**, making this a true scheduler-aware simulation.

## Overview

The simulator now creates actual Kubernetes pods that are scheduled by the real Kubernetes scheduler. This makes the simulation highly realistic by:

1. **Creating virt-launcher pods** for each VM with naming format: `virt-launcher-<vm-id>-<random>`
2. **Using pod annotations** to store fake CPU and memory consumption values
3. **Letting the Kubernetes scheduler** decide where pods are placed on KWOK nodes
4. **Tracking VM locations** by watching where the scheduler places their pods
5. **Migrating pods** when VMs are moved (delete old pod, create new pod, let scheduler place it)
6. **Calculating Prometheus metrics** by aggregating pod annotations on each node

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Prometheus Exporter                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ExporterState                                       │    │
│  │  - Creates pods (unscheduled)                       │    │
│  │  - Watches where scheduler places pods              │    │
│  │  - Updates VM.scheduled_node from pods              │    │
│  │  - Reads pods from nodes for metrics                │    │
│  │  - Aggregates CPU/memory from pod annotations       │    │
│  │  - Exposes /metrics endpoint for Prometheus         │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ PodManager                                          │    │
│  │  - Creates virt-launcher pods (no node assignment)  │    │
│  │  - Manages pod lifecycle                            │    │
│  │  - Handles VM migrations (delete + recreate)        │    │
│  │  - Queries pod scheduling status                    │    │
│  │  - Updates VM tracking based on scheduler           │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Kubernetes Scheduler + API (KWOK)               │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │  Scheduler: Decides pod placement based on:  │          │
│  │  - Resource requests (CPU/memory)             │          │
│  │  - Node capacity                              │          │
│  │  - Current utilization                        │          │
│  └──────────────────────────────────────────────┘          │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ kwok-node-1  │  │ kwok-node-2  │  │ kwok-node-3  │     │
│  │              │  │              │  │              │     │
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │              │     │
│  │ │ virt-    │ │  │ │ virt-    │ │  │              │     │
│  │ │ launcher │ │  │ │ launcher │ │  │              │     │
│  │ │ -vm-1-*  │←─┼──┼─┼ -vm-3-*  │ │  │              │     │
│  │ └──────────┘ │  │ └──────────┘ │  │              │     │
│  │ ┌──────────┐ │  │              │  │              │     │
│  │ │ virt-    │ │  │              │  │              │     │
│  │ │ launcher │ │  │   Scheduler   │  │              │     │
│  │ │ -vm-2-*  │←─┼──┼──decides─────┼──┘              │     │
│  │ └──────────┘ │  │   placement   │                 │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Pod Structure

Each VM gets a corresponding virt-launcher pod **without node assignment** - the Kubernetes scheduler decides placement:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: virt-launcher-vm-1-a7b3c
  namespace: default
  labels:
    kubevirt.io/domain: vm-1
    vm.kubevirt.io/name: vm-1
    app: virt-launcher
  annotations:
    kubevirt.io/vm-cpu-consumption: "0.05"      # CPU consumption (0.0-1.0+)
    kubevirt.io/vm-memory-consumption: "0.08"   # Memory consumption (0.0-1.0)
    kubevirt.io/domain: vm-1
spec:
  # NO nodeName field - let the scheduler decide!
  containers:
  - name: compute
    image: quay.io/kubevirt/virt-launcher:latest
    resources:
      requests:
        cpu: "1"      # Calculated from consumption * 32 cores
        memory: "10Gi"  # Calculated from consumption * 128Gi
  tolerations:
  - key: kwok.x-k8s.io/node
    operator: Equal
    value: fake
    effect: NoSchedule
```

After creation, the Kubernetes scheduler assigns the pod to a node based on:
- Resource requests vs. node capacity
- Current node utilization
- Scheduling policies (e.g., bin-packing, spread)

## Key Components

### 1. VM Class (`node.py`)

Each VM now tracks its pod and scheduler assignment:

```python
@dataclass
class VM:
    id: str
    cpu_consumption: float
    memory_consumption: float
    pod_name: str = ""           # Name of the virt-launcher pod
    scheduled_node: str = ""     # Node assigned by scheduler (updated by watching)
```

### 2. PodManager (`pod_manager.py`)

Manages the lifecycle of virt-launcher pods:

- **`create_pod(vm, node_name=None)`**: Creates a new virt-launcher pod (scheduler assigns node unless node_name is provided)
- **`delete_pod(vm_id)`**: Deletes the pod for a VM
- **`migrate_vm_pod(vm, from_node, to_node=None)`**: Migrates a VM's pod (delete + recreate, scheduler assigns unless to_node is provided)
- **`get_pod_node_assignment(vm_id)`**: Queries where scheduler placed the pod
- **`update_vm_node_assignments(vms)`**: Updates VM.scheduled_node for all VMs by reading pod status
- **`sync_pods_with_vms(vms)`**: Synchronizes pods with VM list (creates/deletes as needed)

### 3. Enhanced Prometheus Exporter (`prometheus_exporter.py`)

Now works with scheduler-assigned pods:

- **`_calculate_node_metrics_from_pods(node_name)`**: Reads all virt-launcher pods on a node and aggregates their consumption
- **`update_node_metrics(node_name)`**: Updates Prometheus metrics by reading pods
- **`load_scenario(nodes)`**:
  1. Collects all VMs across all nodes
  2. Creates pods for all VMs (unscheduled)
  3. Waits for scheduler to assign pods
  4. Updates VM.scheduled_node from pod status
  5. Reorganizes VMs into nodes based on scheduler's decisions
- **`move_vm(vm_id, from_node, to_node)`**:
  1. Deletes old pod
  2. Creates new pod (scheduler assigns it)
  3. Reads where scheduler placed the pod
  4. Moves VM internally to match scheduler's decision

### 4. Scheduler Integration Flow

1. **Initial Placement**:
   - VMs are created with resource requirements
   - Pods are created without node assignment
   - Kubernetes scheduler evaluates available nodes
   - Scheduler assigns pods based on capacity and policies
   - Exporter reads pod assignments and updates VM.scheduled_node

2. **Migration**:
   - Exporter/Simulator decides a VM should move
   - Old pod is deleted
   - New pod is created (no node assignment)
   - Scheduler re-evaluates and assigns to best node
   - Exporter reads new assignment and updates VM location

3. **Metrics Calculation**:
   - Prometheus scrapes `/metrics` endpoint
   - Exporter lists all pods on each KWOK node
   - For each `virt-launcher-*` pod, reads annotations:
     - `kubevirt.io/vm-cpu-consumption`
     - `kubevirt.io/vm-memory-consumption`
   - Aggregates consumption across all pods on the node
   - Calculates pressure metrics from utilization
   - Exposes metrics to Prometheus

## API Endpoints

### `/metrics` (GET)
Prometheus scrape endpoint. Automatically refreshes metrics from pods before serving.

```bash
curl http://localhost:8000/metrics
```

### `/health` (GET)
Health check with pod manager status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "nodes": 5,
  "pod_manager_enabled": true
}
```

### `/refresh` (POST)
Manually refresh metrics for all nodes from pods.

```bash
curl -X POST http://localhost:8000/refresh
```

Response:
```json
{
  "status": "success",
  "refreshed": ["kwok-node-1", "kwok-node-2", "kwok-node-3"],
  "failed": [],
  "total": 3
}
```

### `/feedback` (POST)
Receive VM migration feedback. Now also migrates pods.

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "migrations": [
      {"vm_id": "vm-1", "from_node": "kwok-node-1", "to_node": "kwok-node-2"}
    ]
  }'
```

### `/scenario` (POST)
Load a new scenario. Creates pods for all VMs.

```bash
curl -X POST http://localhost:8000/scenario \
  -H "Content-Type: application/json" \
  -d @scenario.json
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The new dependency is:
- `kubernetes>=28.0.0` - Python Kubernetes client

### 2. Set Up KWOK Cluster

```bash
# Create kind cluster with KWOK
kind create cluster --name node-classifier

# Install KWOK
kubectl apply -f https://github.com/kubernetes-sigs/kwok/releases/download/v0.4.0/kwok.yaml

# Create fake KWOK nodes
kubectl apply -f k8s/kwok-nodes.yaml

# Verify nodes
kubectl get nodes
```

### 3. Configure Kubernetes Access

Ensure your kubeconfig is set up correctly:

```bash
kubectl config current-context
# Should show your KWOK-enabled cluster
```

### 4. Start Prometheus Exporter

```bash
python prometheus_exporter.py --scenario sample_scenarios.json
```

The exporter will:
1. Load the Kubernetes config
2. Initialize the pod manager
3. Load the scenario and create virt-launcher pods
4. Start serving metrics on http://localhost:8000

### 5. Verify Pod Creation

```bash
# List all virt-launcher pods
kubectl get pods -l app=virt-launcher

# Check a specific pod's annotations
kubectl get pod virt-launcher-vm-1-xxxxx -o yaml | grep -A 5 annotations
```

### 6. Start Prometheus

```bash
kubectl apply -f k8s/prometheus.yaml
```

## VM Migration Flow

When a VM is migrated, **the scheduler decides the final placement**:

1. **Simulator** detects overutilization and decides to move VM from node A
2. **Feedback API** receives migration request via `/feedback` endpoint
3. **ExporterState.move_vm()** is called:
   - Finds the VM on the source node
   - Calls **PodManager.migrate_vm_pod()** (without forcing destination)
4. **PodManager.migrate_vm_pod()** executes:
   - Deletes the old pod on node A
   - Creates a new pod **without node assignment**
   - Pod has same CPU/memory annotations but new random suffix
5. **Kubernetes Scheduler** evaluates:
   - Available resources on all nodes
   - Current utilization
   - Scheduling policies
   - **Assigns pod to best node** (might be different from simulator's intended target!)
6. **Exporter queries** pod status to find actual assignment
7. **VM.scheduled_node updated** to reflect scheduler's decision
8. **VM moved internally** to match scheduler's placement
9. **Metrics are refreshed** by reading the new pod distribution
10. **Prometheus scrapes** updated metrics on next scrape interval

**Key Insight**: The simulator suggests migrations, but the Kubernetes scheduler has final say on where VMs actually land!

## Fallback Behavior

If Kubernetes API is not available, the exporter automatically falls back to:
- Direct VM-based metric calculation (original behavior)
- Logs a warning: "Running without pod management - metrics will be calculated from VMs directly"
- All functionality continues to work, just without actual pod creation

## Troubleshooting

### Pods not being created

Check that:
1. Kubernetes config is accessible: `kubectl get nodes`
2. KWOK nodes exist: `kubectl get nodes -l type=kwok`
3. Pod manager initialized: Check logs for "Kubernetes client and pod manager initialized"

### Metrics not updating

1. Check pod annotations:
```bash
kubectl get pod <pod-name> -o jsonpath='{.metadata.annotations}'
```

2. Manually refresh metrics:
```bash
curl -X POST http://localhost:8000/refresh
```

3. Check exporter logs for errors

### Pods stuck in Pending

This is normal for KWOK pods! They won't actually run, but KWOK will report them as Running.

To verify:
```bash
kubectl get pods -o wide
# Should show nodeName as kwok-node-X
```

## Configuration

### Pod Namespace

By default, pods are created in the `default` namespace. To change:

```python
# In prometheus_exporter.py
self.pod_manager = PodManager(namespace="your-namespace", use_in_cluster_config=False)
```

### Resource Requests

Pod resource requests are calculated as:
- CPU: `vm.cpu_consumption * 32` cores (assuming 32 cores per KWOK node)
- Memory: `vm.memory_consumption * 128` Gi (assuming 128Gi per KWOK node)

Adjust in `pod_manager.py` if your KWOK nodes have different capacities.

## Metrics Exposed

All metrics remain the same as before:

- `node_cpu_usage_ratio{node="kwok-node-X"}` - CPU utilization (0.0 to 1.0+)
- `node_cpu_pressure_psi{node="kwok-node-X"}` - CPU pressure (0.0 to 1.0)
- `node_memory_usage_ratio{node="kwok-node-X"}` - Memory utilization (0.0 to 1.0)
- `node_memory_pressure_psi{node="kwok-node-X"}` - Memory pressure (0.0 to 1.0)
- `node_vm_count{node="kwok-node-X"}` - Number of VMs/pods on the node

The difference is that these metrics are now calculated by aggregating pod annotations instead of internal VM objects.

## Benefits of Scheduler-Based Pod Simulation

1. **True Scheduler Integration**: The Kubernetes scheduler makes real placement decisions based on actual resource requests and node capacity
2. **Realistic Behavior**: Simulates how VMs would actually be placed in a real KubeVirt environment
3. **Observable**: Can use `kubectl` to inspect pods, nodes, and resource allocation
4. **Testable**: Can test real controllers, operators, and custom schedulers against the simulated environment
5. **Debuggable**: Pod events and status provide additional debugging information
6. **Policy Testing**: Can test different scheduler policies (e.g., bin-packing vs. spread) and see how they affect VM distribution
7. **Extensible**: Easy to add more KubeVirt-like behavior (e.g., VMI objects, migrations CRDs)
8. **Validation**: The simulator's migration decisions can be validated against what the scheduler actually does

## Future Enhancements

Potential improvements:
- Add VirtualMachineInstance (VMI) custom resources
- Implement VirtualMachineMigration CRs for live migrations
- Add pod events to simulate migration lifecycle
- Support for pod disruption budgets
- Integration with real KubeVirt components
