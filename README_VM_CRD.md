## VirtualMachine Custom Resource

This simulator now includes a **VirtualMachine Custom Resource Definition (CRD)** that mimics KubeVirt's VirtualMachineInstance (VMI) objects in a simplified way.

## Overview

The VirtualMachine CR provides a Kubernetes-native way to represent VMs in the cluster:

- **Spec**: Defines resource requirements (CPU/memory consumption ratios)
- **Annotations**: Track detailed consumption metrics
- **Status**: Shows which pod is executing the VM and where it's scheduled
- **Lifecycle**: Automatically managed by the pod manager

This makes the simulation even more realistic by creating actual Kubernetes resources that can be queried with `kubectl`.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   VirtualMachine CR                      │
│  ┌────────────────────────────────────────────────┐    │
│  │ Spec:                                           │    │
│  │   resources:                                    │    │
│  │     cpu: "0.15"                                 │    │
│  │     memory: "0.20"                              │    │
│  │   running: true                                 │    │
│  │                                                  │    │
│  │ Status:                                          │    │
│  │   phase: Running                                │    │
│  │   nodeName: kwok-node-2                         │    │
│  │   podName: virt-launcher-vm-1-a7b3c             │    │
│  │   cpuConsumption: "0.15"                        │    │
│  │   memoryConsumption: "0.20"                     │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          ├─── Created by: VMManager
                          │
                          ├─── Status updated when:
                          │      • Pod is created
                          │      • Pod is scheduled
                          │      • Pod starts running
                          │      • VM is migrated
                          │
                          └─── Deleted when: VM is removed
```

## VirtualMachine CR Specification

### API Group & Version

- **Group**: `simulation.node-classifier.io`
- **Version**: `v1alpha1`
- **Kind**: `VirtualMachine`
- **Short names**: `vm`, `vms`

### Resource Structure

```yaml
apiVersion: simulation.node-classifier.io/v1alpha1
kind: VirtualMachine
metadata:
  name: vm-1
  namespace: default
  annotations:
    simulation.node-classifier.io/cpu-consumption: "0.15"
    simulation.node-classifier.io/memory-consumption: "0.20"
spec:
  resources:
    cpu: "0.15"         # CPU consumption ratio (0.0-1.0+)
    memory: "0.20"      # Memory consumption ratio (0.0-1.0)
  running: true         # Whether VM should be running

status:
  phase: Running         # Pending | Scheduling | Scheduled | Running | Failed
  nodeName: kwok-node-2  # Node where VM is running
  podName: virt-launcher-vm-1-a7b3c  # Pod executing the VM
  cpuConsumption: "0.15"
  memoryConsumption: "0.20"
  conditions:            # List of conditions (similar to Pod conditions)
  - type: PhaseRunning
    status: "True"
    lastTransitionTime: "2025-01-15T10:30:00Z"
    reason: TransitionedToRunning
    message: "VM transitioned to Running phase"
  createdAt: "2025-01-15T10:29:45Z"
```

### Status Phases

| Phase | Description |
|-------|-------------|
| `Pending` | VM CR created, waiting for pod creation |
| `Scheduling` | Pod created, waiting for scheduler to assign node |
| `Scheduled` | Pod assigned to node, waiting to start |
| `Running` | Pod is running on assigned node |
| `Failed` | Pod failed or VM encountered an error |

## Installation

### 1. Install the CRD

Use the provided setup script:

```bash
./setup_vm_crd.sh
```

Or manually:

```bash
# Install CRD
kubectl apply -f k8s/virtualmachine-crd.yaml

# Wait for CRD to be established
kubectl wait --for condition=established --timeout=60s \
  crd/virtualmachines.simulation.node-classifier.io

# Verify
kubectl get crd virtualmachines.simulation.node-classifier.io
```

### 2. Create Example VMs (Optional)

```bash
kubectl apply -f k8s/example-vms.yaml
```

## Usage

### Creating VMs Manually

```yaml
# small-vm.yaml
apiVersion: simulation.node-classifier.io/v1alpha1
kind: VirtualMachine
metadata:
  name: my-vm
  namespace: default
spec:
  resources:
    cpu: "0.10"
    memory: "0.15"
  running: true
```

```bash
kubectl apply -f small-vm.yaml
```

### Listing VMs

```bash
# List all VMs
kubectl get virtualmachines
kubectl get vm  # short form

# With detailed output
kubectl get vm -o wide

# Watch VM status changes
kubectl get vm -w
```

Example output:

```
NAME                PHASE      NODE          POD                              CPU     MEMORY   AGE
vm-small-1          Running    kwok-node-1   virt-launcher-vm-small-1-x7k9m   0.05    0.08     2m
vm-medium-1         Running    kwok-node-2   virt-launcher-vm-medium-1-p3q8w  0.15    0.20     2m
vm-large-1          Scheduling                virt-launcher-vm-large-1-m9r2n   0.30    0.40     5s
```

### Viewing VM Details

```bash
kubectl describe vm vm-small-1
```

Example output:

```yaml
Name:         vm-small-1
Namespace:    default
API Version:  simulation.node-classifier.io/v1alpha1
Kind:         VirtualMachine
Metadata:
  Annotations:
    simulation.node-classifier.io/cpu-consumption: 0.05
    simulation.node-classifier.io/memory-consumption: 0.08
Spec:
  Resources:
    Cpu:     0.05
    Memory:  0.08
  Running:   true
Status:
  Conditions:
    Type:                 PhasePending
    Status:               True
    Last Transition Time: 2025-01-15T10:29:45Z
    Reason:               TransitionedToPending
    Message:              VM transitioned to Pending phase
    Type:                 PhaseRunning
    Status:               True
    Last Transition Time: 2025-01-15T10:30:12Z
    Reason:               TransitionedToRunning
    Message:              VM transitioned to Running phase
  Cpu Consumption:        0.05
  Created At:             2025-01-15T10:29:45Z
  Memory Consumption:     0.08
  Node Name:              kwok-node-1
  Phase:                  Running
  Pod Name:               virt-launcher-vm-small-1-x7k9m
```

### Deleting VMs

```bash
kubectl delete vm vm-small-1
```

**Note**: Deleting a VM will also delete its associated virt-launcher pod.

## Integration with Simulator

### Automatic VM Creation

When using the simulator, VMs are automatically created as CRs:

```python
from prometheus_exporter import ExporterState
from node import Node, VM

# Create VMs
vms = [
    VM(id="vm-1", cpu_consumption=0.10, memory_consumption=0.15),
    VM(id="vm-2", cpu_consumption=0.20, memory_consumption=0.25),
]

# Load scenario - automatically creates VM CRs
exporter = ExporterState()
exporter.load_scenario([Node("kwok-node-1", vms=vms, ...)])
```

This will create:
1. VirtualMachine CR for each VM
2. virt-launcher pod for each VM
3. Status updates as scheduler assigns pods

### VM Lifecycle Management

The pod manager automatically:

1. **Creates VM CR** when creating a pod
2. **Updates VM status** when:
   - Pod is created (`Scheduling`)
   - Pod is scheduled (`Scheduled` or `Running`)
   - VM is migrated (new pod created)
3. **Deletes VM CR** when pod is deleted

### Status Tracking

```python
from pod_manager import PodManager

pod_manager = PodManager(create_vm_crs=True)

# Create VM and pod
pod_manager.create_pod(vm)

# VM CR is automatically created with status:
#   phase: Scheduling
#   podName: virt-launcher-vm-1-xxxxx

# After scheduler assigns pod
pod_manager.update_vm_node_assignments([vm])

# VM CR status is updated:
#   phase: Running
#   nodeName: kwok-node-2
#   podName: virt-launcher-vm-1-xxxxx
```

## Querying VM Status Programmatically

### Python API

```python
from vm_manager import VMManager

vm_manager = VMManager(namespace="default")

# Get VM status
status = vm_manager.get_vm_status("vm-1")
print(f"Phase: {status.phase}")
print(f"Node: {status.node_name}")
print(f"Pod: {status.pod_name}")

# List all VMs
vms = vm_manager.list_vms()
for vm in vms:
    name = vm["metadata"]["name"]
    phase = vm["status"]["phase"]
    print(f"{name}: {phase}")
```

### kubectl with JSONPath

```bash
# Get all VM names and their nodes
kubectl get vm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.nodeName}{"\n"}{end}'

# Get VMs on a specific node
kubectl get vm -o json | jq '.items[] | select(.status.nodeName=="kwok-node-1") | .metadata.name'

# Get VMs by phase
kubectl get vm -o json | jq '.items[] | select(.status.phase=="Running") | .metadata.name'

# Get CPU/memory consumption for all VMs
kubectl get vm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.cpuConsumption}{"\t"}{.status.memoryConsumption}{"\n"}{end}'
```

## Comparison with KubeVirt VMI

This simplified VirtualMachine CR mimics KubeVirt's VirtualMachineInstance in these ways:

| Feature | KubeVirt VMI | Simulator VM |
|---------|--------------|--------------|
| Custom Resource | ✅ | ✅ |
| Resource specs | CPU/Memory cores | CPU/Memory ratios |
| Status tracking | Yes, detailed | Yes, simplified |
| Pod association | virt-launcher | virt-launcher (simulated) |
| Node placement | Via scheduler | Via scheduler |
| Lifecycle management | Full | Basic |
| Live migration | Yes | Simulated (delete+create) |
| Conditions | Yes | Yes |
| Events | Yes | Via conditions |

**Simplified aspects**:
- No actual VM runtime (no qemu/KVM)
- Resource values are ratios (0.0-1.0) instead of cores/bytes
- Migration is delete+recreate instead of live migration
- No networking, storage, or device specifications
- No sub-resources like /console or /vnc

## Monitoring VMs

### Watch VM Phase Transitions

```bash
kubectl get vm -w
```

### Check VM History

```bash
# View all condition transitions
kubectl get vm vm-1 -o jsonpath='{.status.conditions[*]}' | jq .
```

### Prometheus Integration

VM status is reflected in Prometheus metrics:

```promql
# VMs per node (from pod annotations)
node_vm_count{node="kwok-node-1"}

# CPU usage per node (aggregated from VM annotations)
node_cpu_usage_ratio{node="kwok-node-1"}
```

## Example Workflows

### Create and Monitor VM Lifecycle

```bash
# 1. Create a VM
cat <<EOF | kubectl apply -f -
apiVersion: simulation.node-classifier.io/v1alpha1
kind: VirtualMachine
metadata:
  name: test-vm
spec:
  resources:
    cpu: "0.10"
    memory: "0.15"
  running: true
EOF

# 2. Watch it get scheduled
kubectl get vm test-vm -w

# 3. Check which node it landed on
kubectl get vm test-vm -o jsonpath='{.status.nodeName}'

# 4. Check the pod
POD=$(kubectl get vm test-vm -o jsonpath='{.status.podName}')
kubectl get pod $POD -o wide

# 5. View full status
kubectl describe vm test-vm

# 6. Cleanup
kubectl delete vm test-vm
```

### Simulate VM Migration

```bash
# Current state
kubectl get vm my-vm

# Trigger migration via simulator
# (This deletes old pod, creates new pod, scheduler places it)

# Watch it transition
kubectl get vm my-vm -w
# my-vm   Running     kwok-node-1   ...
# my-vm   Scheduling                ...  (pod deleted, new pod created)
# my-vm   Running     kwok-node-3   ...  (scheduler chose node-3)
```

## Troubleshooting

### VM stuck in Pending

```bash
# Check if CRD is installed
kubectl get crd virtualmachines.simulation.node-classifier.io

# Check if pod was created
kubectl get vm my-vm -o jsonpath='{.status.podName}'
kubectl get pod <pod-name>
```

### VM shows wrong status

```bash
# Manually sync VM status with pod
python -c "
from pod_manager import PodManager
from node import VM

pm = PodManager(create_vm_crs=True)
vm = VM(id='my-vm', cpu_consumption=0.1, memory_consumption=0.15, pod_name='<pod-name>')
pm.update_vm_node_assignments([vm])
"
```

### Delete all VMs

```bash
kubectl delete vm --all

# Or via Python
python -c "
from vm_manager import VMManager
vm_mgr = VMManager()
vm_mgr.cleanup_all_vms()
"
```

## Advanced Usage

### Custom Controllers

You can build custom controllers that watch VirtualMachine resources:

```python
from kubernetes import client, config, watch

config.load_kube_config()
api = client.CustomObjectsApi()
w = watch.Watch()

for event in w.stream(
    api.list_namespaced_custom_object,
    group="simulation.node-classifier.io",
    version="v1alpha1",
    namespace="default",
    plural="virtualmachines"
):
    vm = event["object"]
    event_type = event["type"]  # ADDED, MODIFIED, DELETED

    print(f"{event_type}: {vm['metadata']['name']} -> {vm['status']['phase']}")
```

### Admission Webhooks

You could add admission webhooks to:
- Validate resource values are within 0.0-1.0
- Mutate VMs to add default annotations
- Prevent deletion of running VMs

## Files

- `k8s/virtualmachine-crd.yaml` - CRD definition
- `k8s/example-vms.yaml` - Example VM manifests
- `vm_manager.py` - Python VM CR manager
- `pod_manager.py` - Integrated pod + VM management
- `setup_vm_crd.sh` - Installation script

## Next Steps

1. Install the CRD: `./setup_vm_crd.sh`
2. Start the simulator: `python prometheus_exporter.py`
3. Watch VMs: `kubectl get vm -w`
4. Explore: `kubectl describe vm <name>`
