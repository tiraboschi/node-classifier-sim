# Scheduler Integration Summary

This document summarizes the changes made to integrate Kubernetes scheduler-based pod placement into the VM simulator.

## Overview

The simulator now uses the **real Kubernetes scheduler** to determine where virt-launcher pods (and thus VMs) are placed. This makes the simulation much more realistic and allows testing of actual scheduling behavior.

## Key Changes

### 1. VM Class Enhancement (`node.py`)

**Added fields to track pod and scheduler assignment:**

```python
@dataclass
class VM:
    id: str
    cpu_consumption: float
    memory_consumption: float
    pod_name: str = ""           # Name of the virt-launcher pod executing this VM
    scheduled_node: str = ""     # Node where the pod is scheduled (filled by scheduler)
```

**Why**: VMs need to track which pod represents them and where the scheduler placed that pod.

### 2. PodManager Refactoring (`pod_manager.py`)

**Key changes:**

- `create_pod(vm, node_name=None)`: Now creates pods **without** node assignment by default
  - The `node_name` parameter is optional and only used for forced placement
  - Returns pod name instead of PodInfo
  - Updates `vm.pod_name` automatically

- `migrate_vm_pod(vm, from_node, to_node=None)`: Now lets scheduler decide placement
  - `to_node` is optional - if None, scheduler decides
  - Deletes old pod and creates new pod (unscheduled)
  - Returns success/failure boolean

- `get_pod_node_assignment(vm_id)`: New method to query where scheduler placed a pod
  - Reads pod from Kubernetes API
  - Returns `pod.spec.node_name`

- `update_vm_node_assignments(vms)`: New method to sync VM locations with pod placements
  - Reads all pods for the given VMs
  - Updates each `vm.scheduled_node` from pod status
  - Returns statistics about scheduled/pending/missing pods

- `sync_pods_with_vms(vms)`: Renamed from `sync_pods_with_nodes`
  - Takes a list of VMs instead of node->VMs mapping
  - Creates pods for VMs without forcing node assignment
  - Scheduler handles all placement decisions

### 3. Prometheus Exporter Refactoring (`prometheus_exporter.py`)

**ExporterState.load_scenario(nodes):**

```python
def load_scenario(self, nodes):
    # Collect all VMs from all nodes
    all_vms = [vm for node in nodes for vm in node.vms]

    # Create pods without node assignment
    self.pod_manager.sync_pods_with_vms(all_vms)

    # Wait for scheduler (gives it time to assign pods)
    time.sleep(2)

    # Update VM.scheduled_node from pod status
    self.pod_manager.update_vm_node_assignments(all_vms)

    # Reorganize VMs into nodes based on scheduler's decisions
    for node in nodes:
        node.vms.clear()

    for vm in all_vms:
        if vm.scheduled_node and vm.scheduled_node in self.nodes:
            self.nodes[vm.scheduled_node].vms.append(vm)
```

**ExporterState.move_vm(vm_id, from_node, to_node):**

```python
def move_vm(self, vm_id, from_node, to_node):
    # Find VM on source node
    vm = find_vm(vm_id, from_node)

    # Migrate pod (delete old, create new without node assignment)
    self.pod_manager.migrate_vm_pod(vm, from_node, to_node=None)

    # Wait for scheduler
    time.sleep(1)

    # Read where scheduler actually placed it
    actual_node = self.pod_manager.get_pod_node_assignment(vm_id)

    # Update VM location to match scheduler's decision
    vm.scheduled_node = actual_node
    source_node.vms.remove(vm)
    self.nodes[actual_node].vms.append(vm)
```

## Pod Specification Changes

### Before (Direct Assignment)

```yaml
spec:
  nodeName: kwok-node-1  # ❌ Direct assignment - bypasses scheduler
  containers: [...]
```

### After (Scheduler-Based)

```yaml
spec:
  # ✅ No nodeName - scheduler decides placement
  containers:
  - name: compute
    resources:
      requests:
        cpu: "1"
        memory: "10Gi"
  tolerations:
  - key: kwok.x-k8s.io/node
    value: fake
    effect: NoSchedule
```

## Workflow Comparison

### Old Workflow (Direct Assignment)

```
1. Create VM in simulator
2. Assign VM to node-1 in simulator state
3. Create pod with nodeName=node-1
4. Pod appears on node-1
5. Metrics calculated from pods on node-1
```

### New Workflow (Scheduler-Based)

```
1. Create VM in simulator (no node assignment yet)
2. Create pod WITHOUT nodeName
3. ⭐ Kubernetes scheduler evaluates resources
4. ⭐ Scheduler assigns pod to node-2 (based on capacity)
5. Query pod status to find scheduler's decision
6. Update VM.scheduled_node = node-2
7. Move VM to node-2 in simulator state
8. Metrics calculated from pods on node-2
```

## Migration Workflow

### Old Approach

```
Simulator: "Move VM-1 from node-A to node-B"
  ↓
Delete pod on node-A
  ↓
Create pod with nodeName=node-B
  ↓
VM is now on node-B (forced)
```

### New Approach

```
Simulator: "Move VM-1 away from node-A"
  ↓
Delete pod on node-A
  ↓
Create pod WITHOUT nodeName
  ↓
⭐ Scheduler evaluates all nodes
  ↓
Scheduler: "Best fit is node-C"
  ↓
Read pod.spec.node_name → node-C
  ↓
VM is now on node-C (scheduler's choice)
```

**Note**: The scheduler might place the VM somewhere different from what the simulator intended!

## Benefits

1. **Realistic Scheduling**: Uses actual Kubernetes scheduler logic
2. **Resource-Aware**: Scheduler considers actual resource requests and capacity
3. **Policy Testing**: Can test different scheduling policies
4. **Validation**: Can validate simulator decisions against scheduler behavior
5. **Flexibility**: Can plug in custom schedulers
6. **True-to-Production**: Mimics real KubeVirt/VM scheduling behavior

## Potential Scheduler Behaviors to Observe

1. **Bin-packing**: Scheduler fills up nodes before moving to next one
2. **Spreading**: Scheduler distributes pods evenly across nodes
3. **Resource Conflicts**: Scheduler might reject placement if resources unavailable
4. **Pod Pending**: Pods might stay pending if no node has capacity
5. **Different from Simulator**: Scheduler might choose different node than simulator intended

## Testing Scenarios

### Scenario 1: Initial Placement

```python
# Create 10 VMs with varying resource requirements
vms = [VM(f"vm-{i}", cpu=random(0.1, 0.3), mem=random(0.2, 0.5)) for i in range(10)]

# Load into exporter (creates pods)
exporter.load_scenario(nodes_with_vms)

# Observe: Where did scheduler place each VM?
for vm in vms:
    print(f"{vm.id} scheduled to {vm.scheduled_node}")
```

### Scenario 2: Migration with Scheduler Decision

```python
# Simulator decides to move VM-5 from node-1
exporter.move_vm("vm-5", from_node="kwok-node-1", to_node="kwok-node-2")

# But scheduler might place it on node-3 instead!
vm = find_vm("vm-5")
print(f"Simulator wanted node-2, scheduler chose {vm.scheduled_node}")
```

### Scenario 3: Resource Exhaustion

```python
# Create many VMs that exceed total cluster capacity
# Some pods will remain Pending
stats = pod_manager.update_vm_node_assignments(vms)
print(f"Scheduled: {stats['scheduled']}, Pending: {stats['pending']}")
```

## Implementation Notes

### Timing Considerations

The code uses `time.sleep()` to wait for the scheduler:
- `time.sleep(2)` after initial pod creation
- `time.sleep(1)` after migration

**Why**: The scheduler runs asynchronously. We need to give it time to evaluate and assign pods.

**Alternative**: Could use Kubernetes watches to react immediately when pods are scheduled, but sleep is simpler for this simulation.

### Fallback Behavior

If Kubernetes API is not available:
- Pod manager initialization fails gracefully
- Exporter falls back to VM-based metrics (old behavior)
- Direct VM moves without pod creation

## Future Enhancements

1. **Watch API**: Use Kubernetes watch API instead of polling for pod assignments
2. **Custom Scheduler**: Test with custom scheduler that uses Prometheus metrics
3. **Scheduler Policies**: Configure different scheduling policies and compare results
4. **Descheduler Integration**: Integrate with Kubernetes descheduler for automatic rebalancing
5. **Resource Pressure**: Have scheduler avoid nodes with high pressure metrics
6. **Pod Disruption Budgets**: Test PDB behavior during migrations
7. **Node Affinity/Anti-affinity**: Add VM placement constraints and test scheduler compliance

## Files Modified

1. `node.py`: Added `pod_name` and `scheduled_node` fields to VM class
2. `pod_manager.py`: Refactored to create unscheduled pods and track scheduler assignments
3. `prometheus_exporter.py`: Updated to work with scheduler-based placement
4. `requirements.txt`: Added `kubernetes>=28.0.0` dependency
5. `README_PODS.md`: Updated documentation to explain scheduler integration

## Migration Guide

If you have existing code using the old pod manager:

### Old Code
```python
pod_manager.create_pod(vm, node_name="kwok-node-1")
```

### New Code (Scheduler-Based)
```python
# Create pod - let scheduler decide
pod_name = pod_manager.create_pod(vm)

# Wait for scheduler
time.sleep(1)

# Read scheduler's decision
actual_node = pod_manager.get_pod_node_assignment(vm.id)
vm.scheduled_node = actual_node
```

### Old Code (Migration)
```python
pod_manager.migrate_vm_pod(vm, from_node="node-1", to_node="node-2")
```

### New Code (Scheduler-Based Migration)
```python
# Migrate - let scheduler decide destination
pod_manager.migrate_vm_pod(vm, from_node="node-1", to_node=None)

# Wait and read scheduler's decision
time.sleep(1)
actual_node = pod_manager.get_pod_node_assignment(vm.id)
```
