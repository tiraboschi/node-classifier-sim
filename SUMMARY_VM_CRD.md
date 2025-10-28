# VirtualMachine CRD Implementation Summary

## Overview

Successfully implemented a **VirtualMachine Custom Resource Definition (CRD)** that mimics KubeVirt's VirtualMachineInstance (VMI) objects. VMs are now first-class Kubernetes resources that can be queried with `kubectl`.

## What Was Built

### 1. Custom Resource Definition (`k8s/virtualmachine-crd.yaml`)

A full-featured CRD with:
- **API Group**: `simulation.node-classifier.io/v1alpha1`
- **Resource**: `VirtualMachine` (short: `vm`, `vms`)
- **Spec**: Resource requirements (CPU/memory consumption ratios)
- **Status**: Phase, node assignment, pod reference, consumption metrics
- **Subresources**: Status subresource for proper status updates
- **Printer columns**: Custom columns for `kubectl get vm` output

### 2. VM Manager (`vm_manager.py`)

Python controller for managing VirtualMachine resources:

**Key Methods**:
- `create_vm(vm)` - Create VM custom resource
- `update_vm_status(vm_name, phase, pod_name, node_name)` - Update status
- `get_vm_status(vm_name)` - Query VM status
- `list_vms()` - List all VMs
- `delete_vm(vm_name)` - Delete VM
- `sync_vm_with_pod(vm, pod_name, node_name)` - Sync with pod status
- `sync_vms_with_pods(vms)` - Batch sync multiple VMs
- `cleanup_all_vms()` - Delete all VMs in namespace

**Features**:
- Automatic status phase management
- Condition tracking (similar to Pod conditions)
- Timestamp tracking for creation and transitions
- Annotation-based consumption tracking

### 3. Pod Manager Integration

Enhanced `pod_manager.py` to work with VM CRs:

**Changes**:
- Added `vm_manager` attribute
- New parameter: `create_vm_crs=True` to enable/disable VM CR creation
- `create_pod()` now also creates VM CR and updates status
- `delete_pod()` now also deletes VM CR
- `update_vm_node_assignments()` now updates VM CR statuses

**Lifecycle**:
```
create_pod()        → Create VM CR (Pending) → Create pod → Update VM (Scheduling)
scheduler assigns   → Update VM (Running) with node and pod info
migrate_vm_pod()    → Delete pod → Delete/recreate VM CR → Wait for scheduler
delete_pod()        → Delete pod → Delete VM CR
```

### 4. Example Resources

**`k8s/example-vms.yaml`**: 5 sample VMs:
- `vm-small-1`: Low resources (5% CPU, 8% memory)
- `vm-medium-1`: Moderate resources (15% CPU, 20% memory)
- `vm-large-1`: High resources (30% CPU, 40% memory)
- `vm-compute-intensive`: CPU-heavy (40% CPU, 15% memory)
- `vm-memory-intensive`: Memory-heavy (10% CPU, 50% memory)

### 5. Setup Script (`setup_vm_crd.sh`)

Automated installation script that:
- Checks for kubectl and cluster access
- Installs VirtualMachine CRD
- Waits for CRD to be established
- Optionally installs KWOK nodes
- Optionally creates example VMs
- Provides helpful next steps

### 6. Documentation

Three comprehensive documentation files:
- **`README_VM_CRD.md`**: Complete reference (installation, usage, API, troubleshooting)
- **`VM_CRD_QUICKSTART.md`**: 5-minute quick start guide
- **`SUMMARY_VM_CRD.md`**: This file - implementation summary

## VirtualMachine Resource Structure

```yaml
apiVersion: simulation.node-classifier.io/v1alpha1
kind: VirtualMachine
metadata:
  name: vm-1
  annotations:
    simulation.node-classifier.io/cpu-consumption: "0.15"
    simulation.node-classifier.io/memory-consumption: "0.20"
spec:
  resources:
    cpu: "0.15"
    memory: "0.20"
  running: true

status:
  phase: Running                           # Pending | Scheduling | Scheduled | Running | Failed
  nodeName: kwok-node-2                    # Where VM is running
  podName: virt-launcher-vm-1-a7b3c        # Pod executing the VM
  cpuConsumption: "0.15"
  memoryConsumption: "0.20"
  conditions: [...]                        # Phase transition history
  createdAt: "2025-01-15T10:29:45Z"
```

## Status Phase Transitions

```
Pending → Scheduling → Scheduled → Running
   ↓          ↓           ↓           ↓
   └──────────┴───────────┴───────────┴──→ Failed
```

1. **Pending**: VM CR created, no pod yet
2. **Scheduling**: Pod created, waiting for scheduler
3. **Scheduled**: Scheduler assigned node, pod not running yet
4. **Running**: Pod is running on assigned node
5. **Failed**: Pod failed or error occurred

## Integration with Existing Components

### Prometheus Exporter
- No changes required
- Works transparently with pod manager
- VMs are created automatically when loading scenarios

### Simulator
- No changes required
- VM CRs created/deleted automatically during simulation
- Status updates happen automatically

### Pod Manager
- Enhanced to create/update/delete VM CRs
- Backward compatible (can disable with `create_vm_crs=False`)

## kubectl Commands

```bash
# List VMs
kubectl get vm
kubectl get vm -o wide

# Describe VM
kubectl describe vm vm-1

# Get VM status
kubectl get vm vm-1 -o jsonpath='{.status.phase}'

# Get pod name
kubectl get vm vm-1 -o jsonpath='{.status.podName}'

# Get node name
kubectl get vm vm-1 -o jsonpath='{.status.nodeName}'

# Watch for changes
kubectl get vm -w

# Delete VM
kubectl delete vm vm-1

# Create VM
kubectl apply -f my-vm.yaml
```

## Python API

```python
from vm_manager import VMManager
from node import VM

# Initialize manager
vm_mgr = VMManager(namespace="default")

# Create VM
vm = VM(id="test-vm", cpu_consumption=0.15, memory_consumption=0.20)
vm_mgr.create_vm(vm)

# Update status
vm_mgr.update_vm_status("test-vm", "Running", pod_name="pod-xyz", node_name="node-1")

# Get status
status = vm_mgr.get_vm_status("test-vm")
print(f"Phase: {status.phase}, Node: {status.node_name}")

# List all VMs
vms = vm_mgr.list_vms()

# Delete VM
vm_mgr.delete_vm("test-vm")
```

## Example Workflow

```bash
# 1. Install CRD
./setup_vm_crd.sh

# 2. Start simulator (VMs created automatically)
python prometheus_exporter.py --scenario sample_scenarios.json

# 3. Watch VMs being created and scheduled
kubectl get vm -w

# 4. Check a specific VM
kubectl describe vm vm-1

# 5. See which node it's on
kubectl get vm vm-1 -o jsonpath='{.status.nodeName}'

# 6. Simulate migration (pod deleted/recreated, scheduler decides new node)
# ... trigger via simulator feedback API ...

# 7. Watch VM transition through phases
kubectl get vm vm-1 -w
```

## Comparison with KubeVirt

| Feature | KubeVirt VMI | Simulator VM | Status |
|---------|--------------|--------------|--------|
| Custom Resource | ✅ | ✅ | ✅ |
| Spec with resources | ✅ | ✅ | ✅ |
| Status tracking | ✅ | ✅ | ✅ |
| Pod association | ✅ | ✅ | ✅ |
| Node placement | ✅ | ✅ | ✅ |
| Conditions | ✅ | ✅ | ✅ |
| Annotations | ✅ | ✅ | ✅ |
| Printer columns | ✅ | ✅ | ✅ |
| Live migration | ✅ | ❌ | Simulated (delete+create) |
| Actual VM runtime | ✅ | ❌ | No qemu/KVM |
| Networking config | ✅ | ❌ | Not needed for simulation |
| Storage volumes | ✅ | ❌ | Not needed for simulation |
| Console/VNC | ✅ | ❌ | Not applicable |

## Benefits

1. **Kubernetes-Native**: VMs are real Kubernetes resources
2. **Observable**: Use `kubectl` to inspect VM state
3. **Queryable**: Standard Kubernetes API access
4. **Integrable**: Works with any Kubernetes tool (watches, controllers, etc.)
5. **Realistic**: Mimics KubeVirt's approach to VM management
6. **Status Tracking**: Pod execution and node placement visible in VM status
7. **Condition History**: Track phase transitions over time
8. **Familiar**: Uses standard Kubernetes patterns (CRD, status subresource, conditions)

## Files Created

```
k8s/
  ├── virtualmachine-crd.yaml       # CRD definition
  └── example-vms.yaml               # Sample VM resources

vm_manager.py                        # VM CR controller
pod_manager.py                       # Updated with VM CR support
setup_vm_crd.sh                      # Installation script

README_VM_CRD.md                     # Full documentation
VM_CRD_QUICKSTART.md                 # Quick start guide
SUMMARY_VM_CRD.md                    # This summary
```

## Testing

```bash
# Install CRD
./setup_vm_crd.sh

# Create test VMs
kubectl apply -f k8s/example-vms.yaml

# Verify VMs created
kubectl get vm

# Check details
kubectl describe vm vm-small-1

# Test status updates
python -c "
from vm_manager import VMManager
from node import VM

vm_mgr = VMManager()
vm = VM(id='test-vm', cpu_consumption=0.1, memory_consumption=0.15)
vm_mgr.create_vm(vm)
vm_mgr.update_vm_status('test-vm', 'Running', 'pod-123', 'kwok-node-1')
"

kubectl get vm test-vm -o yaml

# Cleanup
kubectl delete vm --all
```

## Next Steps

Potential enhancements:
1. **Webhooks**: Add admission/validation webhooks for VM resources
2. **Controller**: Build a proper Kubernetes controller with watches
3. **Events**: Generate Kubernetes events for VM lifecycle
4. **Finalizers**: Prevent deletion of running VMs
5. **Owner References**: Link pods to VMs via owner references
6. **Scale Subresource**: Add scale subresource for batch operations
7. **Additional Fields**: Add more KubeVirt-like fields (guest OS, interfaces, etc.)
8. **Metrics**: Expose VM metrics via custom metrics API

## Summary

✅ **Complete implementation** of VirtualMachine CRD mimicking KubeVirt
✅ **Fully integrated** with existing pod manager and simulator
✅ **Production-ready** with proper error handling and logging
✅ **Well-documented** with guides, examples, and reference docs
✅ **Easy to use** with setup script and kubectl commands
✅ **Status tracking** for pod execution and node placement
✅ **Backward compatible** can be disabled if not needed

The simulator now has full Kubernetes-native VM representation! 🎉
