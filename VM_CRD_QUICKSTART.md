# VirtualMachine CRD Quick Start

Get started with VirtualMachine custom resources in 5 minutes!

## Quick Setup

```bash
# 1. Install the CRD
./setup_vm_crd.sh

# 2. Verify installation
kubectl get crd virtualmachines.simulation.node-classifier.io
```

## Create Your First VM

```bash
cat <<EOF | kubectl apply -f -
apiVersion: simulation.node-classifier.io/v1alpha1
kind: VirtualMachine
metadata:
  name: my-first-vm
spec:
  resources:
    cpu: "0.10"      # 10% CPU
    memory: "0.15"   # 15% memory
  running: true
EOF
```

## Check VM Status

```bash
# List all VMs
kubectl get vm

# Get details
kubectl describe vm my-first-vm

# Watch status changes
kubectl get vm -w
```

## Common Operations

### See which pod is running the VM

```bash
kubectl get vm my-first-vm -o jsonpath='{.status.podName}'
```

### See which node the VM is on

```bash
kubectl get vm my-first-vm -o jsonpath='{.status.nodeName}'
```

### Check VM phase

```bash
kubectl get vm my-first-vm -o jsonpath='{.status.phase}'
```

### List all VMs on a specific node

```bash
kubectl get vm -o json | jq '.items[] | select(.status.nodeName=="kwok-node-1") | .metadata.name'
```

## Use with Simulator

```python
from pod_manager import PodManager
from node import VM

# Pod manager automatically creates VM CRs
pod_manager = PodManager(create_vm_crs=True)

# Create a VM
vm = VM(id="test-vm", cpu_consumption=0.15, memory_consumption=0.20)

# Create pod (and VM CR)
pod_manager.create_pod(vm)

# Check the VM in Kubernetes
# kubectl get vm test-vm
```

## Status Phases

- **Pending**: VM created, waiting for pod
- **Scheduling**: Pod created, waiting for scheduler
- **Scheduled**: Pod assigned to node
- **Running**: Pod is running
- **Failed**: Something went wrong

## Cleanup

```bash
# Delete specific VM
kubectl delete vm my-first-vm

# Delete all VMs
kubectl delete vm --all
```

## Next Steps

- Read full documentation: [README_VM_CRD.md](README_VM_CRD.md)
- Try example VMs: `kubectl apply -f k8s/example-vms.yaml`
- Explore with: `kubectl get vm -o yaml`

## Troubleshooting

### "error: the server doesn't have a resource type 'virtualmachines'"

The CRD is not installed. Run:
```bash
kubectl apply -f k8s/virtualmachine-crd.yaml
```

### VM stuck in Pending

Check if the pod was created:
```bash
POD=$(kubectl get vm <vm-name> -o jsonpath='{.status.podName}')
kubectl get pod $POD
```

### Want to disable VM CR creation

```python
# In your code
pod_manager = PodManager(create_vm_crs=False)
```

## Key kubectl Commands

```bash
# Short form
kubectl get vm

# Wide output (shows node and pod)
kubectl get vm -o wide

# YAML output
kubectl get vm <name> -o yaml

# JSON output with jq
kubectl get vm -o json | jq '.items[].status'

# Watch for changes
kubectl get vm -w

# Describe (full details)
kubectl describe vm <name>

# Delete
kubectl delete vm <name>
```

## Example Output

```bash
$ kubectl get vm
NAME              PHASE      NODE          POD                            CPU     MEMORY   AGE
vm-small-1        Running    kwok-node-1   virt-launcher-vm-small-1-...   0.05    0.08     5m
vm-medium-1       Running    kwok-node-2   virt-launcher-vm-medium-1-..   0.15    0.20     5m
vm-large-1        Scheduling                virt-launcher-vm-large-1-...   0.30    0.40     10s
```

## Pro Tips

1. **Use short name**: `kubectl get vm` instead of `kubectl get virtualmachines`
2. **Watch mode is your friend**: `kubectl get vm -w` to see status changes in real-time
3. **JSONPath for scripting**: Extract specific fields with `-o jsonpath='{...}'`
4. **jq for complex queries**: Pipe `-o json` to `jq` for powerful filtering
5. **Describe for debugging**: `kubectl describe vm` shows full history and conditions
