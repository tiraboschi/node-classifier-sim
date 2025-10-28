#!/usr/bin/env python3
"""
Example: VirtualMachine Custom Resource Integration

Demonstrates how VirtualMachine CRs are automatically created and managed
alongside pods and how to query them.
"""

import time
import subprocess
from node import VM
from pod_manager import PodManager
from vm_manager import VMManager

def run_kubectl(cmd):
    """Run a kubectl command and return output."""
    result = subprocess.run(
        f"kubectl {cmd}",
        shell=True,
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

print("=" * 70)
print("VirtualMachine CRD Integration Example")
print("=" * 70)
print()

# Check if CRD is installed
print("🔍 Checking if VirtualMachine CRD is installed...")
crd_check = run_kubectl("get crd virtualmachines.simulation.node-classifier.io 2>/dev/null")
if not crd_check:
    print("❌ VirtualMachine CRD not found!")
    print("   Please run: ./setup_vm_crd.sh")
    exit(1)
print("✓ VirtualMachine CRD is installed")
print()

# Initialize managers
print("🔧 Initializing managers...")
vm_manager = VMManager(namespace="default", use_in_cluster_config=False)
pod_manager = PodManager(namespace="default", use_in_cluster_config=False, create_vm_crs=True)
print("✓ Managers initialized")
print()

# Create some VMs
vms = [
    VM(id="demo-vm-1", cpu_consumption=0.10, memory_consumption=0.15),
    VM(id="demo-vm-2", cpu_consumption=0.20, memory_consumption=0.25),
    VM(id="demo-vm-3", cpu_consumption=0.15, memory_consumption=0.20),
]

print(f"📝 Creating {len(vms)} VMs with pods...")
for vm in vms:
    # Create pod (also creates VM CR)
    pod_name = pod_manager.create_pod(vm)
    if pod_name:
        print(f"  ✓ Created VM CR and pod for {vm.id}")
    else:
        print(f"  ❌ Failed to create {vm.id}")

print()

# Show VMs in Kubernetes
print("📋 Listing VirtualMachine resources:")
print("-" * 70)
vm_output = run_kubectl("get vm -o wide 2>/dev/null")
if vm_output:
    print(vm_output)
else:
    print("  (No VMs found)")
print()

# Wait for scheduler
print("⏳ Waiting for Kubernetes scheduler to assign pods...")
time.sleep(3)
print()

# Update VM node assignments
print("📍 Updating VM node assignments from scheduler...")
stats = pod_manager.update_vm_node_assignments(vms)
print(f"  ✓ Scheduled: {stats['scheduled']}")
print(f"  ⏸ Pending: {stats['pending']}")
print(f"  ❌ Missing: {stats['missing']}")
print()

# Show updated VM status
print("📊 Updated VM Status:")
print("-" * 70)
for vm in vms:
    status = vm_manager.get_vm_status(vm.id)
    if status:
        print(f"  {vm.id:15} Phase: {status.phase:10} Node: {status.node_name or 'N/A':15} Pod: {status.pod_name}")
    else:
        print(f"  {vm.id:15} Status: Unknown")
print()

# Show VMs with kubectl
print("🔍 VMs via kubectl:")
print("-" * 70)
vm_output = run_kubectl("get vm 2>/dev/null")
if vm_output:
    print(vm_output)
print()

# Describe one VM
if vms:
    print(f"📖 Detailed view of {vms[0].id}:")
    print("-" * 70)
    describe_output = run_kubectl(f"describe vm {vms[0].id} 2>/dev/null")
    if describe_output:
        # Show first 30 lines
        lines = describe_output.split('\n')
        for line in lines[:30]:
            print(line)
        if len(lines) > 30:
            print(f"... ({len(lines) - 30} more lines)")
    print()

# Query VM status with JSONPath
print("🔎 Querying VM data with kubectl:")
print("-" * 70)
for vm in vms:
    pod = run_kubectl(f"get vm {vm.id} -o jsonpath='{{.status.podName}}' 2>/dev/null")
    node = run_kubectl(f"get vm {vm.id} -o jsonpath='{{.status.nodeName}}' 2>/dev/null")
    phase = run_kubectl(f"get vm {vm.id} -o jsonpath='{{.status.phase}}' 2>/dev/null")
    cpu = run_kubectl(f"get vm {vm.id} -o jsonpath='{{.status.cpuConsumption}}' 2>/dev/null")
    mem = run_kubectl(f"get vm {vm.id} -o jsonpath='{{.status.memoryConsumption}}' 2>/dev/null")

    print(f"  {vm.id}:")
    print(f"    Phase: {phase}")
    print(f"    Node: {node or 'N/A'}")
    print(f"    Pod: {pod}")
    print(f"    CPU: {cpu}, Memory: {mem}")
print()

# Demonstrate migration
if vms and vms[0].scheduled_node:
    print("🔄 Demonstrating VM migration...")
    print(f"  Current: {vms[0].id} on {vms[0].scheduled_node}")

    old_pod = vms[0].pod_name
    old_node = vms[0].scheduled_node

    # Migrate (delete old pod, create new pod, let scheduler decide)
    success = pod_manager.migrate_vm_pod(vms[0], old_node, to_node=None)

    if success:
        print("  ✓ Migration initiated (pod deleted, new pod created)")
        print("  ⏳ Waiting for scheduler...")
        time.sleep(2)

        # Update assignment
        pod_manager.update_vm_node_assignments([vms[0]])

        # Check new status
        status = vm_manager.get_vm_status(vms[0].id)
        if status:
            print(f"  ✓ Migration complete:")
            print(f"    Old: {old_pod} on {old_node}")
            print(f"    New: {status.pod_name} on {status.node_name}")
            print(f"    Phase: {status.phase}")

            # Show VM status history via conditions
            print(f"  📜 Checking phase transition history...")
            conditions_output = run_kubectl(
                f"get vm {vms[0].id} -o jsonpath='{{.status.conditions[*].type}}' 2>/dev/null"
            )
            if conditions_output:
                print(f"    Conditions: {conditions_output}")
    print()

# Show summary
print("📈 Summary:")
print("-" * 70)
vm_count = run_kubectl("get vm --no-headers 2>/dev/null | wc -l").strip()
pod_count = run_kubectl("get pods -l app=virt-launcher --no-headers 2>/dev/null | wc -l").strip()
print(f"  VirtualMachine CRs: {vm_count}")
print(f"  virt-launcher pods: {pod_count}")
print()

# Cleanup prompt
input("Press Enter to cleanup (delete VMs and pods)...")
print()

print("🧹 Cleaning up...")
cleanup_count = pod_manager.cleanup_all_pods()
print(f"  ✓ Deleted {cleanup_count} pods and VM CRs")
print()

print("=" * 70)
print("Demo Complete!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  • VirtualMachine CRs are created automatically with pods")
print("  • VM status tracks pod placement and scheduler decisions")
print("  • kubectl can query VM state directly")
print("  • VM CRs show pod name and node in their status")
print("  • Conditions track VM phase transitions over time")
print()
