#!/usr/bin/env python3
"""
Example: Scheduler-Based Pod Placement

This example demonstrates how the Kubernetes scheduler decides VM placement
instead of the simulator directly assigning VMs to nodes.
"""

import time
from node import Node, VM
from pod_manager import PodManager

# Initialize pod manager
pod_manager = PodManager(namespace="default", use_in_cluster_config=False)

print("=" * 70)
print("Kubernetes Scheduler Integration Example")
print("=" * 70)

# Create some VMs with different resource requirements
vms = [
    VM(id="vm-small-1", cpu_consumption=0.05, memory_consumption=0.08),
    VM(id="vm-small-2", cpu_consumption=0.06, memory_consumption=0.10),
    VM(id="vm-medium-1", cpu_consumption=0.15, memory_consumption=0.20),
    VM(id="vm-medium-2", cpu_consumption=0.18, memory_consumption=0.22),
    VM(id="vm-large-1", cpu_consumption=0.30, memory_consumption=0.40),
]

print(f"\n📋 Created {len(vms)} VMs with varying resource requirements:")
for vm in vms:
    print(f"  • {vm.id}: CPU={vm.cpu_consumption:.2%}, Memory={vm.memory_consumption:.2%}")

# Create pods for all VMs (without node assignment)
print("\n🚀 Creating virt-launcher pods (scheduler will assign nodes)...")
stats = pod_manager.sync_pods_with_vms(vms)
print(f"  ✓ Created {stats['created']} pods")

# Wait for scheduler to assign pods
print("\n⏳ Waiting for Kubernetes scheduler to assign pods...")
time.sleep(3)

# Update VM node assignments from scheduler decisions
print("\n📍 Reading scheduler's placement decisions...")
assign_stats = pod_manager.update_vm_node_assignments(vms)
print(f"  ✓ Scheduled: {assign_stats['scheduled']}")
print(f"  ⏸ Pending: {assign_stats['pending']}")
print(f"  ❌ Missing: {assign_stats['missing']}")

# Show where each VM was placed
print("\n📊 Final VM Placement (decided by scheduler):")
print("-" * 70)
for vm in vms:
    if vm.scheduled_node:
        print(f"  {vm.id:20} → {vm.scheduled_node:15} (pod: {vm.pod_name})")
    else:
        print(f"  {vm.id:20} → PENDING (pod: {vm.pod_name})")

# Demonstrate migration
if vms[0].scheduled_node:
    print("\n🔄 Demonstrating VM Migration...")
    print(f"  Moving {vms[0].id} from {vms[0].scheduled_node}")

    old_node = vms[0].scheduled_node
    old_pod = vms[0].pod_name

    # Migrate pod (delete old, create new, let scheduler decide)
    success = pod_manager.migrate_vm_pod(
        vms[0],
        from_node=old_node,
        to_node=None  # Let scheduler decide!
    )

    if success:
        # Wait for scheduler
        print("  ⏳ Waiting for scheduler to place new pod...")
        time.sleep(2)

        # Read scheduler's decision
        new_node = pod_manager.get_pod_node_assignment(vms[0].id)
        vms[0].scheduled_node = new_node

        print(f"  ✓ Migration complete:")
        print(f"    Old pod: {old_pod} on {old_node}")
        print(f"    New pod: {vms[0].pod_name} on {new_node}")

        if new_node != old_node:
            print(f"    ℹ️  Scheduler chose a different node!")
        else:
            print(f"    ℹ️  Scheduler chose the same node")

# Show resource distribution across nodes
print("\n📈 Resource Distribution by Node:")
print("-" * 70)

node_resources = {}
for vm in vms:
    if vm.scheduled_node:
        if vm.scheduled_node not in node_resources:
            node_resources[vm.scheduled_node] = {"cpu": 0.0, "memory": 0.0, "count": 0}
        node_resources[vm.scheduled_node]["cpu"] += vm.cpu_consumption
        node_resources[vm.scheduled_node]["memory"] += vm.memory_consumption
        node_resources[vm.scheduled_node]["count"] += 1

for node, resources in sorted(node_resources.items()):
    print(f"  {node}:")
    print(f"    VMs: {resources['count']}")
    print(f"    Total CPU: {resources['cpu']:.2%}")
    print(f"    Total Memory: {resources['memory']:.2%}")

print("\n" + "=" * 70)
print("Key Takeaway: The Kubernetes scheduler made all placement decisions!")
print("=" * 70)

# Cleanup
print("\n🧹 Cleaning up pods...")
cleanup_count = pod_manager.cleanup_all_pods()
print(f"  ✓ Deleted {cleanup_count} pods")
print("\nDone!")
