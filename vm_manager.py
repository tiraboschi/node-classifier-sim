#!/usr/bin/env python3
"""
VirtualMachine Custom Resource Manager

Manages VirtualMachine custom resources that mimic KubeVirt's VMI objects.
Tracks VM resource consumption via annotations and pod execution via status.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timezone
import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from node import VM

logger = logging.getLogger(__name__)

# CRD details
VM_GROUP = "simulation.node-classifier.io"
VM_VERSION = "v1alpha1"
VM_PLURAL = "virtualmachines"


@dataclass
class VMStatus:
    """Status of a VirtualMachine CR."""
    phase: str  # Pending, Scheduling, Scheduled, Running, Failed
    node_name: str = ""
    pod_name: str = ""
    allocated_cpu: str = ""
    allocated_memory: str = ""
    cpu_utilization: str = ""
    memory_utilization: str = ""


class VMManager:
    """
    Manages VirtualMachine custom resources.

    VirtualMachine CRs are similar to KubeVirt's VirtualMachineInstance objects:
    - Spec contains resource requirements (CPU/memory consumption)
    - Annotations track detailed metrics
    - Status tracks which pod is executing the VM and where it's scheduled
    """

    def __init__(self, namespace: str = "default", use_in_cluster_config: bool = False):
        """
        Initialize the VM manager.

        Args:
            namespace: Kubernetes namespace for VMs
            use_in_cluster_config: If True, use in-cluster config, otherwise use kubeconfig
        """
        self.namespace = namespace
        self.vm_registry: Dict[str, VMStatus] = {}  # vm_name -> VMStatus

        # Initialize Kubernetes clients
        try:
            if use_in_cluster_config:
                config.load_incluster_config()
            else:
                config.load_kube_config()

            self.core_v1 = client.CoreV1Api()
            self.custom_api = client.CustomObjectsApi()
            logger.info("VM Manager: Kubernetes client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    def create_vm(self, vm: VM) -> bool:
        """
        Create a VirtualMachine custom resource.

        Args:
            vm: VM object with resource requirements

        Returns:
            True if successful, False otherwise
        """
        # Format memory for Kubernetes
        memory_gi = vm.memory_gi()
        if memory_gi >= 1.0:
            memory_str = f"{memory_gi:.1f}Gi"
        else:
            memory_mi = vm.memory_mi()
            memory_str = f"{int(memory_mi)}Mi"

        vm_manifest = {
            "apiVersion": f"{VM_GROUP}/{VM_VERSION}",
            "kind": "VirtualMachine",
            "metadata": {
                "name": vm.id,
                "namespace": self.namespace,
                "annotations": {
                    "simulation.node-classifier.io/vm-cpu-cores": str(vm.cpu_cores),
                    "simulation.node-classifier.io/vm-memory-bytes": str(vm.memory_bytes),
                    "simulation.node-classifier.io/vm-cpu-utilization": str(vm.cpu_utilization),
                    "simulation.node-classifier.io/vm-memory-utilization": str(vm.memory_utilization),
                }
            },
            "spec": {
                "resources": {
                    "cpu": str(vm.cpu_cores),
                    "memory": memory_str
                },
                "utilization": {
                    "cpu": str(vm.cpu_utilization),
                    "memory": str(vm.memory_utilization)
                },
                "running": True
            },
            "status": {
                "phase": "Pending",
                "allocatedCpu": str(vm.cpu_cores),
                "allocatedMemory": memory_str,
                "cpuUtilization": str(vm.cpu_utilization),
                "memoryUtilization": str(vm.memory_utilization),
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "conditions": []
            }
        }

        try:
            self.custom_api.create_namespaced_custom_object(
                group=VM_GROUP,
                version=VM_VERSION,
                namespace=self.namespace,
                plural=VM_PLURAL,
                body=vm_manifest
            )

            # Register in local state
            self.vm_registry[vm.id] = VMStatus(
                phase="Pending",
                allocated_cpu=str(vm.cpu_cores),
                allocated_memory=memory_str,
                cpu_utilization=str(vm.cpu_utilization),
                memory_utilization=str(vm.memory_utilization)
            )

            logger.info(f"Created VirtualMachine CR: {vm.id}")
            return True

        except ApiException as e:
            if e.status == 409:  # Already exists
                logger.warning(f"VirtualMachine {vm.id} already exists")
                return True
            logger.error(f"Failed to create VirtualMachine {vm.id}: {e}")
            return False

    def update_vm_status(self, vm_name: str, phase: str, pod_name: str = "", node_name: str = "") -> bool:
        """
        Update the status of a VirtualMachine CR.

        Args:
            vm_name: Name of the VM
            phase: New phase (Pending, Scheduling, Scheduled, Running, Failed)
            pod_name: Name of the virt-launcher pod (if any)
            node_name: Name of the node where VM is scheduled (if any)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read current VM
            vm_obj = self.custom_api.get_namespaced_custom_object(
                group=VM_GROUP,
                version=VM_VERSION,
                namespace=self.namespace,
                plural=VM_PLURAL,
                name=vm_name
            )

            # Update status
            if "status" not in vm_obj:
                vm_obj["status"] = {}

            vm_obj["status"]["phase"] = phase

            if pod_name:
                vm_obj["status"]["podName"] = pod_name

            if node_name:
                vm_obj["status"]["nodeName"] = node_name

            # Add condition for phase change
            now = datetime.now(timezone.utc).isoformat()
            condition = {
                "type": f"Phase{phase}",
                "status": "True",
                "lastTransitionTime": now,
                "reason": f"TransitionedTo{phase}",
                "message": f"VM transitioned to {phase} phase"
            }

            if "conditions" not in vm_obj["status"]:
                vm_obj["status"]["conditions"] = []

            vm_obj["status"]["conditions"].append(condition)

            # Update via status subresource
            self.custom_api.patch_namespaced_custom_object_status(
                group=VM_GROUP,
                version=VM_VERSION,
                namespace=self.namespace,
                plural=VM_PLURAL,
                name=vm_name,
                body=vm_obj
            )

            # Update local registry
            if vm_name in self.vm_registry:
                self.vm_registry[vm_name].phase = phase
                self.vm_registry[vm_name].pod_name = pod_name
                self.vm_registry[vm_name].node_name = node_name

            logger.info(f"Updated VM {vm_name} status: phase={phase}, pod={pod_name}, node={node_name}")
            return True

        except ApiException as e:
            logger.error(f"Failed to update VM {vm_name} status: {e}")
            return False

    def get_vm_status(self, vm_name: str) -> Optional[VMStatus]:
        """
        Get the status of a VirtualMachine CR.

        Args:
            vm_name: Name of the VM

        Returns:
            VMStatus if found, None otherwise
        """
        try:
            vm_obj = self.custom_api.get_namespaced_custom_object(
                group=VM_GROUP,
                version=VM_VERSION,
                namespace=self.namespace,
                plural=VM_PLURAL,
                name=vm_name
            )

            status = vm_obj.get("status", {})
            return VMStatus(
                phase=status.get("phase", "Unknown"),
                node_name=status.get("nodeName", ""),
                pod_name=status.get("podName", ""),
                allocated_cpu=status.get("allocatedCpu", ""),
                allocated_memory=status.get("allocatedMemory", ""),
                cpu_utilization=status.get("cpuUtilization", ""),
                memory_utilization=status.get("memoryUtilization", "")
            )

        except ApiException as e:
            logger.error(f"Failed to get VM {vm_name} status: {e}")
            return None

    def list_vms(self) -> List[Dict]:
        """
        List all VirtualMachine CRs in the namespace.

        Returns:
            List of VM objects
        """
        try:
            result = self.custom_api.list_namespaced_custom_object(
                group=VM_GROUP,
                version=VM_VERSION,
                namespace=self.namespace,
                plural=VM_PLURAL
            )
            return result.get("items", [])

        except ApiException as e:
            logger.error(f"Failed to list VMs: {e}")
            return []

    def delete_vm(self, vm_name: str) -> bool:
        """
        Delete a VirtualMachine CR.

        Args:
            vm_name: Name of the VM to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.custom_api.delete_namespaced_custom_object(
                group=VM_GROUP,
                version=VM_VERSION,
                namespace=self.namespace,
                plural=VM_PLURAL,
                name=vm_name
            )

            # Remove from registry
            if vm_name in self.vm_registry:
                del self.vm_registry[vm_name]

            logger.info(f"Deleted VirtualMachine CR: {vm_name}")
            return True

        except ApiException as e:
            logger.error(f"Failed to delete VM {vm_name}: {e}")
            return False

    def sync_vm_with_pod(self, vm: VM, pod_name: str, node_name: str) -> bool:
        """
        Synchronize a VM's status with its pod placement.

        Args:
            vm: VM object
            pod_name: Name of the virt-launcher pod
            node_name: Name of the node where pod is scheduled

        Returns:
            True if successful, False otherwise
        """
        phase = "Running" if node_name else "Scheduling"
        return self.update_vm_status(vm.id, phase, pod_name, node_name)

    def update_vm_from_pod_status(self, vm: VM) -> bool:
        """
        Update VM status by reading its pod's status.

        Args:
            vm: VM object with pod_name set

        Returns:
            True if successful, False otherwise
        """
        if not vm.pod_name:
            return self.update_vm_status(vm.id, "Pending")

        try:
            # Read pod to get current status
            pod = self.core_v1.read_namespaced_pod(
                name=vm.pod_name,
                namespace=self.namespace
            )

            node_name = pod.spec.node_name or ""

            # Determine phase based on pod phase
            if pod.status.phase == "Running":
                phase = "Running"
            elif pod.status.phase == "Pending":
                phase = "Scheduling" if not node_name else "Scheduled"
            elif pod.status.phase in ["Failed", "Unknown"]:
                phase = "Failed"
            else:
                phase = "Pending"

            return self.update_vm_status(vm.id, phase, vm.pod_name, node_name)

        except ApiException as e:
            logger.error(f"Failed to update VM {vm.id} from pod status: {e}")
            return False

    def sync_vms_with_pods(self, vms: List[VM]) -> Dict[str, int]:
        """
        Synchronize all VM statuses with their pod statuses.

        Args:
            vms: List of VM objects

        Returns:
            Statistics: updated, failed
        """
        stats = {"updated": 0, "failed": 0}

        for vm in vms:
            if self.update_vm_from_pod_status(vm):
                stats["updated"] += 1
            else:
                stats["failed"] += 1

        logger.info(f"VM-Pod sync complete: {stats}")
        return stats

    def cleanup_all_vms(self) -> int:
        """
        Delete all VirtualMachine CRs in the namespace.

        Returns:
            Number of VMs deleted
        """
        vms = self.list_vms()
        count = 0

        for vm in vms:
            vm_name = vm["metadata"]["name"]
            if self.delete_vm(vm_name):
                count += 1

        logger.info(f"Cleaned up {count} VMs")
        return count
