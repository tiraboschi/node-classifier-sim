#!/usr/bin/env python3
"""
VirtualMachine Custom Resource Manager

Manages VirtualMachine custom resources that mimic KubeVirt's VMI objects.
Tracks VM resource consumption via annotations and pod execution via status.
Includes built-in utilization-to-pod synchronization.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
import threading
import time
from kubernetes import client, config, watch
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

        # Utilization sync state
        self._utilization_cache: Dict[str, Dict[str, str]] = {}
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_running = False

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

        # Always start utilization sync
        self.start_utilization_sync()

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

            # Check current status
            current_status = vm_obj.get("status", {})
            current_phase = current_status.get("phase", "")
            current_pod = current_status.get("podName", "")
            current_node = current_status.get("nodeName", "")

            # Check current labels
            current_labels = vm_obj.get("metadata", {}).get("labels", {})
            current_node_label = current_labels.get("kubevirt.io/nodeName", "")

            # Check if anything actually changed
            phase_changed = current_phase != phase
            pod_changed = current_pod != pod_name
            node_changed = current_node != node_name
            # Also trigger update if node label is missing but node is assigned
            label_missing = node_name and current_node_label != node_name

            # Skip update if nothing changed and label is correct
            if not (phase_changed or pod_changed or node_changed or label_missing):
                logger.debug(f"Skipping VM {vm_name} status update - no changes detected")
                return True

            # Update status
            if "status" not in vm_obj:
                vm_obj["status"] = {}

            vm_obj["status"]["phase"] = phase

            if pod_name:
                vm_obj["status"]["podName"] = pod_name

            if node_name:
                vm_obj["status"]["nodeName"] = node_name

            # Copy utilization values from spec to status
            if "spec" in vm_obj and "utilization" in vm_obj["spec"]:
                utilization = vm_obj["spec"]["utilization"]
                if "cpu" in utilization:
                    vm_obj["status"]["cpuUtilization"] = utilization["cpu"]
                if "memory" in utilization:
                    vm_obj["status"]["memoryUtilization"] = utilization["memory"]

            # Only add condition if phase changed
            if phase_changed:
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

            # Update kubevirt.io/nodeName label if node changed or label is missing (KubeVirt-style)
            if node_changed or label_missing:
                if "metadata" not in vm_obj:
                    vm_obj["metadata"] = {}
                if "labels" not in vm_obj["metadata"]:
                    vm_obj["metadata"]["labels"] = {}

                if node_name:
                    # Add/update label with node name
                    vm_obj["metadata"]["labels"]["kubevirt.io/nodeName"] = node_name
                else:
                    # Remove label if node is unset
                    vm_obj["metadata"]["labels"].pop("kubevirt.io/nodeName", None)

                # Patch metadata labels separately (not part of status subresource)
                self.custom_api.patch_namespaced_custom_object(
                    group=VM_GROUP,
                    version=VM_VERSION,
                    namespace=self.namespace,
                    plural=VM_PLURAL,
                    name=vm_name,
                    body={"metadata": {"labels": vm_obj["metadata"]["labels"]}}
                )

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

    # ========================================================================
    # Utilization Synchronization (VM spec -> Pod annotations)
    # ========================================================================

    def _get_vm_utilization(self, vm_obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract utilization values from a VM object."""
        spec = vm_obj.get("spec", {})
        utilization = spec.get("utilization", {})
        if not utilization:
            return None
        return {
            "cpu": utilization.get("cpu", ""),
            "memory": utilization.get("memory", "")
        }

    def _sync_vm_utilization_to_pod(self, vm_obj: Dict[str, Any]) -> bool:
        """
        Sync a VirtualMachine's utilization to its pod annotations.

        Args:
            vm_obj: VirtualMachine object

        Returns:
            True if successful, False otherwise
        """
        vm_name = vm_obj.get("metadata", {}).get("name", "")
        if not vm_name:
            return False

        # Get current utilization values
        utilization = self._get_vm_utilization(vm_obj)
        if not utilization:
            logger.debug(f"No utilization spec found for VM {vm_name}")
            return False

        cpu_util = utilization.get("cpu", "")
        memory_util = utilization.get("memory", "")

        # Check if utilization has changed
        cached_util = self._utilization_cache.get(vm_name)
        if cached_util and cached_util.get("cpu") == cpu_util and cached_util.get("memory") == memory_util:
            # No change, skip update
            return True

        # Get pod name from VM status
        status = vm_obj.get("status", {})
        pod_name = status.get("podName", "")

        if not pod_name:
            logger.debug(f"VM {vm_name} has no associated pod yet")
            # Update cache even if no pod exists yet
            self._utilization_cache[vm_name] = utilization
            return False

        # Update pod annotations
        try:
            patch = {
                "metadata": {
                    "annotations": {
                        "simulation.node-classifier.io/vm-cpu-utilization": cpu_util,
                        "simulation.node-classifier.io/vm-memory-utilization": memory_util
                    }
                }
            }

            self.core_v1.patch_namespaced_pod(
                name=pod_name,
                namespace=self.namespace,
                body=patch
            )

            # Update cache
            self._utilization_cache[vm_name] = utilization

            # Also update VM status to reflect new utilization values
            try:
                status_patch = {
                    "status": {
                        "cpuUtilization": cpu_util,
                        "memoryUtilization": memory_util
                    }
                }
                self.custom_api.patch_namespaced_custom_object_status(
                    group=VM_GROUP,
                    version=VM_VERSION,
                    namespace=self.namespace,
                    plural=VM_PLURAL,
                    name=vm_name,
                    body=status_patch
                )
            except ApiException as status_e:
                logger.warning(f"Failed to update VM {vm_name} status utilization: {status_e}")
                # Don't fail the whole operation if status update fails

            logger.info(f"Synced utilization for VM {vm_name} -> pod {pod_name}: "
                       f"cpu={cpu_util}, memory={memory_util}")
            return True

        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Pod {pod_name} not found for VM {vm_name}")
            else:
                logger.error(f"Failed to update pod {pod_name} annotations: {e}")
            return False

    def _watch_vms_for_utilization_changes(self) -> None:
        """
        Watch VirtualMachine resources and sync utilization to pods.
        Runs in a background thread.
        """
        logger.info(f"Starting VM utilization watcher for namespace '{self.namespace}'")
        w = watch.Watch()

        while self._sync_running:
            try:
                stream = w.stream(
                    self.custom_api.list_namespaced_custom_object,
                    group=VM_GROUP,
                    version=VM_VERSION,
                    namespace=self.namespace,
                    plural=VM_PLURAL,
                    timeout_seconds=60
                )

                for event in stream:
                    if not self._sync_running:
                        break

                    event_type = event.get("type", "")
                    vm_obj = event.get("object", {})
                    vm_name = vm_obj.get("metadata", {}).get("name", "")

                    if event_type == "DELETED":
                        # Remove from cache
                        if vm_name in self._utilization_cache:
                            del self._utilization_cache[vm_name]
                    elif event_type in ["ADDED", "MODIFIED"]:
                        # Sync utilization to pod
                        self._sync_vm_utilization_to_pod(vm_obj)

            except ApiException as e:
                if e.status == 410:
                    logger.warning("Watch resource version too old, restarting watch")
                    continue
                else:
                    logger.error(f"API exception in watch loop: {e}")
                    if self._sync_running:
                        time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in watch loop: {e}")
                if self._sync_running:
                    time.sleep(5)

        logger.info("VM utilization watcher stopped")

    def start_utilization_sync(self) -> bool:
        """
        Start background thread to sync VM utilization to pod annotations.

        Returns:
            True if started, False if already running
        """
        if self._sync_running:
            logger.warning("Utilization sync already running")
            return False

        # Perform initial sync
        self.sync_all_utilization()

        # Start watcher thread
        self._sync_running = True
        self._sync_thread = threading.Thread(
            target=self._watch_vms_for_utilization_changes,
            daemon=True,
            name="vm-utilization-sync"
        )
        self._sync_thread.start()
        logger.info("VM utilization sync started in background thread")
        return True

    def stop_utilization_sync(self) -> None:
        """Stop the background utilization sync thread."""
        if not self._sync_running:
            return

        logger.info("Stopping VM utilization sync...")
        self._sync_running = False

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)

        logger.info("VM utilization sync stopped")

    def sync_all_utilization(self) -> Dict[str, int]:
        """
        Perform a one-time sync of all VMs' utilization to their pods.

        Returns:
            Statistics: synced, failed, skipped
        """
        stats = {"synced": 0, "failed": 0, "skipped": 0}

        try:
            vms = self.list_vms()
            logger.info(f"Syncing utilization for {len(vms)} VMs")

            for vm_obj in vms:
                success = self._sync_vm_utilization_to_pod(vm_obj)
                if success:
                    stats["synced"] += 1
                else:
                    vm_name = vm_obj.get("metadata", {}).get("name", "")
                    status = vm_obj.get("status", {})
                    if not status.get("podName"):
                        stats["skipped"] += 1
                    else:
                        stats["failed"] += 1

            logger.info(f"Utilization sync complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to sync utilization: {e}")
            return stats
