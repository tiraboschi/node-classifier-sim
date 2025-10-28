#!/usr/bin/env python3
"""
Pod Manager for VM Simulation

Manages virt-launcher pods for VMs in the KWOK cluster.
Each VM has a corresponding virt-launcher-<vm-name>-<random> pod with annotations
reporting fake memory and CPU consumption values.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import string
import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from node import VM
from vm_manager import VMManager

logger = logging.getLogger(__name__)


@dataclass
class PodInfo:
    """Information about a virt-launcher pod."""
    vm_id: str
    pod_name: str
    node_name: str
    cpu_cores: float
    memory_bytes: int
    cpu_utilization: float
    memory_utilization: float


class PodManager:
    """
    Manages virt-launcher pods for VMs in KWOK cluster.

    Each VM gets a corresponding pod:
    - Name: virt-launcher-<vm-name>-<random>
    - Annotations: cpu_consumption and memory_consumption from VM
    - Scheduled to the KWOK node where the VM is running
    """

    def __init__(self, namespace: str = "default", use_in_cluster_config: bool = False, create_vm_crs: bool = True):
        """
        Initialize the pod manager.

        Args:
            namespace: Kubernetes namespace for pods
            use_in_cluster_config: If True, use in-cluster config, otherwise use kubeconfig
            create_vm_crs: If True, also create VirtualMachine CRs for each VM
        """
        self.namespace = namespace
        self.pod_registry: Dict[str, PodInfo] = {}  # vm_id -> PodInfo
        self.create_vm_crs = create_vm_crs
        self.vm_manager: Optional[VMManager] = None

        # Initialize Kubernetes client
        try:
            if use_in_cluster_config:
                config.load_incluster_config()
            else:
                config.load_kube_config()
            self.v1 = client.CoreV1Api()
            logger.info("Kubernetes client initialized successfully")

            # Initialize VM manager if requested
            if create_vm_crs:
                try:
                    self.vm_manager = VMManager(namespace=namespace, use_in_cluster_config=use_in_cluster_config)
                    logger.info("VM Manager initialized - will create VirtualMachine CRs")
                except Exception as e:
                    logger.warning(f"Could not initialize VM Manager: {e}")
                    logger.warning("Continuing without VirtualMachine CR creation")
                    self.vm_manager = None

        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise

    def _generate_random_suffix(self, length: int = 5) -> str:
        """Generate a random alphanumeric suffix."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def _create_pod_name(self, vm_id: str) -> str:
        """
        Create a pod name in the format: virt-launcher-<vm-name>-<random>

        Args:
            vm_id: VM identifier (e.g., "vm-1")

        Returns:
            Pod name (e.g., "virt-launcher-vm-1-a7b3c")
        """
        suffix = self._generate_random_suffix()
        return f"virt-launcher-{vm_id}-{suffix}"

    def _create_pod_spec(self, vm: VM, node_name: Optional[str] = None) -> client.V1Pod:
        """
        Create a pod specification for a VM.

        Args:
            vm: VM object
            node_name: Optional node name for direct assignment (usually None, let scheduler decide)

        Returns:
            V1Pod specification
        """
        pod_name = self._create_pod_name(vm.id)

        # Create annotations with resource allocation AND utilization
        # Allocation: what the VM has
        # Utilization: what the VM is actually using (for simulation/metrics)
        annotations = {
            "kubevirt.io/domain": vm.id,
            # Resource allocation
            "simulation.node-classifier.io/vm-cpu-cores": str(vm.cpu_cores),
            "simulation.node-classifier.io/vm-memory-bytes": str(vm.memory_bytes),
            # Resource utilization (fake/expected usage for simulation)
            "simulation.node-classifier.io/vm-cpu-utilization": str(vm.cpu_utilization),
            "simulation.node-classifier.io/vm-memory-utilization": str(vm.memory_utilization),
        }

        # Calculate pod resource requests
        # CPU: 1/10 of VM cores (virt-launcher overhead)
        # Memory: 1:1 with VM memory (no overhead for simulation)
        pod_cpu_request = vm.cpu_cores / 10.0
        pod_memory_bytes = vm.memory_bytes

        # Format memory for Kubernetes (use Mi for better precision)
        memory_mi = int(pod_memory_bytes / (1024 ** 2))

        # Create pod spec - let scheduler decide placement unless node_name is specified
        pod_spec = client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="compute",
                    image="quay.io/kubevirt/virt-launcher:latest",
                    resources=client.V1ResourceRequirements(
                        requests={
                            "cpu": f"{pod_cpu_request}",  # 1/10 of VM CPU
                            "memory": f"{memory_mi}Mi"     # 1:1 with VM memory
                        }
                        # No limits - only requests
                    )
                )
            ],
            tolerations=[
                client.V1Toleration(
                    key="kwok.x-k8s.io/node",
                    operator="Equal",
                    value="fake",
                    effect="NoSchedule"
                )
            ]
        )

        # Only set node_name if explicitly provided (for forced placement)
        if node_name:
            pod_spec.node_name = node_name

        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=client.V1ObjectMeta(
                name=pod_name,
                namespace=self.namespace,
                labels={
                    "kubevirt.io/domain": vm.id,
                    "vm.kubevirt.io/name": vm.id,
                    "app": "virt-launcher"
                },
                annotations=annotations
            ),
            spec=pod_spec
        )

        return pod

    def create_pod(self, vm: VM, node_name: Optional[str] = None) -> Optional[str]:
        """
        Create a virt-launcher pod for a VM.
        Also creates a VirtualMachine CR if vm_manager is available.
        Let the scheduler decide placement unless node_name is explicitly provided.

        Args:
            vm: VM object
            node_name: Optional node name for direct assignment (usually None)

        Returns:
            Pod name if successful, None otherwise
        """
        try:
            # Create VirtualMachine CR first if enabled
            if self.vm_manager:
                self.vm_manager.create_vm(vm)
                self.vm_manager.update_vm_status(vm.id, "Pending")

            # Create the pod
            pod_spec = self._create_pod_spec(vm, node_name)
            created_pod = self.v1.create_namespaced_pod(
                namespace=self.namespace,
                body=pod_spec
            )

            pod_name = created_pod.metadata.name

            # Update VM with pod reference
            vm.pod_name = pod_name

            # Register pod (node_name might be empty if not scheduled yet)
            pod_info = PodInfo(
                vm_id=vm.id,
                pod_name=pod_name,
                node_name=node_name or "",
                cpu_cores=vm.cpu_cores,
                memory_bytes=vm.memory_bytes,
                cpu_utilization=vm.cpu_utilization,
                memory_utilization=vm.memory_utilization
            )
            self.pod_registry[vm.id] = pod_info

            # Update VM CR status with pod name
            if self.vm_manager:
                phase = "Scheduled" if node_name else "Scheduling"
                self.vm_manager.update_vm_status(vm.id, phase, pod_name, node_name or "")

            logger.info(f"Created pod {pod_name} for VM {vm.id}" +
                       (f" on node {node_name}" if node_name else " (waiting for scheduler)"))

            return pod_name

        except ApiException as e:
            logger.error(f"Failed to create pod for VM {vm.id}: {e}")
            return None

    def delete_pod(self, vm_id: str) -> bool:
        """
        Delete the virt-launcher pod for a VM.
        Also deletes the VirtualMachine CR if vm_manager is available.

        Args:
            vm_id: VM identifier

        Returns:
            True if successful, False otherwise
        """
        if vm_id not in self.pod_registry:
            logger.warning(f"No pod found for VM {vm_id}")
            return False

        pod_info = self.pod_registry[vm_id]

        try:
            self.v1.delete_namespaced_pod(
                name=pod_info.pod_name,
                namespace=self.namespace,
                grace_period_seconds=0  # Immediate deletion for simulation
            )

            del self.pod_registry[vm_id]
            logger.info(f"Deleted pod {pod_info.pod_name} for VM {vm_id}")

            # Delete VM CR if enabled
            if self.vm_manager:
                self.vm_manager.delete_vm(vm_id)

            return True

        except ApiException as e:
            logger.error(f"Failed to delete pod for VM {vm_id}: {e}")
            return False

    def get_pod_node_assignment(self, vm_id: str) -> Optional[str]:
        """
        Get the node where a VM's pod is currently scheduled.

        Args:
            vm_id: VM identifier

        Returns:
            Node name if pod is scheduled, None otherwise
        """
        if vm_id not in self.pod_registry:
            return None

        pod_info = self.pod_registry[vm_id]

        try:
            pod = self.v1.read_namespaced_pod(
                name=pod_info.pod_name,
                namespace=self.namespace
            )

            return pod.spec.node_name

        except ApiException as e:
            logger.error(f"Failed to read pod for VM {vm_id}: {e}")
            return None

    def update_vm_node_assignments(self, vms: List[VM]) -> Dict[str, int]:
        """
        Update the scheduled_node field for all VMs by reading their pods.
        Also updates VirtualMachine CR statuses if vm_manager is available.

        Args:
            vms: List of VM objects to update

        Returns:
            Statistics: scheduled, pending, missing
        """
        stats = {"scheduled": 0, "pending": 0, "missing": 0}

        for vm in vms:
            if not vm.pod_name:
                stats["missing"] += 1
                continue

            try:
                pod = self.v1.read_namespaced_pod(
                    name=vm.pod_name,
                    namespace=self.namespace
                )

                if pod.spec.node_name:
                    vm.scheduled_node = pod.spec.node_name
                    stats["scheduled"] += 1

                    # Update registry
                    if vm.id in self.pod_registry:
                        self.pod_registry[vm.id].node_name = pod.spec.node_name

                    # Update VM CR status to Running
                    if self.vm_manager:
                        self.vm_manager.update_vm_status(vm.id, "Running", vm.pod_name, pod.spec.node_name)
                else:
                    vm.scheduled_node = ""
                    stats["pending"] += 1

                    # Update VM CR status to Scheduling
                    if self.vm_manager:
                        self.vm_manager.update_vm_status(vm.id, "Scheduling", vm.pod_name, "")

            except ApiException as e:
                logger.warning(f"Failed to read pod {vm.pod_name} for VM {vm.id}: {e}")
                stats["missing"] += 1

        return stats

    def migrate_vm_pod(self, vm: VM, from_node: str, to_node: Optional[str] = None) -> bool:
        """
        Migrate a VM's pod by deleting the old pod and creating a new one.
        The new pod will be scheduled by the Kubernetes scheduler unless to_node is specified.

        Args:
            vm: VM object
            from_node: Source node name (for logging)
            to_node: Optional destination node name (None = let scheduler decide)

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Migrating VM {vm.id} pod from {from_node}" +
                   (f" to {to_node}" if to_node else " (scheduler will assign)"))

        # Delete old pod
        if not self.delete_pod(vm.id):
            logger.error(f"Failed to delete old pod for VM {vm.id}")
            return False

        # Create new pod (let scheduler decide placement unless to_node specified)
        pod_name = self.create_pod(vm, to_node)
        if pod_name is None:
            logger.error(f"Failed to create new pod for VM {vm.id}")
            return False

        logger.info(f"Successfully migrated VM {vm.id} pod" +
                   (f" to {to_node}" if to_node else " (waiting for scheduler)"))
        return True

    def update_pod_annotations(self, vm_id: str, cpu_consumption: float, memory_consumption: float) -> bool:
        """
        Update the resource consumption annotations on a pod.

        Args:
            vm_id: VM identifier
            cpu_consumption: New CPU consumption value
            memory_consumption: New memory consumption value

        Returns:
            True if successful, False otherwise
        """
        if vm_id not in self.pod_registry:
            logger.warning(f"No pod found for VM {vm_id}")
            return False

        pod_info = self.pod_registry[vm_id]

        try:
            # Patch the pod annotations
            patch = {
                "metadata": {
                    "annotations": {
                        "kubevirt.io/vm-cpu-consumption": str(cpu_consumption),
                        "kubevirt.io/vm-memory-consumption": str(memory_consumption)
                    }
                }
            }

            self.v1.patch_namespaced_pod(
                name=pod_info.pod_name,
                namespace=self.namespace,
                body=patch
            )

            # Update local registry
            pod_info.cpu_consumption = cpu_consumption
            pod_info.memory_consumption = memory_consumption

            logger.info(f"Updated annotations for pod {pod_info.pod_name}")
            return True

        except ApiException as e:
            logger.error(f"Failed to update pod annotations for VM {vm_id}: {e}")
            return False

    def get_pod_info(self, vm_id: str) -> Optional[PodInfo]:
        """Get pod information for a VM."""
        return self.pod_registry.get(vm_id)

    def list_pods(self) -> List[PodInfo]:
        """List all managed pods."""
        return list(self.pod_registry.values())

    def cleanup_all_pods(self) -> int:
        """
        Delete all managed pods.

        Returns:
            Number of pods deleted
        """
        count = 0
        vm_ids = list(self.pod_registry.keys())

        for vm_id in vm_ids:
            if self.delete_pod(vm_id):
                count += 1

        logger.info(f"Cleaned up {count} pods")
        return count

    def sync_pods_with_vms(self, vms: List[VM]) -> Dict[str, int]:
        """
        Synchronize pods with the current VM list.
        Creates pods for VMs that don't have them, deletes pods for VMs that are gone.
        Does NOT manage node assignment - that's the scheduler's job.

        Args:
            vms: List of all VMs that should have pods

        Returns:
            Dictionary with statistics: created, deleted, updated
        """
        stats = {"created": 0, "deleted": 0, "updated": 0}

        # Build set of current VM IDs
        current_vm_ids = {vm.id for vm in vms}
        vm_by_id = {vm.id: vm for vm in vms}

        # Find pods that need to be deleted (VMs no longer exist)
        for vm_id in list(self.pod_registry.keys()):
            if vm_id not in current_vm_ids:
                if self.delete_pod(vm_id):
                    stats["deleted"] += 1

        # Create pods for VMs that don't have them
        for vm in vms:
            if not vm.pod_name or vm.id not in self.pod_registry:
                # Pod doesn't exist - create it (scheduler will assign node)
                if self.create_pod(vm):
                    stats["created"] += 1
            else:
                # Pod exists - update annotations if consumption changed
                pod_info = self.pod_registry[vm.id]
                if (pod_info.cpu_consumption != vm.cpu_consumption or
                    pod_info.memory_consumption != vm.memory_consumption):
                    if self.update_pod_annotations(vm.id, vm.cpu_consumption, vm.memory_consumption):
                        stats["updated"] += 1

        logger.info(f"Pod sync complete: {stats}")
        return stats