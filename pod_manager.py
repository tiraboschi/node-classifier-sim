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
import threading
import time
from kubernetes import client, config, watch
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

    def __init__(self, namespace: str = "default", use_in_cluster_config: bool = False,
                 create_vm_crs: bool = True, enable_migration_controller: bool = False):
        """
        Initialize the pod manager.

        Args:
            namespace: Kubernetes namespace for pods
            use_in_cluster_config: If True, use in-cluster config, otherwise use kubeconfig
            create_vm_crs: If True, also create VirtualMachine CRs for each VM
            enable_migration_controller: If True, start migration controller (default: False)
        """
        self.namespace = namespace
        self.pod_registry: Dict[str, PodInfo] = {}  # vm_id -> PodInfo
        self.create_vm_crs = create_vm_crs
        self.vm_manager: Optional[VMManager] = None

        # Migration controller state
        self._migration_controller_thread: Optional[threading.Thread] = None
        self._migration_controller_running = False
        self._migrations_in_progress: set = set()  # Track VMs currently migrating
        self._migration_lock = threading.Lock()
        self._enable_migration_controller = enable_migration_controller

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
                    self.vm_manager = VMManager(
                        namespace=namespace,
                        use_in_cluster_config=use_in_cluster_config
                    )
                    logger.info("VM Manager initialized - will create VirtualMachine CRs")
                    logger.info("VM utilization auto-sync enabled (always on)")
                except Exception as e:
                    logger.warning(f"Could not initialize VM Manager: {e}")
                    logger.warning("Continuing without VirtualMachine CR creation")
                    self.vm_manager = None

            # Start migration controller if enabled
            if enable_migration_controller:
                self.start_migration_controller()
            else:
                logger.info("Migration controller disabled for this PodManager instance")

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

    def _create_pod_spec(self, vm: VM, node_name: Optional[str] = None,
                         exclude_node: Optional[str] = None) -> client.V1Pod:
        """
        Create a pod specification for a VM.

        Args:
            vm: VM object
            node_name: Optional node name for direct assignment (usually None, let scheduler decide)
            exclude_node: Optional node name to exclude via anti-affinity (for live migration)

        Returns:
            V1Pod specification
        """
        pod_name = self._create_pod_name(vm.id)

        # Create annotations with resource allocation AND utilization
        # Allocation: what the VM has
        # Utilization: what the VM is actually using (for simulation/metrics)
        annotations = {
            "kubevirt.io/domain": vm.id,
            # Descheduler annotation - tells descheduler this pod supports background eviction
            "descheduler.alpha.kubernetes.io/request-evict-only": "true",
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

        # Add anti-affinity rule to exclude source node (like KubeVirt live migration)
        if exclude_node:
            pod_spec.affinity = client.V1Affinity(
                node_affinity=client.V1NodeAffinity(
                    required_during_scheduling_ignored_during_execution=client.V1NodeSelector(
                        node_selector_terms=[
                            client.V1NodeSelectorTerm(
                                match_expressions=[
                                    client.V1NodeSelectorRequirement(
                                        key="kubernetes.io/hostname",
                                        operator="NotIn",
                                        values=[exclude_node]
                                    )
                                ]
                            )
                        ]
                    )
                )
            )
            logger.info(f"Added anti-affinity rule to exclude node {exclude_node} (KubeVirt live migration)")

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
                annotations=annotations,
                # Add finalizer to protect against eviction (like KubeVirt)
                # This prevents immediate deletion and allows webhook to intercept
                finalizers=["kubevirt.io/migration-protection"]
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
        Simulate KubeVirt live migration: create target pod with anti-affinity, then delete source.

        This mimics KubeVirt's behavior where:
        1. A new target pod is created with anti-affinity to the source node
        2. Both source and target pods run simultaneously during migration
        3. After migration completes, the source pod is deleted

        Args:
            vm: VM object
            from_node: Source node name
            to_node: Optional destination node name (None = let scheduler decide with anti-affinity)

        Returns:
            True if successful, False otherwise
        """
        import time

        if vm.id not in self.pod_registry:
            logger.error(f"Cannot migrate VM {vm.id}: no existing pod found")
            return False

        source_pod_info = self.pod_registry[vm.id]
        source_pod_name = source_pod_info.pod_name

        logger.info(f"🔄 Starting KubeVirt-style live migration for VM {vm.id} from {from_node}")
        logger.info(f"   Source pod: {source_pod_name} on {from_node}")

        # Step 1: Create target pod with anti-affinity to source node
        # This ensures it won't be scheduled on the same node
        logger.info(f"   Creating target pod with anti-affinity to node {from_node}")

        try:
            # Create pod spec with anti-affinity to source node
            target_pod_spec = self._create_pod_spec(
                vm,
                node_name=to_node,  # None if scheduler should decide
                exclude_node=from_node  # Anti-affinity: don't schedule on source node
            )

            created_pod = self.v1.create_namespaced_pod(
                namespace=self.namespace,
                body=target_pod_spec
            )

            target_pod_name = created_pod.metadata.name
            logger.info(f"   Target pod created: {target_pod_name}")

        except ApiException as e:
            logger.error(f"Failed to create target pod for VM {vm.id}: {e}")
            return False

        # Step 2: Wait for target pod to be scheduled
        logger.info(f"   Waiting for target pod to be scheduled...")
        target_node = None
        for i in range(30):  # Wait up to 30 seconds
            try:
                pod = self.v1.read_namespaced_pod(
                    name=target_pod_name,
                    namespace=self.namespace
                )
                if pod.spec.node_name:
                    target_node = pod.spec.node_name
                    logger.info(f"   Target pod scheduled to node: {target_node}")
                    break
            except ApiException:
                pass
            time.sleep(1)

        if not target_node:
            logger.error(f"Target pod {target_pod_name} was not scheduled within 30 seconds")
            # Clean up target pod
            try:
                self.v1.delete_namespaced_pod(
                    name=target_pod_name,
                    namespace=self.namespace,
                    grace_period_seconds=0
                )
            except:
                pass
            return False

        # Verify target node is different from source node
        if target_node == from_node:
            logger.error(f"⚠️  Target pod scheduled to SAME node {target_node}! Anti-affinity rule failed!")
            logger.error(f"   This should not happen in KubeVirt live migration")
            # Clean up target pod
            try:
                self.v1.delete_namespaced_pod(
                    name=target_pod_name,
                    namespace=self.namespace,
                    grace_period_seconds=0
                )
            except:
                pass
            return False

        # Step 3: Simulate migration in progress (both pods running)
        logger.info(f"   Migration in progress: source on {from_node}, target on {target_node}")
        logger.info(f"   Both pods running simultaneously (simulating live migration)")
        time.sleep(0.5)  # Brief pause to simulate migration

        # Step 4: Delete source pod (migration complete)
        # IMPORTANT: Delete source pod BEFORE updating VM CR
        # This ensures webhook can correctly identify source pod (VM node == pod node)
        # First remove finalizer to allow deletion
        logger.info(f"   Migration complete, removing finalizer from source pod {source_pod_name}")
        finalizer_removed = self._remove_pod_finalizer(source_pod_name)
        if finalizer_removed:
            # Wait briefly for API server to propagate the finalizer removal
            # This prevents race condition where webhook reads stale pod state
            time.sleep(0.5)

        # Now delete the source pod
        logger.info(f"   Deleting source pod {source_pod_name}")
        try:
            self.v1.delete_namespaced_pod(
                name=source_pod_name,
                namespace=self.namespace,
                grace_period_seconds=0
            )
            logger.info(f"   Source pod {source_pod_name} deleted successfully")
        except ApiException as e:
            logger.warning(f"Failed to delete source pod {source_pod_name}: {e}")
            # Continue anyway, target is running

        # Step 5: Update VM to point to target pod (after source is deleted)
        vm.pod_name = target_pod_name
        vm.scheduled_node = target_node

        # Update registry
        self.pod_registry[vm.id] = PodInfo(
            vm_id=vm.id,
            pod_name=target_pod_name,
            node_name=target_node,
            cpu_cores=vm.cpu_cores,
            memory_bytes=vm.memory_bytes,
            cpu_utilization=vm.cpu_utilization,
            memory_utilization=vm.memory_utilization
        )

        # Update VM CR status
        if self.vm_manager:
            self.vm_manager.update_vm_status(vm.id, "Running", target_pod_name, target_node)

        logger.info(f"✅ Successfully migrated VM {vm.id}: {from_node} → {target_node}")
        logger.info(f"   Old pod: {source_pod_name} (deleted)")
        logger.info(f"   New pod: {target_pod_name} (running)")
        return True

    def update_pod_annotations(self, vm_id: str, cpu_utilization: float = None, memory_utilization: float = None) -> bool:
        """
        Update the resource utilization annotations on a pod.

        Args:
            vm_id: VM identifier
            cpu_utilization: New CPU utilization value (optional)
            memory_utilization: New memory utilization value (optional)

        Returns:
            True if successful, False otherwise
        """
        if vm_id not in self.pod_registry:
            logger.warning(f"No pod found for VM {vm_id}")
            return False

        pod_info = self.pod_registry[vm_id]

        try:
            # Build annotations patch
            annotations = {}

            if cpu_utilization is not None:
                annotations["simulation.node-classifier.io/vm-cpu-utilization"] = str(cpu_utilization)
                pod_info.cpu_utilization = cpu_utilization

            if memory_utilization is not None:
                annotations["simulation.node-classifier.io/vm-memory-utilization"] = str(memory_utilization)
                pod_info.memory_utilization = memory_utilization

            if not annotations:
                logger.warning(f"No annotations to update for VM {vm_id}")
                return False

            # Patch the pod annotations
            patch = {
                "metadata": {
                    "annotations": annotations
                }
            }

            self.v1.patch_namespaced_pod(
                name=pod_info.pod_name,
                namespace=self.namespace,
                body=patch
            )

            logger.info(f"Updated utilization annotations for pod {pod_info.pod_name}: {annotations}")
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
                # Clear any cached node assignment to ensure fresh scheduling
                vm.scheduled_node = ""
                if self.create_pod(vm):
                    stats["created"] += 1
            else:
                # Pod exists - update annotations if utilization changed
                pod_info = self.pod_registry[vm.id]
                if (pod_info.cpu_utilization != vm.cpu_utilization or
                    pod_info.memory_utilization != vm.memory_utilization):
                    if self.update_pod_annotations(vm.id, vm.cpu_utilization, vm.memory_utilization):
                        stats["updated"] += 1

        logger.info(f"Pod sync complete: {stats}")
        return stats
    # ========================================================================
    # Migration Controller (watches for pod evictions and triggers migrations)
    # ========================================================================

    def _handle_pod_eviction(self, pod_name: str, vm_id: str, source_node: str):
        """
        Handle eviction of a virt-launcher pod by triggering live migration.
        This is called by the migration controller when it detects a pod with deletionTimestamp.

        Args:
            pod_name: Name of the pod being evicted
            vm_id: VM identifier
            source_node: Node where the pod is currently running
        """
        try:
            logger.info(f"🚀 Eviction detected for pod {pod_name} (VM: {vm_id}) on {source_node}")

            # Get pod information from Kubernetes
            try:
                pod = self.v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            except ApiException as e:
                logger.error(f"Cannot read pod {pod_name}: {e}")
                self._remove_pod_finalizer(pod_name)
                return

            # Extract VM resource info from pod annotations
            annotations = pod.metadata.annotations or {}
            try:
                cpu_cores = float(annotations.get("simulation.node-classifier.io/vm-cpu-cores", "1.0"))
                memory_bytes = int(annotations.get("simulation.node-classifier.io/vm-memory-bytes", "1073741824"))
                cpu_utilization = float(annotations.get("simulation.node-classifier.io/vm-cpu-utilization", "0.5"))
                memory_utilization = float(annotations.get("simulation.node-classifier.io/vm-memory-utilization", "0.5"))
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse VM resource annotations from pod {pod_name}: {e}")
                self._remove_pod_finalizer(pod_name)
                return

            # Create VM object for migration
            from node import VM
            vm = VM(
                id=vm_id,
                cpu_cores=cpu_cores,
                memory_bytes=memory_bytes,
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization
            )
            vm.pod_name = pod_name
            vm.scheduled_node = source_node

            # Register pod in registry
            pod_info = PodInfo(
                vm_id=vm_id,
                pod_name=pod_name,
                node_name=source_node,
                cpu_cores=cpu_cores,
                memory_bytes=memory_bytes,
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization
            )
            self.pod_registry[vm_id] = pod_info

            # Perform live migration
            success = self._migrate_vm_pod_for_eviction(vm, source_node, to_node=None)

            if success:
                logger.info(f"✅ Live migration completed for VM {vm_id}")
            else:
                logger.error(f"❌ Live migration failed for VM {vm_id}")
                self._remove_pod_finalizer(pod_name)

        except Exception as e:
            logger.error(f"Error handling eviction for VM {vm_id}: {e}", exc_info=True)
            self._remove_pod_finalizer(pod_name)
        finally:
            with self._migration_lock:
                self._migrations_in_progress.discard(vm_id)

    def _migrate_vm_pod_for_eviction(self, vm: VM, from_node: str, to_node: Optional[str] = None) -> bool:
        """
        Migrate VM pod when triggered by eviction (deletionTimestamp set).
        Similar to migrate_vm_pod but removes finalizer instead of deleting the pod.
        """
        if vm.id not in self.pod_registry:
            logger.error(f"Cannot migrate VM {vm.id}: no existing pod found")
            return False

        source_pod_info = self.pod_registry[vm.id]
        source_pod_name = source_pod_info.pod_name

        logger.info(f"🔄 Starting KubeVirt-style live migration for VM {vm.id} from {from_node}")
        logger.info(f"   Source pod: {source_pod_name} on {from_node}")

        # Create target pod with anti-affinity
        logger.info(f"   Creating target pod with anti-affinity to node {from_node}")
        try:
            target_pod_spec = self._create_pod_spec(vm, node_name=to_node, exclude_node=from_node)
            created_pod = self.v1.create_namespaced_pod(namespace=self.namespace, body=target_pod_spec)
            target_pod_name = created_pod.metadata.name
            logger.info(f"   Target pod created: {target_pod_name}")
        except ApiException as e:
            logger.error(f"Failed to create target pod for VM {vm.id}: {e}")
            return False

        # Wait for target pod to be scheduled
        logger.info(f"   Waiting for target pod to be scheduled...")
        target_node = None
        for i in range(30):
            try:
                pod = self.v1.read_namespaced_pod(name=target_pod_name, namespace=self.namespace)
                if pod.spec.node_name:
                    target_node = pod.spec.node_name
                    logger.info(f"   Target pod scheduled to node: {target_node}")
                    break
            except ApiException:
                pass
            time.sleep(1)

        if not target_node:
            logger.error(f"Target pod {target_pod_name} was not scheduled within 30 seconds")
            try:
                self.v1.delete_namespaced_pod(name=target_pod_name, namespace=self.namespace, grace_period_seconds=0)
            except:
                pass
            return False

        if target_node == from_node:
            logger.error(f"⚠️  Target pod scheduled to SAME node {target_node}! Anti-affinity rule failed!")
            try:
                self.v1.delete_namespaced_pod(name=target_pod_name, namespace=self.namespace, grace_period_seconds=0)
            except:
                pass
            return False

        # Simulate migration
        logger.info(f"   Migration in progress: source on {from_node}, target on {target_node}")
        time.sleep(0.5)

        # Update VM to point to target pod
        vm.pod_name = target_pod_name
        vm.scheduled_node = target_node

        self.pod_registry[vm.id] = PodInfo(
            vm_id=vm.id,
            pod_name=target_pod_name,
            node_name=target_node,
            cpu_cores=vm.cpu_cores,
            memory_bytes=vm.memory_bytes,
            cpu_utilization=vm.cpu_utilization,
            memory_utilization=vm.memory_utilization
        )

        if self.vm_manager:
            self.vm_manager.update_vm_status(vm.id, "Running", target_pod_name, target_node)

        # Remove finalizer from source pod (allows deletion to proceed)
        logger.info(f"   Migration complete, removing finalizer from source pod {source_pod_name}")
        self._remove_pod_finalizer(source_pod_name)

        logger.info(f"✅ Successfully migrated VM {vm.id}: {from_node} → {target_node}")
        logger.info(f"   Old pod: {source_pod_name} (finalizer removed, will be deleted)")
        logger.info(f"   New pod: {target_pod_name} (running)")
        return True

    def _remove_pod_finalizer(self, pod_name: str) -> bool:
        """Remove the migration finalizer from a pod."""
        try:
            import json as json_module
            from kubernetes.client import ApiClient

            pod = self.v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            finalizers = pod.metadata.finalizers or []
            if "kubevirt.io/migration-protection" in finalizers:
                finalizers.remove("kubevirt.io/migration-protection")
                # Use JSON patch for reliable finalizer removal
                patch = [{"op": "replace", "path": "/metadata/finalizers", "value": finalizers}]

                # Call API directly with correct content type
                api_client = self.v1.api_client
                api_client.call_api(
                    f'/api/v1/namespaces/{self.namespace}/pods/{pod_name}',
                    'PATCH',
                    header_params={'Content-Type': 'application/json-patch+json'},
                    body=patch,
                    response_type='V1Pod',
                    auth_settings=['BearerToken'],
                    _return_http_data_only=True
                )
                logger.info(f"   Removed finalizer from pod {pod_name}")
                return True
            return False
        except ApiException as e:
            logger.error(f"Failed to remove finalizer from pod {pod_name}: {e}")
            return False

    def _clear_vm_evacuation_marker(self, vm_id: str) -> bool:
        """Clear the evacuationNodeName from VM CR status."""
        try:
            from kubernetes import client as k8s_client
            custom_api = k8s_client.CustomObjectsApi()

            patch = {
                "status": {
                    "evacuationNodeName": None
                }
            }
            custom_api.patch_namespaced_custom_object_status(
                group="simulation.node-classifier.io",
                version="v1alpha1",
                namespace=self.namespace,
                plural="virtualmachines",
                name=vm_id,
                body=patch
            )
            logger.info(f"Cleared evacuationNodeName from VM {vm_id}")
            return True
        except ApiException as e:
            logger.error(f"Failed to clear evacuationNodeName from VM {vm_id}: {e}")
            return False

    def _handle_vm_evacuation(self, vm_id: str, pod_name: str, evacuation_node: str):
        """
        Handle VM evacuation by triggering live migration.
        This is called when VM CR has status.evacuationNodeName set.

        Args:
            vm_id: VM identifier
            pod_name: Name of the pod to migrate
            evacuation_node: Node to evacuate from
        """
        try:
            logger.info(f"🚀 Handling evacuation for VM {vm_id} from node {evacuation_node}")

            # Get pod information
            try:
                pod = self.v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            except ApiException as e:
                logger.error(f"Cannot read pod {pod_name}: {e}")
                self._clear_vm_evacuation_marker(vm_id)
                return

            # Extract VM resource info from pod annotations
            annotations = pod.metadata.annotations or {}
            try:
                cpu_cores = float(annotations.get("simulation.node-classifier.io/vm-cpu-cores", "1.0"))
                memory_bytes = int(annotations.get("simulation.node-classifier.io/vm-memory-bytes", "1073741824"))
                cpu_utilization = float(annotations.get("simulation.node-classifier.io/vm-cpu-utilization", "0.5"))
                memory_utilization = float(annotations.get("simulation.node-classifier.io/vm-memory-utilization", "0.5"))
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to parse VM resource annotations from pod {pod_name}: {e}")
                self._clear_vm_evacuation_marker(vm_id)
                return

            # Create VM object for migration
            from node import VM
            vm = VM(
                id=vm_id,
                cpu_cores=cpu_cores,
                memory_bytes=memory_bytes,
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization
            )
            vm.pod_name = pod_name
            vm.scheduled_node = evacuation_node

            # Register pod in registry
            pod_info = PodInfo(
                vm_id=vm_id,
                pod_name=pod_name,
                node_name=evacuation_node,
                cpu_cores=cpu_cores,
                memory_bytes=memory_bytes,
                cpu_utilization=cpu_utilization,
                memory_utilization=memory_utilization
            )
            self.pod_registry[vm_id] = pod_info

            # Perform live migration (doesn't delete source pod, scheduler will handle it)
            success = self.migrate_vm_pod(vm, evacuation_node, to_node=None)

            if success:
                logger.info(f"✅ Live migration completed for VM {vm_id}")
                # Clear evacuation marker from VM CR
                self._clear_vm_evacuation_marker(vm_id)
            else:
                logger.error(f"❌ Live migration failed for VM {vm_id}")
                self._clear_vm_evacuation_marker(vm_id)

        except Exception as e:
            logger.error(f"Error handling evacuation for VM {vm_id}: {e}", exc_info=True)
            self._clear_vm_evacuation_marker(vm_id)
        finally:
            with self._migration_lock:
                self._migrations_in_progress.discard(vm_id)

    def _migration_controller_loop(self):
        """
        Watch for VM CRs with evacuationNodeName and trigger migrations.
        This is the main controller loop that implements KubeVirt-style eviction handling.

        KubeVirt flow:
        1. Webhook sets status.evacuationNodeName on VM CR
        2. Controller sees evacuationNodeName and triggers migration
        3. After migration, controller clears evacuationNodeName
        """
        logger.info(f"Migration controller started for namespace '{self.namespace}'")
        w = watch.Watch()

        # We need CustomObjectsApi to watch VM CRs
        from kubernetes import client as k8s_client
        custom_api = k8s_client.CustomObjectsApi()

        while self._migration_controller_running:
            try:
                stream = w.stream(
                    custom_api.list_namespaced_custom_object,
                    group="simulation.node-classifier.io",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="virtualmachines",
                    timeout_seconds=60
                )

                for event in stream:
                    if not self._migration_controller_running:
                        break

                    event_type = event.get("type", "")
                    vm_cr = event.get("object")

                    if not vm_cr or event_type not in ["ADDED", "MODIFIED"]:
                        continue

                    vm_id = vm_cr.get("metadata", {}).get("name", "")
                    vm_status = vm_cr.get("status", {})
                    evacuation_node = vm_status.get("evacuationNodeName", "")

                    # Check if VM is marked for evacuation
                    if not evacuation_node:
                        continue

                    # Get VM's current pod and node
                    pod_name = vm_status.get("podName", "")
                    if not pod_name:
                        logger.warning(f"VM {vm_id} marked for evacuation but has no pod, clearing evacuationNodeName")
                        self._clear_vm_evacuation_marker(vm_id)
                        continue

                    # Check if migration already in progress
                    with self._migration_lock:
                        if vm_id in self._migrations_in_progress:
                            logger.debug(f"Migration already in progress for VM {vm_id}, skipping")
                            continue
                        self._migrations_in_progress.add(vm_id)

                    logger.info(f"🚀 VM {vm_id} marked for evacuation from node {evacuation_node}")

                    # Trigger migration in background thread
                    migration_thread = threading.Thread(
                        target=self._handle_vm_evacuation,
                        args=(vm_id, pod_name, evacuation_node),
                        daemon=True,
                        name=f"migrate-{vm_id}"
                    )
                    migration_thread.start()

            except ApiException as e:
                if e.status == 410:
                    logger.warning("Watch resource version too old, restarting watch")
                    continue
                else:
                    logger.error(f"API exception in migration controller: {e}")
                    if self._migration_controller_running:
                        time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error in migration controller: {e}", exc_info=True)
                if self._migration_controller_running:
                    time.sleep(5)

        logger.info("Migration controller stopped")

    def start_migration_controller(self) -> bool:
        """
        Start the migration controller in a background thread.
        The controller watches for pod evictions and triggers live migrations.
        """
        if self._migration_controller_running:
            logger.warning("Migration controller already running")
            return False

        self._migration_controller_running = True
        self._migration_controller_thread = threading.Thread(
            target=self._migration_controller_loop,
            daemon=True,
            name="migration-controller"
        )
        self._migration_controller_thread.start()
        logger.info("Migration controller thread started")
        return True

    def stop_migration_controller(self):
        """Stop the migration controller."""
        if not self._migration_controller_running:
            return

        logger.info("Stopping migration controller...")
        self._migration_controller_running = False
        if self._migration_controller_thread and self._migration_controller_thread.is_alive():
            self._migration_controller_thread.join(timeout=10)
        logger.info("Migration controller stopped")
