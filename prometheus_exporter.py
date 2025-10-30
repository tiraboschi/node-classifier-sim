#!/usr/bin/env python3
"""
Synthetic Metrics Exporter for Node Classifier Simulation

Exposes Prometheus metrics for KWOK nodes with dynamic VM-based resource consumption.
Metrics are calculated by reading virt-launcher pod annotations on KWOK nodes.
Receives feedback from the simulator to update metrics based on VM migrations.
"""

from flask import Flask, Response, request, jsonify
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import json
import logging
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from node import Node, VM
from scenario_loader import ScenarioLoader
from pod_manager import PodManager
from vm_manager import VMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Prometheus metrics registry
registry = CollectorRegistry()

# Define metrics (one per node, labeled by node name)
cpu_usage_gauge = Gauge(
    'node_cpu_usage_ratio',
    'CPU usage ratio (0.0 to 1.0+, can exceed for overload)',
    ['node'],
    registry=registry
)

cpu_pressure_gauge = Gauge(
    'node_cpu_pressure_psi',
    'CPU Pressure Stall Information (0.0 to 1.0)',
    ['node'],
    registry=registry
)

memory_usage_gauge = Gauge(
    'node_memory_usage_ratio',
    'Memory usage ratio (0.0 to 1.0)',
    ['node'],
    registry=registry
)

memory_pressure_gauge = Gauge(
    'node_memory_pressure_psi',
    'Memory Pressure Stall Information (0.0 to 1.0)',
    ['node'],
    registry=registry
)

vm_count_gauge = Gauge(
    'node_vm_count',
    'Number of VMs running on the node',
    ['node'],
    registry=registry
)

# OpenShift-compatible counter metrics (node-exporter style)
# Use 'instance' label as OpenShift expects
cpu_seconds_counter = Counter(
    'node_cpu_seconds_total',
    'Total CPU seconds by mode (simulated node-exporter metric)',
    ['instance', 'mode'],
    registry=registry
)

cpu_pressure_counter = Counter(
    'node_pressure_cpu_waiting_seconds_total',
    'Total CPU pressure waiting seconds (simulated PSI metric)',
    ['instance'],
    registry=registry
)

# Node role label metric (kube-state-metrics style)
node_role_gauge = Gauge(
    'kube_node_role',
    'Node role label (1 for worker nodes)',
    ['node', 'role', 'instance'],
    registry=registry
)


@dataclass
class ExporterState:
    """Thread-safe state for the exporter."""
    nodes: Dict[str, Node] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pod_manager: Optional[PodManager] = None
    vm_manager: Optional[VMManager] = None
    k8s_client: Optional[client.CoreV1Api] = None
    custom_api: Optional[client.CustomObjectsApi] = None
    vm_controller_running: bool = False
    vm_controller_thread: Optional[threading.Thread] = None
    # Track last update time for each node to calculate counter increments
    last_update_time: Dict[str, float] = field(default_factory=dict)
    # Node configuration
    node_cpu_cores: float = 32.0  # 32 CPU cores per KWOK node

    def __post_init__(self):
        """Initialize Kubernetes client and pod manager."""
        try:
            # Try in-cluster config first (when running as pod), fallback to kubeconfig
            use_in_cluster = False
            try:
                config.load_incluster_config()
                use_in_cluster = True
                logger.info("Using in-cluster Kubernetes config")
            except Exception:
                config.load_kube_config()
                logger.info("Using local kubeconfig")

            self.k8s_client = client.CoreV1Api()
            self.custom_api = client.CustomObjectsApi()
            self.pod_manager = PodManager(namespace="default", use_in_cluster_config=use_in_cluster)
            self.vm_manager = VMManager(namespace="default", use_in_cluster_config=use_in_cluster)
            logger.info("Kubernetes client, pod manager, and VM manager initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Kubernetes client: {e}")
            logger.warning("Running without pod management - metrics will be calculated from VMs directly")

    def _calculate_node_metrics_from_pods(self, node_name: str) -> Optional[Dict[str, float]]:
        """
        Calculate node metrics by reading pod annotations on the KWOK node.

        Returns:
            Dictionary with cpu_usage, memory_usage, cpu_pressure, memory_pressure, vm_count
            or None if pod reading is not available
        """
        if self.k8s_client is None:
            return None

        try:
            # List all pods on this node
            pods = self.k8s_client.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={node_name}"
            )

            total_cpu = 0.0
            total_memory = 0.0
            vm_count = 0

            # Aggregate consumption from virt-launcher pods
            for pod in pods.items:
                # Only process virt-launcher pods
                if not pod.metadata.name.startswith("virt-launcher-"):
                    continue

                annotations = pod.metadata.annotations or {}

                # Read VM resource allocation and utilization from annotations
                try:
                    cpu_cores = float(annotations.get("simulation.node-classifier.io/vm-cpu-cores", "0"))
                    memory_bytes = float(annotations.get("simulation.node-classifier.io/vm-memory-bytes", "0"))
                    cpu_util = float(annotations.get("simulation.node-classifier.io/vm-cpu-utilization", "0"))
                    mem_util = float(annotations.get("simulation.node-classifier.io/vm-memory-utilization", "0"))

                    # Calculate actual consumption: cores * utilization / node_capacity
                    # Assuming 32 cores and 128Gi per node
                    node_cpu_cores = 32.0
                    node_memory_bytes = 128 * 1024 ** 3

                    actual_cpu_consumption = (cpu_cores * cpu_util) / node_cpu_cores
                    actual_memory_consumption = (memory_bytes * mem_util) / node_memory_bytes

                    total_cpu += actual_cpu_consumption
                    total_memory += actual_memory_consumption
                    vm_count += 1
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid consumption values in pod {pod.metadata.name}: {e}")
                    continue

            # Calculate pressure from utilization
            # CPU can exceed 1.0, memory cannot
            cpu_usage = min(1.0, total_cpu)
            memory_usage = min(1.0, total_memory)

            # Calculate pressure using the same logic as Node.update_metrics_from_vms()
            if total_cpu <= 1.0:
                cpu_pressure = ScenarioLoader.calculate_pressure_from_utilization(total_cpu)
            else:
                base_pressure = ScenarioLoader.calculate_pressure_from_utilization(1.0)
                overload_factor = total_cpu - 1.0
                cpu_pressure = min(1.0, base_pressure + overload_factor * 0.5)

            memory_pressure = ScenarioLoader.calculate_pressure_from_utilization(memory_usage)

            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "cpu_pressure": cpu_pressure,
                "memory_pressure": memory_pressure,
                "vm_count": vm_count
            }

        except ApiException as e:
            logger.error(f"Error reading pods for node {node_name}: {e}")
            return None

    def update_node_metrics(self, node_name: str):
        """Update Prometheus metrics for a node by reading pod annotations."""
        with self.lock:
            # Try to calculate metrics from pods
            metrics = self._calculate_node_metrics_from_pods(node_name)

            if metrics:
                # Get current time for counter calculations
                current_time = time.time()
                last_time = self.last_update_time.get(node_name, current_time)
                elapsed_seconds = current_time - last_time

                # Update Prometheus gauge metrics from pod data
                cpu_usage_gauge.labels(node=node_name).set(metrics["cpu_usage"])
                cpu_pressure_gauge.labels(node=node_name).set(metrics["cpu_pressure"])
                memory_usage_gauge.labels(node=node_name).set(metrics["memory_usage"])
                memory_pressure_gauge.labels(node=node_name).set(metrics["memory_pressure"])
                vm_count_gauge.labels(node=node_name).set(metrics["vm_count"])

                # Update node-exporter compatible counter metrics
                # Only increment if we have a previous timestamp (not first scrape)
                if node_name in self.last_update_time and elapsed_seconds > 0:
                    # Node-exporter style: counters increment at 1 second/second
                    # We simulate an "aggregate CPU" that represents the node's average utilization
                    # This way rate() will return values in the 0-1 range matching the recording rules

                    # Round elapsed_seconds to eliminate floating point precision errors
                    elapsed_seconds = round(elapsed_seconds, 9)

                    # Calculate busy time components first
                    busy_seconds = round(elapsed_seconds * metrics["cpu_usage"], 9)
                    user_seconds = round(busy_seconds * 0.6, 9)
                    system_seconds = round(busy_seconds * 0.4, 9)

                    # Idle time: ensure total sums to elapsed_seconds to avoid floating point errors
                    # This prevents rate() from returning values > 1.0 or < 0.0
                    idle_seconds = round(elapsed_seconds - user_seconds - system_seconds, 9)

                    # Increment counters
                    cpu_seconds_counter.labels(instance=node_name, mode='idle').inc(idle_seconds)
                    cpu_seconds_counter.labels(instance=node_name, mode='user').inc(user_seconds)
                    cpu_seconds_counter.labels(instance=node_name, mode='system').inc(system_seconds)

                    # CPU pressure counter: pressure is already a ratio (0.0-1.0)
                    # Increment by pressure ratio × elapsed time
                    pressure_seconds = round(metrics["cpu_pressure"] * elapsed_seconds, 9)
                    cpu_pressure_counter.labels(instance=node_name).inc(pressure_seconds)

                # Set node role (always set, not incremented)
                node_role_gauge.labels(node=node_name, role='worker', instance=node_name).set(1)

                # Update last update time
                self.last_update_time[node_name] = current_time

                # Update internal node state if it exists
                if node_name in self.nodes:
                    node = self.nodes[node_name]
                    node.cpu_usage = metrics["cpu_usage"]
                    node.cpu_pressure = metrics["cpu_pressure"]
                    node.memory_usage = metrics["memory_usage"]
                    node.memory_pressure = metrics["memory_pressure"]

                logger.debug(f"Updated metrics for {node_name} from pods: {metrics}")
            else:
                # Fallback to VM-based metrics if available
                if node_name in self.nodes:
                    node = self.nodes[node_name]
                    node.update_metrics_from_vms()
                    self._update_prometheus_metrics_from_node(node)
                    logger.debug(f"Updated metrics for {node_name} from VMs (fallback)")

    def _update_prometheus_metrics_from_node(self, node: Node):
        """Update Prometheus metrics directly from a Node object (fallback)."""
        cpu_usage_gauge.labels(node=node.name).set(node.cpu_usage)
        cpu_pressure_gauge.labels(node=node.name).set(node.cpu_pressure)
        memory_usage_gauge.labels(node=node.name).set(node.memory_usage)
        memory_pressure_gauge.labels(node=node.name).set(node.memory_pressure)
        vm_count_gauge.labels(node=node.name).set(node.vm_count)

    def get_node(self, name: str) -> Optional[Node]:
        """Get a node by name."""
        with self.lock:
            return self.nodes.get(name)

    def get_all_nodes(self) -> List[Node]:
        """Get all nodes."""
        with self.lock:
            return list(self.nodes.values())

    def _sync_vm_crs_to_pods(self):
        """
        Synchronize VirtualMachine CRs with virt-launcher pods.
        Creates pods for VMs that don't have them, updates VM status for existing pods.
        """
        if not self.vm_manager or not self.pod_manager or not self.custom_api:
            return

        try:
            # List all VirtualMachine CRs
            vms_list = self.custom_api.list_namespaced_custom_object(
                group="simulation.node-classifier.io",
                version="v1alpha1",
                namespace="default",
                plural="virtualmachines"
            )

            for vm_obj in vms_list.get("items", []):
                vm_name = vm_obj["metadata"]["name"]
                spec = vm_obj.get("spec", {})
                status = vm_obj.get("status", {})

                # Parse VM resources from spec
                resources = spec.get("resources", {})
                utilization = spec.get("utilization", {})

                cpu_cores = float(resources.get("cpu", "1.0"))
                memory_str = resources.get("memory", "2Gi")

                # Parse memory string (e.g., "4Gi", "2048Mi")
                if memory_str.endswith("Gi"):
                    memory_bytes = int(float(memory_str[:-2]) * 1024**3)
                elif memory_str.endswith("Mi"):
                    memory_bytes = int(float(memory_str[:-2]) * 1024**2)
                else:
                    memory_bytes = int(memory_str)

                cpu_util = float(utilization.get("cpu", "0.5"))
                memory_util = float(utilization.get("memory", "0.5"))

                # Create VM object
                vm = VM(
                    id=vm_name,
                    cpu_cores=cpu_cores,
                    memory_bytes=memory_bytes,
                    cpu_utilization=cpu_util,
                    memory_utilization=memory_util
                )

                # Check if pod exists for this VM
                pod_name = status.get("podName", "")

                if not pod_name or not self._pod_exists(pod_name):
                    # No pod or pod doesn't exist - create one
                    logger.info(f"Creating virt-launcher pod for VM {vm_name}")
                    created_pod_name = self.pod_manager.create_pod(vm)
                    if created_pod_name:
                        vm.pod_name = created_pod_name
                        # Update VM status to Scheduling
                        self.vm_manager.update_vm_status(vm_name, "Scheduling", created_pod_name)
                else:
                    # Pod exists - update VM status from pod
                    vm.pod_name = pod_name
                    self._update_vm_status_from_pod(vm, pod_name)

        except ApiException as e:
            logger.error(f"Failed to sync VM CRs to pods: {e}")
        except Exception as e:
            logger.error(f"Unexpected error syncing VM CRs: {e}", exc_info=True)

    def _pod_exists(self, pod_name: str) -> bool:
        """Check if a pod exists."""
        try:
            self.k8s_client.read_namespaced_pod(name=pod_name, namespace="default")
            return True
        except ApiException:
            return False

    def _update_vm_status_from_pod(self, vm: VM, pod_name: str):
        """Update VM status based on pod status."""
        try:
            pod = self.k8s_client.read_namespaced_pod(name=pod_name, namespace="default")
            node_name = pod.spec.node_name or ""

            # Update VM's scheduled_node field
            vm.scheduled_node = node_name

            # Determine phase
            if pod.status.phase == "Running":
                phase = "Running"
            elif pod.status.phase == "Pending":
                phase = "Scheduling" if not node_name else "Scheduled"
            elif pod.status.phase in ["Failed", "Unknown"]:
                phase = "Failed"
            else:
                phase = "Pending"

            self.vm_manager.update_vm_status(vm.id, phase, pod_name, node_name)

        except ApiException as e:
            logger.error(f"Failed to read pod {pod_name}: {e}")

    def _vm_controller_loop(self):
        """Background loop that watches VirtualMachine CRs and creates pods."""
        import time

        logger.info("VM controller started")

        while self.vm_controller_running:
            try:
                self._sync_vm_crs_to_pods()
            except Exception as e:
                logger.error(f"Error in VM controller loop: {e}", exc_info=True)

            # Sleep for a bit before next sync
            time.sleep(5)

        logger.info("VM controller stopped")

    def start_vm_controller(self):
        """Start the VM controller in a background thread."""
        if self.vm_controller_running:
            logger.warning("VM controller already running")
            return

        if not self.vm_manager or not self.pod_manager:
            logger.warning("Cannot start VM controller: managers not initialized")
            return

        self.vm_controller_running = True
        self.vm_controller_thread = threading.Thread(
            target=self._vm_controller_loop,
            daemon=True,
            name="vm-controller"
        )
        self.vm_controller_thread.start()
        logger.info("VM controller thread started")

    def stop_vm_controller(self):
        """Stop the VM controller."""
        if not self.vm_controller_running:
            return

        self.vm_controller_running = False
        if self.vm_controller_thread:
            self.vm_controller_thread.join(timeout=10)
        logger.info("VM controller stopped")

    def load_scenario(self, nodes: List[Node]):
        """Load a complete scenario and create pods for all VMs."""
        with self.lock:
            self.nodes.clear()
            for node in nodes:
                self.nodes[node.name] = node

            # Collect all VMs from all nodes
            all_vms = []
            for node in nodes:
                all_vms.extend(node.vms)

            # Create pods for all VMs if pod manager is available
            # The scheduler will decide where to place them
            if self.pod_manager:
                logger.info(f"Creating virt-launcher pods for {len(all_vms)} VMs (scheduler will assign nodes)...")
                stats = self.pod_manager.sync_pods_with_vms(all_vms)
                logger.info(f"Pod sync complete: {stats}")

                # Wait a bit for scheduler to assign pods, then update VM node assignments
                import time
                time.sleep(2)  # Give scheduler time to work
                assign_stats = self.pod_manager.update_vm_node_assignments(all_vms)
                logger.info(f"VM node assignments updated: {assign_stats}")

                # Reorganize VMs into nodes based on actual scheduler assignments
                for node in nodes:
                    node.vms.clear()

                for vm in all_vms:
                    if vm.scheduled_node:
                        if vm.scheduled_node in self.nodes:
                            self.nodes[vm.scheduled_node].vms.append(vm)
                        else:
                            logger.warning(f"VM {vm.id} scheduled to unknown node {vm.scheduled_node}")

            # Update metrics for all nodes
            for node in nodes:
                self.update_node_metrics(node.name)

    def move_vm(self, vm_id: str, from_node: str, to_node: str) -> bool:
        """
        Move a VM from one node to another.
        This triggers pod recreation - the scheduler should place it on the target node
        based on resource availability.

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            source = self.nodes.get(from_node)

            if source is None:
                logger.error(f"Cannot move VM: source={from_node} not found")
                return False

            # Find the VM
            vm_to_move = None
            for vm in source.vms:
                if vm.id == vm_id:
                    vm_to_move = vm
                    break

            if vm_to_move is None:
                logger.error(f"VM {vm_id} not found on node {from_node}")
                return False

            # Migrate the pod if pod manager is available
            # Delete old pod and create new one - scheduler will assign it
            if self.pod_manager:
                logger.info(f"Migrating VM {vm_id}: deleting old pod, scheduler will assign new one")
                success = self.pod_manager.migrate_vm_pod(vm_to_move, from_node, to_node)
                if not success:
                    logger.error(f"Failed to migrate pod for VM {vm_id}")
                    return False

                # Wait for scheduler to assign the new pod
                import time
                time.sleep(1)

                # Update VM's node assignment from the pod
                actual_node = self.pod_manager.get_pod_node_assignment(vm_id)
                if actual_node:
                    vm_to_move.scheduled_node = actual_node
                    logger.info(f"VM {vm_id} pod scheduled to {actual_node}")

                    # Move VM in internal state to match scheduler's decision
                    source.vms.remove(vm_to_move)
                    if actual_node in self.nodes:
                        self.nodes[actual_node].vms.append(vm_to_move)

                        # Update metrics for affected nodes
                        self.update_node_metrics(from_node)
                        self.update_node_metrics(actual_node)

                        logger.info(f"Moved VM {vm_id} from {from_node} to {actual_node}")
                        return True
                    else:
                        logger.error(f"VM {vm_id} scheduled to unknown node {actual_node}")
                        # Re-add to source to avoid losing the VM
                        source.vms.append(vm_to_move)
                        return False
                else:
                    logger.error(f"VM {vm_id} pod not scheduled after migration")
                    return False
            else:
                # Fallback: direct move without pod manager
                dest = self.nodes.get(to_node)
                if dest is None:
                    logger.error(f"Cannot move VM: dest={to_node} not found")
                    return False

                source.vms.remove(vm_to_move)
                dest.vms.append(vm_to_move)
                vm_to_move.scheduled_node = to_node

                logger.info(f"Moved VM {vm_id} from {from_node} to {to_node} (without pod manager)")
                return True


# Global state
state = ExporterState()


@app.route('/metrics')
def metrics():
    """
    Prometheus scrape endpoint.
    Refreshes metrics from pods before serving.
    """
    # Discover all KWOK nodes and update metrics for them
    updated_nodes = set()
    if state.k8s_client:
        try:
            nodes = state.k8s_client.list_node(label_selector="type=kwok")
            logger.info(f"Found {len(nodes.items)} KWOK nodes to update metrics")
            for node in nodes.items:
                node_name = node.metadata.name
                logger.info(f"Updating metrics for {node_name}")
                state.update_node_metrics(node_name)
                updated_nodes.add(node_name)
        except Exception as e:
            logger.error(f"Failed to list KWOK nodes: {e}", exc_info=True)
    else:
        logger.warning("k8s_client not available, cannot discover KWOK nodes")

    # Also refresh metrics for any nodes already in state that weren't updated above
    for node_name in list(state.nodes.keys()):
        if node_name not in updated_nodes:
            state.update_node_metrics(node_name)
            updated_nodes.add(node_name)

    logger.info(f"Metrics endpoint: updated {len(updated_nodes)} nodes")
    return Response(generate_latest(registry), mimetype='text/plain')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'nodes': len(state.nodes),
        'pod_manager_enabled': state.pod_manager is not None
    })


@app.route('/refresh', methods=['POST'])
def refresh_metrics():
    """
    Manually refresh metrics for all nodes from pods.

    Returns statistics about the refresh operation.
    """
    try:
        refreshed = []
        failed = []

        for node_name in list(state.nodes.keys()):
            try:
                state.update_node_metrics(node_name)
                refreshed.append(node_name)
            except Exception as e:
                logger.error(f"Failed to refresh metrics for {node_name}: {e}")
                failed.append(node_name)

        return jsonify({
            'status': 'success',
            'refreshed': refreshed,
            'failed': failed,
            'total': len(refreshed) + len(failed)
        })

    except Exception as e:
        logger.error(f"Error refreshing metrics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Receive feedback from the simulator about VM migrations.

    Expected JSON:
    {
        "migrations": [
            {"vm_id": "vm-1", "from_node": "node-1", "to_node": "node-2"},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'migrations' not in data:
            return jsonify({'error': 'Missing migrations field'}), 400

        migrations = data['migrations']
        results = []

        for migration in migrations:
            vm_id = migration.get('vm_id')
            from_node = migration.get('from_node')
            to_node = migration.get('to_node')

            if not all([vm_id, from_node, to_node]):
                results.append({
                    'vm_id': vm_id,
                    'success': False,
                    'error': 'Missing required fields'
                })
                continue

            success = state.move_vm(vm_id, from_node, to_node)
            results.append({
                'vm_id': vm_id,
                'from_node': from_node,
                'to_node': to_node,
                'success': success
            })

        return jsonify({'results': results})

    except Exception as e:
        logger.error(f"Error processing feedback: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/scenario', methods=['POST'])
def load_scenario():
    """
    Load a new scenario.

    Expected JSON:
    {
        "nodes": [
            {
                "name": "node-1",
                "cpu_usage": 0.5,
                "cpu_pressure": 0.1,
                "memory_usage": 0.6,
                "memory_pressure": 0.15,
                "vms": [...]
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'nodes' not in data:
            return jsonify({'error': 'Missing nodes field'}), 400

        nodes = [Node.from_dict(node_data) for node_data in data['nodes']]
        state.load_scenario(nodes)

        logger.info(f"Loaded scenario with {len(nodes)} nodes")
        return jsonify({
            'status': 'success',
            'nodes_loaded': len(nodes),
            'total_vms': sum(node.vm_count for node in nodes)
        })

    except Exception as e:
        logger.error(f"Error loading scenario: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/scenario', methods=['GET'])
def get_scenario():
    """Get the current scenario state."""
    nodes = state.get_all_nodes()
    return jsonify({
        'nodes': [node.to_dict() for node in nodes]
    })


@app.route('/nodes/<node_name>', methods=['GET'])
def get_node(node_name: str):
    """Get details of a specific node."""
    node = state.get_node(node_name)
    if node is None:
        return jsonify({'error': f'Node {node_name} not found'}), 404

    return jsonify(node.to_dict())


def initialize_from_file(scenario_file: str):
    """Initialize the exporter with a scenario from a file."""
    try:
        logger.info(f"Loading initial scenario from {scenario_file}")
        loader = ScenarioLoader(scenario_file)

        # Get the first available scenario
        scenarios = loader.list_scenarios()
        if not scenarios:
            logger.warning("No scenarios found in file, starting empty")
            return

        scenario_name = scenarios[0]
        nodes = loader.load_scenario(scenario_name)
        state.load_scenario(nodes)

        logger.info(f"Initialized with scenario '{scenario_name}' ({len(nodes)} nodes)")

    except Exception as e:
        logger.error(f"Error initializing from file: {e}", exc_info=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prometheus Metrics Exporter for Node Simulation')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--scenario', type=str, default='sample_scenarios.json',
                        help='Initial scenario file to load')

    args = parser.parse_args()

    # Initialize with scenario if provided
    initialize_from_file(args.scenario)

    # Start VM controller to watch for VirtualMachine CRs
    state.start_vm_controller()

    logger.info(f"Starting Prometheus Exporter on {args.host}:{args.port}")
    try:
        app.run(host=args.host, port=args.port)
    finally:
        # Clean shutdown
        state.stop_vm_controller()