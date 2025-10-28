#!/usr/bin/env python3
"""
Synthetic Metrics Exporter for Node Classifier Simulation

Exposes Prometheus metrics for KWOK nodes with dynamic VM-based resource consumption.
Metrics are calculated by reading virt-launcher pod annotations on KWOK nodes.
Receives feedback from the simulator to update metrics based on VM migrations.
"""

from flask import Flask, Response, request, jsonify
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import json
import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from node import Node, VM
from scenario_loader import ScenarioLoader
from pod_manager import PodManager

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


@dataclass
class ExporterState:
    """Thread-safe state for the exporter."""
    nodes: Dict[str, Node] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pod_manager: Optional[PodManager] = None
    k8s_client: Optional[client.CoreV1Api] = None

    def __post_init__(self):
        """Initialize Kubernetes client and pod manager."""
        try:
            config.load_kube_config()
            self.k8s_client = client.CoreV1Api()
            self.pod_manager = PodManager(namespace="default", use_in_cluster_config=False)
            logger.info("Kubernetes client and pod manager initialized")
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
                # Update Prometheus metrics from pod data
                cpu_usage_gauge.labels(node=node_name).set(metrics["cpu_usage"])
                cpu_pressure_gauge.labels(node=node_name).set(metrics["cpu_pressure"])
                memory_usage_gauge.labels(node=node_name).set(metrics["memory_usage"])
                memory_pressure_gauge.labels(node=node_name).set(metrics["memory_pressure"])
                vm_count_gauge.labels(node=node_name).set(metrics["vm_count"])

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
    # Refresh metrics for all known nodes
    for node_name in list(state.nodes.keys()):
        state.update_node_metrics(node_name)

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

    logger.info(f"Starting Prometheus Exporter on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)