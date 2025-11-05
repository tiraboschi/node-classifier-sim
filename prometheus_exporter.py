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
    """Thread-safe state for the exporter - READ-ONLY metrics collection."""
    nodes: Dict[str, Node] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    k8s_client: Optional[client.CoreV1Api] = None
    # Track last update time for each node to calculate counter increments
    last_update_time: Dict[str, float] = field(default_factory=dict)
    # Node configuration
    node_cpu_cores: float = 32.0  # 32 CPU cores per KWOK node

    def __post_init__(self):
        """Initialize Kubernetes client for READ-ONLY metrics collection."""
        try:
            # Try in-cluster config first (when running as pod), fallback to kubeconfig
            try:
                config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes config")
            except Exception:
                config.load_kube_config()
                logger.info("Using local kubeconfig")

            self.k8s_client = client.CoreV1Api()
            logger.info("Kubernetes client initialized for metrics collection (read-only)")
        except Exception as e:
            logger.error(f"Could not initialize Kubernetes client: {e}")
            logger.error("Metrics exporter requires Kubernetes access to read pods")
            raise

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

    def load_scenario(self, nodes: List[Node]):
        """
        Load a scenario for metrics tracking.

        NOTE: This does NOT create pods - that's the vm-controller's job.
        This only initializes the internal node state for fallback metrics.
        """
        with self.lock:
            self.nodes.clear()
            for node in nodes:
                self.nodes[node.name] = node

            logger.info(f"Loaded scenario with {len(nodes)} nodes (read-only)")

            # Update metrics for all nodes
            for node in nodes:
                self.update_node_metrics(node.name)

    def move_vm(self, vm_id: str, from_node: str, to_node: str) -> bool:
        """
        DEPRECATED: Metrics exporter no longer manages VMs.
        VM migrations are handled by eviction-webhook.

        This method is kept for backward compatibility but does nothing.
        """
        logger.warning(f"move_vm() called but metrics-exporter is read-only. Use eviction-webhook for migrations.")
        return False


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
        'k8s_client_ready': state.k8s_client is not None
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

    parser = argparse.ArgumentParser(description='Prometheus Metrics Exporter (Read-Only)')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--scenario', type=str, default='sample_scenarios.json',
                        help='Initial scenario file to load (optional, for fallback metrics)')

    args = parser.parse_args()

    # Initialize with scenario if provided (optional - only for fallback metrics)
    if args.scenario:
        try:
            initialize_from_file(args.scenario)
        except Exception as e:
            logger.warning(f"Could not load scenario file: {e}")
            logger.warning("Continuing without scenario - will read metrics from pods only")

    logger.info("=" * 60)
    logger.info("Prometheus Metrics Exporter - READ-ONLY MODE")
    logger.info("This component ONLY reads pods and exports metrics")
    logger.info("VM/Pod management is handled by vm-controller")
    logger.info("Migrations are handled by eviction-webhook")
    logger.info("=" * 60)
    logger.info(f"Starting Prometheus Exporter on {args.host}:{args.port}")

    app.run(host=args.host, port=args.port)