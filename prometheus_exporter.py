#!/usr/bin/env python3
"""
Synthetic Metrics Exporter for Node Classifier Simulation

Exposes Prometheus metrics for KWOK nodes with dynamic VM-based resource consumption.
Receives feedback from the simulator to update metrics based on VM migrations.
"""

from flask import Flask, Response, request, jsonify
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import json
import logging

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


@dataclass
class ExporterState:
    """Thread-safe state for the exporter."""
    nodes: Dict[str, Node] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_node(self, node: Node):
        """Update a node's state and recalculate metrics."""
        with self.lock:
            self.nodes[node.name] = node
            node.update_metrics_from_vms()
            self._update_prometheus_metrics(node)

    def get_node(self, name: str) -> Optional[Node]:
        """Get a node by name."""
        with self.lock:
            return self.nodes.get(name)

    def get_all_nodes(self) -> List[Node]:
        """Get all nodes."""
        with self.lock:
            return list(self.nodes.values())

    def load_scenario(self, nodes: List[Node]):
        """Load a complete scenario."""
        with self.lock:
            self.nodes.clear()
            for node in nodes:
                self.nodes[node.name] = node
                # Only update metrics from VMs if VMs are present
                # Otherwise, use the metrics from the scenario data
                if node.vms:
                    node.update_metrics_from_vms()
                self._update_prometheus_metrics(node)

    def _update_prometheus_metrics(self, node: Node):
        """Update Prometheus metrics for a node."""
        cpu_usage_gauge.labels(node=node.name).set(node.cpu_usage)
        cpu_pressure_gauge.labels(node=node.name).set(node.cpu_pressure)
        memory_usage_gauge.labels(node=node.name).set(node.memory_usage)
        memory_pressure_gauge.labels(node=node.name).set(node.memory_pressure)
        vm_count_gauge.labels(node=node.name).set(node.vm_count)

    def move_vm(self, vm_id: str, from_node: str, to_node: str) -> bool:
        """
        Move a VM from one node to another.

        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            source = self.nodes.get(from_node)
            dest = self.nodes.get(to_node)

            if source is None or dest is None:
                logger.error(f"Cannot move VM: source={from_node} or dest={to_node} not found")
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

            # Check memory constraint
            current_memory = sum(vm.memory_consumption for vm in dest.vms)
            if current_memory + vm_to_move.memory_consumption > 1.0:
                logger.warning(f"Cannot move VM {vm_id}: would exceed memory on {to_node}")
                return False

            # Perform the move
            source.vms.remove(vm_to_move)
            dest.vms.append(vm_to_move)

            # Update metrics
            source.update_metrics_from_vms()
            dest.update_metrics_from_vms()
            self._update_prometheus_metrics(source)
            self._update_prometheus_metrics(dest)

            logger.info(f"Moved VM {vm_id} from {from_node} to {to_node}")
            return True


# Global state
state = ExporterState()


@app.route('/metrics')
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(registry), mimetype='text/plain')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'nodes': len(state.nodes)})


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