#!/usr/bin/env python3
"""
Closed-Loop Simulation CLI with Prometheus Integration

Runs the node classifier simulator in closed-loop mode:
1. Loads metrics from Prometheus
2. Classifies nodes
3. Simulates VM migrations
4. Sends feedback to metrics exporter
5. Waits for metrics to update
6. Repeats
"""

import argparse
import sys
import time
import logging
from typing import Optional

from prometheus_loader import PrometheusLoader
from prometheus_feedback import PrometheusFeedbackClient
from classifier import NodeClassifier, ThresholdMode, ThresholdConfig
from simulator import Simulator, SimulationConfig
from algorithms import get_default_algorithms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClosedLoopSimulator:
    """
    Runs the node classifier in closed-loop mode with Prometheus.
    """

    def __init__(
        self,
        prometheus_url: str,
        exporter_url: str,
        algorithm_name: str,
        threshold_mode: str = 'Medium',
        use_recording_rules: bool = False
    ):
        """
        Initialize the closed-loop simulator.

        Args:
            prometheus_url: Prometheus API URL
            exporter_url: Metrics exporter URL
            algorithm_name: Name of classification algorithm to use
            threshold_mode: Threshold mode (Low, Medium, High, Asymmetric)
            use_recording_rules: Whether to use recording rules
        """
        self.prom_loader = PrometheusLoader(prometheus_url)
        self.feedback_client = PrometheusFeedbackClient(exporter_url)
        self.use_recording_rules = use_recording_rules

        # Get algorithm
        algorithms = {alg.name: alg for alg in get_default_algorithms()}
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(algorithms.keys())}")

        algorithm = algorithms[algorithm_name]

        # Create classifier
        threshold_config = ThresholdConfig.from_mode(ThresholdMode[threshold_mode.upper()])
        self.classifier = NodeClassifier(algorithm=algorithm, threshold_config=threshold_config)

        # Simulation config
        self.sim_config = SimulationConfig()

        # State
        self.simulator: Optional[Simulator] = None
        self.current_step = 0

    def check_connectivity(self) -> bool:
        """
        Check if Prometheus and exporter are reachable.

        Returns:
            True if both are healthy, False otherwise
        """
        logger.info("Checking connectivity...")

        prom_ok = self.prom_loader.check_health()
        if not prom_ok:
            logger.error(f"Prometheus is not reachable")
            return False

        exporter_ok = self.feedback_client.check_health()
        if not exporter_ok:
            logger.error(f"Metrics exporter is not reachable")
            return False

        logger.info("Connectivity check passed")
        return True

    def initialize(self) -> bool:
        """
        Initialize the simulator by loading initial state from Prometheus.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing simulator from Prometheus metrics...")

        try:
            nodes = self._load_merged_state()

            if not nodes:
                logger.error("No nodes loaded")
                return False

            logger.info(f"Loaded {len(nodes)} nodes with {sum(node.vm_count for node in nodes)} total VMs")

            # Create simulator
            self.simulator = Simulator(
                initial_nodes=nodes,
                classifier=self.classifier,
                config=self.sim_config
            )

            self.current_step = 0
            return True

        except Exception as e:
            logger.error(f"Failed to initialize: {e}", exc_info=True)
            return False

    def _load_merged_state(self):
        """
        Load state by merging Prometheus metrics with exporter VM state.

        Returns:
            List of Node objects with metrics from Prometheus and VMs from exporter
        """
        from node import Node, VM

        # Get metrics from Prometheus
        prom_nodes = self.prom_loader.load_nodes(use_recording_rules=self.use_recording_rules)
        logger.info(f"Loaded {len(prom_nodes)} nodes from Prometheus")

        # Get VM state from exporter
        exporter_state = self.feedback_client.get_current_state()
        if not exporter_state or 'nodes' not in exporter_state:
            logger.warning("No VM state from exporter, using metrics-only nodes")
            return prom_nodes

        logger.info(f"Loaded VM state for {len(exporter_state['nodes'])} nodes from exporter")

        # Create mapping of exporter nodes by name
        exporter_nodes_map = {node_data['name']: node_data for node_data in exporter_state['nodes']}

        # Merge: use Prometheus metrics but exporter VMs
        merged_nodes = []
        for prom_node in prom_nodes:
            if prom_node.name in exporter_nodes_map:
                exporter_node_data = exporter_nodes_map[prom_node.name]
                vms = [VM.from_dict(vm_data) for vm_data in exporter_node_data.get('vms', [])]

                # Create node with Prometheus metrics but exporter VMs
                merged_node = Node(
                    name=prom_node.name,
                    cpu_usage=prom_node.cpu_usage,
                    cpu_pressure=prom_node.cpu_pressure,
                    memory_usage=prom_node.memory_usage,
                    memory_pressure=prom_node.memory_pressure,
                    vms=vms
                )
                merged_nodes.append(merged_node)
                logger.debug(f"Merged {prom_node.name}: {len(vms)} VMs")
            else:
                # No VMs for this node in exporter
                merged_nodes.append(prom_node)

        return merged_nodes

    def run_step(self) -> bool:
        """
        Run one simulation step:
        1. Classify nodes
        2. Determine VM migrations
        3. Send feedback to exporter
        4. Wait for metrics to stabilize

        Returns:
            True if step completed, False if simulation should stop
        """
        if self.simulator is None:
            logger.error("Simulator not initialized")
            return False

        self.current_step += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP {self.current_step}")
        logger.info(f"{'='*60}")

        # Run simulation step
        step_result = self.simulator.step()

        # Display results
        logger.info(f"Moved {step_result.total_vms_moved} VMs")

        # Display classification
        for result in step_result.classification_results:
            logger.info(f"  {result.node.name}: {result.category.name} (score={result.score:.3f})")

        # Display moves
        if step_result.moves:
            logger.info("Migrations:")
            for move in step_result.moves:
                logger.info(f"  {move.vm_id}: {move.from_node} -> {move.to_node}")

        # Send feedback to exporter
        if step_result.moves:
            logger.info("Sending feedback to metrics exporter...")
            success = self.feedback_client.send_migrations(step_result.moves)
            if not success:
                logger.warning("Failed to send feedback to exporter")
                return False

            # Wait for metrics to update
            logger.info("Waiting for metrics to stabilize (15s)...")
            time.sleep(15)

            # Reload merged state (metrics + VMs)
            logger.info("Reloading state from Prometheus and exporter...")
            nodes = self._load_merged_state()
            if not nodes:
                logger.error("Failed to reload state")
                return False

            # Update simulator with new state
            self.simulator.reset(nodes)

        else:
            logger.info("No migrations needed - system is balanced!")
            return False

        return True

    def run(self, max_steps: int = 10, step_delay: int = 0) -> bool:
        """
        Run the closed-loop simulation.

        Args:
            max_steps: Maximum number of simulation steps
            step_delay: Delay between steps (in addition to metric stabilization)

        Returns:
            True if successful, False otherwise
        """
        if not self.check_connectivity():
            return False

        if not self.initialize():
            return False

        logger.info(f"Starting closed-loop simulation (max {max_steps} steps)")

        for step in range(max_steps):
            should_continue = self.run_step()

            if not should_continue:
                logger.info("Simulation converged or stopped")
                break

            if step_delay > 0:
                logger.info(f"Waiting {step_delay}s before next step...")
                time.sleep(step_delay)

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMULATION COMPLETE")
        logger.info(f"{'='*60}")

        summary = self.simulator.get_move_summary()
        logger.info(f"Total steps: {self.current_step}")
        logger.info(f"Total migrations: {summary['total_moves']}")
        logger.info(f"Source nodes: {summary['unique_source_nodes']}")
        logger.info(f"Destination nodes: {summary['unique_destination_nodes']}")

        return True


def main():
    parser = argparse.ArgumentParser(
        description='Closed-loop node classifier simulation with Prometheus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python cli_prometheus.py

  # Use specific algorithm
  python cli_prometheus.py --algorithm "Weighted Average Distance"

  # Use recording rules
  python cli_prometheus.py --recording-rules

  # Run with custom URLs
  python cli_prometheus.py --prometheus http://localhost:9090 --exporter http://localhost:8000

  # Run with more steps
  python cli_prometheus.py --max-steps 20 --step-delay 5
        """
    )

    parser.add_argument(
        '--prometheus',
        default='http://localhost:9090',
        help='Prometheus URL (default: http://localhost:9090)'
    )

    parser.add_argument(
        '--exporter',
        default='http://localhost:8000',
        help='Metrics exporter URL (default: http://localhost:8000)'
    )

    parser.add_argument(
        '--algorithm',
        default='Ideal Point Positive Distance',
        help='Classification algorithm name (default: Ideal Point Positive Distance)'
    )

    parser.add_argument(
        '--threshold',
        default='Medium',
        choices=['Low', 'Medium', 'High', 'Asymmetric'],
        help='Threshold mode (default: Medium)'
    )

    parser.add_argument(
        '--recording-rules',
        action='store_true',
        help='Use Prometheus recording rules instead of raw metrics'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        default=10,
        help='Maximum simulation steps (default: 10)'
    )

    parser.add_argument(
        '--step-delay',
        type=int,
        default=0,
        help='Additional delay between steps in seconds (default: 0)'
    )

    parser.add_argument(
        '--list-algorithms',
        action='store_true',
        help='List available algorithms and exit'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List algorithms
    if args.list_algorithms:
        print("Available algorithms:")
        for alg in get_default_algorithms():
            print(f"  - {alg.name}")
        return 0

    # Run simulation
    try:
        sim = ClosedLoopSimulator(
            prometheus_url=args.prometheus,
            exporter_url=args.exporter,
            algorithm_name=args.algorithm,
            threshold_mode=args.threshold,
            use_recording_rules=args.recording_rules
        )

        success = sim.run(
            max_steps=args.max_steps,
            step_delay=args.step_delay
        )

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())