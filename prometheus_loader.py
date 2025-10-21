"""
Prometheus Query Client for Node Classifier Simulation

Loads node metrics from a Prometheus instance and converts them to Node objects.
Supports both raw metrics and recording rules.
"""

from typing import List, Optional, Dict, Any
import requests
from datetime import datetime
import logging

from node import Node, VM

logger = logging.getLogger(__name__)


class PrometheusLoader:
    """
    Loads node metrics from Prometheus and converts to Node objects.
    """

    def __init__(self, prometheus_url: str = 'http://localhost:9090'):
        """
        Initialize the Prometheus loader.

        Args:
            prometheus_url: Base URL for Prometheus API (e.g., http://localhost:9090)
        """
        self.prometheus_url = prometheus_url.rstrip('/')
        self.api_url = f"{self.prometheus_url}/api/v1"

    def query(self, promql: str) -> Dict[str, Any]:
        """
        Execute a PromQL query.

        Args:
            promql: PromQL query string

        Returns:
            Query result as dictionary

        Raises:
            requests.RequestException: If the query fails
        """
        url = f"{self.api_url}/query"
        params = {'query': promql}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data['status'] != 'success':
                raise ValueError(f"Prometheus query failed: {data.get('error', 'Unknown error')}")

            return data['data']

        except requests.RequestException as e:
            logger.error(f"Failed to query Prometheus: {e}")
            raise

    def load_nodes(self, use_recording_rules: bool = False) -> List[Node]:
        """
        Load all nodes from Prometheus.

        Args:
            use_recording_rules: If True, use pre-aggregated recording rules instead of raw metrics

        Returns:
            List of Node objects with current metrics
        """
        if use_recording_rules:
            return self._load_nodes_from_recording_rules()
        else:
            return self._load_nodes_from_raw_metrics()

    def _load_nodes_from_raw_metrics(self) -> List[Node]:
        """
        Load nodes from raw Prometheus metrics.

        Queries:
        - node_cpu_usage_ratio{node=~".+"}
        - node_cpu_pressure_psi{node=~".+"}
        - node_memory_usage_ratio{node=~".+"}
        - node_memory_pressure_psi{node=~".+"}
        - node_vm_count{node=~".+"}

        Returns:
            List of Node objects
        """
        # Define metric queries
        metric_queries = {
            'cpu_usage': 'node_cpu_usage_ratio',
            'cpu_pressure': 'node_cpu_pressure_psi',
            'memory_usage': 'node_memory_usage_ratio',
            'memory_pressure': 'node_memory_pressure_psi',
            'vm_count': 'node_vm_count'
        }

        # Fetch all metrics
        metric_data: Dict[str, Dict[str, float]] = {}

        for metric_name, promql in metric_queries.items():
            try:
                result = self.query(promql)
                metric_data[metric_name] = self._parse_metric_result(result)
            except Exception as e:
                logger.error(f"Failed to fetch {metric_name}: {e}")
                metric_data[metric_name] = {}

        # Get all node names
        all_nodes = set()
        for metrics in metric_data.values():
            all_nodes.update(metrics.keys())

        if not all_nodes:
            logger.warning("No nodes found in Prometheus")
            return []

        # Build Node objects
        nodes = []
        for node_name in sorted(all_nodes):
            try:
                node = Node(
                    name=node_name,
                    cpu_usage=metric_data['cpu_usage'].get(node_name, 0.0),
                    cpu_pressure=metric_data['cpu_pressure'].get(node_name, 0.0),
                    memory_usage=metric_data['memory_usage'].get(node_name, 0.0),
                    memory_pressure=metric_data['memory_pressure'].get(node_name, 0.0),
                    vms=[]  # VMs are not tracked in Prometheus (only count)
                )
                nodes.append(node)
                logger.debug(f"Loaded node {node_name}: CPU={node.cpu_usage:.2f}, Mem={node.memory_usage:.2f}")

            except ValueError as e:
                logger.error(f"Invalid metrics for node {node_name}: {e}")
                continue

        logger.info(f"Loaded {len(nodes)} nodes from Prometheus")
        return nodes

    def _load_nodes_from_recording_rules(self) -> List[Node]:
        """
        Load nodes from Prometheus recording rules.

        Expects recording rules like:
        - node:cpu_usage:ratio
        - node:cpu_pressure:psi
        - node:memory_usage:ratio
        - node:memory_pressure:psi

        Returns:
            List of Node objects
        """
        # Define recording rule metric names
        metric_queries = {
            'cpu_usage': 'node:cpu_usage:ratio',
            'cpu_pressure': 'node:cpu_pressure:psi',
            'memory_usage': 'node:memory_usage:ratio',
            'memory_pressure': 'node:memory_pressure:psi',
        }

        # Fetch all metrics
        metric_data: Dict[str, Dict[str, float]] = {}

        for metric_name, promql in metric_queries.items():
            try:
                result = self.query(promql)
                metric_data[metric_name] = self._parse_metric_result(result)
            except Exception as e:
                logger.warning(f"Failed to fetch recording rule {metric_name}: {e}")
                metric_data[metric_name] = {}

        # Get all node names
        all_nodes = set()
        for metrics in metric_data.values():
            all_nodes.update(metrics.keys())

        if not all_nodes:
            logger.warning("No nodes found in Prometheus recording rules, falling back to raw metrics")
            return self._load_nodes_from_raw_metrics()

        # Build Node objects
        nodes = []
        for node_name in sorted(all_nodes):
            try:
                node = Node(
                    name=node_name,
                    cpu_usage=metric_data['cpu_usage'].get(node_name, 0.0),
                    cpu_pressure=metric_data['cpu_pressure'].get(node_name, 0.0),
                    memory_usage=metric_data['memory_usage'].get(node_name, 0.0),
                    memory_pressure=metric_data['memory_pressure'].get(node_name, 0.0),
                    vms=[]
                )
                nodes.append(node)

            except ValueError as e:
                logger.error(f"Invalid metrics for node {node_name}: {e}")
                continue

        logger.info(f"Loaded {len(nodes)} nodes from recording rules")
        return nodes

    def _parse_metric_result(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Parse Prometheus query result into a dict of node -> value.

        Args:
            result: Prometheus query result data

        Returns:
            Dictionary mapping node name to metric value
        """
        metrics = {}

        if result['resultType'] != 'vector':
            logger.warning(f"Unexpected result type: {result['resultType']}")
            return metrics

        for item in result['result']:
            # Extract node label
            node_name = item['metric'].get('node')
            if not node_name:
                logger.warning(f"Metric missing 'node' label: {item['metric']}")
                continue

            # Extract value
            try:
                value = float(item['value'][1])
                metrics[node_name] = value
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse value for {node_name}: {e}")
                continue

        return metrics

    def check_health(self) -> bool:
        """
        Check if Prometheus is reachable and healthy.

        Returns:
            True if Prometheus is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_prometheus_info(self) -> Optional[Dict[str, Any]]:
        """
        Get Prometheus build information.

        Returns:
            Build info dictionary or None if failed
        """
        try:
            result = self.query('prometheus_build_info')
            return result
        except Exception as e:
            logger.error(f"Failed to get Prometheus info: {e}")
            return None


if __name__ == '__main__':
    """Simple test/demo of the Prometheus loader."""
    import argparse

    parser = argparse.ArgumentParser(description='Load nodes from Prometheus')
    parser.add_argument('--url', default='http://localhost:9090',
                        help='Prometheus URL')
    parser.add_argument('--recording-rules', action='store_true',
                        help='Use recording rules instead of raw metrics')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    loader = PrometheusLoader(args.url)

    # Check health
    if not loader.check_health():
        print(f"ERROR: Prometheus at {args.url} is not reachable")
        exit(1)

    print(f"Connected to Prometheus at {args.url}")

    # Load nodes
    nodes = loader.load_nodes(use_recording_rules=args.recording_rules)

    if not nodes:
        print("No nodes found")
        exit(0)

    # Display nodes
    print(f"\nFound {len(nodes)} nodes:\n")
    print(f"{'Node':<20} {'CPU Usage':<12} {'CPU PSI':<12} {'Mem Usage':<12} {'Mem PSI':<12}")
    print("-" * 68)

    for node in nodes:
        print(f"{node.name:<20} "
              f"{node.cpu_usage:<12.3f} "
              f"{node.cpu_pressure:<12.3f} "
              f"{node.memory_usage:<12.3f} "
              f"{node.memory_pressure:<12.3f}")