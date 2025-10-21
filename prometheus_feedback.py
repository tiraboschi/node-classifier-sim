"""
Prometheus Feedback Client

Sends VM migration feedback to the metrics exporter to update metrics dynamically.
"""

import requests
import logging
from typing import List
from simulator import VMMove

logger = logging.getLogger(__name__)


class PrometheusFeedbackClient:
    """
    Client for sending feedback to the metrics exporter.
    """

    def __init__(self, exporter_url: str = 'http://localhost:8000'):
        """
        Initialize the feedback client.

        Args:
            exporter_url: Base URL for the metrics exporter (e.g., http://localhost:8000)
        """
        self.exporter_url = exporter_url.rstrip('/')

    def send_migrations(self, moves: List[VMMove]) -> bool:
        """
        Send VM migration events to the exporter.

        Args:
            moves: List of VMMove objects

        Returns:
            True if successful, False otherwise
        """
        if not moves:
            logger.debug("No migrations to send")
            return True

        migrations = [
            {
                'vm_id': move.vm_id,
                'from_node': move.from_node,
                'to_node': move.to_node
            }
            for move in moves
        ]

        try:
            url = f"{self.exporter_url}/feedback"
            response = requests.post(url, json={'migrations': migrations}, timeout=10)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Sent {len(migrations)} migrations to exporter")

            # Check if any failed
            if 'results' in result:
                failed = [r for r in result['results'] if not r.get('success', False)]
                if failed:
                    logger.warning(f"{len(failed)} migrations failed: {failed}")
                    return False

            return True

        except requests.RequestException as e:
            logger.error(f"Failed to send migrations to exporter: {e}")
            return False

    def load_scenario(self, scenario_data: dict) -> bool:
        """
        Load a new scenario into the exporter.

        Args:
            scenario_data: Dictionary with 'nodes' key containing node data

        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.exporter_url}/scenario"
            response = requests.post(url, json=scenario_data, timeout=10)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Loaded scenario: {result.get('nodes_loaded', 0)} nodes, {result.get('total_vms', 0)} VMs")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to load scenario: {e}")
            return False

    def get_current_state(self) -> dict:
        """
        Get the current state from the exporter.

        Returns:
            Dictionary with current node states or empty dict if failed
        """
        try:
            url = f"{self.exporter_url}/scenario"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get current state: {e}")
            return {}

    def check_health(self) -> bool:
        """
        Check if the exporter is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{self.exporter_url}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200

        except requests.RequestException:
            return False