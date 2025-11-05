#!/usr/bin/env python3
"""
VM Controller for Node Classifier Simulation

Watches VirtualMachine CRs and manages virt-launcher pods.
Creates pods for VMs that don't have them, updates VM status based on pod state.
Does NOT handle migrations - that's the eviction-webhook's responsibility.
"""

import logging
import time
import threading
from typing import Optional
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

from pod_manager import PodManager
from vm_manager import VMManager
from node import VM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VMController:
    """
    Watches VirtualMachine CRs and manages virt-launcher pods.

    Responsibilities:
    - Create virt-launcher pods for VMs that don't have them
    - Update VM status based on pod state
    - Keep VM CRs in sync with pod state

    Does NOT:
    - Handle migrations (eviction-webhook does this)
    - Export metrics (metrics-exporter does this)
    """

    def __init__(self, namespace: str = "default", use_in_cluster_config: bool = False):
        """
        Initialize the VM controller.

        Args:
            namespace: Kubernetes namespace for VMs and pods
            use_in_cluster_config: If True, use in-cluster config, otherwise use kubeconfig
        """
        self.namespace = namespace
        self.controller_running = False
        self.controller_thread: Optional[threading.Thread] = None

        # Initialize Kubernetes clients
        try:
            if use_in_cluster_config:
                config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes config")
            else:
                config.load_kube_config()
                logger.info("Using local kubeconfig")

            self.k8s_client = client.CoreV1Api()
            self.custom_api = client.CustomObjectsApi()

            # Initialize managers (without migration controller - that's in eviction-webhook)
            self.pod_manager = PodManager(
                namespace=namespace,
                use_in_cluster_config=use_in_cluster_config,
                create_vm_crs=False,  # VM CRs already exist
                enable_migration_controller=False  # Migration handled by eviction-webhook
            )

            self.vm_manager = VMManager(
                namespace=namespace,
                use_in_cluster_config=use_in_cluster_config
            )

            logger.info("VM controller initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize VM controller: {e}")
            raise

    def _pod_exists(self, pod_name: str) -> bool:
        """Check if a pod exists."""
        try:
            self.k8s_client.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            return True
        except ApiException:
            return False

    def _update_vm_status_from_pod(self, vm_id: str, pod_name: str):
        """Update VM status based on pod status."""
        try:
            pod = self.k8s_client.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            node_name = pod.spec.node_name or ""

            # Determine phase
            if pod.status.phase == "Running":
                phase = "Running"
            elif pod.status.phase == "Pending":
                phase = "Scheduling" if not node_name else "Scheduled"
            elif pod.status.phase in ["Failed", "Unknown"]:
                phase = "Failed"
            else:
                phase = "Pending"

            self.vm_manager.update_vm_status(vm_id, phase, pod_name, node_name)

        except ApiException as e:
            logger.error(f"Failed to read pod {pod_name}: {e}")

    def _sync_vm_crs_to_pods(self):
        """
        Synchronize VirtualMachine CRs with virt-launcher pods.
        ONLY creates pods for VMs that don't have them.
        NEVER deletes or recreates existing pods - migrations are handled by eviction-webhook.
        """
        try:
            # List all VirtualMachine CRs
            vms_list = self.custom_api.list_namespaced_custom_object(
                group="simulation.node-classifier.io",
                version="v1alpha1",
                namespace=self.namespace,
                plural="virtualmachines"
            )

            for vm_obj in vms_list.get("items", []):
                vm_name = vm_obj["metadata"]["name"]
                spec = vm_obj.get("spec", {})
                status = vm_obj.get("status", {})

                # Check if VM is being evacuated/migrated
                evacuation_node = status.get("evacuationNodeName", "")
                if evacuation_node:
                    # Migration in progress - don't create pod, eviction-webhook handles it
                    logger.debug(f"VM {vm_name} is being evacuated from {evacuation_node}, skipping")
                    continue

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
                    logger.info(f"Creating missing pod for VM {vm_name}")
                    vm.scheduled_node = ""  # Let scheduler decide
                    created_pod_name = self.pod_manager.create_pod(vm)
                    if created_pod_name:
                        vm.pod_name = created_pod_name
                        # Update VM status to Scheduling
                        self.vm_manager.update_vm_status(vm_name, "Scheduling", created_pod_name)
                else:
                    # Pod exists - just update VM status from pod
                    self._update_vm_status_from_pod(vm_name, pod_name)

        except ApiException as e:
            logger.error(f"Failed to sync VM CRs to pods: {e}")
        except Exception as e:
            logger.error(f"Unexpected error syncing VM CRs: {e}", exc_info=True)

    def _controller_loop(self):
        """Main controller loop that watches VirtualMachine CRs and creates pods."""
        logger.info(f"VM controller started for namespace '{self.namespace}'")

        while self.controller_running:
            try:
                self._sync_vm_crs_to_pods()
            except Exception as e:
                logger.error(f"Error in VM controller loop: {e}", exc_info=True)

            # Sleep for a bit before next sync
            time.sleep(5)

        logger.info("VM controller stopped")

    def start(self):
        """Start the VM controller in a background thread."""
        if self.controller_running:
            logger.warning("VM controller already running")
            return

        self.controller_running = True
        self.controller_thread = threading.Thread(
            target=self._controller_loop,
            daemon=True,
            name="vm-controller"
        )
        self.controller_thread.start()
        logger.info("VM controller thread started")

    def stop(self):
        """Stop the VM controller."""
        if not self.controller_running:
            return

        logger.info("Stopping VM controller...")
        self.controller_running = False
        if self.controller_thread:
            self.controller_thread.join(timeout=10)
        logger.info("VM controller stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='VM Controller for Node Classifier Simulation')
    parser.add_argument('--namespace', type=str, default='default', help='Kubernetes namespace')
    parser.add_argument('--in-cluster', action='store_true', help='Use in-cluster config')

    args = parser.parse_args()

    # Create and start controller
    controller = VMController(
        namespace=args.namespace,
        use_in_cluster_config=args.in_cluster
    )

    logger.info(f"Starting VM controller for namespace '{args.namespace}'")
    controller.start()

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        controller.stop()


if __name__ == "__main__":
    main()
