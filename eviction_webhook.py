#!/usr/bin/env python3
"""
KubeVirt-style Eviction Webhook

Intercepts pod eviction requests for virt-launcher pods and denies deletion
if the pod has a migration finalizer. The actual migration is handled by the
migration controller in pod_manager.py.
"""

import json
import logging
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
class WebhookState:
    def __init__(self):
        self.k8s_client: Optional[client.CoreV1Api] = None
        self.custom_api: Optional[client.CustomObjectsApi] = None
        self.pod_manager: Optional['PodManager'] = None
        self.namespace = "default"

    def initialize(self, use_in_cluster_config: bool = False):
        """Initialize Kubernetes client and migration controller."""
        try:
            if use_in_cluster_config:
                config.load_incluster_config()
            else:
                config.load_kube_config()

            self.k8s_client = client.CoreV1Api()
            self.custom_api = client.CustomObjectsApi()

            # Initialize PodManager to run migration controller
            # NOTE: We don't create VMs here - that's vm-controller's job
            # We ONLY handle evictions and migrations
            from pod_manager import PodManager
            self.pod_manager = PodManager(
                namespace=self.namespace,
                use_in_cluster_config=use_in_cluster_config,
                create_vm_crs=True,  # Need VM manager for migration status updates
                enable_migration_controller=True  # ONLY the webhook runs migration controller
            )

            logger.info("Webhook state initialized successfully")
            logger.info("Migration controller started")
        except Exception as e:
            logger.error(f"Failed to initialize webhook state: {e}")
            raise

state = WebhookState()


def is_virt_launcher_pod(pod_name: str, labels: Dict[str, str]) -> bool:
    """Check if a pod is a virt-launcher pod."""
    return (pod_name.startswith("virt-launcher-") or
            labels.get("app") == "virt-launcher" or
            "kubevirt.io/domain" in labels)


def get_vm_from_pod(pod_name: str, labels: Dict[str, str]) -> Optional[str]:
    """Extract VM ID from pod name or labels."""
    # Try to get from labels first
    vm_id = labels.get("kubevirt.io/domain") or labels.get("vm.kubevirt.io/name")
    if vm_id:
        return vm_id

    # Try to extract from pod name: virt-launcher-<vm-id>-<random>
    if pod_name.startswith("virt-launcher-"):
        parts = pod_name.split("-")
        if len(parts) >= 3:
            # Remove "virt", "launcher", and last random suffix
            return "-".join(parts[2:-1])

    return None


@app.route('/mutate', methods=['POST'])
def mutate_webhook():
    """
    Mutating webhook endpoint for pod evictions.

    This intercepts DELETE requests for virt-launcher pods and denies deletion
    if the pod has the migration-protection finalizer. The migration controller
    in pod_manager.py watches for deletionTimestamp and handles the actual migration.
    """
    try:
        admission_review = request.get_json()

        # Extract admission request
        req = admission_review.get("request", {})
        uid = req.get("uid")
        namespace = req.get("namespace", "default")
        operation = req.get("operation", "")

        # Get object being deleted
        old_object = req.get("oldObject", {})
        metadata = old_object.get("metadata", {})
        pod_name = metadata.get("name", "")
        labels = metadata.get("labels", {})
        finalizers = metadata.get("finalizers", [])

        logger.info(f"Webhook received {operation} request for pod {pod_name} in namespace {namespace}")

        # Only intercept DELETE operations
        if operation != "DELETE":
            return jsonify({
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": True
                }
            })

        # Check if this is a virt-launcher pod
        if not is_virt_launcher_pod(pod_name, labels):
            logger.info(f"Pod {pod_name} is not a virt-launcher pod, allowing deletion")
            return jsonify({
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": True
                }
            })

        # Extract VM ID for logging
        vm_id = get_vm_from_pod(pod_name, labels)
        if not vm_id:
            logger.warning(f"Cannot determine VM ID from pod {pod_name}, allowing deletion")
            return jsonify({
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": True
                }
            })

        # Check current pod state from API (not oldObject which may be stale)
        try:
            current_pod = state.k8s_client.read_namespaced_pod(name=pod_name, namespace=namespace)
            current_finalizers = current_pod.metadata.finalizers or []
        except ApiException as e:
            logger.warning(f"Cannot read current pod state for {pod_name}: {e}, using oldObject")
            current_finalizers = finalizers

        # If finalizer is removed, migration is complete - allow deletion
        if "kubevirt.io/migration-protection" not in current_finalizers:
            logger.info(f"Pod {pod_name} has no finalizer (migration complete), allowing deletion")
            return jsonify({
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": uid,
                    "allowed": True
                }
            })

        # Get pod's node (from the pod we already read)
        pod_node = current_pod.spec.node_name if hasattr(current_pod, 'spec') and current_pod.spec else None

        # Check if this is the target pod (KubeVirt-style check)
        # Query the VM CR to get the VM's current node
        try:
            vm_cr = state.custom_api.get_namespaced_custom_object(
                group="simulation.node-classifier.io",
                version="v1alpha1",
                namespace=namespace,
                plural="virtualmachines",
                name=vm_id
            )
            vm_status = vm_cr.get("status", {})
            vm_node = vm_status.get("nodeName", "")

            # If VM's node != pod's node, this is a target pod (migration destination)
            if pod_node and vm_node and vm_node != pod_node:
                logger.info(f"⚠️  Denying eviction of target pod {pod_name}")
                logger.info(f"   VM {vm_id} is on node {vm_node}, pod is on node {pod_node}")
                logger.info(f"   This is the migration target - eviction denied")
                return jsonify({
                    "apiVersion": "admission.k8s.io/v1",
                    "kind": "AdmissionReview",
                    "response": {
                        "uid": uid,
                        "allowed": False,
                        "status": {
                            "code": 403,
                            "message": f"Eviction request for target pod (migration in progress for VM {vm_id})"
                        }
                    }
                })
        except ApiException as e:
            if e.status != 404:
                logger.warning(f"Failed to read VM CR for {vm_id}: {e}")
            # Continue - if we can't read VM, still deny based on finalizer

        # Finalizer is present - deny deletion and mark VM for evacuation
        # This follows KubeVirt's approach: deny eviction, mark VMI with evacuationNodeName
        logger.info(f"🛑 Pod {pod_name} (VM: {vm_id}) has finalizer - denying eviction")
        logger.info(f"   Marking VM for evacuation from node {pod_node}")

        # Mark VM CR with evacuationNodeName (KubeVirt-style)
        try:
            # Read current VM object
            vm_obj = state.custom_api.get_namespaced_custom_object(
                group="simulation.node-classifier.io",
                version="v1alpha1",
                namespace=namespace,
                plural="virtualmachines",
                name=vm_id
            )

            # Update status with evacuation node
            if "status" not in vm_obj:
                vm_obj["status"] = {}
            vm_obj["status"]["evacuationNodeName"] = pod_node

            # Patch via status subresource (send full VM object)
            state.custom_api.patch_namespaced_custom_object_status(
                group="simulation.node-classifier.io",
                version="v1alpha1",
                namespace=namespace,
                plural="virtualmachines",
                name=vm_id,
                body=vm_obj
            )
            logger.info(f"   Marked VM {vm_id} for evacuation from node {pod_node}")
        except ApiException as e:
            logger.error(f"Failed to mark VM {vm_id} for evacuation: {e}")
            # Continue anyway - deny the eviction even if we can't mark VM

        # Return KubeVirt-compatible error message that descheduler recognizes
        # The descheduler looks for "Eviction triggered evacuation" to know it should back off
        return jsonify({
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": uid,
                "allowed": False,
                "status": {
                    "code": 403,
                    "message": f"Eviction triggered evacuation of VMI \"{namespace}/{vm_id}\""
                }
            }
        })

    except Exception as e:
        logger.error(f"Error in webhook: {e}", exc_info=True)
        return jsonify({
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": req.get("uid", ""),
                "allowed": True,  # Allow on error to avoid blocking
                "status": {
                    "message": f"Webhook error: {str(e)}"
                }
            }
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "k8s_client": state.k8s_client is not None
    })


def main():
    import argparse

    parser = argparse.ArgumentParser(description='KubeVirt-style Eviction Webhook')
    parser.add_argument('--port', type=int, default=8443, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--cert', type=str, help='Path to TLS certificate')
    parser.add_argument('--key', type=str, help='Path to TLS key')
    parser.add_argument('--in-cluster', action='store_true', help='Use in-cluster config')

    args = parser.parse_args()

    # Initialize state
    state.initialize(use_in_cluster_config=args.in_cluster)

    logger.info(f"Starting KubeVirt-style eviction webhook on {args.host}:{args.port}")

    # Start webhook server with TLS if cert/key provided
    if args.cert and args.key:
        logger.info("Starting webhook with TLS")
        app.run(
            host=args.host,
            port=args.port,
            ssl_context=(args.cert, args.key)
        )
    else:
        logger.warning("Starting webhook WITHOUT TLS (not recommended for production)")
        app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()