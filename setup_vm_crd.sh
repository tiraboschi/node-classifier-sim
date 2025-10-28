#!/bin/bash
# Setup script for VirtualMachine CRD installation

set -e

echo "========================================="
echo "VirtualMachine CRD Setup"
echo "========================================="
echo

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot access Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

echo "✓ Kubernetes cluster is accessible"
echo

# Install the VirtualMachine CRD
echo "📦 Installing VirtualMachine CRD..."
kubectl apply -f k8s/virtualmachine-crd.yaml

echo "⏳ Waiting for CRD to be established..."
kubectl wait --for condition=established --timeout=60s crd/virtualmachines.simulation.node-classifier.io

echo "✓ VirtualMachine CRD installed successfully"
echo

# Check if KWOK nodes exist
echo "🔍 Checking for KWOK nodes..."
KWOK_NODES=$(kubectl get nodes -l type=kwok --no-headers 2>/dev/null | wc -l || echo "0")

if [ "$KWOK_NODES" -eq 0 ]; then
    echo "⚠️  No KWOK nodes found. Installing KWOK nodes..."
    kubectl apply -f k8s/kwok-nodes.yaml
    echo "✓ KWOK nodes installed"
else
    echo "✓ Found $KWOK_NODES KWOK nodes"
fi

echo

# Optionally create example VMs
read -p "Do you want to create example VirtualMachine resources? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📝 Creating example VirtualMachine resources..."
    kubectl apply -f k8s/example-vms.yaml
    echo "✓ Example VMs created"
    echo
    echo "View them with: kubectl get virtualmachines"
fi

echo
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo
echo "Available commands:"
echo "  kubectl get virtualmachines         # List all VMs"
echo "  kubectl get vm                      # Short form"
echo "  kubectl describe vm <name>          # Get VM details"
echo "  kubectl get vm -o wide              # See pod and node info"
echo
echo "Next steps:"
echo "  1. Start the Prometheus exporter: python prometheus_exporter.py"
echo "  2. VMs will be created automatically from scenarios"
echo "  3. Watch VM status: kubectl get vm -w"
echo
