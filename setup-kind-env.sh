#!/bin/bash
set -e

echo "========================================"
echo "Node Classifier Closed-Loop Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
KIND_CLUSTER_NAME="node-classifier-sim"
PROMETHEUS_OPERATOR_VERSION="v0.71.0"
CONTAINER_RUNTIME=""

# Helper functions
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

detect_container_runtime() {
    info "Detecting container runtime..."

    # Check if user specified via environment variable
    if [ -n "$KIND_EXPERIMENTAL_PROVIDER" ]; then
        CONTAINER_RUNTIME="$KIND_EXPERIMENTAL_PROVIDER"
        info "Using runtime from KIND_EXPERIMENTAL_PROVIDER: $CONTAINER_RUNTIME"
        return
    fi

    # Auto-detect: prefer Docker, fallback to Podman
    if command -v docker &> /dev/null && docker ps &> /dev/null 2>&1; then
        CONTAINER_RUNTIME="docker"
        info "Detected Docker"
    elif command -v podman &> /dev/null; then
        CONTAINER_RUNTIME="podman"
        info "Detected Podman"

        # Check if podman machine is running (on macOS/Windows)
        if command -v podman-machine &> /dev/null || [[ "$OSTYPE" == "darwin"* ]]; then
            if ! podman machine list 2>/dev/null | grep -q "Currently running"; then
                warn "Podman machine may not be running"
                info "Starting podman machine..."
                podman machine start || warn "Failed to start podman machine automatically"
            fi
        fi
    else
        error "Neither Docker nor Podman found or running"
        error "Please install Docker (https://docs.docker.com/get-docker/)"
        error "Or install Podman (https://podman.io/getting-started/installation)"
        exit 1
    fi

    export KIND_EXPERIMENTAL_PROVIDER="$CONTAINER_RUNTIME"
    debug "Set KIND_EXPERIMENTAL_PROVIDER=$CONTAINER_RUNTIME"
}

configure_podman() {
    if [ "$CONTAINER_RUNTIME" != "podman" ]; then
        return
    fi

    info "Configuring Podman for KIND..."

    # Check Podman version
    PODMAN_VERSION=$(podman --version | awk '{print $3}')
    info "Podman version: $PODMAN_VERSION"

    # Enable podman socket if not already running
    if ! podman info --format '{{.Host.RemoteSocket.Path}}' &> /dev/null; then
        warn "Podman socket not detected"
    fi

    # Check if running rootless
    if podman info --format '{{.Host.Security.Rootless}}' 2>/dev/null | grep -q "true"; then
        info "Running Podman in rootless mode"
    fi

    # Verify podman can create containers
    if ! podman ps &> /dev/null; then
        error "Podman is not working correctly"
        error "Try: podman machine init && podman machine start"
        exit 1
    fi
}

check_prereqs() {
    info "Checking prerequisites..."

    # Detect and configure container runtime
    detect_container_runtime
    configure_podman

    # Check for kind
    if ! command -v kind &> /dev/null; then
        error "kind not found. Please install from https://kind.sigs.k8s.io/"
        error "Recommended: KIND v0.30.0+"
        error "Install: curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64"
        exit 1
    fi

    # Check KIND version (should be v0.30.0+ recommended, v0.20.0+ minimum for Podman)
    KIND_VERSION=$(kind version | grep -oP 'v\d+\.\d+\.\d+' | head -1)
    info "KIND version: $KIND_VERSION"

    # Parse version for comparison
    KIND_MAJOR=$(echo $KIND_VERSION | cut -d. -f1 | sed 's/v//')
    KIND_MINOR=$(echo $KIND_VERSION | cut -d. -f2)

    # Warn if less than 0.30.0
    if [ "$KIND_MAJOR" -eq 0 ] && [ "$KIND_MINOR" -lt 30 ]; then
        warn "KIND version $KIND_VERSION detected. v0.30.0+ is recommended."
        warn "You may experience compatibility issues."

        if [ "$KIND_MINOR" -lt 20 ]; then
            error "KIND version too old (< v0.20.0). Podman support requires v0.20.0+"
            error "Please upgrade: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
            exit 1
        fi
    fi

    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Please install from https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi

    # Check for helm (optional but recommended)
    if ! command -v helm &> /dev/null; then
        warn "helm not found. Will use kubectl for Prometheus Operator installation."
        HELM_AVAILABLE=false
    else
        HELM_AVAILABLE=true
    fi

    info "Prerequisites check passed (runtime: $CONTAINER_RUNTIME)"
}

create_kind_cluster() {
    info "Creating KIND cluster..."

    # Check if cluster already exists
    if kind get clusters | grep -q "^${KIND_CLUSTER_NAME}$"; then
        warn "Cluster '${KIND_CLUSTER_NAME}' already exists"

        # If SKIP_CLUSTER_RECREATE is set, skip recreation
        if [ "${SKIP_CLUSTER_RECREATE:-false}" = "true" ]; then
            info "Skipping cluster recreation (SKIP_CLUSTER_RECREATE=true)"
            return
        fi

        # If running non-interactively, skip recreation by default
        if [ ! -t 0 ]; then
            info "Non-interactive mode detected, using existing cluster"
            return
        fi

        read -p "Delete and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Deleting existing cluster..."
            kind delete cluster --name "${KIND_CLUSTER_NAME}"
        else
            info "Using existing cluster"
            return
        fi
    fi

    info "Creating new KIND cluster with config..."
    kind create cluster --config k8s/kind-config.yaml --name "${KIND_CLUSTER_NAME}"

    info "Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=120s
}

install_kwok() {
    info "Installing KWOK..."

    # Install KWOK controller
    kubectl apply -f https://github.com/kubernetes-sigs/kwok/releases/download/v0.4.0/kwok.yaml

    info "Waiting for KWOK controller to be ready..."
    kubectl wait --for=condition=Available deployment/kwok-controller -n kube-system --timeout=120s

    info "Creating KWOK fake nodes..."
    kubectl apply -f k8s/kwok-nodes.yaml

    info "Waiting for KWOK nodes to be ready..."
    sleep 5
    kubectl wait --for=condition=Ready node -l type=kwok --timeout=60s

    info "Labeling KWOK nodes as schedulable for KubeVirt..."
    kubectl label nodes -l type=kwok kubevirt.io/schedulable=true --overwrite

    info "Patching DaemonSets to avoid KWOK nodes..."
    # Patch kindnet to only run on real nodes
    kubectl patch daemonset kindnet -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/affinity", "value": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [{"key": "type", "operator": "NotIn", "values": ["kwok"]}]}]}}}}]'

    # Patch kube-proxy to only run on real nodes
    kubectl patch daemonset kube-proxy -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/affinity", "value": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [{"key": "type", "operator": "NotIn", "values": ["kwok"]}]}]}}}}]'

    info "Cleaning up pending pods on KWOK nodes..."
    # Delete any pending pods scheduled to KWOK nodes
    kubectl delete pod -n kube-system --field-selector=status.phase=Pending --force --grace-period=0 2>/dev/null || true

    info "Installing KWOK pod stages (to move pods to Running state)..."
    kubectl apply -f k8s/kwok-pod-stage.yaml
}

install_vm_crd() {
    info "Installing VirtualMachine CRD..."

    # Install the CRD
    kubectl apply -f k8s/virtualmachine-crd.yaml

    info "Waiting for VirtualMachine CRD to be established..."
    kubectl wait --for condition=established --timeout=60s crd/virtualmachines.simulation.node-classifier.io

    info "VirtualMachine CRD installed successfully"
}

install_prometheus_operator() {
    info "Installing Prometheus Operator..."

    # Create monitoring namespace
    kubectl apply -f k8s/namespace.yaml

    if [ "$HELM_AVAILABLE" = true ]; then
        info "Using Helm to install Prometheus Operator..."
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update

        helm install prometheus-operator prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
            --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
            --set prometheus.service.type=NodePort \
            --set prometheus.service.nodePort=30090 \
            --set prometheus-node-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key=type \
            --set prometheus-node-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator=NotIn \
            --set prometheus-node-exporter.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].values[0]=kwok \
            --timeout 10m \
            --wait
    else
        info "Using kubectl to install Prometheus Operator..."
        kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/${PROMETHEUS_OPERATOR_VERSION}/bundle.yaml

        # Wait for operator
        kubectl wait --for=condition=Available deployment/prometheus-operator -n default --timeout=120s

        # Apply Prometheus instance
        kubectl apply -f k8s/prometheus.yaml
    fi

    info "Applying Prometheus recording rules..."
    kubectl apply -f k8s/prometheus-rules.yaml

    info "Patching Prometheus node-exporter to avoid KWOK nodes..."
    # Wait for node-exporter DaemonSet to be created
    sleep 5

    # Try to add kwok to existing type exclusion list (if it exists)
    kubectl patch daemonset prometheus-operator-prometheus-node-exporter -n monitoring --type='json' -p='[{"op": "add", "path": "/spec/template/spec/affinity/nodeAffinity/requiredDuringSchedulingIgnoredDuringExecution/nodeSelectorTerms/0/matchExpressions/1/values/-", "value": "kwok"}]' 2>/dev/null || \
    # If that fails, try to create the affinity from scratch
    kubectl patch daemonset prometheus-operator-prometheus-node-exporter -n monitoring --type='json' -p='[{"op": "add", "path": "/spec/template/spec/affinity", "value": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [{"key": "type", "operator": "NotIn", "values": ["kwok"]}]}]}}}}]' 2>/dev/null || true

    # Clean up any pending pods on KWOK nodes in monitoring namespace
    kubectl delete pod -n monitoring --field-selector=status.phase=Pending --force --grace-period=0 2>/dev/null || true
}

install_descheduler() {
    info "Installing Descheduler..."

    # Create namespaces
    info "Creating kube-descheduler namespace..."
    kubectl create namespace kube-descheduler --dry-run=client -o yaml | kubectl apply -f -

    info "Creating openshift-kube-descheduler-operator namespace (for softtainter leader election)..."
    kubectl create namespace openshift-kube-descheduler-operator --dry-run=client -o yaml | kubectl apply -f -

    # Install KubeDescheduler CRD (required by softtainter)
    info "Installing KubeDescheduler CRD..."
    kubectl apply -f k8s/kubedescheduler-crd.yaml

    info "Waiting for KubeDescheduler CRD to be established..."
    kubectl wait --for condition=established --timeout=60s crd/kubedeschedulers.operator.openshift.io

    # Create KubeDescheduler instance
    info "Creating KubeDescheduler instance..."
    kubectl apply -f k8s/kubedescheduler-instance.yaml

    # Generate TLS certificate
    info "Generating TLS certificates for descheduler..."
    TLS_DIR=$(mktemp -d)
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$TLS_DIR/tls.key" \
        -out "$TLS_DIR/tls.crt" \
        -days 365 \
        -subj "/CN=descheduler.kube-descheduler.svc" \
        2>/dev/null

    kubectl create secret tls kube-descheduler-serving-cert \
        -n kube-descheduler \
        --cert="$TLS_DIR/tls.crt" \
        --key="$TLS_DIR/tls.key" \
        --dry-run=client -o yaml | kubectl apply -f -

    rm -rf "$TLS_DIR"

    # Create descheduler policy ConfigMap
    info "Creating descheduler policy ConfigMap..."
    kubectl apply -f k8s/descheduler-policy.yaml

    # Generate Prometheus TLS certificates for nginx proxy
    # Sign with Kubernetes CA so descheduler automatically trusts it
    info "Generating Prometheus proxy TLS certificates signed by Kubernetes CA..."
    PROM_TLS_DIR=$(mktemp -d)

    # Get Kubernetes CA certificate from the cluster (works with any container runtime)
    # The CA cert is available in ConfigMap kube-root-ca.crt in every namespace
    kubectl get configmap kube-root-ca.crt -n kube-system -o jsonpath='{.data.ca\.crt}' > "$PROM_TLS_DIR/k8s-ca.crt"

    # Get the CA private key from the control plane
    # Detect container runtime (docker or podman)
    if command -v docker &> /dev/null && docker ps &> /dev/null; then
        CONTAINER_RUNTIME="docker"
    elif command -v podman &> /dev/null; then
        CONTAINER_RUNTIME="podman"
    else
        error "Neither docker nor podman found or accessible"
        exit 1
    fi

    $CONTAINER_RUNTIME exec ${KIND_CLUSTER_NAME}-control-plane cat /etc/kubernetes/pki/ca.key > "$PROM_TLS_DIR/k8s-ca.key"

    # Create server certificate configuration
    cat > "$PROM_TLS_DIR/server.conf" << 'EOF'
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = prometheus

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = prometheus-operator-kube-p-prometheus
DNS.2 = prometheus-operator-kube-p-prometheus.monitoring
DNS.3 = prometheus-operator-kube-p-prometheus.monitoring.svc
DNS.4 = prometheus-operator-kube-p-prometheus.monitoring.svc.cluster.local
DNS.5 = localhost
IP.1 = 127.0.0.1
EOF

    # Generate server key and CSR
    openssl req -newkey rsa:2048 -nodes \
        -keyout "$PROM_TLS_DIR/tls.key" \
        -out "$PROM_TLS_DIR/server.csr" \
        -config "$PROM_TLS_DIR/server.conf" \
        2>/dev/null

    # Sign server certificate with Kubernetes CA
    openssl x509 -req \
        -in "$PROM_TLS_DIR/server.csr" \
        -CA "$PROM_TLS_DIR/k8s-ca.crt" \
        -CAkey "$PROM_TLS_DIR/k8s-ca.key" \
        -CAcreateserial \
        -out "$PROM_TLS_DIR/tls.crt" \
        -days 365 \
        -extensions v3_req \
        -extfile "$PROM_TLS_DIR/server.conf" \
        2>/dev/null

    # Create prometheus-web-tls Secret (signed by Kubernetes CA)
    info "Creating prometheus-web-tls Secret (signed by Kubernetes CA)..."
    kubectl create secret tls prometheus-web-tls \
        -n kube-descheduler \
        --cert="$PROM_TLS_DIR/tls.crt" \
        --key="$PROM_TLS_DIR/tls.key" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Create nginx proxy configuration
    info "Creating nginx proxy configuration..."
    cat > "$PROM_TLS_DIR/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream prometheus {
        server prometheus-operator-kube-p-prometheus.monitoring.svc.cluster.local:9090;
    }

    server {
        listen 443 ssl;
        server_name localhost 127.0.0.1;

        ssl_certificate /tls/tls.crt;
        ssl_certificate_key /tls/tls.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        location / {
            proxy_pass http://prometheus;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
EOF

    kubectl create configmap prometheus-proxy-config \
        -n kube-descheduler \
        --from-file=nginx.conf="$PROM_TLS_DIR/nginx.conf" \
        --dry-run=client -o yaml | kubectl apply -f -

    rm -rf "$PROM_TLS_DIR"

    # Label KWOK nodes with kubevirt.io/schedulable=true
    info "Labeling KWOK nodes with kubevirt.io/schedulable=true..."
    kubectl get nodes -l type=kwok -o name | xargs -I {} kubectl label {} kubevirt.io/schedulable=true --overwrite

    # Apply descheduler deployment
    info "Deploying descheduler and softtainter..."
    kubectl apply -f k8s/descheduler.yaml

    info "Waiting for descheduler to be ready..."
    kubectl wait --for=condition=Available deployment/descheduler -n kube-descheduler --timeout=120s || warn "Descheduler deployment not ready yet"

    info "Waiting for softtainter to be ready..."
    kubectl wait --for=condition=Available deployment/softtainter -n kube-descheduler --timeout=120s || warn "Softtainter deployment not ready yet"
}

build_and_deploy_exporter() {
    info "Building metrics exporter ConfigMap..."

    # Create ConfigMap with Python code
    kubectl create configmap metrics-exporter-code \
        --from-file=prometheus_exporter.py \
        --from-file=node.py \
        --from-file=scenario_loader.py \
        --from-file=pod_manager.py \
        --from-file=vm_manager.py \
        -n monitoring \
        --dry-run=client -o yaml | kubectl apply -f -

    info "Deploying metrics exporter..."
    kubectl apply -f k8s/metrics-exporter.yaml

    info "Waiting for metrics exporter to be ready..."
    kubectl wait --for=condition=Available deployment/metrics-exporter -n monitoring --timeout=120s
}

deploy_vm_controller() {
    info "Deploying VM controller..."

    # Create ConfigMap with Python code
    kubectl create configmap vm-controller-code \
        --from-file=vm_controller.py \
        --from-file=pod_manager.py \
        --from-file=vm_manager.py \
        --from-file=node.py \
        -n monitoring \
        --dry-run=client -o yaml | kubectl apply -f -

    info "Deploying VM controller deployment..."
    kubectl apply -f k8s/vm-controller.yaml

    info "Waiting for VM controller to be ready..."
    kubectl wait --for=condition=Available deployment/vm-controller -n monitoring --timeout=120s
}

deploy_eviction_webhook() {
    info "Deploying KubeVirt-style eviction webhook..."

    # Step 1: Generate TLS certificates for webhook
    info "Generating TLS certificates for eviction webhook..."
    WEBHOOK_TLS_DIR=$(mktemp -d)

    # Generate CA key and certificate
    openssl genrsa -out "$WEBHOOK_TLS_DIR/ca.key" 2048 2>/dev/null
    openssl req -x509 -new -nodes \
        -key "$WEBHOOK_TLS_DIR/ca.key" \
        -subj "/CN=Eviction Webhook CA" \
        -days 365 \
        -out "$WEBHOOK_TLS_DIR/ca.crt" \
        2>/dev/null

    # Generate server key
    openssl genrsa -out "$WEBHOOK_TLS_DIR/tls.key" 2048 2>/dev/null

    # Generate certificate signing request
    cat > "$WEBHOOK_TLS_DIR/csr.conf" << EOF
[req]
req_extensions = v3_req
distinguished_name = req_distinguished_name
[req_distinguished_name]
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names
[alt_names]
DNS.1 = eviction-webhook
DNS.2 = eviction-webhook.monitoring
DNS.3 = eviction-webhook.monitoring.svc
DNS.4 = eviction-webhook.monitoring.svc.cluster.local
EOF

    # Create CSR
    openssl req -new \
        -key "$WEBHOOK_TLS_DIR/tls.key" \
        -subj "/CN=eviction-webhook.monitoring.svc" \
        -config "$WEBHOOK_TLS_DIR/csr.conf" \
        -out "$WEBHOOK_TLS_DIR/tls.csr" \
        2>/dev/null

    # Sign the certificate
    openssl x509 -req \
        -in "$WEBHOOK_TLS_DIR/tls.csr" \
        -CA "$WEBHOOK_TLS_DIR/ca.crt" \
        -CAkey "$WEBHOOK_TLS_DIR/ca.key" \
        -CAcreateserial \
        -out "$WEBHOOK_TLS_DIR/tls.crt" \
        -days 365 \
        -extensions v3_req \
        -extfile "$WEBHOOK_TLS_DIR/csr.conf" \
        2>/dev/null

    # Step 2: Create Kubernetes secret with certificates
    info "Creating webhook TLS secret..."
    kubectl create secret tls eviction-webhook-certs \
        --cert="$WEBHOOK_TLS_DIR/tls.crt" \
        --key="$WEBHOOK_TLS_DIR/tls.key" \
        --namespace=monitoring \
        --dry-run=client -o yaml | kubectl apply -f -

    # Step 3: Create ConfigMap with webhook code
    info "Creating webhook code ConfigMap..."
    kubectl create configmap eviction-webhook-code \
        --from-file=eviction_webhook.py \
        --namespace=monitoring \
        --dry-run=client -o yaml | kubectl apply -f -

    # Step 4: Deploy webhook resources
    info "Deploying webhook deployment and service..."
    kubectl apply -f k8s/eviction-webhook.yaml

    # Step 5: Wait for webhook to be ready
    info "Waiting for webhook to be ready..."
    kubectl wait --for=condition=Available deployment/eviction-webhook -n monitoring --timeout=120s

    # Step 6: Update MutatingWebhookConfiguration with CA bundle
    info "Configuring webhook with CA bundle..."
    CA_BUNDLE=$(cat "$WEBHOOK_TLS_DIR/ca.crt" | base64 | tr -d '\n')

    kubectl patch mutatingwebhookconfiguration eviction-webhook \
        --type='json' \
        -p="[{'op': 'replace', 'path': '/webhooks/0/clientConfig/caBundle', 'value':'${CA_BUNDLE}'}]"

    # Cleanup
    rm -rf "$WEBHOOK_TLS_DIR"

    info "Eviction webhook deployed successfully"
}

verify_installation() {
    info "Verifying installation..."

    echo ""
    info "Cluster nodes:"
    kubectl get nodes

    echo ""
    info "VirtualMachine CRD:"
    kubectl get crd virtualmachines.simulation.node-classifier.io

    echo ""
    info "Monitoring pods:"
    kubectl get pods -n monitoring

    echo ""
    info "Descheduler pods:"
    kubectl get pods -n kube-descheduler

    echo ""
    info "Eviction webhook:"
    kubectl get pods -l app=eviction-webhook -n monitoring

    echo ""
    info "Services:"
    kubectl get svc -n monitoring

    echo ""
    info "Prometheus recording rules:"
    kubectl get prometheusrules -n monitoring

    echo ""
    info "Webhook configuration:"
    kubectl get mutatingwebhookconfiguration eviction-webhook
}

print_access_info() {
    echo ""
    echo "========================================"
    echo "Setup Complete!"
    echo "========================================"
    echo ""
    info "Installed components:"
    echo "  ✓ KIND cluster with KWOK nodes"
    echo "  ✓ VirtualMachine CRD"
    echo "  ✓ Prometheus Operator"
    echo "  ✓ Descheduler"
    echo "  ✓ Metrics Exporter"
    echo "  ✓ Eviction Webhook (KubeVirt-style)"
    echo ""
    info "Access endpoints:"
    echo "  Prometheus:       http://localhost:9090"
    echo "  Metrics Exporter: http://localhost:8000"
    echo ""
    info "Test connectivity:"
    echo "  curl http://localhost:8000/health"
    echo "  curl http://localhost:8000/metrics"
    echo "  curl http://localhost:9090/-/healthy"
    echo ""
    info "Work with VirtualMachines:"
    echo "  kubectl get vm                     # List VMs"
    echo "  kubectl apply -f k8s/example-vms.yaml  # Create example VMs"
    echo "  kubectl describe vm <name>         # Get VM details"
    echo ""
    info "Test eviction webhook (triggers live migration):"
    echo "  kubectl delete pod <virt-launcher-pod-name>"
    echo "  kubectl logs -l app=eviction-webhook -n monitoring -f  # Watch migration logs"
    echo ""
    info "Load metrics into Prometheus:"
    echo "  python prometheus_loader.py --url http://localhost:9090"
    echo ""
    info "Run closed-loop simulation:"
    echo "  python cli_prometheus.py --prometheus http://localhost:9090 --exporter http://localhost:8000"
    echo ""
    info "Cleanup:"
    echo "  kind delete cluster --name ${KIND_CLUSTER_NAME}"
    echo ""
}

# Main execution
main() {
    check_prereqs
    create_kind_cluster
    install_kwok
    install_vm_crd
    install_prometheus_operator
    install_descheduler
    deploy_vm_controller
    build_and_deploy_exporter
    deploy_eviction_webhook
    verify_installation
    print_access_info
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi