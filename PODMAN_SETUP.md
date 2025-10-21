# Using KIND with Podman

This guide explains how to use the node classifier simulation environment with Podman instead of Docker.

## Why Podman?

- **Rootless**: Run containers without root privileges
- **Docker-compatible**: Drop-in replacement for Docker CLI
- **Daemonless**: No background daemon required
- **Kubernetes-native**: Better integration with Kubernetes tools

## Prerequisites

### Install Podman

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install podman
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install podman
```

**macOS:**
```bash
brew install podman
```

**Verify installation:**
```bash
podman --version
```

### Initialize Podman (macOS/Windows)

On macOS and Windows, Podman runs in a VM:

```bash
# Initialize the VM
podman machine init

# Start the VM
podman machine start

# Verify it's running
podman machine list
```

### Install KIND

```bash
# Download KIND v0.30.0 (recommended)
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.30.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Verify
kind version
```

**Note**:
- KIND v0.30.0+ is recommended for best compatibility
- Minimum v0.20.0 required for Podman support

## Running with Podman

### Method 1: Automatic Detection (Recommended)

The setup script automatically detects Podman if Docker is not available:

```bash
./setup-kind-env.sh
```

**Expected output:**
```
[INFO] Detecting container runtime...
[INFO] Detected Podman
[INFO] Podman version: 4.x.x
[INFO] Running Podman in rootless mode
[INFO] Prerequisites check passed (runtime: podman)
```

### Method 2: Explicit Configuration

Force Podman even if Docker is available:

```bash
export KIND_EXPERIMENTAL_PROVIDER=podman
./setup-kind-env.sh
```

### Method 3: Direct KIND Command

```bash
export KIND_EXPERIMENTAL_PROVIDER=podman
kind create cluster --config k8s/kind-config.yaml --name node-classifier-sim
```

## Podman-Specific Configuration

### Rootless Mode

KIND with Podman runs in rootless mode by default. This is more secure but has some limitations:

**Advantages:**
- No root privileges required
- Better security isolation
- Works well for development

**Limitations:**
- Port binding < 1024 requires extra configuration
- Some volume mounts may have permission issues

### Networking

Podman uses different network backends:

**Linux:**
- Default: CNI (Container Network Interface)
- Alternative: netavark (newer, recommended)

**macOS/Windows:**
- Uses Podman machine with port forwarding

**Verify networking:**
```bash
podman info --format '{{.Host.NetworkBackend}}'
```

### Volume Mounts

Podman handles volumes differently than Docker:

**Rootless mode considerations:**
- Volumes are owned by your user (not root)
- SELinux labels may need adjustment on RHEL/Fedora

**If you encounter permission issues:**
```bash
# On SELinux systems
sudo setsebool -P container_manage_cgroup true
```

## Troubleshooting

### Issue: "Cannot connect to Podman socket"

**Solution:**
```bash
# Start podman socket (systemd)
systemctl --user start podman.socket
systemctl --user enable podman.socket

# Verify
podman info
```

### Issue: "Error: short-name resolution"

Podman may prompt for registry selection.

**Solution:**
```bash
# Configure default registry
cat > ~/.config/containers/registries.conf <<EOF
[registries.search]
registries = ['docker.io']
EOF
```

### Issue: "Failed to create cluster"

**Check Podman status:**
```bash
podman ps
podman info
```

**Check KIND compatibility:**
```bash
kind version
# Should be v0.20.0 or newer
```

**Try with verbose logging:**
```bash
export KIND_EXPERIMENTAL_PROVIDER=podman
kind create cluster --config k8s/kind-config.yaml --name node-classifier-sim -v 5
```

### Issue: "Port already in use"

Podman may have containers running from previous attempts.

**Solution:**
```bash
# List all containers
podman ps -a

# Remove old KIND containers
podman rm -f $(podman ps -a --filter "label=io.x-k8s.kind.cluster" -q)

# Or clean everything
kind delete cluster --name node-classifier-sim
podman system prune -a
```

### Issue: "Cannot access localhost:9090"

On macOS/Windows with Podman machine, port forwarding may need manual setup.

**Solution:**
```bash
# Check if ports are forwarded
podman machine inspect | grep PortForwarding

# Manually forward ports (if needed)
kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
kubectl port-forward -n monitoring svc/metrics-exporter 8000:8000 &
```

## Performance Considerations

### Podman vs Docker

| Aspect | Podman | Docker |
|--------|--------|--------|
| **Startup** | Slightly slower (no daemon) | Faster (daemon running) |
| **Resource usage** | Lower (no daemon) | Higher (daemon overhead) |
| **Rootless** | Native support | Requires setup |
| **Compatibility** | High (Docker CLI compatible) | N/A |

### Optimization Tips

**Use newer Podman versions:**
```bash
# Fedora/RHEL
sudo dnf upgrade podman

# Ubuntu (use official PPA for latest)
sudo add-apt-repository ppa:projectatomic/ppa
sudo apt-get update
sudo apt-get upgrade podman
```

**Enable cgroups v2 (if available):**
```bash
# Check current version
cat /sys/fs/cgroup/cgroup.controllers

# If empty, you're on v1
# Upgrade to v2 for better performance (requires reboot)
sudo grubby --update-kernel=ALL --args="systemd.unified_cgroup_hierarchy=1"
sudo reboot
```

**Use faster storage driver:**
```bash
# Check current driver
podman info --format '{{.Store.GraphDriverName}}'

# overlay is fastest (should be default)
# If using vfs, consider switching to overlay
```

## Verifying Podman Setup

Run this complete test:

```bash
#!/bin/bash

echo "=== Podman Setup Verification ==="

# Test 1: Podman installed
echo -n "Podman installed: "
if command -v podman &> /dev/null; then
    podman --version
else
    echo "FAILED - not installed"
    exit 1
fi

# Test 2: Podman working
echo -n "Podman functional: "
if podman ps &> /dev/null; then
    echo "OK"
else
    echo "FAILED - cannot list containers"
    exit 1
fi

# Test 3: KIND installed
echo -n "KIND installed: "
if command -v kind &> /dev/null; then
    kind version
else
    echo "FAILED - not installed"
    exit 1
fi

# Test 4: Create test cluster
echo "Creating test cluster..."
export KIND_EXPERIMENTAL_PROVIDER=podman
kind create cluster --name test-podman

if [ $? -eq 0 ]; then
    echo "✓ Test cluster created successfully"
    kind delete cluster --name test-podman
    echo "✓ Test cluster deleted"
else
    echo "✗ Failed to create test cluster"
    exit 1
fi

echo "=== All checks passed! ==="
```

Save as `test-podman.sh`, make executable, and run:
```bash
chmod +x test-podman.sh
./test-podman.sh
```

## Running the Full Setup with Podman

Once Podman is configured:

```bash
# 1. Ensure Podman is running (macOS/Windows)
podman machine start

# 2. Run setup (auto-detects Podman)
./setup-kind-env.sh

# 3. Verify
kubectl get nodes
curl http://localhost:9090/-/healthy
curl http://localhost:8000/health

# 4. Run simulation
python cli_prometheus.py
```

## Switching Between Docker and Podman

You can switch between runtimes:

**Use Docker:**
```bash
export KIND_EXPERIMENTAL_PROVIDER=docker
kind create cluster --config k8s/kind-config.yaml --name node-classifier-sim
```

**Use Podman:**
```bash
export KIND_EXPERIMENTAL_PROVIDER=podman
kind create cluster --config k8s/kind-config.yaml --name node-classifier-sim
```

**Reset (auto-detect):**
```bash
unset KIND_EXPERIMENTAL_PROVIDER
./setup-kind-env.sh
```

## Additional Resources

- [KIND with Podman documentation](https://kind.sigs.k8s.io/docs/user/rootless/)
- [Podman documentation](https://docs.podman.io/)
- [Podman rootless setup](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md)
- [KIND Podman provider](https://kind.sigs.k8s.io/docs/user/configuration/#experimental-podman-provider)

## Known Issues

### SELinux on Fedora/RHEL

If you encounter permission errors:

```bash
# Temporarily disable SELinux (testing only)
sudo setenforce 0

# Permanent fix: adjust SELinux policies
sudo setsebool -P container_manage_cgroup true
```

### cgroups v1 vs v2

KIND works better with cgroups v2. Check your version:

```bash
mount | grep cgroup
```

If using v1 and experiencing issues, consider upgrading your kernel.

### Podman Machine on macOS

Podman machine uses QEMU. For better performance:

```bash
# Increase resources
podman machine set --cpus 4 --memory 8192

# Restart
podman machine stop
podman machine start
```

## Getting Help

If you encounter issues:

1. Check Podman logs:
   ```bash
   podman info
   podman system df
   ```

2. Check KIND logs:
   ```bash
   kind export logs /tmp/kind-logs --name node-classifier-sim
   ```

3. Check our troubleshooting section in PROMETHEUS_SETUP.md

4. File an issue with:
   - Podman version: `podman --version`
   - KIND version: `kind version`
   - OS: `uname -a`
   - Error logs