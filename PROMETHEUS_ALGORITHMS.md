# Prometheus Algorithm Mapping

This document maps Python algorithm names to their corresponding Prometheus recording rules.

## Architecture

The descheduler plugin queries a **single metric per node** from Prometheus. All classification logic is expressed as PromQL recording rules defined in `k8s/prometheus-rules.yaml`, not in the descheduler code.

```
┌──────────────┐       ┌─────────────┐       ┌────────────────┐
│   Metrics    │──────▶│ Prometheus  │──────▶│  Descheduler   │
│   Exporter   │       │  Recording  │       │    Plugin      │
│              │       │    Rules    │       │                │
└──────────────┘       └─────────────┘       └────────────────┘
                              │
                              │ Computes algorithm scores
                              │ via PromQL expressions
                              ▼
                    One score per node [0-1]
```

## Algorithm to Metric Mapping

| Python Algorithm Name | Prometheus Metric | Description |
|----------------------|-------------------|-------------|
| **Weighted Average** | `descheduler:node:weighted_average:score` | Simple weighted average of CPU and memory metrics |
| **Max Metric** | `descheduler:node:max_metric:score` | Maximum of all four metrics (CPU usage, CPU pressure, memory usage, memory pressure) |
| **Euclidean Distance** | `descheduler:node:euclidean_distance:score` | Euclidean distance from origin (0,0,0,0) |
| **Pressure Focused** | `descheduler:node:pressure_focused:score` | Average of CPU and memory pressure only |
| **Weighted RMS Positive Deviation** | `descheduler:node:weighted_rms_positive_deviation:score` | Root mean square of weighted positive deviations from ideal |
| **Weighted Mean Square Positive Deviation** | `descheduler:node:weighted_mean_square_positive_deviation:score` | Mean square (without sqrt) of weighted positive deviations |
| **Linear Weighted Positive Deviation** | `descheduler:node:linear_weighted_positive_deviation:score` | Linear sum of weighted positive deviations |
| **Pareto Front (NSGA-II)** | `descheduler:node:pareto_front:score` | Simplified multi-objective scoring (sum of normalized objectives) |
| **Centroid Distance** | `descheduler:node:centroid_distance:score` | Distance from cluster centroid (average point) |
| **Directional Centroid Distance** | `descheduler:node:directional_centroid_distance:score` | Manhattan distance from centroid |
| **Variance Minimization** | `descheduler:node:variance_minimization:score` | Variance of CPU and memory usage from cluster average |
| **Directional Variance Minimization** | `descheduler:node:directional_variance_minimization:score` | Variance of all 4 metrics from cluster average |
| **Critical Dimension Focus** | `descheduler:node:critical_dimension_focus:score` | Maximum positive deviation across all dimensions |
| **Ideal Point Positive Distance** | `descheduler:node:ideal_point_positive_distance:score` | Euclidean distance from ideal using only positive deviations |
| **Linear Amplified IPPD (k=1.0)** | `descheduler:node:linear_amplified_ippd_k1:score` | IPPD amplified by 1.0x, clamped to [0,1] |
| **Linear Amplified IPPD (k=3.0)** | `descheduler:node:linear_amplified_ippd_k3:score` | IPPD amplified by 3.0x, clamped to [0,1] |
| **Linear Amplified IPPD (k=5.0)** | `descheduler:node:linear_amplified_ippd_k5:score` | IPPD amplified by 5.0x, clamped to [0,1] |
| **CPU Focused** | `descheduler:node:cpu_focused:score` | 75% weight on CPU metrics, 25% on memory |
| **Memory Focused** | `descheduler:node:memory_focused:score` | 25% weight on CPU metrics, 75% on memory |

## Base Metrics

All algorithms are computed from these base metrics:

- `descheduler:node:cpu_usage:ratio` - CPU utilization [0-1]
- `descheduler:node:cpu_pressure:psi` - CPU pressure stall information [0-1]
- `descheduler:node:memory_usage:ratio` - Memory utilization [0-1]
- `descheduler:node:memory_pressure:psi` - Memory pressure stall information [0-1]

## Cluster Averages (Ideal Point)

- `descheduler:cluster:cpu_usage:avg` - Average CPU usage across all nodes
- `descheduler:cluster:cpu_pressure:avg` - Average CPU pressure across all nodes
- `descheduler:cluster:memory_usage:avg` - Average memory usage across all nodes
- `descheduler:cluster:memory_pressure:avg` - Average memory pressure across all nodes

## Positive Deviations

Used by distance-based algorithms:

- `descheduler:node:cpu_usage:positive_deviation` - max(0, cpu_usage - cluster_avg)
- `descheduler:node:cpu_pressure:positive_deviation` - max(0, cpu_pressure - cluster_avg)
- `descheduler:node:memory_usage:positive_deviation` - max(0, memory_usage - cluster_avg)
- `descheduler:node:memory_pressure:positive_deviation` - max(0, memory_pressure - cluster_avg)

## Usage in Descheduler

The descheduler plugin configuration would look like:

```yaml
apiVersion: descheduler/v1alpha2
kind: DeschedulerPolicy
profiles:
  - name: ProfileName
    pluginConfig:
      - name: RemovePodsViolatingNodeTaints
        args:
          targetThresholds:
            cpu: 50
            memory: 50
          metricsDataSource:
            prometheusAddress: "http://prometheus:9090"
            # Query this metric for node scores
            query: "descheduler:node:ideal_point_positive_distance:score"
```

## Testing Algorithm Performance

Compare algorithm performance:

```bash
# Test with Ideal Point Positive Distance
python cli_prometheus.py --algorithm "Ideal Point Positive Distance"

# The descheduler would query:
# descheduler:node:ideal_point_positive_distance:score

# Test with Linear Amplified IPPD (k=3.0)
python cli_prometheus.py --algorithm "Linear Amplified Ideal Point Positive Distance (k=3.0)"

# The descheduler would query:
# descheduler:node:linear_amplified_ippd_k3:score
```

## Query Examples

Fetch scores for all nodes with a specific algorithm:

```bash
# Ideal Point Positive Distance
curl -s 'http://localhost:9090/api/v1/query?query=descheduler:node:ideal_point_positive_distance:score' | jq

# CPU Focused
curl -s 'http://localhost:9090/api/v1/query?query=descheduler:node:cpu_focused:score' | jq

# Linear Amplified IPPD (k=3.0)
curl -s 'http://localhost:9090/api/v1/query?query=descheduler:node:linear_amplified_ippd_k3:score' | jq
```

## Recording Rule Groups

All rules are defined in `k8s/prometheus-rules.yaml` and organized into logical groups:

1. **`descheduler.base.metrics`** - Raw node metrics (CPU/memory usage and PSI pressure)
2. **`descheduler.cluster.averages`** - Cluster-wide averages (ideal point) and standard deviations
3. **`descheduler.node.deviations`** - Positive deviations from cluster average
4. **`descheduler.algorithms.simple`** - Simple averaging algorithms (5 algorithms)
5. **`descheduler.algorithms.distance`** - Distance-based algorithms (6 algorithms)
6. **`descheduler.algorithms.rms`** - RMS-based algorithms (2 algorithms)
7. **`descheduler.algorithms.linear`** - Linear combination algorithms (1 algorithm)
8. **`descheduler.algorithms.advanced`** - Advanced multi-dimensional algorithms (4 algorithms)
9. **`descheduler.algorithms.meta`** - Meta-algorithms (Pareto Front)
10. **`descheduler.cluster.stats`** - Cluster-wide statistics (VM counts, node categories)
11. **`descheduler.alerts`** - Alert rules for monitoring

## Notes

- All scores are in the range **[0-1]** where:
  - **0** = ideal/balanced node
  - **1** = highly imbalanced/overloaded node

- The descheduler should evict workloads from nodes with **higher scores**

- Recording rules update every **15 seconds** by default

- For production, adjust the `interval` in the PrometheusRule based on your scrape interval and data freshness requirements