from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import math
from node import Node

class ClassificationAlgorithm(ABC):
    """Abstract base class for node classification algorithms."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs

    @abstractmethod
    def calculate_score(self, node: Node) -> float:
        """Calculate a score (0-1) for a node where 0 = least loaded, 1 = most loaded."""
        pass

    def classify_nodes(self, nodes: List[Node]) -> List[tuple[Node, float]]:
        """Classify nodes and return sorted list (least to most loaded)."""
        scored_nodes = [(node, self.calculate_score(node)) for node in nodes]
        return sorted(scored_nodes, key=lambda x: x[1])

    def get_params(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return self.params.copy()

    def set_param(self, key: str, value: Any):
        """Set a parameter value."""
        self.params[key] = value

class WeightedAverageAlgorithm(ClassificationAlgorithm):
    """Simple weighted average of all metrics."""

    def __init__(self, cpu_weight=0.25, cpu_pressure_weight=0.25,
                 memory_weight=0.25, memory_pressure_weight=0.25):
        super().__init__("Weighted Average",
                        cpu_weight=cpu_weight,
                        cpu_pressure_weight=cpu_pressure_weight,
                        memory_weight=memory_weight,
                        memory_pressure_weight=memory_pressure_weight)

    def calculate_score(self, node: Node) -> float:
        return (node.cpu_usage * self.params['cpu_weight'] +
                node.cpu_pressure * self.params['cpu_pressure_weight'] +
                node.memory_usage * self.params['memory_weight'] +
                node.memory_pressure * self.params['memory_pressure_weight'])

class MaxMetricAlgorithm(ClassificationAlgorithm):
    """Uses the maximum metric value as the score."""

    def __init__(self):
        super().__init__("Max Metric")

    def calculate_score(self, node: Node) -> float:
        return max(node.cpu_usage, node.cpu_pressure,
                  node.memory_usage, node.memory_pressure)

class EuclideanDistanceAlgorithm(ClassificationAlgorithm):
    """Uses Euclidean distance from origin (0,0,0,0) as load metric."""

    def __init__(self):
        super().__init__("Euclidean Distance")

    def calculate_score(self, node: Node) -> float:
        return math.sqrt(node.cpu_usage**2 + node.cpu_pressure**2 +
                        node.memory_usage**2 + node.memory_pressure**2) / 2.0

class PressureFocusedAlgorithm(ClassificationAlgorithm):
    """Prioritizes pressure metrics over usage metrics."""

    def __init__(self, pressure_multiplier=2.0):
        super().__init__("Pressure Focused", pressure_multiplier=pressure_multiplier)

    def calculate_score(self, node: Node) -> float:
        pressure_score = (node.cpu_pressure + node.memory_pressure) / 2.0
        usage_score = (node.cpu_usage + node.memory_usage) / 2.0

        weighted_pressure = pressure_score * self.params['pressure_multiplier']
        combined = (weighted_pressure + usage_score) / (self.params['pressure_multiplier'] + 1)

        return min(combined, 1.0)

class ResourceTypeAlgorithm(ClassificationAlgorithm):
    """Focuses on either CPU or Memory resources."""

    def __init__(self, resource_type="cpu", usage_weight=0.7, pressure_weight=0.3):
        super().__init__(f"{resource_type.upper()} Focused",
                        resource_type=resource_type,
                        usage_weight=usage_weight,
                        pressure_weight=pressure_weight)

    def calculate_score(self, node: Node) -> float:
        if self.params['resource_type'] == "cpu":
            return (node.cpu_usage * self.params['usage_weight'] +
                   node.cpu_pressure * self.params['pressure_weight'])
        else:  # memory
            return (node.memory_usage * self.params['usage_weight'] +
                   node.memory_pressure * self.params['pressure_weight'])

class WeightedRMSPositiveDeviationAlgorithm(ClassificationAlgorithm):
    """Weighted Root Mean Square of Positive Deviation algorithm.

    For each dimension, calculates cluster average and computes positive deviations
    (values above average). Then calculates weighted RMS of these positive deviations.
    Weights: 0.15 for usage metrics, 0.35 for pressure metrics.
    """

    def __init__(self):
        super().__init__("Weighted RMS Positive Deviation",
                        cpu_usage_weight=0.15,
                        cpu_pressure_weight=0.35,
                        memory_usage_weight=0.15,
                        memory_pressure_weight=0.35)

    def classify_nodes(self, nodes: List[Node]) -> List[Tuple[Node, float]]:
        """Classify nodes using weighted RMS of positive deviations."""
        if not nodes:
            return []

        # Calculate cluster averages for each dimension
        cpu_usage_avg = sum(node.cpu_usage for node in nodes) / len(nodes)
        cpu_pressure_avg = sum(node.cpu_pressure for node in nodes) / len(nodes)
        memory_usage_avg = sum(node.memory_usage for node in nodes) / len(nodes)
        memory_pressure_avg = sum(node.memory_pressure for node in nodes) / len(nodes)

        # Calculate weighted RMS of positive deviations for each node
        scored_nodes = []
        for node in nodes:
            # Calculate positive deviations (0 if below average, actual deviation if above)
            cpu_usage_dev = max(0, node.cpu_usage - cpu_usage_avg)
            cpu_pressure_dev = max(0, node.cpu_pressure - cpu_pressure_avg)
            memory_usage_dev = max(0, node.memory_usage - memory_usage_avg)
            memory_pressure_dev = max(0, node.memory_pressure - memory_pressure_avg)

            # Apply weights and calculate weighted sum of squares
            weighted_sum_squares = (
                self.params['cpu_usage_weight'] * (cpu_usage_dev ** 2) +
                self.params['cpu_pressure_weight'] * (cpu_pressure_dev ** 2) +
                self.params['memory_usage_weight'] * (memory_usage_dev ** 2) +
                self.params['memory_pressure_weight'] * (memory_pressure_dev ** 2)
            )

            # Calculate weighted RMS (root mean square)
            total_weight = (self.params['cpu_usage_weight'] +
                          self.params['cpu_pressure_weight'] +
                          self.params['memory_usage_weight'] +
                          self.params['memory_pressure_weight'])

            score = (weighted_sum_squares / total_weight) ** 0.5
            scored_nodes.append((node, score))

        # Sort by score (ascending - least loaded first)
        return sorted(scored_nodes, key=lambda x: x[1])

    def calculate_score(self, node: Node) -> float:
        """Calculate score for a single node (requires context of other nodes)."""
        # This method is called when we don't have cluster context
        # Fall back to simple weighted average of all metrics
        return (node.cpu_usage * self.params['cpu_usage_weight'] +
                node.cpu_pressure * self.params['cpu_pressure_weight'] +
                node.memory_usage * self.params['memory_usage_weight'] +
                node.memory_pressure * self.params['memory_pressure_weight'])

class ParetoFrontAlgorithm(ClassificationAlgorithm):
    """Pareto Front algorithm inspired by NSGA-II for multi-objective optimization.

    Identifies non-dominated solutions across all four metrics and ranks nodes
    based on Pareto front level and crowding distance. Nodes in the first front
    represent optimal trade-offs between resource metrics.
    """

    def __init__(self):
        super().__init__("Pareto Front (NSGA-II)")

    def dominates(self, node1: Node, node2: Node) -> bool:
        """Check if node1 dominates node2 (node1 is worse in all objectives)."""
        # For load classification, higher values are worse, so node1 dominates node2 if
        # node1 has all metrics <= node2's metrics and at least one is strictly less
        objectives1 = [node1.cpu_usage, node1.cpu_pressure, node1.memory_usage, node1.memory_pressure]
        objectives2 = [node2.cpu_usage, node2.cpu_pressure, node2.memory_usage, node2.memory_pressure]

        # Check if node1 is at least as good as node2 in all objectives
        at_least_as_good = all(obj1 <= obj2 for obj1, obj2 in zip(objectives1, objectives2))
        # Check if node1 is strictly better in at least one objective
        strictly_better = any(obj1 < obj2 for obj1, obj2 in zip(objectives1, objectives2))

        return at_least_as_good and strictly_better

    def fast_non_dominated_sort(self, nodes: List[Node]) -> List[List[int]]:
        """Perform fast non-dominated sorting to identify Pareto fronts."""
        n = len(nodes)
        fronts = [[]]
        dominated_count = [0] * n  # Number of solutions that dominate solution i
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by solution i

        # For each solution, find which solutions it dominates and count dominating solutions
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(nodes[i], nodes[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(nodes[j], nodes[i]):
                        dominated_count[i] += 1

            # If no solution dominates this one, it belongs to the first front
            if dominated_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        front_index = 0
        while front_index < len(fronts) and fronts[front_index]:
            next_front = []
            for i in fronts[front_index]:
                for j in dominated_solutions[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front_index += 1

        return fronts

    def calculate_crowding_distance(self, nodes: List[Node], front: List[int]) -> List[float]:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            return [float('inf')] * len(front)

        distances = [0.0] * len(front)
        objectives = [
            [nodes[i].cpu_usage for i in front],
            [nodes[i].cpu_pressure for i in front],
            [nodes[i].memory_usage for i in front],
            [nodes[i].memory_pressure for i in front]
        ]

        # For each objective
        for obj_idx, obj_values in enumerate(objectives):
            # Sort indices by objective value
            sorted_indices = sorted(range(len(front)), key=lambda i: obj_values[i])

            # Set boundary solutions to infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Calculate crowding distance for intermediate solutions
            obj_range = max(obj_values) - min(obj_values)
            if obj_range > 0:
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    distances[idx] += (obj_values[next_idx] - obj_values[prev_idx]) / obj_range

        return distances

    def classify_nodes(self, nodes: List[Node]) -> List[Tuple[Node, float]]:
        """Classify nodes using Pareto front ranking with crowding distance."""
        if not nodes:
            return []

        # Perform non-dominated sorting
        fronts = self.fast_non_dominated_sort(nodes)

        # Assign normalized scores between 0 and 1 based on front level and crowding distance
        scored_nodes = []
        total_fronts = len(fronts)

        for front_level, front in enumerate(fronts):
            # Calculate crowding distances for this front
            distances = self.calculate_crowding_distance(nodes, front)

            # Normalize front level to [0, 1] range
            # Best front (0) gets score close to 0, worst front gets score close to 1
            front_score = front_level / max(1, total_fronts - 1) if total_fronts > 1 else 0.0

            # Calculate crowding distance contribution (smaller is better for diversity)
            for i, node_idx in enumerate(front):
                if distances[i] == float('inf'):
                    # Boundary solutions get slightly better scores for diversity
                    crowding_contribution = 0.0
                else:
                    # Normalize crowding distance contribution to small range [0, 0.1]
                    max_dist = max(d for d in distances if d != float('inf')) if any(d != float('inf') for d in distances) else 1.0
                    if max_dist > 0:
                        crowding_contribution = 0.1 * (1.0 - distances[i] / max_dist)
                    else:
                        crowding_contribution = 0.05

                # Combine front score (primary) with crowding distance (secondary)
                final_score = front_score + crowding_contribution

                # Ensure score is in [0, 1] range
                final_score = max(0.0, min(1.0, final_score))
                scored_nodes.append((nodes[node_idx], final_score))

        # Sort by score (ascending - better solutions have lower scores)
        return sorted(scored_nodes, key=lambda x: x[1])

    def calculate_score(self, node: Node) -> float:
        """Calculate score for a single node (fallback when no cluster context)."""
        # When called individually, use average of all metrics as fallback
        return (node.cpu_usage + node.cpu_pressure + node.memory_usage + node.memory_pressure) / 4.0

class CentroidDistanceAlgorithm(ClassificationAlgorithm):
    """Centroid-based classification using distance from cluster center.

    Calculates the multidimensional centroid (center) of all nodes in the cluster,
    then ranks nodes based on their Euclidean distance from this center point.
    Nodes closer to the center are considered more balanced, while nodes farther
    from the center are considered outliers with higher load.

    NOTE: This algorithm treats under-utilized and over-utilized nodes the same way
    if they are at equal distance from the center.
    """

    def __init__(self):
        super().__init__("Centroid Distance")

    def classify_nodes(self, nodes: List[Node]) -> List[Tuple[Node, float]]:
        """Classify nodes based on distance from cluster centroid."""
        if not nodes:
            return []

        # Calculate cluster centroid (average of all nodes in 4D space)
        centroid_cpu_usage = sum(node.cpu_usage for node in nodes) / len(nodes)
        centroid_cpu_pressure = sum(node.cpu_pressure for node in nodes) / len(nodes)
        centroid_memory_usage = sum(node.memory_usage for node in nodes) / len(nodes)
        centroid_memory_pressure = sum(node.memory_pressure for node in nodes) / len(nodes)

        # Calculate distance from centroid for each node
        scored_nodes = []
        for node in nodes:
            # Euclidean distance in 4D space from centroid
            distance = math.sqrt(
                (node.cpu_usage - centroid_cpu_usage) ** 2 +
                (node.cpu_pressure - centroid_cpu_pressure) ** 2 +
                (node.memory_usage - centroid_memory_usage) ** 2 +
                (node.memory_pressure - centroid_memory_pressure) ** 2
            )
            scored_nodes.append((node, distance))

        # Normalize scores to [0, 1] range
        if scored_nodes:
            max_distance = max(score for _, score in scored_nodes)
            if max_distance > 0:
                scored_nodes = [(node, score / max_distance) for node, score in scored_nodes]

        # Sort by distance (ascending - nodes closer to center are less loaded)
        return sorted(scored_nodes, key=lambda x: x[1])

    def calculate_score(self, node: Node) -> float:
        """Calculate score for a single node (fallback when no cluster context)."""
        # When called individually, use Euclidean distance from ideal center (0.5, 0.5, 0.5, 0.5)
        # This assumes a balanced node would be at 50% of all metrics
        ideal_center = 0.5
        return math.sqrt(
            (node.cpu_usage - ideal_center) ** 2 +
            (node.cpu_pressure - ideal_center) ** 2 +
            (node.memory_usage - ideal_center) ** 2 +
            (node.memory_pressure - ideal_center) ** 2
        ) / 2.0  # Divide by 2 to normalize to [0, 1] range

class VarianceMinimizationAlgorithm(ClassificationAlgorithm):
    """Variance Minimization - balances cluster by penalizing nodes that increase variance.

    Calculates how much each node contributes to the overall variance across all metrics.
    Nodes that deviate most from cluster mean get higher scores (should offload VMs).
    This ensures all nodes are "equally happy" by minimizing differences.

    Pressure metrics are weighted higher (2x) since they indicate actual resource contention
    and unhappy workloads, not just utilization.
    """

    def __init__(self):
        super().__init__("Variance Minimization",
                        cpu_usage_weight=1.0,
                        cpu_pressure_weight=2.0,  # Pressure is an alarm bell
                        memory_usage_weight=1.0,
                        memory_pressure_weight=2.0)  # Pressure is an alarm bell

    def classify_nodes(self, nodes: List[Node]) -> List[Tuple[Node, float]]:
        """Classify nodes based on their contribution to cluster variance."""
        if not nodes:
            return []

        # Calculate cluster means
        n = len(nodes)
        cpu_usage_mean = sum(node.cpu_usage for node in nodes) / n
        cpu_pressure_mean = sum(node.cpu_pressure for node in nodes) / n
        memory_usage_mean = sum(node.memory_usage for node in nodes) / n
        memory_pressure_mean = sum(node.memory_pressure for node in nodes) / n

        # For each node, calculate weighted squared deviation from mean
        scored_nodes = []
        for node in nodes:
            # Squared deviations from cluster mean
            cpu_usage_dev_sq = (node.cpu_usage - cpu_usage_mean) ** 2
            cpu_pressure_dev_sq = (node.cpu_pressure - cpu_pressure_mean) ** 2
            memory_usage_dev_sq = (node.memory_usage - memory_usage_mean) ** 2
            memory_pressure_dev_sq = (node.memory_pressure - memory_pressure_mean) ** 2

            # Apply weights (pressure metrics weighted 2x)
            weighted_variance_contribution = (
                self.params['cpu_usage_weight'] * cpu_usage_dev_sq +
                self.params['cpu_pressure_weight'] * cpu_pressure_dev_sq +
                self.params['memory_usage_weight'] * memory_usage_dev_sq +
                self.params['memory_pressure_weight'] * memory_pressure_dev_sq
            )

            # Take square root to get a distance-like metric
            score = math.sqrt(weighted_variance_contribution)
            scored_nodes.append((node, score))

        # Normalize scores to [0, 1] range
        if scored_nodes:
            max_score = max(score for _, score in scored_nodes)
            if max_score > 0:
                scored_nodes = [(node, score / max_score) for node, score in scored_nodes]

        # Sort by score (ascending - nodes closest to mean are most balanced)
        return sorted(scored_nodes, key=lambda x: x[1])

    def calculate_score(self, node: Node) -> float:
        """Calculate score for a single node (fallback when no cluster context)."""
        # When called individually, use weighted average with pressure emphasis
        return (
            node.cpu_usage * self.params['cpu_usage_weight'] +
            node.cpu_pressure * self.params['cpu_pressure_weight'] +
            node.memory_usage * self.params['memory_usage_weight'] +
            node.memory_pressure * self.params['memory_pressure_weight']
        ) / (self.params['cpu_usage_weight'] +
             self.params['cpu_pressure_weight'] +
             self.params['memory_usage_weight'] +
             self.params['memory_pressure_weight'])

class DirectionalVarianceMinimizationAlgorithm(ClassificationAlgorithm):
    """Directional Variance Minimization - balances cluster by penalizing nodes above mean.

    Similar to Variance Minimization but only penalizes positive deviations from cluster mean.
    Nodes below the mean get score of 0 (underutilized, should accept VMs).
    Nodes above the mean get scored based on how far above they are (overutilized, should offload).

    This ensures the algorithm correctly identifies overutilized vs underutilized nodes
    for load-aware rebalancing. Pressure metrics weighted 2x as "alarm bells".
    """

    def __init__(self):
        super().__init__("Directional Variance Minimization",
                        cpu_usage_weight=1.0,
                        cpu_pressure_weight=2.0,  # Pressure is an alarm bell
                        memory_usage_weight=1.0,
                        memory_pressure_weight=2.0)  # Pressure is an alarm bell

    def classify_nodes(self, nodes: List[Node]) -> List[Tuple[Node, float]]:
        """Classify nodes based on their positive deviation from cluster mean.

        Only penalizes nodes ABOVE the cluster mean (overutilized).
        Nodes below the mean get score of 0 (underutilized, available for VMs).
        """
        if not nodes:
            return []

        # Calculate cluster means
        n = len(nodes)
        cpu_usage_mean = sum(node.cpu_usage for node in nodes) / n
        cpu_pressure_mean = sum(node.cpu_pressure for node in nodes) / n
        memory_usage_mean = sum(node.memory_usage for node in nodes) / n
        memory_pressure_mean = sum(node.memory_pressure for node in nodes) / n

        # For each node, calculate weighted positive deviation from mean
        scored_nodes = []
        for node in nodes:
            # Only count positive deviations (node above mean)
            cpu_usage_dev = max(0, node.cpu_usage - cpu_usage_mean)
            cpu_pressure_dev = max(0, node.cpu_pressure - cpu_pressure_mean)
            memory_usage_dev = max(0, node.memory_usage - memory_usage_mean)
            memory_pressure_dev = max(0, node.memory_pressure - memory_pressure_mean)

            # Apply weights (pressure metrics weighted 2x)
            weighted_deviation = (
                self.params['cpu_usage_weight'] * cpu_usage_dev +
                self.params['cpu_pressure_weight'] * cpu_pressure_dev +
                self.params['memory_usage_weight'] * memory_usage_dev +
                self.params['memory_pressure_weight'] * memory_pressure_dev
            )

            scored_nodes.append((node, weighted_deviation))

        # Normalize scores to [0, 1] range
        if scored_nodes:
            max_score = max(score for _, score in scored_nodes)
            if max_score > 0:
                scored_nodes = [(node, score / max_score) for node, score in scored_nodes]

        # Sort by score (ascending - nodes at/below mean ranked first)
        return sorted(scored_nodes, key=lambda x: x[1])

    def calculate_score(self, node: Node) -> float:
        """Calculate score for a single node (fallback when no cluster context)."""
        # When called individually, use weighted average with pressure emphasis
        return (
            node.cpu_usage * self.params['cpu_usage_weight'] +
            node.cpu_pressure * self.params['cpu_pressure_weight'] +
            node.memory_usage * self.params['memory_usage_weight'] +
            node.memory_pressure * self.params['memory_pressure_weight']
        ) / (self.params['cpu_usage_weight'] +
             self.params['cpu_pressure_weight'] +
             self.params['memory_usage_weight'] +
             self.params['memory_pressure_weight'])


class DirectionalCentroidDistanceAlgorithm(ClassificationAlgorithm):
    """Directional Centroid Distance - measures positive deviation from cluster center.

    Calculates the multidimensional centroid (center) of all nodes in the cluster,
    then ranks nodes based on how much they exceed the center in each dimension.
    Only positive deviations (above center) contribute to the score.

    This ensures overutilized nodes get higher scores than underutilized nodes,
    solving the limitation of standard Centroid Distance.
    """

    def __init__(self):
        super().__init__("Directional Centroid Distance")

    def classify_nodes(self, nodes: List[Node]) -> List[Tuple[Node, float]]:
        """Classify nodes based on positive deviation from cluster centroid."""
        if not nodes:
            return []

        # Calculate cluster centroid (average of all nodes in 4D space)
        centroid_cpu_usage = sum(node.cpu_usage for node in nodes) / len(nodes)
        centroid_cpu_pressure = sum(node.cpu_pressure for node in nodes) / len(nodes)
        centroid_memory_usage = sum(node.memory_usage for node in nodes) / len(nodes)
        centroid_memory_pressure = sum(node.memory_pressure for node in nodes) / len(nodes)

        # Calculate directional distance from centroid for each node
        scored_nodes = []
        for node in nodes:
            # Only positive deviations count (nodes above center)
            # Nodes below center get 0 for that dimension
            cpu_usage_dev = max(0, node.cpu_usage - centroid_cpu_usage)
            cpu_pressure_dev = max(0, node.cpu_pressure - centroid_cpu_pressure)
            memory_usage_dev = max(0, node.memory_usage - centroid_memory_usage)
            memory_pressure_dev = max(0, node.memory_pressure - centroid_memory_pressure)

            # Euclidean distance in 4D space using only positive deviations
            distance = math.sqrt(
                cpu_usage_dev ** 2 +
                cpu_pressure_dev ** 2 +
                memory_usage_dev ** 2 +
                memory_pressure_dev ** 2
            )
            scored_nodes.append((node, distance))

        # Normalize scores to [0, 1] range
        if scored_nodes:
            max_distance = max(score for _, score in scored_nodes)
            if max_distance > 0:
                scored_nodes = [(node, score / max_distance) for node, score in scored_nodes]

        # Sort by distance (ascending - nodes below/at center get lower scores)
        return sorted(scored_nodes, key=lambda x: x[1])

    def calculate_score(self, node: Node) -> float:
        """Calculate score for a single node (fallback when no cluster context)."""
        # When called individually, use positive deviation from ideal center (0.5, 0.5, 0.5, 0.5)
        # This assumes a balanced node would be at 50% of all metrics
        ideal_center = 0.5
        cpu_usage_dev = max(0, node.cpu_usage - ideal_center)
        cpu_pressure_dev = max(0, node.cpu_pressure - ideal_center)
        memory_usage_dev = max(0, node.memory_usage - ideal_center)
        memory_pressure_dev = max(0, node.memory_pressure - ideal_center)

        return math.sqrt(
            cpu_usage_dev ** 2 +
            cpu_pressure_dev ** 2 +
            memory_usage_dev ** 2 +
            memory_pressure_dev ** 2
        ) / 2.0  # Divide by 2 to normalize to [0, 1] range

def get_default_algorithms() -> List[ClassificationAlgorithm]:
    """Get a list of default classification algorithms."""
    return [
        WeightedAverageAlgorithm(),
        MaxMetricAlgorithm(),
        EuclideanDistanceAlgorithm(),
        PressureFocusedAlgorithm(),
        WeightedRMSPositiveDeviationAlgorithm(),
        ParetoFrontAlgorithm(),
        CentroidDistanceAlgorithm(),
        DirectionalCentroidDistanceAlgorithm(),
        VarianceMinimizationAlgorithm(),
        DirectionalVarianceMinimizationAlgorithm(),
        ResourceTypeAlgorithm("cpu"),
        ResourceTypeAlgorithm("memory")
    ]