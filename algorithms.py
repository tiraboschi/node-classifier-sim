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

def get_default_algorithms() -> List[ClassificationAlgorithm]:
    """Get a list of default classification algorithms."""
    return [
        WeightedAverageAlgorithm(),
        MaxMetricAlgorithm(),
        EuclideanDistanceAlgorithm(),
        PressureFocusedAlgorithm(),
        WeightedRMSPositiveDeviationAlgorithm(),
        ResourceTypeAlgorithm("cpu"),
        ResourceTypeAlgorithm("memory")
    ]