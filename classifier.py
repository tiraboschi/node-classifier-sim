from enum import Enum
from typing import List, Dict, Tuple
from dataclasses import dataclass
from node import Node
from algorithms import ClassificationAlgorithm

class UtilizationCategory(Enum):
    """Categories for node utilization classification."""
    UNDER_UTILIZED = "under-utilized"
    APPROPRIATELY_UTILIZED = "appropriately-utilized"
    OVER_UTILIZED = "over-utilized"

class ThresholdMode(Enum):
    """Different threshold configuration modes."""
    LOW = "Low (10%:10%)"
    MEDIUM = "Medium (20%:20%)"
    HIGH = "High (30%:30%)"
    ASYMMETRIC_LOW = "AsymmetricLow (0%:10%)"
    ASYMMETRIC_MEDIUM = "AsymmetricMedium (0%:20%)"
    ASYMMETRIC_HIGH = "AsymmetricHigh (0%:30%)"

@dataclass
class ThresholdConfig:
    """Configuration for dynamic threshold calculation."""
    lower_percentage: float  # Percentage below average for under-utilized threshold
    upper_percentage: float  # Percentage above average for over-utilized threshold

    @classmethod
    def from_mode(cls, mode: ThresholdMode) -> 'ThresholdConfig':
        """Create threshold config from predefined mode."""
        configs = {
            ThresholdMode.LOW: cls(0.10, 0.10),
            ThresholdMode.MEDIUM: cls(0.20, 0.20),
            ThresholdMode.HIGH: cls(0.30, 0.30),
            ThresholdMode.ASYMMETRIC_LOW: cls(0.0, 0.10),
            ThresholdMode.ASYMMETRIC_MEDIUM: cls(0.0, 0.20),
            ThresholdMode.ASYMMETRIC_HIGH: cls(0.0, 0.30),
        }
        return configs[mode]

@dataclass
class ClassificationResult:
    """Result of node classification including category and thresholds used."""
    node: Node
    score: float
    category: UtilizationCategory
    cluster_average: float
    under_threshold: float
    over_threshold: float

class NodeClassifier:
    """Classifies nodes into utilization categories using dynamic thresholds."""

    def __init__(self, algorithm: ClassificationAlgorithm, threshold_config: ThresholdConfig):
        self.algorithm = algorithm
        self.threshold_config = threshold_config

    def classify_nodes(self, nodes: List[Node]) -> List[ClassificationResult]:
        """
        Classify nodes into utilization categories using dynamic thresholds.

        Returns nodes sorted by score (least to most loaded).
        """
        # Get scores from the classification algorithm
        scored_nodes = self.algorithm.classify_nodes(nodes)

        # Calculate cluster average
        total_score = sum(score for _, score in scored_nodes)
        cluster_average = total_score / len(scored_nodes) if scored_nodes else 0.0

        # Calculate dynamic thresholds
        under_threshold = cluster_average * (1 - self.threshold_config.lower_percentage)
        over_threshold = cluster_average * (1 + self.threshold_config.upper_percentage)

        # Classify each node
        results = []
        for node, score in scored_nodes:
            if score <= under_threshold:
                category = UtilizationCategory.UNDER_UTILIZED
            elif score >= over_threshold:
                category = UtilizationCategory.OVER_UTILIZED
            else:
                category = UtilizationCategory.APPROPRIATELY_UTILIZED

            results.append(ClassificationResult(
                node=node,
                score=score,
                category=category,
                cluster_average=cluster_average,
                under_threshold=under_threshold,
                over_threshold=over_threshold
            ))

        return results

    def get_category_summary(self, results: List[ClassificationResult]) -> Dict[UtilizationCategory, int]:
        """Get count of nodes in each category."""
        summary = {category: 0 for category in UtilizationCategory}
        for result in results:
            summary[result.category] += 1
        return summary

    def get_threshold_info(self, results: List[ClassificationResult]) -> Dict[str, float]:
        """Get threshold information from classification results."""
        if not results:
            return {}

        # All results have the same thresholds, so take from first
        first_result = results[0]
        return {
            "cluster_average": first_result.cluster_average,
            "under_threshold": first_result.under_threshold,
            "over_threshold": first_result.over_threshold,
            "lower_percentage": self.threshold_config.lower_percentage * 100,
            "upper_percentage": self.threshold_config.upper_percentage * 100
        }

def get_category_color(category: UtilizationCategory) -> str:
    """Get color code for category visualization."""
    colors = {
        UtilizationCategory.UNDER_UTILIZED: "#4CAF50",     # Green
        UtilizationCategory.APPROPRIATELY_UTILIZED: "#FFC107",  # Yellow/Orange
        UtilizationCategory.OVER_UTILIZED: "#F44336",      # Red
    }
    return colors[category]

def get_category_symbol(category: UtilizationCategory) -> str:
    """Get symbol for category representation."""
    symbols = {
        UtilizationCategory.UNDER_UTILIZED: "▼",
        UtilizationCategory.APPROPRIATELY_UTILIZED: "■",
        UtilizationCategory.OVER_UTILIZED: "▲",
    }
    return symbols[category]