from dataclasses import dataclass
from typing import Dict, Any
import json

@dataclass
class Node:
    """Represents a Kubernetes node with resource usage metrics."""

    name: str
    cpu_usage: float  # 0.0 to 1.0
    cpu_pressure: float  # 0.0 to 1.0 (PSI CPU pressure)
    memory_usage: float  # 0.0 to 1.0
    memory_pressure: float  # 0.0 to 1.0 (PSI memory pressure)

    def __post_init__(self):
        """Validate that all metrics are within valid ranges."""
        for field_name, value in [
            ("cpu_usage", self.cpu_usage),
            ("cpu_pressure", self.cpu_pressure),
            ("memory_usage", self.memory_usage),
            ("memory_pressure", self.memory_pressure)
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "name": self.name,
            "cpu_usage": self.cpu_usage,
            "cpu_pressure": self.cpu_pressure,
            "memory_usage": self.memory_usage,
            "memory_pressure": self.memory_pressure
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary representation."""
        return cls(
            name=data["name"],
            cpu_usage=data["cpu_usage"],
            cpu_pressure=data["cpu_pressure"],
            memory_usage=data["memory_usage"],
            memory_pressure=data["memory_pressure"]
        )

    def get_metric(self, metric_name: str) -> float:
        """Get a specific metric value by name."""
        metric_map = {
            "cpu_usage": self.cpu_usage,
            "cpu_pressure": self.cpu_pressure,
            "memory_usage": self.memory_usage,
            "memory_pressure": self.memory_pressure
        }
        if metric_name not in metric_map:
            raise ValueError(f"Unknown metric: {metric_name}")
        return metric_map[metric_name]

    def get_available_metrics(self) -> list[str]:
        """Get list of available metric names."""
        return ["cpu_usage", "cpu_pressure", "memory_usage", "memory_pressure"]