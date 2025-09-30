import json
from typing import List, Dict, Any
from pathlib import Path
from node import Node

class ScenarioLoader:
    """Loads and manages node scenarios from JSON files."""

    @staticmethod
    def load_scenario(file_path: str) -> Dict[str, List[Node]]:
        """
        Load scenarios from a JSON file.

        Expected JSON format:
        {
            "scenario_name": [
                {
                    "name": "node1",
                    "cpu_usage": 0.5,
                    "cpu_pressure": 0.2,
                    "memory_usage": 0.7,
                    "memory_pressure": 0.1
                },
                ...
            ],
            ...
        }
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            scenarios = {}
            for scenario_name, nodes_data in data.items():
                nodes = [Node.from_dict(node_data) for node_data in nodes_data]
                scenarios[scenario_name] = nodes

            return scenarios

        except FileNotFoundError:
            raise FileNotFoundError(f"Scenario file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")
        except KeyError as e:
            raise ValueError(f"Missing required field in scenario data: {e}")

    @staticmethod
    def save_scenario(scenarios: Dict[str, List[Node]], file_path: str):
        """Save scenarios to a JSON file."""
        data = {}
        for scenario_name, nodes in scenarios.items():
            data[scenario_name] = [node.to_dict() for node in nodes]

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def calculate_pressure_from_utilization(utilization: float) -> float:
        """
        Calculate realistic pressure based on utilization.

        Rules:
        - Under 70% utilization: pressure is almost 0 (0-5%)
        - Over 70% utilization: pressure grows, reaching up to 30% at 100% utilization
        """
        if utilization < 0.7:
            # Low utilization: minimal pressure (0-5%)
            return min(0.05, utilization * 0.07)
        else:
            # High utilization: pressure grows from 0 to 30%
            # Linear growth from 70% utilization onwards
            pressure_factor = (utilization - 0.7) / 0.3  # 0 to 1 for 70% to 100%
            return pressure_factor * 0.3  # Scale to max 30%

    @staticmethod
    def create_sample_scenarios() -> Dict[str, List[Node]]:
        """Create sample scenarios with realistic utilization-pressure relationships."""
        import random

        def create_realistic_node(name: str, cpu_target: float, mem_target: float,
                                variance: float = 0.05) -> Node:
            """Create a node with realistic pressure values based on utilization."""
            # Add some variance to targets
            cpu_usage = max(0.0, min(1.0, cpu_target + random.uniform(-variance, variance)))
            mem_usage = max(0.0, min(1.0, mem_target + random.uniform(-variance, variance)))

            # Calculate realistic pressures
            cpu_pressure = ScenarioLoader.calculate_pressure_from_utilization(cpu_usage)
            mem_pressure = ScenarioLoader.calculate_pressure_from_utilization(mem_usage)

            # Add small random variance to pressure (±10% of calculated value)
            cpu_pressure += random.uniform(-0.01, 0.01)
            mem_pressure += random.uniform(-0.01, 0.01)

            # Ensure pressures stay within bounds
            cpu_pressure = max(0.0, min(1.0, cpu_pressure))
            mem_pressure = max(0.0, min(1.0, mem_pressure))

            return Node(name, cpu_usage, cpu_pressure, mem_usage, mem_pressure)

        # Set seed for reproducible scenarios
        random.seed(42)

        return {
            "light_load": [
                create_realistic_node("node-1", 0.15, 0.20),
                create_realistic_node("node-2", 0.25, 0.30),
                create_realistic_node("node-3", 0.10, 0.15),
                create_realistic_node("node-4", 0.18, 0.25),
                create_realistic_node("node-5", 0.12, 0.18),
                create_realistic_node("node-6", 0.22, 0.28),
            ],
            "mixed_load": [
                create_realistic_node("node-1", 0.45, 0.60),
                create_realistic_node("node-2", 0.75, 0.40),  # High CPU, moderate memory
                create_realistic_node("node-3", 0.35, 0.80),  # Moderate CPU, high memory
                create_realistic_node("node-4", 0.25, 0.65),
                create_realistic_node("node-5", 0.85, 0.55),  # High CPU
                create_realistic_node("node-6", 0.55, 0.45),
                create_realistic_node("node-7", 0.65, 0.75),
                create_realistic_node("node-8", 0.40, 0.85),  # High memory
            ],
            "heavy_load": [
                create_realistic_node("node-1", 0.85, 0.90),
                create_realistic_node("node-2", 0.90, 0.85),
                create_realistic_node("node-3", 0.80, 0.95),
                create_realistic_node("node-4", 0.88, 0.82),
                create_realistic_node("node-5", 0.75, 0.78),
                create_realistic_node("node-6", 0.92, 0.88),
                create_realistic_node("node-7", 0.87, 0.83),
                create_realistic_node("node-8", 0.83, 0.93),
                create_realistic_node("node-9", 0.89, 0.86),
                create_realistic_node("node-10", 0.78, 0.80),
            ],
            "realistic_progression": [
                # Show progression from low to high utilization with realistic pressure
                create_realistic_node("node-1", 0.10, 0.15),  # Very low - minimal pressure
                create_realistic_node("node-2", 0.30, 0.35),  # Low - minimal pressure
                create_realistic_node("node-3", 0.50, 0.55),  # Medium - minimal pressure
                create_realistic_node("node-4", 0.65, 0.68),  # Getting higher - still minimal pressure
                create_realistic_node("node-5", 0.72, 0.75),  # Above 70% - pressure starts
                create_realistic_node("node-6", 0.80, 0.82),  # High - noticeable pressure
                create_realistic_node("node-7", 0.90, 0.88),  # Very high - significant pressure
                create_realistic_node("node-8", 0.95, 0.92),  # Near max - high pressure
            ]
        }

    @staticmethod
    def generate_sample_file(file_path: str = "sample_scenarios.json"):
        """Generate a sample JSON file with scenarios."""
        scenarios = ScenarioLoader.create_sample_scenarios()
        ScenarioLoader.save_scenario(scenarios, file_path)
        return file_path