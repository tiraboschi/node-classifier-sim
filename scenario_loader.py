import json
from typing import List, Dict, Any
from pathlib import Path
from node import Node, VM
import random

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
        - 70-90% utilization: pressure grows moderately
        - 90-100% utilization: pressure grows exponentially (approaching saturation)

        This reflects reality: as resources approach limits, pressure increases dramatically.
        """
        if utilization < 0.7:
            # Low utilization: minimal pressure (0-5%)
            return min(0.05, utilization * 0.07)
        elif utilization < 0.9:
            # Moderate utilization: linear growth from ~0 to ~0.20
            # At 70%: ~0.0, At 90%: ~0.20
            pressure_factor = (utilization - 0.7) / 0.2  # 0 to 1 for 70% to 90%
            return pressure_factor * 0.20
        else:
            # High utilization (90-100%): exponential growth from 0.20 to 0.80
            # Use quadratic/exponential curve to show pressure rising sharply
            pressure_factor = (utilization - 0.9) / 0.1  # 0 to 1 for 90% to 100%
            # Quadratic growth: pressure rises sharply as we approach 100%
            return 0.20 + (pressure_factor ** 2) * 0.60  # 0.20 at 90%, up to 0.80 at 100%

    @staticmethod
    def create_sample_scenarios() -> Dict[str, List[Node]]:
        """Create sample scenarios with realistic utilization-pressure relationships."""
        import random

        # Global VM counter for unique IDs across all nodes
        vm_id_counter = 0

        def create_realistic_node(name: str, cpu_target: float, mem_target: float,
                                variance: float = 0.05,
                                vm_cpu_percent: float = 0.02,
                                vm_memory_percent: float = 0.04) -> Node:
            """
            Create a node with realistic pressure values based on utilization.
            Creates VMs with random consumption that sum to approximately the target utilization.

            Args:
                name: Node name
                cpu_target: Target CPU utilization (0.0-1.0)
                mem_target: Target memory utilization (0.0-1.0)
                variance: Random variance to add (default 5%)
                vm_cpu_percent: Maximum CPU consumption per VM (default 2%)
                vm_memory_percent: Maximum memory consumption per VM (default 4%)
            """
            nonlocal vm_id_counter

            # Add some variance to targets
            cpu_target_with_variance = max(0.0, min(1.0, cpu_target + random.uniform(-variance, variance)))
            mem_target_with_variance = max(0.0, min(1.0, mem_target + random.uniform(-variance, variance)))

            # Create VMs with random consumption until we reach target utilization
            vms = []
            cpu_accumulated = 0.0
            mem_accumulated = 0.0

            # Calculate target ratio to create VMs that maintain it
            target_ratio = cpu_target_with_variance / mem_target_with_variance if mem_target_with_variance > 0 else float('inf')

            # Keep creating VMs until we approach both targets
            safety_counter = 0
            while safety_counter < 100:
                # Calculate remaining needed
                cpu_remaining = cpu_target_with_variance - cpu_accumulated
                mem_remaining = mem_target_with_variance - mem_accumulated

                # Stop if both targets are reached (within 10% of max VM consumption)
                if cpu_remaining < vm_cpu_percent * 0.1 and mem_remaining < vm_memory_percent * 0.1:
                    break

                # Create VM maintaining the target ratio to avoid overshooting memory
                if mem_target_with_variance == 0:
                    # Only CPU needed
                    vm_cpu = random.uniform(0, min(vm_cpu_percent, cpu_remaining + vm_cpu_percent * 0.5))
                    vm_memory = 0
                elif cpu_target_with_variance == 0:
                    # Only memory needed
                    vm_cpu = 0
                    vm_memory = random.uniform(0, min(vm_memory_percent, mem_remaining + vm_memory_percent * 0.5))
                else:
                    # Both resources needed - create VM maintaining ratio
                    vm_cpu = random.uniform(0, min(vm_cpu_percent, cpu_remaining + vm_cpu_percent * 0.5))
                    vm_memory = vm_cpu / target_ratio

                    # If memory would exceed max or remaining + buffer, scale down
                    if vm_memory > vm_memory_percent or vm_memory > mem_remaining + vm_memory_percent * 0.5:
                        vm_memory = min(vm_memory_percent, mem_remaining + vm_memory_percent * 0.5)
                        vm_cpu = vm_memory * target_ratio

                        # If CPU now exceeds limits, scale down again
                        if vm_cpu > vm_cpu_percent:
                            vm_cpu = vm_cpu_percent
                            vm_memory = vm_cpu / target_ratio

                # Ensure valid values
                vm_cpu = max(0, vm_cpu)
                vm_memory = max(0, vm_memory)

                if vm_cpu <= 0 and vm_memory <= 0:
                    break

                # Create the VM
                vm_id_counter += 1
                vm = VM(
                    id=f"vm-{vm_id_counter}",
                    cpu_consumption=vm_cpu,
                    memory_consumption=vm_memory
                )
                vms.append(vm)

                cpu_accumulated += vm_cpu
                mem_accumulated += vm_memory
                safety_counter += 1

            # Calculate actual utilization from VMs
            cpu_usage = min(1.0, cpu_accumulated)
            mem_usage = min(1.0, mem_accumulated)

            # Calculate realistic pressures
            cpu_pressure = ScenarioLoader.calculate_pressure_from_utilization(cpu_usage)
            mem_pressure = ScenarioLoader.calculate_pressure_from_utilization(mem_usage)

            # Add small random variance to pressure (±10% of calculated value)
            cpu_pressure += random.uniform(-0.01, 0.01)
            mem_pressure += random.uniform(-0.01, 0.01)

            # Ensure pressures stay within bounds
            cpu_pressure = max(0.0, min(1.0, cpu_pressure))
            mem_pressure = max(0.0, min(1.0, mem_pressure))

            return Node(name, cpu_usage, cpu_pressure, mem_usage, mem_pressure, vms)

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
                create_realistic_node("node-1", 0.45, 0.50),
                create_realistic_node("node-2", 0.75, 0.35),  # High CPU, moderate memory
                create_realistic_node("node-3", 0.35, 0.65),  # Moderate CPU, higher memory
                create_realistic_node("node-4", 0.25, 0.45),
                create_realistic_node("node-5", 0.85, 0.40),  # High CPU, moderate memory
                create_realistic_node("node-6", 0.55, 0.30),
                create_realistic_node("node-7", 0.65, 0.55),
                create_realistic_node("node-8", 0.40, 0.70),  # Higher memory
                create_realistic_node("node-9", 0.30, 0.35),  # Low-moderate load
                create_realistic_node("node-10", 0.90, 0.50), # Very high CPU, moderate memory
                create_realistic_node("node-11", 0.50, 0.25), # Moderate CPU, lower memory
                create_realistic_node("node-12", 0.70, 0.60), # High CPU, moderate-high memory
            ],
            "heavy_load": [
                create_realistic_node("node-1", 0.85, 0.65),
                create_realistic_node("node-2", 0.90, 0.70),
                create_realistic_node("node-3", 0.80, 0.75),
                create_realistic_node("node-4", 0.88, 0.60),
                create_realistic_node("node-5", 0.75, 0.55),
                create_realistic_node("node-6", 0.92, 0.68),
                create_realistic_node("node-7", 0.87, 0.72),
                create_realistic_node("node-8", 0.83, 0.78),
                create_realistic_node("node-9", 0.89, 0.66),
                create_realistic_node("node-10", 0.78, 0.58),
            ],
            "simple_progression": [
                # Show progression from low to high utilization with realistic pressure
                create_realistic_node("node-1", 0.10, 0.15),  # Very low - minimal pressure
                create_realistic_node("node-2", 0.30, 0.35),  # Low - minimal pressure
                create_realistic_node("node-3", 0.50, 0.45),  # Medium - minimal pressure
                create_realistic_node("node-4", 0.65, 0.55),  # Getting higher - still minimal pressure
                create_realistic_node("node-5", 0.72, 0.60),  # Above 70% - pressure starts
                create_realistic_node("node-6", 0.80, 0.65),  # High - noticeable pressure
                create_realistic_node("node-7", 0.90, 0.70),  # Very high - significant pressure
                create_realistic_node("node-8", 0.95, 0.75),  # Near max - high pressure
            ]
        }

    @staticmethod
    def generate_sample_file(file_path: str = "sample_scenarios.json"):
        """Generate a sample JSON file with scenarios."""
        scenarios = ScenarioLoader.create_sample_scenarios()
        ScenarioLoader.save_scenario(scenarios, file_path)
        return file_path