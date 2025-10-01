from dataclasses import dataclass, field
from typing import Dict, Any, List
import json


@dataclass
class VM:
    """Represents a Virtual Machine with its resource consumption."""
    id: str  # Unique identifier (e.g., "vm-1", "vm-2")
    cpu_consumption: float  # CPU consumption (0.0 to vm_cpu_percent)
    memory_consumption: float  # Memory consumption (0.0 to vm_memory_percent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert VM to dictionary representation."""
        return {
            "id": self.id,
            "cpu_consumption": self.cpu_consumption,
            "memory_consumption": self.memory_consumption
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VM':
        """Create VM from dictionary representation."""
        return cls(
            id=data["id"],
            cpu_consumption=data["cpu_consumption"],
            memory_consumption=data["memory_consumption"]
        )


@dataclass
class Node:
    """Represents a Kubernetes node with resource usage metrics."""

    name: str
    cpu_usage: float  # 0.0+, can exceed 1.0 (compressible resource)
    cpu_pressure: float  # 0.0 to 1.0 (PSI CPU pressure)
    memory_usage: float  # 0.0 to 1.0 (non-compressible, hard limit)
    memory_pressure: float  # 0.0 to 1.0 (PSI memory pressure)
    vms: List[VM] = field(default_factory=list)  # List of VMs running on this node

    @property
    def vm_count(self) -> int:
        """Get the number of VMs running on this node."""
        return len(self.vms)

    def __post_init__(self):
        """Validate that initial metrics are within valid ranges.
        Note: CPU can exceed 1.0 during simulation (compressible), but starts <= 1.0."""
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
            "memory_pressure": self.memory_pressure,
            "vms": [vm.to_dict() for vm in self.vms]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create node from dictionary representation."""
        # Handle backward compatibility: if vms not present, create empty list
        vms_data = data.get("vms", [])
        vms = [VM.from_dict(vm_data) for vm_data in vms_data]

        return cls(
            name=data["name"],
            cpu_usage=data["cpu_usage"],
            cpu_pressure=data["cpu_pressure"],
            memory_usage=data["memory_usage"],
            memory_pressure=data["memory_pressure"],
            vms=vms
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

    def update_metrics_from_vms(self):
        """
        Update CPU and memory utilization based on actual VMs running on this node.
        Also updates pressure based on utilization using the standard formula.

        Note:
        - CPU is compressible: VMs can demand >100%, but node caps at 100% (VMs slow down)
        - CPU pressure reflects total demand (can exceed 100% internally for pressure calc)
        - Memory is NOT compressible: capped at 100% (hard limit enforced by descheduler)
        - Pressure increases dramatically as utilization approaches/exceeds limits
        """
        # Calculate total CPU demand (can exceed 1.0)
        cpu_demand = sum(vm.cpu_consumption for vm in self.vms)

        # CPU usage displayed is capped at 1.0 (100%)
        self.cpu_usage = min(1.0, cpu_demand)

        # Memory is hard capped at 1.0
        self.memory_usage = min(1.0, sum(vm.memory_consumption for vm in self.vms))

        # Update pressure based on actual demand/utilization
        # CPU pressure uses the actual demand (can exceed 1.0) to show overload
        # This reflects VMs slowing down when demand > capacity
        from scenario_loader import ScenarioLoader

        # For CPU: if demand > 1.0, pressure should be very high
        # Scale pressure: at 1.0 demand = normal pressure, >1.0 = increased pressure
        if cpu_demand <= 1.0:
            self.cpu_pressure = ScenarioLoader.calculate_pressure_from_utilization(cpu_demand)
        else:
            # When overloaded (demand > 100%), pressure increases sharply
            # Base pressure at 100% + additional pressure for overload
            base_pressure = ScenarioLoader.calculate_pressure_from_utilization(1.0)
            # Scale additional pressure: 110% demand = more pressure than 100%
            overload_factor = cpu_demand - 1.0  # How much over 100%
            # Additional pressure grows with overload (capped at 1.0 total)
            self.cpu_pressure = min(1.0, base_pressure + overload_factor * 0.5)

        self.memory_pressure = ScenarioLoader.calculate_pressure_from_utilization(self.memory_usage)

    def sync_vms_to_utilization(self, target_cpu: float, target_mem: float,
                                vm_cpu_max: float = 0.06, vm_mem_max: float = 0.04,
                                vm_id_counter: int = 0):
        """
        Synchronize VMs to match target CPU and memory utilization.

        Adds or removes VMs to reach the target utilization levels.

        Args:
            target_cpu: Target CPU utilization (0.0+, can exceed 1.0)
            target_mem: Target memory utilization (0.0 to 1.0)
            vm_cpu_max: Maximum CPU consumption per VM (default 2%)
            vm_mem_max: Maximum memory consumption per VM (default 4%)
            vm_id_counter: Counter for generating unique VM IDs

        Returns:
            Updated vm_id_counter
        """
        import random

        # For simplicity and reliability, always clear and regenerate VMs from scratch
        # This ensures we always hit the exact target values
        self.vms.clear()
        current_cpu = 0.0
        current_mem = 0.0

        # Special case: if both targets are zero, we're done
        if target_cpu == 0 and target_mem == 0:
            self.update_metrics_from_vms()
            return vm_id_counter

        # Calculate the target ratio for new VMs
        target_ratio = target_cpu / target_mem if target_mem > 0 else float('inf')

        # Add VMs until we reach the targets
        safety_counter = 0
        while (current_cpu < target_cpu or current_mem < target_mem) and safety_counter < 100:
            # Calculate remaining needed
            cpu_remaining = max(0, target_cpu - current_cpu)
            mem_remaining = max(0, target_mem - current_mem)

            # Stop if both are very close (within 1% of max VM consumption)
            if cpu_remaining < vm_cpu_max * 0.01 and mem_remaining < vm_mem_max * 0.01:
                break

            # Create VM maintaining the target ratio
            if target_mem == 0:
                # Only CPU needed
                vm_cpu = random.uniform(0, min(vm_cpu_max, cpu_remaining))
                vm_memory = 0
            elif target_cpu == 0:
                # Only memory needed
                vm_cpu = 0
                vm_memory = random.uniform(0, min(vm_mem_max, mem_remaining))
            else:
                # Maintain target ratio
                vm_cpu = random.uniform(0, min(vm_cpu_max, cpu_remaining))
                vm_memory = vm_cpu / target_ratio

                # If memory exceeds limits, scale down
                if vm_memory > vm_mem_max or vm_memory > mem_remaining:
                    vm_memory = min(vm_mem_max, mem_remaining)
                    vm_cpu = vm_memory * target_ratio

                    # If CPU now exceeds limits, scale down again
                    if vm_cpu > vm_cpu_max or vm_cpu > cpu_remaining:
                        vm_cpu = min(vm_cpu_max, cpu_remaining)
                        vm_memory = vm_cpu / target_ratio

            # Ensure we're adding something
            if vm_cpu <= 0 and vm_memory <= 0:
                break

            # Create the VM
            vm_id_counter += 1
            vm = VM(
                id=f"vm-{vm_id_counter}",
                cpu_consumption=max(0, vm_cpu),
                memory_consumption=max(0, vm_memory)
            )
            self.vms.append(vm)

            current_cpu += vm.cpu_consumption
            current_mem += vm.memory_consumption
            safety_counter += 1

        # Update metrics to reflect new VMs
        self.update_metrics_from_vms()

        return vm_id_counter