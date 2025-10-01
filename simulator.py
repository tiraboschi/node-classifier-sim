"""Finite state simulator for node classification with VM redistribution."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from copy import deepcopy
import random

from node import Node
from classifier import NodeClassifier, UtilizationCategory, ClassificationResult
from algorithms import ClassificationAlgorithm


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    vm_cpu_percent: float = 0.02  # CPU consumption per VM (2%)
    vm_memory_percent: float = 0.04  # Memory consumption per VM (4%)
    max_vms_per_step: int = 5  # Maximum VMs to move per simulation step
    max_vms_per_node: int = 2  # Maximum VMs to move from a single node per step


@dataclass
class VMMove:
    """Record of a VM move during descheduling."""
    step: int
    vm_id: str
    from_node: str
    to_node: str
    from_node_score: float
    to_node_score: float


@dataclass
class SimulationStep:
    """State of the simulation at a specific step."""
    step_number: int
    nodes: List[Node]
    classification_results: List[ClassificationResult]
    moves: List[VMMove]
    total_vms_moved: int


class Simulator:
    """
    Finite state simulator for node classification with VM redistribution.

    The simulator:
    1. Classifies nodes using the selected algorithm
    2. Redistributes VMs from overutilized to underutilized nodes (descheduler)
    3. Updates node metrics based on new VM counts
    4. Repeats for each simulation step
    """

    def __init__(
        self,
        initial_nodes: List[Node],
        classifier: NodeClassifier,
        config: Optional[SimulationConfig] = None
    ):
        """
        Initialize the simulator.

        Args:
            initial_nodes: Initial node configuration
            classifier: Node classifier with algorithm and threshold config
            config: Simulation configuration (uses defaults if None)
        """
        self.classifier = classifier
        self.config = config or SimulationConfig()

        # Deep copy nodes to avoid modifying originals
        self.nodes = deepcopy(initial_nodes)

        # Simulation state
        self.current_step = 0
        self.history: List[SimulationStep] = []
        self.move_log: List[VMMove] = []

        # Perform initial classification
        self.current_classification = self.classifier.classify_nodes(self.nodes)

    def step(self) -> SimulationStep:
        """
        Execute one simulation step:
        1. Classify nodes
        2. Run descheduler to move VMs
        3. Update node metrics
        4. Store step in history

        Returns:
            SimulationStep containing the state after this step
        """
        self.current_step += 1

        # Step 1: Classify nodes with current metrics
        self.current_classification = self.classifier.classify_nodes(self.nodes)

        # Step 2: Run descheduler
        step_moves = self._run_descheduler()

        # Step 3: Update node metrics based on actual VMs
        for node in self.nodes:
            node.update_metrics_from_vms()

        # Step 4: Re-classify with updated metrics
        self.current_classification = self.classifier.classify_nodes(self.nodes)

        # Store step in history
        step_state = SimulationStep(
            step_number=self.current_step,
            nodes=deepcopy(self.nodes),
            classification_results=deepcopy(self.current_classification),
            moves=step_moves,
            total_vms_moved=len(step_moves)
        )
        self.history.append(step_state)

        return step_state

    def _run_descheduler(self) -> List[VMMove]:
        """
        Run the descheduler to redistribute VMs.

        Algorithm:
        1. Identify overutilized and underutilized nodes
        2. Starting from most overutilized, move VMs to random underutilized nodes
        3. Respect per-node and per-step limits

        Returns:
            List of VM moves performed during this step
        """
        moves: List[VMMove] = []

        # Separate nodes by category
        overutilized = []
        underutilized = []

        for result in self.current_classification:
            if result.category == UtilizationCategory.OVER_UTILIZED:
                overutilized.append(result)
            elif result.category == UtilizationCategory.UNDER_UTILIZED:
                underutilized.append(result)

        # If no overutilized or no underutilized nodes, nothing to do
        if not overutilized or not underutilized:
            return moves

        # Sort overutilized by score (highest first - most overutilized)
        overutilized.sort(key=lambda r: r.score, reverse=True)

        # Track VMs moved this step
        vms_moved_this_step = 0

        # Process overutilized nodes
        for over_result in overutilized:
            if vms_moved_this_step >= self.config.max_vms_per_step:
                break  # Reached global limit

            # Find the actual node object
            over_node = None
            for node in self.nodes:
                if node.name == over_result.node.name:
                    over_node = node
                    break

            if over_node is None or over_node.vm_count == 0:
                continue

            # Move up to max_vms_per_node from this node
            vms_moved_from_node = 0

            while (vms_moved_from_node < self.config.max_vms_per_node and
                   vms_moved_this_step < self.config.max_vms_per_step and
                   over_node.vm_count > 0 and
                   underutilized):

                # Pick a random underutilized node
                under_result = random.choice(underutilized)

                # Find the actual node object
                under_node = None
                for node in self.nodes:
                    if node.name == under_result.node.name:
                        under_node = node
                        break

                if under_node is None:
                    continue

                # Pick a random VM from the overutilized node
                vm_to_move = random.choice(over_node.vms)

                # Check if destination node can accept this VM (memory constraint)
                # Memory is not compressible, so we must ensure it doesn't exceed 1.0
                current_memory = sum(vm.memory_consumption for vm in under_node.vms)
                would_exceed_memory = (current_memory + vm_to_move.memory_consumption) > 1.0

                if would_exceed_memory:
                    # Cannot move this VM to this node, try next iteration
                    # (will try different underutilized node or different VM)
                    continue

                # Move the VM object from source to destination
                over_node.vms.remove(vm_to_move)
                under_node.vms.append(vm_to_move)

                # Record the move
                move = VMMove(
                    step=self.current_step,
                    vm_id=vm_to_move.id,
                    from_node=over_node.name,
                    to_node=under_node.name,
                    from_node_score=over_result.score,
                    to_node_score=under_result.score
                )
                moves.append(move)
                self.move_log.append(move)

                vms_moved_from_node += 1
                vms_moved_this_step += 1

        return moves

    def reset(self, initial_nodes: List[Node]):
        """
        Reset the simulator with new initial nodes.

        Args:
            initial_nodes: New initial node configuration
        """
        self.nodes = deepcopy(initial_nodes)
        self.current_step = 0
        self.history = []
        self.move_log = []
        self.current_classification = self.classifier.classify_nodes(self.nodes)

    def get_current_state(self) -> Dict:
        """
        Get the current simulation state.

        Returns:
            Dictionary containing current step, nodes, and classification
        """
        return {
            'step': self.current_step,
            'nodes': self.nodes,
            'classification': self.current_classification,
            'total_moves': len(self.move_log)
        }

    def get_step_history(self, step_number: int) -> Optional[SimulationStep]:
        """
        Get the state at a specific step.

        Args:
            step_number: Step number to retrieve (1-indexed)

        Returns:
            SimulationStep if found, None otherwise
        """
        for step in self.history:
            if step.step_number == step_number:
                return step
        return None

    def get_move_summary(self) -> Dict:
        """
        Get a summary of all VM moves.

        Returns:
            Dictionary with move statistics
        """
        if not self.move_log:
            return {
                'total_moves': 0,
                'unique_source_nodes': 0,
                'unique_destination_nodes': 0,
                'moves_by_step': {}
            }

        source_nodes = set(move.from_node for move in self.move_log)
        dest_nodes = set(move.to_node for move in self.move_log)

        moves_by_step = {}
        for move in self.move_log:
            if move.step not in moves_by_step:
                moves_by_step[move.step] = 0
            moves_by_step[move.step] += 1

        return {
            'total_moves': len(self.move_log),
            'unique_source_nodes': len(source_nodes),
            'unique_destination_nodes': len(dest_nodes),
            'moves_by_step': moves_by_step
        }