#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
from copy import deepcopy
from scenario_loader import ScenarioLoader
from algorithms import get_default_algorithms, ClassificationAlgorithm
from node import Node
from classifier import NodeClassifier, ThresholdMode, ThresholdConfig, UtilizationCategory, get_category_symbol
from simulator import Simulator, SimulationConfig

class NodeClassifierCLI:
    """Command-line interface for the node classifier simulator."""

    def __init__(self):
        self.algorithms = get_default_algorithms()

    def run_classification(self, nodes: List[Node], algorithm: ClassificationAlgorithm) -> List[tuple[Node, float]]:
        """Run classification on nodes using the given algorithm."""
        return algorithm.classify_nodes(nodes)

    def print_results(self, scenario_name: str, algorithm_name: str, results: List[tuple[Node, float]]):
        """Print classification results in a formatted way."""
        print(f"\n=== {scenario_name} - {algorithm_name} ===")
        print(f"{'Rank':<4} {'Node':<12} {'Score':<8} {'CPU%':<6} {'CPUP':<6} {'MEM%':<6} {'MEMP':<6}")
        print("-" * 56)

        for rank, (node, score) in enumerate(results, 1):
            print(f"{rank:<4} {node.name:<12} {score:.3f}    "
                  f"{node.cpu_usage:.2f}   {node.cpu_pressure:.2f}   "
                  f"{node.memory_usage:.2f}   {node.memory_pressure:.2f}")

    def run_all_algorithms(self, scenarios_file: str):
        """Run all algorithms on all scenarios from the file."""
        try:
            scenarios = ScenarioLoader.load_scenario(scenarios_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading scenarios: {e}")
            return

        for scenario_name, nodes in scenarios.items():
            print(f"\n{'='*60}")
            print(f"SCENARIO: {scenario_name}")
            print(f"{'='*60}")

            for algorithm in self.algorithms:
                results = self.run_classification(nodes, algorithm)
                self.print_results(scenario_name, algorithm.name, results)

    def generate_sample_file(self, output_file: str = "sample_scenarios.json"):
        """Generate a sample scenarios file."""
        file_path = ScenarioLoader.generate_sample_file(output_file)
        print(f"Sample scenarios file generated: {file_path}")
        return file_path

    def list_algorithms(self):
        """List all available algorithms."""
        print("Available algorithms:")
        for i, algorithm in enumerate(self.algorithms, 1):
            print(f"{i}. {algorithm.name}")
            if algorithm.get_params():
                print(f"   Parameters: {algorithm.get_params()}")

    def get_threshold_mode(self, mode_str: str) -> ThresholdMode:
        """Convert string to ThresholdMode enum."""
        mode_map = {
            "low": ThresholdMode.LOW,
            "medium": ThresholdMode.MEDIUM,
            "high": ThresholdMode.HIGH,
            "asym-low": ThresholdMode.ASYMMETRIC_LOW,
            "asym-medium": ThresholdMode.ASYMMETRIC_MEDIUM,
            "asym-high": ThresholdMode.ASYMMETRIC_HIGH,
        }
        return mode_map[mode_str]

    def print_classification_results(self, scenario_name: str, algorithm_name: str,
                                   results: list, threshold_info: dict):
        """Print three-bucket classification results."""
        print(f"\n=== {scenario_name} - {algorithm_name} ===")
        print(f"Cluster Average: {threshold_info['cluster_average']:.3f}")
        print(f"Under-utilized threshold: {threshold_info['under_threshold']:.3f} "
              f"(-{threshold_info['lower_percentage']:.0f}%)")
        print(f"Over-utilized threshold: {threshold_info['over_threshold']:.3f} "
              f"(+{threshold_info['upper_percentage']:.0f}%)")
        print()

        # Group by category
        categories = {category: [] for category in UtilizationCategory}
        for result in results:
            categories[result.category].append(result)

        # Print summary
        total_nodes = len(results)
        for category in UtilizationCategory:
            count = len(categories[category])
            percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
            symbol = get_category_symbol(category)
            print(f"{symbol} {category.value}: {count} nodes ({percentage:.1f}%)")

        print()

        # Print detailed results
        print(f"{'Cat':<3} {'Node':<12} {'Score':<8} {'CPU%':<6} {'CPUP':<6} {'MEM%':<6} {'MEMP':<6}")
        print("-" * 58)

        for result in results:
            symbol = get_category_symbol(result.category)
            print(f"{symbol:<3} {result.node.name:<12} {result.score:.3f}    "
                  f"{result.node.cpu_usage:.2f}   {result.node.cpu_pressure:.2f}   "
                  f"{result.node.memory_usage:.2f}   {result.node.memory_pressure:.2f}")

    def run_classification_analysis(self, scenarios_file: str, threshold_mode: ThresholdMode):
        """Run three-bucket classification analysis on scenarios."""
        try:
            scenarios = ScenarioLoader.load_scenario(scenarios_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading scenarios: {e}")
            return

        threshold_config = ThresholdConfig.from_mode(threshold_mode)
        print(f"Using threshold mode: {threshold_mode.value}")
        print(f"Lower threshold: -{threshold_config.lower_percentage*100:.0f}% of cluster average")
        print(f"Upper threshold: +{threshold_config.upper_percentage*100:.0f}% of cluster average")

        for scenario_name, nodes in scenarios.items():
            print(f"\n{'='*70}")
            print(f"SCENARIO: {scenario_name}")
            print(f"{'='*70}")

            for algorithm in self.algorithms:
                classifier = NodeClassifier(algorithm, threshold_config)
                results = classifier.classify_nodes(nodes)
                threshold_info = classifier.get_threshold_info(results)
                self.print_classification_results(scenario_name, algorithm.name,
                                                results, threshold_info)

    def _has_overutilized_nodes(self, classification_results: List) -> bool:
        """Check if there are any overutilized nodes in the classification results."""
        for result in classification_results:
            if result.category == UtilizationCategory.OVER_UTILIZED:
                return True
        return False

    def _calculate_std_dev(self, nodes: List[Node]) -> Tuple[float, float]:
        """Calculate standard deviation of CPU and memory utilization across nodes."""
        cpu_values = [node.cpu_usage for node in nodes]
        memory_values = [node.memory_usage for node in nodes]

        cpu_std = statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0.0
        memory_std = statistics.stdev(memory_values) if len(memory_values) > 1 else 0.0

        return cpu_std, memory_std

    def benchmark_algorithms(self, scenarios_file: str, threshold_mode: ThresholdMode,
                           max_iterations: int = 100):
        """
        Benchmark all algorithms by running simulations and measuring:
        - Steps to convergence (no overutilized nodes)
        - Standard deviation of CPU and memory utilization

        Uses a ranking-based scoring system where lower rank = better performance.

        Args:
            scenarios_file: Path to scenarios JSON file
            threshold_mode: Threshold mode to use for classification
            max_iterations: Maximum number of simulation steps (default: 100)
        """
        try:
            scenarios = ScenarioLoader.load_scenario(scenarios_file)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading scenarios: {e}")
            return

        threshold_config = ThresholdConfig.from_mode(threshold_mode)

        print(f"\n{'='*100}")
        print(f"BENCHMARK: Convergence Analysis with Ranking-Based Scoring")
        print(f"{'='*100}")
        print(f"Threshold mode: {threshold_mode.value}")
        print(f"Max iterations: {max_iterations}")
        print(f"Scoring: Lower rank = better (1st place = 1 point, 2nd place = 2 points, etc.)")
        print(f"{'='*100}\n")

        for scenario_name, initial_nodes in scenarios.items():
            print(f"\n{'='*100}")
            print(f"SCENARIO: {scenario_name}")
            print(f"{'='*100}")

            results = []

            # Run all algorithms
            for algorithm in self.algorithms:
                # Create classifier and simulator with fresh copy of nodes
                nodes_copy = deepcopy(initial_nodes)
                classifier = NodeClassifier(algorithm, threshold_config)
                sim_config = SimulationConfig()
                simulator = Simulator(nodes_copy, classifier, sim_config)

                # Run simulation until convergence or max iterations
                converged = False
                steps = 0

                for step in range(max_iterations):
                    # Check current state before stepping
                    current_classification = simulator.current_classification

                    if not self._has_overutilized_nodes(current_classification):
                        converged = True
                        steps = step
                        break

                    # Execute one simulation step
                    simulator.step()
                    steps = step + 1

                # Final check after last step
                if not converged:
                    final_classification = simulator.current_classification
                    if not self._has_overutilized_nodes(final_classification):
                        converged = True

                # Calculate standard deviations of final state
                final_nodes = simulator.nodes
                cpu_std, memory_std = self._calculate_std_dev(final_nodes)

                # Store results
                # Disqualify if algorithm took 0 steps (scenario already balanced, algorithm didn't do work)
                disqualified = (steps == 0 and converged)

                results.append({
                    'algorithm': algorithm.name,
                    'steps': steps if converged else max_iterations + 1,  # +1 penalty for non-convergence
                    'converged': converged,
                    'disqualified': disqualified,
                    'cpu_std': cpu_std,
                    'memory_std': memory_std
                })

            # Rank algorithms on each metric (lower is better)
            # Separate valid converged, disqualified, and non-converged algorithms
            valid_results = [r for r in results if r['converged'] and not r['disqualified']]
            disqualified_results = [r for r in results if r['disqualified']]
            non_converged_results = [r for r in results if not r['converged']]

            # Steps ranking (fewer steps = better = lower rank)
            # Valid algorithms get normal ranks
            sorted_by_steps = sorted(valid_results, key=lambda r: r['steps'])
            for rank, result in enumerate(sorted_by_steps, 1):
                result['steps_rank'] = rank

            # Disqualified and non-converged algorithms get penalized ranks (worse than all valid)
            base_penalty_rank = len(valid_results) + 1

            # Disqualified algorithms (0 steps) get first penalty tier
            for idx, result in enumerate(disqualified_results):
                result['steps_rank'] = base_penalty_rank + idx * 10

            # Non-converged algorithms get worst penalty tier
            disqualified_penalty = base_penalty_rank + len(disqualified_results) * 10
            for idx, result in enumerate(sorted(non_converged_results, key=lambda r: r['steps'])):
                result['steps_rank'] = disqualified_penalty + idx * 10  # Large penalty gap

            # CPU std deviation ranking (lower std = better = lower rank)
            # Only rank valid algorithms normally
            sorted_by_cpu = sorted(valid_results, key=lambda r: r['cpu_std'])
            for rank, result in enumerate(sorted_by_cpu, 1):
                result['cpu_rank'] = rank

            # Disqualified algorithms get penalized CPU ranks
            for idx, result in enumerate(sorted(disqualified_results, key=lambda r: r['cpu_std'])):
                result['cpu_rank'] = base_penalty_rank + idx * 10

            # Non-converged algorithms get worst penalized CPU ranks
            for idx, result in enumerate(sorted(non_converged_results, key=lambda r: r['cpu_std'])):
                result['cpu_rank'] = disqualified_penalty + idx * 10

            # Memory std deviation ranking (lower std = better = lower rank)
            # Only rank valid algorithms normally
            sorted_by_mem = sorted(valid_results, key=lambda r: r['memory_std'])
            for rank, result in enumerate(sorted_by_mem, 1):
                result['mem_rank'] = rank

            # Disqualified algorithms get penalized memory ranks
            for idx, result in enumerate(sorted(disqualified_results, key=lambda r: r['memory_std'])):
                result['mem_rank'] = base_penalty_rank + idx * 10

            # Non-converged algorithms get worst penalized memory ranks
            for idx, result in enumerate(sorted(non_converged_results, key=lambda r: r['memory_std'])):
                result['mem_rank'] = disqualified_penalty + idx * 10

            # Calculate total score (sum of ranks - lower is better)
            for result in results:
                result['total_score'] = result['steps_rank'] + result['cpu_rank'] + result['mem_rank']

            # Sort by total score
            results.sort(key=lambda r: r['total_score'])

            # Print detailed results
            print(f"\n{'Rank':<5} {'Algorithm':<35} {'Steps':<8} {'Step':<6} {'CPU σ':<12} {'CPU':<5} {'MEM σ':<12} {'MEM':<5} {'Total':<6}")
            print(f"{'':5} {'':35} {'':8} {'Rank':<6} {'':12} {'Rank':<5} {'':12} {'Rank':<5} {'Score':<6}")
            print("-" * 100)

            for overall_rank, result in enumerate(results, 1):
                if result['disqualified']:
                    status = "⊘"  # Disqualified
                    steps_str = "0 (DQ)"
                elif result['converged']:
                    status = "✓"
                    steps_str = str(result['steps'])
                else:
                    status = "✗"
                    steps_str = f">{max_iterations}"

                print(f"{overall_rank:<5} {result['algorithm']:<35} {steps_str:<8} "
                      f"{result['steps_rank']:<6} {result['cpu_std']:.8f}  {result['cpu_rank']:<5} "
                      f"{result['memory_std']:.8f}  {result['mem_rank']:<5} {result['total_score']:<6}")

            # Print winners (only from valid results)
            print("\n" + "-" * 100)
            if valid_results:
                print("WINNERS:")
                winner = min(valid_results, key=lambda r: r['total_score'])
                print(f"🏆 Overall Best: {winner['algorithm']} (Total Score: {winner['total_score']})")

                best_steps = min(valid_results, key=lambda r: r['steps_rank'])
                print(f"⚡ Fastest Convergence: {best_steps['algorithm']} ({best_steps['steps']} steps)")

                best_cpu = min(valid_results, key=lambda r: r['cpu_rank'])
                print(f"📊 Best CPU Balance: {best_cpu['algorithm']} (σ={best_cpu['cpu_std']:.8f})")

                best_mem = min(valid_results, key=lambda r: r['mem_rank'])
                print(f"💾 Best Memory Balance: {best_mem['algorithm']} (σ={best_mem['memory_std']:.8f})")
            else:
                print("No valid algorithms (all were disqualified or failed to converge)")

            # Convergence stats
            valid_count = len(valid_results)
            disqualified_count = len(disqualified_results)
            failed_count = len(non_converged_results)
            print(f"\n✓ Valid: {valid_count}/{len(results)} algorithms")
            if disqualified_count > 0:
                print(f"⊘ Disqualified (0 steps): {disqualified_count}/{len(results)} algorithms")
            if failed_count > 0:
                print(f"✗ Failed to converge: {failed_count}/{len(results)} algorithms")

            print("=" * 100)

def main():
    parser = argparse.ArgumentParser(
        description="Kubernetes Node Classification Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --generate-sample
  python cli.py --file sample_scenarios.json
  python cli.py --list-algorithms
        """
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="JSON file containing node scenarios"
    )

    parser.add_argument(
        "--generate-sample", "-g",
        action="store_true",
        help="Generate a sample scenarios file"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="sample_scenarios.json",
        help="Output file for sample generation (default: sample_scenarios.json)"
    )

    parser.add_argument(
        "--list-algorithms", "-l",
        action="store_true",
        help="List available classification algorithms"
    )

    parser.add_argument(
        "--classify", "-c",
        action="store_true",
        help="Run three-bucket classification (under/appropriate/over-utilized)"
    )

    parser.add_argument(
        "--threshold-mode", "-t",
        type=str,
        choices=["low", "medium", "high", "asym-low", "asym-medium", "asym-high"],
        default="asym-low",
        help="Threshold mode for classification (default: asym-low)"
    )

    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run convergence benchmark on all algorithms with ranking-based scoring"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum simulation steps for benchmark (default: 100)"
    )

    args = parser.parse_args()

    cli = NodeClassifierCLI()

    if args.generate_sample:
        cli.generate_sample_file(args.output)
        return

    if args.list_algorithms:
        cli.list_algorithms()
        return

    if args.file:
        if not Path(args.file).exists():
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)

        threshold_mode = cli.get_threshold_mode(args.threshold_mode)

        if args.benchmark:
            # Run benchmark analysis
            cli.benchmark_algorithms(args.file, threshold_mode, args.max_iterations)
        elif args.classify:
            # Run three-bucket classification
            cli.run_classification_analysis(args.file, threshold_mode)
        else:
            # Run standard ranking classification
            cli.run_all_algorithms(args.file)
        return

    parser.print_help()

if __name__ == "__main__":
    main()