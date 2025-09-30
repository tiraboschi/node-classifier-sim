#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List
from scenario_loader import ScenarioLoader
from algorithms import get_default_algorithms, ClassificationAlgorithm
from node import Node
from classifier import NodeClassifier, ThresholdMode, ThresholdConfig, UtilizationCategory, get_category_symbol

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

        if args.classify:
            # Run three-bucket classification
            threshold_mode = cli.get_threshold_mode(args.threshold_mode)
            cli.run_classification_analysis(args.file, threshold_mode)
        else:
            # Run standard ranking classification
            cli.run_all_algorithms(args.file)
        return

    parser.print_help()

if __name__ == "__main__":
    main()