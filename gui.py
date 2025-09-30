#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict, Optional
from node import Node
from algorithms import get_default_algorithms, ClassificationAlgorithm
from scenario_loader import ScenarioLoader
from classifier import NodeClassifier, ThresholdMode, ThresholdConfig, UtilizationCategory, get_category_color
import matplotlib.colors as mcolors

class NodeClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kubernetes Node Classification Simulator")
        self.root.geometry("1400x800")

        self.algorithms = get_default_algorithms()
        self.current_scenario: List[Node] = []
        self.scenario_name = "Current Scenario"

        self.setup_ui()
        self.load_sample_data()

    def setup_ui(self):
        """Setup the user interface."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        # File operations
        file_frame = ttk.LabelFrame(left_panel, text="File Operations")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Load Scenarios", command=self.load_file).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(file_frame, text="Save Scenarios", command=self.save_file).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(file_frame, text="Load Sample", command=self.load_sample_data).pack(side=tk.LEFT, padx=5, pady=5)

        # Scenario selection
        scenario_frame = ttk.LabelFrame(left_panel, text="Scenario")
        scenario_frame.pack(fill=tk.X, pady=(0, 10))

        self.scenario_var = tk.StringVar()
        self.scenario_combo = ttk.Combobox(scenario_frame, textvariable=self.scenario_var, state="readonly")
        self.scenario_combo.pack(fill=tk.X, padx=5, pady=5)
        self.scenario_combo.bind('<<ComboboxSelected>>', self.on_scenario_change)

        # Node list and editing
        nodes_frame = ttk.LabelFrame(left_panel, text="Nodes")
        nodes_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Node list
        self.nodes_tree = ttk.Treeview(nodes_frame, columns=('CPU%', 'CPUP', 'MEM%', 'MEMP'), show='tree headings', height=6)
        self.nodes_tree.heading('#0', text='Node Name')
        self.nodes_tree.heading('CPU%', text='CPU%')
        self.nodes_tree.heading('CPUP', text='CPU P')
        self.nodes_tree.heading('MEM%', text='MEM%')
        self.nodes_tree.heading('MEMP', text='MEM P')

        self.nodes_tree.column('#0', width=80)
        self.nodes_tree.column('CPU%', width=50)
        self.nodes_tree.column('CPUP', width=50)
        self.nodes_tree.column('MEM%', width=50)
        self.nodes_tree.column('MEMP', width=50)

        self.nodes_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.nodes_tree.bind('<<TreeviewSelect>>', self.on_node_select)

        # Node editing controls
        edit_frame = ttk.Frame(nodes_frame)
        edit_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(edit_frame, text="Add Node", command=self.add_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_frame, text="Remove Node", command=self.remove_node).pack(side=tk.LEFT, padx=2)

        # Node parameter editing
        params_frame = ttk.LabelFrame(left_panel, text="Edit Selected Node")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        self.param_vars = {}
        self.param_scales = {}

        for i, (param, label) in enumerate([
            ('cpu_usage', 'CPU Usage'),
            ('cpu_pressure', 'CPU Pressure'),
            ('memory_usage', 'Memory Usage'),
            ('memory_pressure', 'Memory Pressure')
        ]):
            frame = ttk.Frame(params_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)

            ttk.Label(frame, text=f"{label}:", width=12).pack(side=tk.LEFT)

            var = tk.DoubleVar()
            self.param_vars[param] = var

            scale = tk.Scale(frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                           variable=var, command=lambda val, p=param: self.update_node_param(p, val))
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.param_scales[param] = scale

        # Algorithm selection
        algo_frame = ttk.LabelFrame(left_panel, text="Classification Algorithm")
        algo_frame.pack(fill=tk.X, pady=(0, 10))

        self.algo_var = tk.StringVar()
        self.algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var, state="readonly")
        self.algo_combo['values'] = [algo.name for algo in self.algorithms]
        # Set default to Euclidean Distance (index 2)
        euclidean_index = 2  # EuclideanDistanceAlgorithm is third in the list
        self.algo_combo.current(euclidean_index)
        self.algo_combo.pack(fill=tk.X, padx=5, pady=5)
        self.algo_combo.bind('<<ComboboxSelected>>', self.update_visualization)

        # Three-bucket classification controls
        bucket_frame = ttk.LabelFrame(left_panel, text="Three-Bucket Classification")
        bucket_frame.pack(fill=tk.X)

        # Enable three-bucket classification checkbox
        self.use_buckets_var = tk.BooleanVar(value=True)
        bucket_check = ttk.Checkbutton(bucket_frame, text="Enable three-bucket classification",
                                     variable=self.use_buckets_var, command=self.update_visualization)
        bucket_check.pack(anchor=tk.W, padx=5, pady=2)

        # Threshold mode selection
        threshold_frame = ttk.Frame(bucket_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(threshold_frame, text="Threshold Mode:").pack(side=tk.LEFT)
        self.threshold_var = tk.StringVar(value="AsymmetricLow (0%:10%)")
        threshold_combo = ttk.Combobox(threshold_frame, textvariable=self.threshold_var,
                                     state="readonly", width=20)
        threshold_combo['values'] = [mode.value for mode in ThresholdMode]
        threshold_combo.pack(side=tk.LEFT, padx=5)
        threshold_combo.bind('<<ComboboxSelected>>', self.update_visualization)

        # Right panel for visualization
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib figure with two subplots
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar for zoom/pan functionality
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_panel)
        self.toolbar.update()

        # Connect mouse events for drag functionality
        self.canvas.mpl_connect('button_press_event', self.on_plot_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_plot_motion)
        self.canvas.mpl_connect('button_release_event', self.on_plot_release)

        # Results panel
        results_frame = ttk.LabelFrame(right_panel, text="Classification Results")
        results_frame.pack(fill=tk.X, pady=(10, 0))

        self.results_tree = ttk.Treeview(results_frame, columns=('Node', 'Bucket', 'Score', 'CPU%', 'CPUP', 'MEM%', 'MEMP'), show='tree headings', height=8)
        self.results_tree.heading('#0', text='Rank')
        self.results_tree.heading('Node', text='Node Name')
        self.results_tree.heading('Bucket', text='Bucket')
        self.results_tree.heading('Score', text='Score')
        self.results_tree.heading('CPU%', text='CPU%')
        self.results_tree.heading('CPUP', text='CPU P')
        self.results_tree.heading('MEM%', text='MEM%')
        self.results_tree.heading('MEMP', text='MEM P')

        # Set column widths - make node name and bucket wider
        self.results_tree.column('#0', width=50)  # Rank
        self.results_tree.column('Node', width=80)  # Node Name
        self.results_tree.column('Bucket', width=70)  # Bucket
        self.results_tree.column('Score', width=60)  # Score
        self.results_tree.column('CPU%', width=50)  # CPU%
        self.results_tree.column('CPUP', width=50)  # CPU P
        self.results_tree.column('MEM%', width=50)  # MEM%
        self.results_tree.column('MEMP', width=50)  # MEM P

        self.results_tree.pack(fill=tk.X, padx=5, pady=5)
        self.results_tree.bind('<<TreeviewSelect>>', self.on_results_select)

        self.current_scenarios = {}
        self.selected_node_id = None
        self.plot_data = {}  # Store plot data for click detection
        self.classified_nodes = []  # Store current classification results
        self.dragging_node = None  # Track which node is being dragged
        self.drag_start_pos = None  # Starting position of drag
        self.dragging_plot = None  # Track which plot (0 or 1) is being dragged
        # References for two plots
        self.scatter_plots = [None, None]  # References to scatter plot objects
        self.rank_annotations = [[], []]  # References to rank annotations for each plot
        self.name_annotations = [[], []]  # References to name annotations for each plot
        # Centralized selection state management
        self._selection_manager_active = False
        self._current_selected_node_name = None

    def load_sample_data(self):
        """Load sample scenarios."""
        self.current_scenarios = ScenarioLoader.create_sample_scenarios()
        self.update_scenario_combo()
        # Set default to heavy_load scenario
        scenario_names = list(self.current_scenarios.keys())
        if 'heavy_load' in scenario_names:
            heavy_load_index = scenario_names.index('heavy_load')
            self.scenario_combo.current(heavy_load_index)
        else:
            self.scenario_combo.current(0)
        self.on_scenario_change()

    def update_scenario_combo(self):
        """Update the scenario combobox with available scenarios."""
        scenario_names = list(self.current_scenarios.keys())
        self.scenario_combo['values'] = scenario_names

    def on_scenario_change(self, event=None):
        """Handle scenario selection change."""
        scenario_name = self.scenario_var.get()
        if scenario_name and scenario_name in self.current_scenarios:
            self.current_scenario = self.current_scenarios[scenario_name]
            self.scenario_name = scenario_name
            self.update_nodes_tree()
            self.update_visualization()

    def update_nodes_tree(self):
        """Update the nodes tree view."""
        for item in self.nodes_tree.get_children():
            self.nodes_tree.delete(item)

        for i, node in enumerate(self.current_scenario):
            self.nodes_tree.insert('', tk.END, iid=str(i), text=node.name,
                                 values=(f"{node.cpu_usage:.2f}",
                                        f"{node.cpu_pressure:.2f}",
                                        f"{node.memory_usage:.2f}",
                                        f"{node.memory_pressure:.2f}"))

    def on_node_select(self, event=None):
        """Handle node selection in the tree."""
        if self._selection_manager_active:
            return

        selected = self.nodes_tree.selection()
        if selected:
            self.selected_node_id = int(selected[0])
            node = self.current_scenario[self.selected_node_id]

            # Update parameter controls
            self.param_vars['cpu_usage'].set(node.cpu_usage)
            self.param_vars['cpu_pressure'].set(node.cpu_pressure)
            self.param_vars['memory_usage'].set(node.memory_usage)
            self.param_vars['memory_pressure'].set(node.memory_pressure)

            # Use selection manager to coordinate updates
            self._set_selected_node(node.name)

    def _set_selected_node(self, node_name):
        """Centralized node selection manager to prevent circular updates."""
        if self._selection_manager_active or self._current_selected_node_name == node_name:
            return

        self._selection_manager_active = True
        self._current_selected_node_name = node_name

        try:
            # Find the node's rank in the classified results
            if not self.classified_nodes:
                return

            for rank_index, (classified_node, score) in enumerate(self.classified_nodes):
                if classified_node.name == node_name:
                    # Update plots highlighting
                    self._update_plot_highlighting(rank_index)

                    # Update results table selection
                    self._update_results_selection(rank_index)

                    # Update nodes panel selection
                    self._update_nodes_panel_selection(node_name)
                    break
        finally:
            self._selection_manager_active = False

    def _update_plot_highlighting(self, rank_index):
        """Update plot highlighting without triggering events."""
        # Reset all node names to normal weight first
        for plot_idx in range(2):
            if plot_idx < len(self.name_annotations):
                for i, annotation in enumerate(self.name_annotations[plot_idx]):
                    if i == rank_index:
                        annotation.set_fontweight('bold')
                        annotation.set_fontsize(8)
                    else:
                        annotation.set_fontweight('normal')
                        annotation.set_fontsize(7)
        self.canvas.draw_idle()

    def _update_results_selection(self, rank_index):
        """Update results table selection without triggering events."""
        # Clear previous selections
        for item in self.results_tree.selection():
            self.results_tree.selection_remove(item)

        # Select the corresponding row
        items = self.results_tree.get_children()
        if 0 <= rank_index < len(items):
            item_id = items[rank_index]
            self.results_tree.selection_set(item_id)
            self.results_tree.focus(item_id)
            self.results_tree.see(item_id)

    def _update_nodes_panel_selection(self, node_name):
        """Update nodes panel selection without triggering events."""
        # Find the node in the current scenario
        for i, node in enumerate(self.current_scenario):
            if node.name == node_name:
                # Clear previous selection
                for item in self.nodes_tree.selection():
                    self.nodes_tree.selection_remove(item)

                # Select the node
                self.nodes_tree.selection_set(str(i))
                self.nodes_tree.focus(str(i))
                self.nodes_tree.see(str(i))

                # Update selected node ID and parameter controls
                self.selected_node_id = i
                self.param_vars['cpu_usage'].set(node.cpu_usage)
                self.param_vars['cpu_pressure'].set(node.cpu_pressure)
                self.param_vars['memory_usage'].set(node.memory_usage)
                self.param_vars['memory_pressure'].set(node.memory_pressure)
                break

    def update_node_param(self, param_name: str, value: str):
        """Update a node parameter."""
        if self.selected_node_id is not None:
            try:
                float_value = float(value)
                node = self.current_scenario[self.selected_node_id]

                if param_name == 'cpu_usage':
                    node.cpu_usage = float_value
                elif param_name == 'cpu_pressure':
                    node.cpu_pressure = float_value
                elif param_name == 'memory_usage':
                    node.memory_usage = float_value
                elif param_name == 'memory_pressure':
                    node.memory_pressure = float_value

                self.update_nodes_tree()
                self.update_visualization()

            except ValueError:
                pass

    def add_node(self):
        """Add a new node to the current scenario."""
        node_count = len(self.current_scenario)
        new_node = Node(f"node-{node_count + 1}", 0.1, 0.1, 0.1, 0.1)
        self.current_scenario.append(new_node)
        self.update_nodes_tree()
        self.update_visualization()

    def remove_node(self):
        """Remove the selected node."""
        if self.selected_node_id is not None and self.current_scenario:
            del self.current_scenario[self.selected_node_id]
            self.selected_node_id = None
            self.update_nodes_tree()
            self.update_visualization()

    def get_current_algorithm(self) -> ClassificationAlgorithm:
        """Get the currently selected algorithm."""
        algo_name = self.algo_var.get()
        for algo in self.algorithms:
            if algo.name == algo_name:
                return algo
        return self.algorithms[0]

    def get_current_threshold_mode(self) -> ThresholdMode:
        """Get the currently selected threshold mode."""
        threshold_str = self.threshold_var.get()
        for mode in ThresholdMode:
            if mode.value == threshold_str:
                return mode
        return ThresholdMode.ASYMMETRIC_LOW  # Default

    def create_three_bucket_colormap(self):
        """Create a custom colormap for three-bucket classification."""
        # Define colors for each section
        # Under-utilized: blue to light blue
        # Appropriately-utilized: light blue to green
        # Over-utilized: orange to red

        colors = [
            '#0066CC',  # Dark blue (under-utilized start)
            '#66B3FF',  # Light blue (under-utilized end / appropriately-utilized start)
            '#00CC66',  # Green (appropriately-utilized end)
            '#FF9933',  # Orange (over-utilized start)
            '#FF3333'   # Red (over-utilized end)
        ]

        # Create positions for the color segments
        # We'll have 5 segments: under (0-0.33), transition (0.33-0.34), appropriate (0.34-0.66), transition (0.66-0.67), over (0.67-1.0)
        positions = [0.0, 0.33, 0.34, 0.66, 1.0]

        # Create the colormap
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'three_bucket', list(zip(positions, colors))
        )
        return cmap

    def update_visualization(self, event=None):
        """Update the dual 2D visualization and results."""
        if not self.current_scenario:
            return

        self.fig.clear()

        # Create two subplots side by side
        ax1 = self.fig.add_subplot(121)  # Left plot: CPU vs Memory usage
        ax2 = self.fig.add_subplot(122)  # Right plot: CPU vs Memory pressure

        # Get algorithm and classify nodes
        algorithm = self.get_current_algorithm()
        use_buckets = self.use_buckets_var.get()

        if use_buckets:
            # Use three-bucket classification
            threshold_mode = self.get_current_threshold_mode()
            threshold_config = ThresholdConfig.from_mode(threshold_mode)
            classifier = NodeClassifier(algorithm, threshold_config)
            classification_results = classifier.classify_nodes(self.current_scenario)

            # Convert classification results to format compatible with existing code
            classified_nodes = [(result.node, result.score) for result in classification_results]
            self.classified_nodes = classified_nodes  # Store for click detection

            # Get threshold information for the colorbar
            threshold_info = classifier.get_threshold_info(classification_results)
            cluster_average = threshold_info['cluster_average']
            under_threshold = threshold_info['under_threshold']
            over_threshold = threshold_info['over_threshold']

            # Use actual scores instead of categorical mapping for better visualization
            color_values = [result.score for result in classification_results]

            # Find min and max scores for colorbar range
            min_score = min(color_values)
            max_score = max(color_values)
            score_range = max_score - min_score

            # Add some padding to the range
            if score_range > 0:
                padding = score_range * 0.1
                vmin = max(0, min_score - padding)
                vmax = min(1, max_score + padding)
            else:
                vmin, vmax = 0, 1

            # Use custom colormap
            cmap = self.create_three_bucket_colormap()
            title_suffix = f" - Three-Bucket ({threshold_mode.value})"
            colorbar_label = 'Load Score'
        else:
            # Use regular classification
            classified_nodes = algorithm.classify_nodes(self.current_scenario)
            self.classified_nodes = classified_nodes  # Store for click detection
            color_values = [score for _, score in classified_nodes]
            cmap = 'RdYlGn_r'
            title_suffix = ""
            colorbar_label = 'Load Score (0=low, 1=high)'

        # Extract data for both plots
        labels = [node.name for node, _ in classified_nodes]

        # Plot 1: CPU Usage vs Memory Usage
        x1_data = [node.cpu_usage for node, _ in classified_nodes]
        y1_data = [node.memory_usage for node, _ in classified_nodes]

        # Plot 2: CPU Pressure vs Memory Pressure
        x2_data = [node.cpu_pressure for node, _ in classified_nodes]
        y2_data = [node.memory_pressure for node, _ in classified_nodes]

        # Store plot data for click detection (using first plot's data as primary)
        self.plot_data = {
            'plot1': {
                'x_data': x1_data,
                'y_data': y1_data,
                'labels': labels,
                'x_param': 'cpu_usage',
                'y_param': 'memory_usage'
            },
            'plot2': {
                'x_data': x2_data,
                'y_data': y2_data,
                'labels': labels,
                'x_param': 'cpu_pressure',
                'y_param': 'memory_pressure'
            }
        }

        # Helper function for formatting labels
        def format_axis_label(param_name):
            """Format axis label with proper capitalization."""
            formatted = param_name.replace('_', ' ').title()
            formatted = formatted.replace('Cpu', 'CPU')
            return formatted

        # Create both scatter plots with appropriate vmin/vmax
        if use_buckets:
            # Use dynamic range for three-bucket mode
            self.scatter_plots[0] = ax1.scatter(x1_data, y1_data, c=color_values, cmap=cmap, s=150, alpha=0.7,
                                              edgecolors='black', linewidths=0.5, vmin=vmin, vmax=vmax)
            self.scatter_plots[1] = ax2.scatter(x2_data, y2_data, c=color_values, cmap=cmap, s=150, alpha=0.7,
                                              edgecolors='black', linewidths=0.5, vmin=vmin, vmax=vmax)
        else:
            # Use fixed range for regular mode
            self.scatter_plots[0] = ax1.scatter(x1_data, y1_data, c=color_values, cmap=cmap, s=150, alpha=0.7,
                                              edgecolors='black', linewidths=0.5, vmin=0, vmax=1)
            self.scatter_plots[1] = ax2.scatter(x2_data, y2_data, c=color_values, cmap=cmap, s=150, alpha=0.7,
                                              edgecolors='black', linewidths=0.5, vmin=0, vmax=1)

        # Clear previous annotations
        self.rank_annotations = [[], []]
        self.name_annotations = [[], []]

        # Add annotations to both plots
        for plot_idx, (ax, x_data, y_data) in enumerate([
            (ax1, x1_data, y1_data),
            (ax2, x2_data, y2_data)
        ]):
            for i, (label, (node, score)) in enumerate(zip(labels, classified_nodes)):
                rank = i + 1
                # Add rank number in center of dot
                rank_ann = ax.annotate(str(rank), (x_data[i], y_data[i]), ha='center', va='center',
                                     fontsize=9, fontweight='bold', color='white')
                self.rank_annotations[plot_idx].append(rank_ann)

                # Add node name as offset label
                name_ann = ax.annotate(label, (x_data[i], y_data[i]), xytext=(8, 8),
                                     textcoords='offset points', fontsize=7)
                self.name_annotations[plot_idx].append(name_ann)

        # Configure first plot (Usage)
        ax1.set_xlabel('CPU Usage')
        ax1.set_ylabel('Memory Usage')
        ax1.set_title('Resource Usage')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)

        # Configure second plot (Pressure)
        ax2.set_xlabel('CPU Pressure')
        ax2.set_ylabel('Memory Pressure')
        ax2.set_title('Resource Pressure (PSI)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(-0.05, 1.05)

        # Add a single color bar on the right side, outside both plots
        # Create a new axes for the colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Position the colorbar to the right of the rightmost plot
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="8%", pad=0.15)
        cbar = self.fig.colorbar(self.scatter_plots[0], cax=cax)
        cbar.set_label(colorbar_label, fontsize=10)

        # For three-bucket mode, add threshold markers and scale
        if use_buckets:
            # Calculate node counts for each bucket
            under_count = sum(1 for result in classification_results if result.category == UtilizationCategory.UNDER_UTILIZED)
            appropriate_count = sum(1 for result in classification_results if result.category == UtilizationCategory.APPROPRIATELY_UTILIZED)
            over_count = sum(1 for result in classification_results if result.category == UtilizationCategory.OVER_UTILIZED)

            # Set up colorbar with actual score values and node counts
            # Create ticks at important points: min, under_threshold, cluster_average, over_threshold, max
            ticks = [under_threshold, cluster_average, over_threshold]
            tick_labels = [
                f'{under_threshold:.3f}\nUnder ({under_count})',
                f'{cluster_average:.3f}\nAvg ({appropriate_count})',
                f'{over_threshold:.3f}\nOver ({over_count})'
            ]

            # Add min and max if they're different from thresholds
            if vmin < under_threshold - 0.01:
                ticks.insert(0, vmin)
                tick_labels.insert(0, f'{vmin:.3f}')
            if vmax > over_threshold + 0.01:
                ticks.append(vmax)
                tick_labels.append(f'{vmax:.3f}')

            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels, fontsize=9, fontweight='bold')

            # Add a summary text box showing total counts
            total_nodes = len(classification_results)
            summary_text = f'Total: {total_nodes} nodes\n▼ Under: {under_count}\n■ Approp: {appropriate_count}\n▲ Over: {over_count}'

            # Position the text box at the bottom of the colorbar
            cbar_ax = cbar.ax
            cbar_ax.text(1.8, 0.02, summary_text, transform=cbar_ax.transAxes,
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
                        verticalalignment='bottom')

            # Add horizontal lines at thresholds (if not already added)
            if not hasattr(cbar_ax, '_threshold_lines_added'):
                cbar_ax.axhline(y=under_threshold, color='blue', linestyle='--', linewidth=1, alpha=0.8)
                cbar_ax.axhline(y=cluster_average, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
                cbar_ax.axhline(y=over_threshold, color='red', linestyle='--', linewidth=1, alpha=0.8)
                cbar_ax._threshold_lines_added = True

        # Set overall title
        self.fig.suptitle(f'{self.scenario_name} - {algorithm.name}{title_suffix}', fontsize=14)

        # Adjust layout to prevent overlap
        self.fig.tight_layout()

        self.canvas.draw()

        # Update results tree
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        if use_buckets:
            # Show three-bucket classification results with categories
            for rank, result in enumerate(classification_results, 1):
                category_symbol = "▼" if result.category == UtilizationCategory.UNDER_UTILIZED else "■" if result.category == UtilizationCategory.APPROPRIATELY_UTILIZED else "▲"
                category_name = result.category.value.replace('-', ' ').title()
                self.results_tree.insert('', tk.END, text=f"{rank}",
                                       values=(result.node.name,
                                              f"{category_symbol} {category_name}",
                                              f"{result.score:.3f}",
                                              f"{result.node.cpu_usage:.2f}",
                                              f"{result.node.cpu_pressure:.2f}",
                                              f"{result.node.memory_usage:.2f}",
                                              f"{result.node.memory_pressure:.2f}"))
        else:
            # Show regular ranking results without bucket information
            for rank, (node, score) in enumerate(classified_nodes, 1):
                self.results_tree.insert('', tk.END, text=f"{rank}",
                                       values=(node.name,
                                              "-",  # No bucket in regular mode
                                              f"{score:.3f}",
                                              f"{node.cpu_usage:.2f}",
                                              f"{node.cpu_pressure:.2f}",
                                              f"{node.memory_usage:.2f}",
                                              f"{node.memory_pressure:.2f}"))

    def load_file(self):
        """Load scenarios from a JSON file."""
        filename = filedialog.askopenfilename(
            title="Load Scenarios",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.current_scenarios = ScenarioLoader.load_scenario(filename)
                self.update_scenario_combo()
                if self.current_scenarios:
                    self.scenario_combo.current(0)
                    self.on_scenario_change()
                messagebox.showinfo("Success", f"Loaded {len(self.current_scenarios)} scenarios")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def save_file(self):
        """Save current scenarios to a JSON file."""
        filename = filedialog.asksaveasfilename(
            title="Save Scenarios",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                ScenarioLoader.save_scenario(self.current_scenarios, filename)
                messagebox.showinfo("Success", f"Scenarios saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def find_closest_node(self, x, y, plot_key):
        """Find the closest node to the given coordinates."""
        if not self.plot_data or plot_key not in self.plot_data:
            return -1

        plot_data = self.plot_data[plot_key]
        min_distance = float('inf')
        closest_index = -1

        for i, (node_x, node_y) in enumerate(zip(plot_data['x_data'], plot_data['y_data'])):
            distance = ((x - node_x) ** 2 + (y - node_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        # Set a reasonable threshold for click detection
        click_threshold = 0.05  # 5% of the axis range
        if min_distance < click_threshold:
            return closest_index
        return -1

    def on_plot_press(self, event):
        """Handle mouse button press on the scatter plot."""
        if event.inaxes is None or not self.plot_data:
            return

        # Only respond to left mouse button clicks
        if event.button != 1:
            return

        # Skip if navigation toolbar is in zoom/pan mode
        if self.toolbar.mode != '':
            return

        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return

        # Determine which plot was clicked by checking the axes
        plot_key = None
        plot_index = None

        # Get the current axes from the figure
        axes = self.fig.get_axes()
        if len(axes) >= 2:
            if event.inaxes == axes[0]:  # Left plot (usage)
                plot_key = 'plot1'
                plot_index = 0
            elif event.inaxes == axes[1]:  # Right plot (pressure)
                plot_key = 'plot2'
                plot_index = 1

        if plot_key is None:
            return

        closest_index = self.find_closest_node(click_x, click_y, plot_key)
        if closest_index >= 0:
            self.dragging_node = closest_index
            self.dragging_plot = plot_index
            self.drag_start_pos = (click_x, click_y)
            # Set cursor to indicate dragging
            self.canvas.get_tk_widget().config(cursor="hand2")

    def on_plot_motion(self, event):
        """Handle mouse motion on the scatter plot."""
        if self.dragging_node is None or event.inaxes is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Constrain to plot bounds
        new_x = max(0.0, min(1.0, event.xdata))
        new_y = max(0.0, min(1.0, event.ydata))

        # Update the node's parameters based on current axes
        self.update_node_from_plot_position(self.dragging_node, new_x, new_y)

    def on_plot_release(self, event):
        """Handle mouse button release on the scatter plot."""
        if self.dragging_node is not None:
            # Restore normal cursor
            self.canvas.get_tk_widget().config(cursor="")

            # Check if this was a significant drag or just a click
            was_drag = False
            if self.drag_start_pos and event.xdata is not None and event.ydata is not None:
                start_x, start_y = self.drag_start_pos
                drag_distance = ((event.xdata - start_x) ** 2 + (event.ydata - start_y) ** 2) ** 0.5

                if drag_distance >= 0.02:  # Significant drag occurred
                    was_drag = True
                    # Update full visualization to reflect new rankings and colors
                    self.update_visualization()
                else:
                    # Just a click - highlight in results
                    self.highlight_node_in_results(self.dragging_node)

            self.dragging_node = None
            self.dragging_plot = None
            self.drag_start_pos = None

    def update_node_from_plot_position(self, node_index, x_value, y_value):
        """Update node parameters based on its position in the plot."""
        if not self.classified_nodes or node_index >= len(self.classified_nodes) or self.dragging_plot is None:
            return

        # Get the actual node object (not from classified list, but from scenario)
        node_from_classified = self.classified_nodes[node_index][0]

        # Find this node in the current scenario
        actual_node = None
        for node in self.current_scenario:
            if node.name == node_from_classified.name:
                actual_node = node
                break

        if actual_node is None:
            return

        # Update the parameters based on which plot is being dragged
        plot_key = f'plot{self.dragging_plot + 1}'
        if plot_key not in self.plot_data:
            return

        plot_info = self.plot_data[plot_key]
        x_param = plot_info['x_param']
        y_param = plot_info['y_param']

        # Set the new values
        if x_param == 'cpu_usage':
            actual_node.cpu_usage = x_value
        elif x_param == 'cpu_pressure':
            actual_node.cpu_pressure = x_value
        elif x_param == 'memory_usage':
            actual_node.memory_usage = x_value
        elif x_param == 'memory_pressure':
            actual_node.memory_pressure = x_value

        if y_param == 'cpu_usage':
            actual_node.cpu_usage = y_value
        elif y_param == 'cpu_pressure':
            actual_node.cpu_pressure = y_value
        elif y_param == 'memory_usage':
            actual_node.memory_usage = y_value
        elif y_param == 'memory_pressure':
            actual_node.memory_pressure = y_value

        # During dragging, only update the specific dot position and node tree
        # Don't re-run full visualization to avoid moving other nodes
        self.update_single_node_position(node_index, x_value, y_value)
        self.update_nodes_tree()

        # Update parameter controls if this node is selected
        if self.selected_node_id is not None:
            selected_node = self.current_scenario[self.selected_node_id]
            if selected_node.name == actual_node.name:
                self.param_vars['cpu_usage'].set(actual_node.cpu_usage)
                self.param_vars['cpu_pressure'].set(actual_node.cpu_pressure)
                self.param_vars['memory_usage'].set(actual_node.memory_usage)
                self.param_vars['memory_pressure'].set(actual_node.memory_pressure)

    def update_single_node_position(self, node_index, x_value, y_value):
        """Update only the dragged node's position without re-classifying all nodes."""
        if self.dragging_plot is None or self.scatter_plots[self.dragging_plot] is None:
            return

        plot_key = f'plot{self.dragging_plot + 1}'
        if plot_key not in self.plot_data:
            return

        # Update the plot data for this specific node
        plot_data = self.plot_data[plot_key]
        if node_index < len(plot_data['x_data']):
            plot_data['x_data'][node_index] = x_value
            plot_data['y_data'][node_index] = y_value

            # Update the scatter plot points for the dragged plot
            scatter_plot = self.scatter_plots[self.dragging_plot]
            offsets = scatter_plot.get_offsets()
            if node_index < len(offsets):
                # Update only the dragged point
                offsets[node_index] = [x_value, y_value]
                scatter_plot.set_offsets(offsets)

                # Update the rank annotation position
                if node_index < len(self.rank_annotations[self.dragging_plot]):
                    self.rank_annotations[self.dragging_plot][node_index].set_position((x_value, y_value))

                # Update the node name annotation position
                if node_index < len(self.name_annotations[self.dragging_plot]):
                    self.name_annotations[self.dragging_plot][node_index].set_position((x_value, y_value))

                # Also update the corresponding node in the other plot with its current values
                other_plot = 1 - self.dragging_plot
                other_plot_key = f'plot{other_plot + 1}'
                if other_plot_key in self.plot_data:
                    # Get updated node values
                    node_from_classified = self.classified_nodes[node_index][0]
                    actual_node = None
                    for node in self.current_scenario:
                        if node.name == node_from_classified.name:
                            actual_node = node
                            break

                    if actual_node:
                        other_plot_info = self.plot_data[other_plot_key]
                        new_x = actual_node.get_metric(other_plot_info['x_param'])
                        new_y = actual_node.get_metric(other_plot_info['y_param'])

                        # Update other plot data
                        other_plot_info['x_data'][node_index] = new_x
                        other_plot_info['y_data'][node_index] = new_y

                        # Update other scatter plot
                        other_scatter = self.scatter_plots[other_plot]
                        other_offsets = other_scatter.get_offsets()
                        if node_index < len(other_offsets):
                            other_offsets[node_index] = [new_x, new_y]
                            other_scatter.set_offsets(other_offsets)

                            # Update annotations in other plot
                            if node_index < len(self.rank_annotations[other_plot]):
                                self.rank_annotations[other_plot][node_index].set_position((new_x, new_y))
                            if node_index < len(self.name_annotations[other_plot]):
                                self.name_annotations[other_plot][node_index].set_position((new_x, new_y))

                # Redraw only the canvas
                self.canvas.draw_idle()

    def highlight_node_in_results(self, plot_index):
        """Highlight the clicked node using centralized selection manager."""
        if 'plot1' in self.plot_data and 0 <= plot_index < len(self.plot_data['plot1']['labels']):
            node_name = self.plot_data['plot1']['labels'][plot_index]
            self._set_selected_node(node_name)



    def on_results_select(self, event=None):
        """Handle selection in the classification results tree."""
        if self._selection_manager_active:
            return

        selected = self.results_tree.selection()
        if selected:
            item_id = selected[0]
            items = self.results_tree.get_children()
            if item_id in items:
                result_index = list(items).index(item_id)
                if result_index < len(self.classified_nodes):
                    node_name = self.classified_nodes[result_index][0].name
                    self._set_selected_node(node_name)

def main():
    root = tk.Tk()
    app = NodeClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()