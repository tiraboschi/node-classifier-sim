#!/usr/bin/env python3
"""
Unit tests for the Node class.
"""

import unittest
from node import Node


class TestNode(unittest.TestCase):
    """Test cases for the Node class."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_node = Node("test-node", 0.5, 0.2, 0.7, 0.1)

    def test_valid_node_creation(self):
        """Test creating a valid node."""
        node = Node("node-1", 0.3, 0.1, 0.6, 0.05)
        self.assertEqual(node.name, "node-1")
        self.assertEqual(node.cpu_usage, 0.3)
        self.assertEqual(node.cpu_pressure, 0.1)
        self.assertEqual(node.memory_usage, 0.6)
        self.assertEqual(node.memory_pressure, 0.05)

    def test_node_creation_with_edge_values(self):
        """Test node creation with boundary values."""
        # Test with minimum values
        node_min = Node("min-node", 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(node_min.cpu_usage, 0.0)
        self.assertEqual(node_min.cpu_pressure, 0.0)
        self.assertEqual(node_min.memory_usage, 0.0)
        self.assertEqual(node_min.memory_pressure, 0.0)

        # Test with maximum values
        node_max = Node("max-node", 1.0, 1.0, 1.0, 1.0)
        self.assertEqual(node_max.cpu_usage, 1.0)
        self.assertEqual(node_max.cpu_pressure, 1.0)
        self.assertEqual(node_max.memory_usage, 1.0)
        self.assertEqual(node_max.memory_pressure, 1.0)

    def test_invalid_metric_values(self):
        """Test that invalid metric values raise ValueError."""
        # Test negative values
        with self.assertRaises(ValueError):
            Node("negative-cpu", -0.1, 0.2, 0.5, 0.1)

        with self.assertRaises(ValueError):
            Node("negative-cpu-pressure", 0.5, -0.1, 0.5, 0.1)

        with self.assertRaises(ValueError):
            Node("negative-memory", 0.5, 0.2, -0.1, 0.1)

        with self.assertRaises(ValueError):
            Node("negative-memory-pressure", 0.5, 0.2, 0.5, -0.1)

        # Test values greater than 1.0
        with self.assertRaises(ValueError):
            Node("high-cpu", 1.1, 0.2, 0.5, 0.1)

        with self.assertRaises(ValueError):
            Node("high-cpu-pressure", 0.5, 1.1, 0.5, 0.1)

        with self.assertRaises(ValueError):
            Node("high-memory", 0.5, 0.2, 1.1, 0.1)

        with self.assertRaises(ValueError):
            Node("high-memory-pressure", 0.5, 0.2, 0.5, 1.1)

    def test_empty_name(self):
        """Test that empty node name raises ValueError."""
        with self.assertRaises(ValueError):
            Node("", 0.5, 0.2, 0.7, 0.1)

    def test_get_metric_method(self):
        """Test the get_metric method."""
        node = Node("test-node", 0.3, 0.1, 0.6, 0.05)

        self.assertEqual(node.get_metric("cpu_usage"), 0.3)
        self.assertEqual(node.get_metric("cpu_pressure"), 0.1)
        self.assertEqual(node.get_metric("memory_usage"), 0.6)
        self.assertEqual(node.get_metric("memory_pressure"), 0.05)

    def test_get_metric_invalid_param(self):
        """Test get_metric with invalid parameter name."""
        with self.assertRaises(ValueError):
            self.valid_node.get_metric("invalid_metric")

    def test_from_dict_method(self):
        """Test creating node from dictionary."""
        node_dict = {
            "name": "dict-node",
            "cpu_usage": 0.4,
            "cpu_pressure": 0.15,
            "memory_usage": 0.8,
            "memory_pressure": 0.2
        }

        node = Node.from_dict(node_dict)
        self.assertEqual(node.name, "dict-node")
        self.assertEqual(node.cpu_usage, 0.4)
        self.assertEqual(node.cpu_pressure, 0.15)
        self.assertEqual(node.memory_usage, 0.8)
        self.assertEqual(node.memory_pressure, 0.2)

    def test_from_dict_missing_fields(self):
        """Test from_dict with missing required fields."""
        incomplete_dict = {
            "name": "incomplete-node",
            "cpu_usage": 0.4,
            # Missing cpu_pressure, memory_usage, memory_pressure
        }

        with self.assertRaises(KeyError):
            Node.from_dict(incomplete_dict)

    def test_from_dict_invalid_values(self):
        """Test from_dict with invalid metric values."""
        invalid_dict = {
            "name": "invalid-node",
            "cpu_usage": 1.5,  # Invalid: > 1.0
            "cpu_pressure": 0.15,
            "memory_usage": 0.8,
            "memory_pressure": 0.2
        }

        with self.assertRaises(ValueError):
            Node.from_dict(invalid_dict)

    def test_to_dict_method(self):
        """Test converting node to dictionary."""
        node = Node("dict-test", 0.35, 0.12, 0.75, 0.18)
        node_dict = node.to_dict()

        expected_dict = {
            "name": "dict-test",
            "cpu_usage": 0.35,
            "cpu_pressure": 0.12,
            "memory_usage": 0.75,
            "memory_pressure": 0.18
        }

        self.assertEqual(node_dict, expected_dict)

    def test_round_trip_dict_conversion(self):
        """Test that from_dict(to_dict()) preserves node data."""
        original = Node("round-trip", 0.42, 0.13, 0.67, 0.09)
        node_dict = original.to_dict()
        reconstructed = Node.from_dict(node_dict)

        self.assertEqual(original.name, reconstructed.name)
        self.assertEqual(original.cpu_usage, reconstructed.cpu_usage)
        self.assertEqual(original.cpu_pressure, reconstructed.cpu_pressure)
        self.assertEqual(original.memory_usage, reconstructed.memory_usage)
        self.assertEqual(original.memory_pressure, reconstructed.memory_pressure)

    def test_node_string_representation(self):
        """Test string representation of node."""
        node = Node("str-test", 0.5, 0.2, 0.7, 0.1)
        str_repr = str(node)

        self.assertIn("str-test", str_repr)
        self.assertIn("CPU", str_repr)
        self.assertIn("0.5", str_repr)
        self.assertIn("0.2", str_repr)
        self.assertIn("0.7", str_repr)
        self.assertIn("0.1", str_repr)

    def test_node_equality(self):
        """Test node equality comparison."""
        node1 = Node("same-node", 0.5, 0.2, 0.7, 0.1)
        node2 = Node("same-node", 0.5, 0.2, 0.7, 0.1)
        node3 = Node("different-node", 0.5, 0.2, 0.7, 0.1)
        node4 = Node("same-node", 0.6, 0.2, 0.7, 0.1)  # Different metrics

        # Same name and metrics should be equal
        self.assertEqual(node1, node2)

        # Different names should not be equal
        self.assertNotEqual(node1, node3)

        # Same name but different metrics should not be equal
        self.assertNotEqual(node1, node4)


if __name__ == '__main__':
    unittest.main()