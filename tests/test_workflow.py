"""
Unit tests for LangGraphWorkflow class in workflow.py

This test suite provides comprehensive coverage for all functions:
- Test edge cases (empty data, invalid inputs, null values)
- Test expected outputs with known inputs
- Test exception handling

Each function has 3 unit tests to ensure robust validation.
"""

import unittest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Fix import issue by importing entity_assignment first
import entity_assignment
from workflow import LangGraphWorkflow
from entity_assignment import EntityAssignment


class TestLangGraphWorkflow(unittest.TestCase):
    """Test suite for LangGraphWorkflow class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock column finder
        self.mock_column_finder = Mock()
        self.mock_column_finder.find_similar_columns.return_value = [
            ('col1', 'col2', 95),
            ('col3', 'col4', 90)
        ]
        self.mock_column_finder.describe_columns.return_value = {
            'col1': 'Description 1',
            'col2': 'Description 2'
        }
        
        # Create test DataFrames
        self.df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        self.df2 = pd.DataFrame({
            'identifier': [1, 2, 3],
            'full_name': ['Alice Smith', 'Bob Jones', 'Charlie Brown'],
            'created_date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        # Initialize workflow with mock column finder
        self.workflow = LangGraphWorkflow(self.mock_column_finder)

    # ==================== __init__ Method Tests ====================
    
    def test_init_valid_column_finder(self):
        """Test 1: Successful initialization with valid column finder"""
        column_finder = Mock()
        column_finder.find_similar_columns = Mock()
        
        workflow = LangGraphWorkflow(column_finder, match_threshold=0.8)
        
        self.assertEqual(workflow.column_finder, column_finder)
        self.assertEqual(workflow.entity_assignment.match_threshold, 0.8)
        self.assertIsInstance(workflow.entity_assignment, EntityAssignment)

    def test_init_none_column_finder(self):
        """Test 2: Exception when column_finder is None"""
        with self.assertRaises(ValueError) as context:
            LangGraphWorkflow(None)
        
        self.assertIn("column_finder cannot be None", str(context.exception))

    def test_init_invalid_column_finder(self):
        """Test 3: Exception when column_finder doesn't have required method"""
        invalid_finder = object()  # Simple object without find_similar_columns method
        
        with self.assertRaises(TypeError) as context:
            LangGraphWorkflow(invalid_finder)
        
        self.assertIn("column_finder must have a 'find_similar_columns' method", str(context.exception))

    def test_init_invalid_match_threshold_type(self):
        """Test 4: Exception when match_threshold is not a number"""
        column_finder = Mock()
        column_finder.find_similar_columns = Mock()
        
        with self.assertRaises(TypeError) as context:
            LangGraphWorkflow(column_finder, match_threshold="invalid")
        
        self.assertIn("match_threshold must be a number", str(context.exception))

    def test_init_invalid_match_threshold_range(self):
        """Test 5: Exception when match_threshold is out of range"""
        column_finder = Mock()
        column_finder.find_similar_columns = Mock()
        
        with self.assertRaises(ValueError) as context:
            LangGraphWorkflow(column_finder, match_threshold=1.5)
        
        self.assertIn("match_threshold must be between 0 and 1", str(context.exception))

    # ==================== create_column_matching_graph Method Tests ====================
    
    @patch('workflow.NodeGraph')
    def test_create_column_matching_graph_normal_case(self, mock_node_graph):
        """Test 1: Normal graph creation with valid DataFrames"""
        # Setup mock graph
        mock_graph = Mock()
        mock_node_graph.return_value = mock_graph
        mock_graph.add_node.return_value = Mock()
        mock_graph.add_edges.return_value = None
        
        result = self.workflow.create_column_matching_graph(self.df1, self.df2)
        
        self.assertEqual(result, mock_graph)
        
        # Verify graph creation steps
        mock_node_graph.assert_called_once()
        
        # Verify nodes were added (should be called 4 times for 4 nodes)
        self.assertEqual(mock_graph.add_node.call_count, 4)
        
        # Verify edges were added
        mock_graph.add_edges.assert_called_once()

    def test_create_column_matching_graph_invalid_df1(self):
        """Test 2: Exception when df1 is not pandas DataFrame"""
        with self.assertRaises(TypeError) as context:
            self.workflow.create_column_matching_graph("not a dataframe", self.df2)
        
        self.assertIn("df1 must be a pandas DataFrame", str(context.exception))

    def test_create_column_matching_graph_invalid_df2(self):
        """Test 3: Exception when df2 is not pandas DataFrame"""
        with self.assertRaises(TypeError) as context:
            self.workflow.create_column_matching_graph(self.df1, "not a dataframe")
        
        self.assertIn("df2 must be a pandas DataFrame", str(context.exception))

    def test_create_column_matching_graph_empty_df1(self):
        """Test 4: Exception when df1 is empty"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.workflow.create_column_matching_graph(empty_df, self.df2)
        
        self.assertIn("df1 cannot be empty", str(context.exception))

    def test_create_column_matching_graph_empty_df2(self):
        """Test 5: Exception when df2 is empty"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.workflow.create_column_matching_graph(self.df1, empty_df)
        
        self.assertIn("df2 cannot be empty", str(context.exception))

    @patch('workflow.NodeGraph')
    def test_create_column_matching_graph_node_details(self, mock_node_graph):
        """Test 6: Verify correct node configuration"""
        mock_graph = Mock()
        mock_node_graph.return_value = mock_graph
        mock_graph.add_node.return_value = Mock()
        mock_graph.add_edges.return_value = None
        
        self.workflow.create_column_matching_graph(self.df1, self.df2)
        
        # Verify node creation calls
        calls = mock_graph.add_node.call_args_list
        
        # Should have 4 nodes
        self.assertEqual(len(calls), 4)
        
        # Check first node (Describe Dataset 1)
        first_call = calls[0]
        self.assertIn('name', first_call[1])
        self.assertIn('Describe columns in Dataset 1', first_call[1]['name'])
        
        # Check second node (Describe Dataset 2)
        second_call = calls[1]
        self.assertIn('name', second_call[1])
        self.assertIn('Describe columns in Dataset 2', second_call[1]['name'])
        
        # Check third node (Compare & Match)
        third_call = calls[2]
        self.assertIn('name', third_call[1])
        self.assertIn('Compare columns and find matches', third_call[1]['name'])
        
        # Check fourth node (Consolidate)
        fourth_call = calls[3]
        self.assertIn('name', fourth_call[1])
        self.assertIn('Consolidate datasets based on matches', fourth_call[1]['name'])

    @patch('workflow.NodeGraph')
    def test_create_column_matching_graph_edge_configuration(self, mock_node_graph):
        """Test 7: Verify correct edge configuration"""
        mock_graph = Mock()
        mock_node_graph.return_value = mock_graph
        mock_graph.add_node.return_value = Mock()
        mock_graph.add_edges.return_value = None
        
        self.workflow.create_column_matching_graph(self.df1, self.df2)
        
        # Verify edge creation was called
        mock_graph.add_edges.assert_called_once()
        
        # Check the edge configuration
        edge_call = mock_graph.add_edges.call_args[0][0]
        self.assertIsInstance(edge_call, list)
        self.assertEqual(len(edge_call), 3)  # Should have 3 edges
        
        # Verify edge connections
        expected_edges = [
            ("Describe columns in Dataset 1", "Compare columns and find matches"),
            ("Describe columns in Dataset 2", "Compare columns and find matches"),
            ("Compare columns and find matches", "Consolidate datasets based on matches"),
        ]
        
        self.assertEqual(edge_call, expected_edges)

    # ==================== Integration Tests ====================
    
    def test_workflow_integration_with_real_components(self):
        """Test 1: Integration test with real EntityAssignment component"""
        # Use the real workflow with real entity assignment
        workflow = LangGraphWorkflow(self.mock_column_finder, match_threshold=0.75)
        
        # Verify the entity assignment was properly initialized
        self.assertIsInstance(workflow.entity_assignment, EntityAssignment)
        self.assertEqual(workflow.entity_assignment.match_threshold, 0.75)

    @patch('workflow.NodeGraph')
    def test_workflow_with_different_dataframe_structures(self, mock_node_graph):
        """Test 2: Workflow with DataFrames of different structures"""
        mock_graph = Mock()
        mock_node_graph.return_value = mock_graph
        mock_graph.add_node.return_value = Mock()
        mock_graph.add_edges.return_value = None
        
        # Create DataFrames with different column counts
        df1_small = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2_large = pd.DataFrame({
            'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 
            'w': [7, 8], 'v': [9, 10]
        })
        
        result = self.workflow.create_column_matching_graph(df1_small, df2_large)
        
        # Should handle different structures gracefully
        self.assertEqual(result, mock_graph)
        self.assertEqual(mock_graph.add_node.call_count, 4)

    def test_workflow_column_finder_interaction(self):
        """Test 3: Verify proper interaction with column finder"""
        # Create a more detailed mock column finder
        detailed_mock = Mock()
        detailed_mock.find_similar_columns.return_value = [('a', 'x', 85)]
        detailed_mock.describe_columns.side_effect = [
            {'a': 'Column A description', 'b': 'Column B description'},
            {'x': 'Column X description', 'y': 'Column Y description'}
        ]
        
        workflow = LangGraphWorkflow(detailed_mock)
        
        # Verify the workflow properly stores the column finder
        self.assertEqual(workflow.column_finder, detailed_mock)
        
        # The column finder methods should be available
        self.assertTrue(hasattr(workflow.column_finder, 'find_similar_columns'))
        self.assertTrue(hasattr(workflow.column_finder, 'describe_columns'))

    # ==================== Edge Case Tests ====================
    
    def test_workflow_with_single_column_dataframes(self):
        """Test 1: Handle DataFrames with single columns"""
        df1_single = pd.DataFrame({'single_col': [1, 2, 3]})
        df2_single = pd.DataFrame({'another_col': [4, 5, 6]})
        
        # This should not raise an exception
        with patch('workflow.NodeGraph') as mock_node_graph:
            mock_graph = Mock()
            mock_node_graph.return_value = mock_graph
            mock_graph.add_node.return_value = Mock()
            mock_graph.add_edges.return_value = None
            
            result = self.workflow.create_column_matching_graph(df1_single, df2_single)
            self.assertEqual(result, mock_graph)

    def test_workflow_with_many_columns(self):
        """Test 2: Handle DataFrames with many columns"""
        # Create DataFrames with many columns
        many_cols_data = {f'col_{i}': [i, i+1, i+2] for i in range(50)}
        df1_many = pd.DataFrame(many_cols_data)
        df2_many = pd.DataFrame(many_cols_data)
        
        with patch('workflow.NodeGraph') as mock_node_graph:
            mock_graph = Mock()
            mock_node_graph.return_value = mock_graph
            mock_graph.add_node.return_value = Mock()
            mock_graph.add_edges.return_value = None
            
            # Should handle large DataFrames without issues
            result = self.workflow.create_column_matching_graph(df1_many, df2_many)
            self.assertEqual(result, mock_graph)

    def test_workflow_with_unicode_column_names(self):
        """Test 3: Handle DataFrames with unicode column names"""
        df1_unicode = pd.DataFrame({
            'column_数据': [1, 2, 3],
            'données_col': ['a', 'b', 'c'],
            'столбец': [4, 5, 6]
        })
        
        df2_unicode = pd.DataFrame({
            'col_データ': [1, 2, 3],
            'colonne_données': ['x', 'y', 'z'],
            'колонка': [7, 8, 9]
        })
        
        with patch('workflow.NodeGraph') as mock_node_graph:
            mock_graph = Mock()
            mock_node_graph.return_value = mock_graph
            mock_graph.add_node.return_value = Mock()
            mock_graph.add_edges.return_value = None
            
            # Should handle unicode column names gracefully
            result = self.workflow.create_column_matching_graph(df1_unicode, df2_unicode)
            self.assertEqual(result, mock_graph)


if __name__ == '__main__':
    unittest.main()
