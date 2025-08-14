"""
Unit tests for SimilarColumnFinder class in entity_resolution.py

This test suite provides comprehensive coverage for all functions in the SimilarColumnFinder class:
- Test edge cases (empty data, invalid inputs, null values)
- Test expected outputs with known inputs
- Test exception handling

Each function has 3 unit tests to ensure robust validation.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock PromptTemplate for testing
class MockPromptTemplate:
    def __init__(self, template):
        self.template = template
    
    def format(self, **kwargs):
        return self.template.format(**kwargs)
    
    @classmethod
    def from_template(cls, template):
        return cls(template)

# Mock OpenAI for testing
class MockOpenAI:
    def __init__(self, temperature=0.1):
        self.temperature = temperature
    
    def invoke(self, prompt):
        return "Mock LLM response"

# Patch the imports to use our mocks
sys.modules['langchain.prompts'] = Mock()
sys.modules['langchain.prompts'].PromptTemplate = MockPromptTemplate
sys.modules['langchain.llms'] = Mock()
sys.modules['langchain.llms'].OpenAI = MockOpenAI

from entity_resolution import SimilarColumnFinder


class TestSimilarColumnFinder(unittest.TestCase):
    """Test suite for SimilarColumnFinder class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock LLM for testing
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = "Customer identifier, likely a unique integer value"
        
        # Create test DataFrames
        self.df1 = pd.DataFrame({
            'cust_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        self.df2 = pd.DataFrame({
            'customer_number': [1, 2, 3],
            'full_name': ['Alice Smith', 'Bob Jones', 'Charlie Brown'],
            'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        # Initialize SimilarColumnFinder with mock LLM
        self.finder = SimilarColumnFinder(llm=self.mock_llm)

    # ==================== __init__ Method Tests ====================
    
    def test_init_with_valid_llm(self):
        """Test 1: Successful initialization with valid LLM"""
        llm = Mock()
        llm.invoke = Mock(return_value="test description")
        
        finder = SimilarColumnFinder(llm=llm)
        
        self.assertEqual(finder.llm, llm)
        self.assertIsNotNone(finder.column_description_prompt)
        self.assertTrue(hasattr(finder.column_description_prompt, 'format'))

    def test_init_with_none_llm(self):
        """Test 2: Exception when LLM is None"""
        with self.assertRaises(ValueError) as context:
            SimilarColumnFinder(llm=None)
        
        self.assertIn("LLM instance cannot be None", str(context.exception))

    def test_init_with_invalid_llm(self):
        """Test 3: Exception when LLM doesn't have invoke method"""
        invalid_llm = "not an llm"
        
        with self.assertRaises(TypeError) as context:
            SimilarColumnFinder(llm=invalid_llm)
        
        self.assertIn("LLM must have an 'invoke' method", str(context.exception))

    def test_init_with_invalid_prompt_template(self):
        """Test 4: Exception when custom prompt template is invalid"""
        llm = Mock()
        llm.invoke = Mock(return_value="test")
        
        # Create an invalid prompt template (one without format method)
        invalid_prompt = object()  # Simple object without format method
        
        with self.assertRaises(TypeError) as context:
            SimilarColumnFinder(llm=llm, column_description_prompt=invalid_prompt)
        
        self.assertIn("column_description_prompt must be a PromptTemplate", str(context.exception))

    # ==================== describe_columns Method Tests ====================
    
    def test_describe_columns_normal_case(self):
        """Test 1: Normal operation with valid DataFrame and columns"""
        self.mock_llm.invoke.return_value = "Test description"
        columns = ['cust_id', 'name']
        
        result = self.finder.describe_columns(self.df1, columns)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn('cust_id', result)
        self.assertIn('name', result)
        self.assertEqual(result['cust_id'], "Test description")
        self.assertEqual(result['name'], "Test description")
        
        # Verify LLM was called for each column
        self.assertEqual(self.mock_llm.invoke.call_count, 2)

    def test_describe_columns_invalid_dataframe(self):
        """Test 2: Exception when DataFrame is not pandas DataFrame"""
        with self.assertRaises(TypeError) as context:
            self.finder.describe_columns("not a dataframe", ['col1'])
        
        self.assertIn("df must be a pandas DataFrame", str(context.exception))

    def test_describe_columns_empty_dataframe(self):
        """Test 3: Exception when DataFrame is empty"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.finder.describe_columns(empty_df, ['col1'])
        
        self.assertIn("DataFrame cannot be empty", str(context.exception))

    def test_describe_columns_invalid_columns_type(self):
        """Test 4: Exception when columns is not a list"""
        with self.assertRaises(TypeError) as context:
            self.finder.describe_columns(self.df1, "not a list")
        
        self.assertIn("columns must be a list", str(context.exception))

    def test_describe_columns_empty_columns_list(self):
        """Test 5: Exception when columns list is empty"""
        with self.assertRaises(ValueError) as context:
            self.finder.describe_columns(self.df1, [])
        
        self.assertIn("columns list cannot be empty", str(context.exception))

    def test_describe_columns_missing_columns(self):
        """Test 6: Exception when columns don't exist in DataFrame"""
        with self.assertRaises(ValueError) as context:
            self.finder.describe_columns(self.df1, ['nonexistent_column'])
        
        self.assertIn("The following columns are not found in the DataFrame", str(context.exception))

    def test_describe_columns_llm_failure(self):
        """Test 7: Exception when LLM fails to generate description"""
        self.mock_llm.invoke.side_effect = Exception("LLM Error")
        
        with self.assertRaises(RuntimeError) as context:
            self.finder.describe_columns(self.df1, ['cust_id'])
        
        self.assertIn("Failed to generate description for column 'cust_id'", str(context.exception))

    # ==================== find_similar_columns Method Tests ====================
    
    @patch('entity_resolution.fuzz')
    def test_find_similar_columns_normal_case(self, mock_fuzz):
        """Test 1: Normal operation with similar columns"""
        # Setup mock responses
        self.mock_llm.invoke.side_effect = [
            "Customer identifier", "Customer name", "Order date",  # df1 descriptions
            "Customer identifier", "Customer full name", "Transaction date"  # df2 descriptions
        ]
        mock_fuzz.ratio.return_value = 95  # High similarity score for all comparisons
        
        result = self.finder.find_similar_columns(self.df1, self.df2, threshold=80)
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)  # Should find some matches above threshold
        
        # Check that results are tuples with correct structure
        for match in result:
            self.assertIsInstance(match, tuple)
            self.assertEqual(len(match), 3)
            self.assertIsInstance(match[0], str)  # col1
            self.assertIsInstance(match[1], str)  # col2
            self.assertIsInstance(match[2], (int, float))  # score

    def test_find_similar_columns_invalid_df1(self):
        """Test 2: Exception when df1 is not pandas DataFrame"""
        with self.assertRaises(TypeError) as context:
            self.finder.find_similar_columns("not a dataframe", self.df2)
        
        self.assertIn("df1 must be a pandas DataFrame", str(context.exception))

    def test_find_similar_columns_invalid_df2(self):
        """Test 3: Exception when df2 is not pandas DataFrame"""
        with self.assertRaises(TypeError) as context:
            self.finder.find_similar_columns(self.df1, "not a dataframe")
        
        self.assertIn("df2 must be a pandas DataFrame", str(context.exception))

    def test_find_similar_columns_empty_df1(self):
        """Test 4: Exception when df1 is empty"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.finder.find_similar_columns(empty_df, self.df2)
        
        self.assertIn("df1 cannot be empty", str(context.exception))

    def test_find_similar_columns_empty_df2(self):
        """Test 5: Exception when df2 is empty"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.finder.find_similar_columns(self.df1, empty_df)
        
        self.assertIn("df2 cannot be empty", str(context.exception))

    def test_find_similar_columns_invalid_threshold_type(self):
        """Test 6: Exception when threshold is not a number"""
        with self.assertRaises(TypeError) as context:
            self.finder.find_similar_columns(self.df1, self.df2, threshold="invalid")
        
        self.assertIn("threshold must be a number", str(context.exception))

    def test_find_similar_columns_invalid_threshold_range(self):
        """Test 7: Exception when threshold is out of valid range"""
        with self.assertRaises(ValueError) as context:
            self.finder.find_similar_columns(self.df1, self.df2, threshold=150)
        
        self.assertIn("threshold must be between 0 and 100", str(context.exception))

    @patch('entity_resolution.fuzz')
    def test_find_similar_columns_no_matches(self, mock_fuzz):
        """Test 8: No matches found below threshold"""
        # Setup mock responses with low similarity
        self.mock_llm.invoke.side_effect = [
            "Customer identifier", "Customer name", "Order date",
            "Product code", "Description", "Price"
        ]
        mock_fuzz.ratio.return_value = 30  # Low similarity scores
        
        result = self.finder.find_similar_columns(self.df1, self.df2, threshold=80)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)  # No matches above threshold

    # ==================== validate_column_matches Method Tests ====================
    
    def test_validate_column_matches_normal_case(self):
        """Test 1: Normal operation with valid column matches"""
        column_matches = [('cust_id', 'customer_number', 95.0)]
        
        result = self.finder.validate_column_matches(self.df1, self.df2, column_matches)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        match_result = result[0]
        self.assertIn('col1', match_result)
        self.assertIn('col2', match_result)
        self.assertIn('semantic_similarity', match_result)
        self.assertIn('data_type_analysis', match_result)
        self.assertIn('value_overlap_analysis', match_result)
        self.assertIn('overall_validation_score', match_result)
        self.assertIn('recommendation', match_result)

    def test_validate_column_matches_invalid_df1(self):
        """Test 2: Exception when df1 is not pandas DataFrame"""
        column_matches = [('col1', 'col2', 95.0)]
        
        with self.assertRaises(TypeError) as context:
            self.finder.validate_column_matches("not a dataframe", self.df2, column_matches)
        
        self.assertIn("df1 must be a pandas DataFrame", str(context.exception))

    def test_validate_column_matches_invalid_column_matches(self):
        """Test 3: Exception when column_matches is not a list"""
        with self.assertRaises(TypeError) as context:
            self.finder.validate_column_matches(self.df1, self.df2, "not a list")
        
        self.assertIn("column_matches must be a list", str(context.exception))

    def test_validate_column_matches_invalid_sample_size(self):
        """Test 4: Exception when sample_size is not an integer"""
        column_matches = [('col1', 'col2', 95.0)]
        
        with self.assertRaises(TypeError) as context:
            self.finder.validate_column_matches(self.df1, self.df2, column_matches, sample_size="invalid")
        
        self.assertIn("sample_size must be an integer", str(context.exception))

    def test_validate_column_matches_negative_sample_size(self):
        """Test 5: Exception when sample_size is negative"""
        column_matches = [('col1', 'col2', 95.0)]
        
        with self.assertRaises(ValueError) as context:
            self.finder.validate_column_matches(self.df1, self.df2, column_matches, sample_size=-1)
        
        self.assertIn("sample_size must be positive", str(context.exception))

    def test_validate_column_matches_invalid_match_structure(self):
        """Test 6: Exception when column match structure is invalid"""
        invalid_matches = [('col1', 'col2')]  # Missing score
        
        with self.assertRaises(ValueError) as context:
            self.finder.validate_column_matches(self.df1, self.df2, invalid_matches)
        
        self.assertIn("must be a tuple of (col1, col2, score)", str(context.exception))

    def test_validate_column_matches_nonexistent_column(self):
        """Test 7: Handles nonexistent columns gracefully"""
        column_matches = [('nonexistent_col', 'customer_number', 95.0)]
        
        result = self.finder.validate_column_matches(self.df1, self.df2, column_matches)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        match_result = result[0]
        self.assertEqual(match_result['overall_validation_score'], 0)
        self.assertIn('COLUMN_NOT_FOUND', match_result['validation_flags'])

    # ==================== Helper Method Tests ====================
    
    def test_analyze_data_type_compatibility_numeric_types(self):
        """Test 1: Data type compatibility for numeric types"""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([10, 20, 30, 40, 50])
        
        result = self.finder._analyze_data_type_compatibility(series1, series2)
        
        self.assertIsInstance(result, dict)
        self.assertIn('compatible', result)
        self.assertIn('compatibility_score', result)
        self.assertTrue(result['compatible'])
        self.assertEqual(result['compatibility_score'], 100)  # Perfect numeric match

    def test_analyze_data_type_compatibility_empty_series(self):
        """Test 2: Data type compatibility for empty series"""
        empty_series1 = pd.Series([], dtype=object)
        empty_series2 = pd.Series([], dtype=object)
        
        result = self.finder._analyze_data_type_compatibility(empty_series1, empty_series2)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['series1_type'], 'empty')
        self.assertEqual(result['series2_type'], 'empty')
        self.assertEqual(result['compatibility_score'], 0)

    def test_analyze_data_type_compatibility_incompatible_types(self):
        """Test 3: Data type compatibility for incompatible types"""
        series1 = pd.Series([1, 2, 3])  # Numeric
        series2 = pd.Series(['a', 'b', 'c'])  # Text
        
        result = self.finder._analyze_data_type_compatibility(series1, series2)
        
        self.assertIsInstance(result, dict)
        self.assertLess(result['compatibility_score'], 70)  # Should be low compatibility

    def test_calculate_value_overlap_identical_values(self):
        """Test 1: Value overlap for identical series"""
        series1 = pd.Series(['A', 'B', 'C'])
        series2 = pd.Series(['A', 'B', 'C'])
        
        result = self.finder._calculate_value_overlap(series1, series2)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['overlap_percentage'], 100.0)
        self.assertEqual(result['jaccard_similarity'], 1.0)

    def test_calculate_value_overlap_no_overlap(self):
        """Test 2: Value overlap for completely different series"""
        series1 = pd.Series(['A', 'B', 'C'])
        series2 = pd.Series(['X', 'Y', 'Z'])
        
        result = self.finder._calculate_value_overlap(series1, series2)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['overlap_percentage'], 0.0)
        self.assertEqual(result['jaccard_similarity'], 0.0)

    def test_calculate_value_overlap_empty_series(self):
        """Test 3: Value overlap for empty series"""
        empty1 = pd.Series([], dtype=object)
        empty2 = pd.Series([], dtype=object)
        
        result = self.finder._calculate_value_overlap(empty1, empty2)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['overlap_percentage'], 0)
        self.assertEqual(result['jaccard_similarity'], 0)
        self.assertIn('both_empty', result['overlap_details'])


if __name__ == '__main__':
    unittest.main()
