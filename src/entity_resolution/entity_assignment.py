"""
Entity Assignment Module for Entity Resolution
=============================================

This module provides classes for consolidating and assigning entities based on 
column matches from entity resolution. It handles the decision-making logic for 
whether to join or concatenate datasets based on match quality.

Classes:
    SemanticConsolidation: Analyzes datasets to derive semantic patterns and mappings
    EntityAssignment: Handles dataset consolidation and entity assignment logic

Dependencies:
    - pandas: For dataset manipulation
    - numpy: For statistical calculations
    - re: For regular expression pattern matching
    - collections: For frequency counting
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict


class SemanticConsolidation:
    """
    A class that analyzes datasets to derive semantic patterns, common prefixes/suffixes,
    and semantic mappings for intelligent column name consolidation.
    
    This class dynamically learns from the provided datasets rather than using 
    hardcoded patterns, making it adaptable to different domain-specific naming conventions.
    """
    
    def __init__(self, min_frequency_threshold=0.1):
        """
        Initialize the SemanticConsolidation analyzer.
        
        Args:
            min_frequency_threshold (float): Minimum frequency (as ratio) for a pattern 
                                           to be considered significant (default 0.1)
                                           
        Raises:
            ValueError: If min_frequency_threshold is not between 0 and 1
            TypeError: If min_frequency_threshold is not a number
        """
        if not isinstance(min_frequency_threshold, (int, float)):
            raise TypeError("min_frequency_threshold must be a number")
        if not (0 <= min_frequency_threshold <= 1):
            raise ValueError("min_frequency_threshold must be between 0 and 1")
            
        self.min_frequency_threshold = min_frequency_threshold
        self.semantic_mappings = {}
        self.common_prefixes = []
        self.common_suffixes = []
        self._analyzed_columns = set()
    
    def analyze_datasets(self, *datasets):
        """
        Analyze one or more datasets to derive semantic patterns.
        
        Args:
            *datasets: Variable number of pandas DataFrames to analyze
            
        Returns:
            dict: Analysis results with derived patterns
            
        Raises:
            ValueError: If no datasets provided or all datasets are empty
            TypeError: If any dataset is not a pandas DataFrame
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")
            
        all_columns = []
        
        # Collect all column names from all datasets
        for i, df in enumerate(datasets):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Dataset {i} must be a pandas DataFrame")
            if df.empty:
                continue  # Skip empty datasets but don't fail
            all_columns.extend(df.columns.tolist())
        
        if not all_columns:
            raise ValueError("All provided datasets are empty")
        
        # Remove duplicates while preserving order
        unique_columns = list(dict.fromkeys(all_columns))
        self._analyzed_columns.update(unique_columns)
        
        # Analyze patterns
        self._derive_common_prefixes(unique_columns)
        self._derive_common_suffixes(unique_columns)
        self._derive_semantic_mappings(unique_columns)
        
        return {
            'total_columns_analyzed': len(unique_columns),
            'semantic_mappings': self.semantic_mappings,
            'common_prefixes': self.common_prefixes,
            'common_suffixes': self.common_suffixes
        }
    
    def _derive_common_prefixes(self, columns):
        """
        Derive common prefixes from column names.
        
        Args:
            columns (list): List of column names to analyze
        """
        prefix_counter = Counter()
        
        for col in columns:
            col_lower = col.lower()
            # Extract potential prefixes (up to first underscore or first 8 chars)
            if '_' in col_lower:
                prefix = col_lower.split('_')[0] + '_'
                prefix_counter[prefix] += 1
            elif len(col_lower) > 4:
                # Try common prefix lengths
                for length in [3, 4, 5, 6]:
                    if length < len(col_lower):
                        prefix = col_lower[:length] + '_'
                        # Only consider if it looks like a meaningful prefix
                        if re.match(r'^[a-z]+_$', prefix):
                            prefix_counter[prefix] += 1
        
        # Filter prefixes by frequency threshold
        total_columns = len(columns)
        min_occurrences = max(1, int(total_columns * self.min_frequency_threshold))
        
        self.common_prefixes = [
            prefix for prefix, count in prefix_counter.items() 
            if count >= min_occurrences
        ]
        
        # Sort by frequency (most common first)
        self.common_prefixes.sort(
            key=lambda x: prefix_counter[x], reverse=True
        )
    
    def _derive_common_suffixes(self, columns):
        """
        Derive common suffixes from column names.
        
        Args:
            columns (list): List of column names to analyze
        """
        suffix_counter = Counter()
        
        for col in columns:
            col_lower = col.lower()
            # Extract potential suffixes (from last underscore or last 8 chars)
            if '_' in col_lower:
                suffix = '_' + col_lower.split('_')[-1]
                suffix_counter[suffix] += 1
            elif len(col_lower) > 4:
                # Try common suffix lengths
                for length in [2, 3, 4, 5]:
                    if length < len(col_lower):
                        suffix = '_' + col_lower[-length:]
                        # Only consider if it looks like a meaningful suffix
                        if re.match(r'^_[a-z]+$', suffix):
                            suffix_counter[suffix] += 1
        
        # Filter suffixes by frequency threshold
        total_columns = len(columns)
        min_occurrences = max(1, int(total_columns * self.min_frequency_threshold))
        
        self.common_suffixes = [
            suffix for suffix, count in suffix_counter.items() 
            if count >= min_occurrences
        ]
        
        # Sort by frequency (most common first)
        self.common_suffixes.sort(
            key=lambda x: suffix_counter[x], reverse=True
        )
    
    def _derive_semantic_mappings(self, columns):
        """
        Derive semantic mappings by clustering similar column names.
        
        Args:
            columns (list): List of column names to analyze
        """
        # Define base semantic categories with common variations
        base_categories = {
            'identifier': ['id', 'identifier', 'number', 'num', 'key', 'pk', 'uid'],
            'name': ['name', 'title', 'label', 'description', 'desc'],
            'date': ['date', 'time', 'timestamp', 'created', 'updated', 'modified'],
            'amount': ['amount', 'value', 'price', 'cost', 'total', 'sum', 'balance'],
            'address': ['address', 'location', 'addr', 'street', 'city', 'state'],
            'contact': ['phone', 'telephone', 'mobile', 'email', 'mail'],
            'status': ['status', 'state', 'condition', 'flag', 'active', 'enabled']
        }
        
        # Group columns by semantic similarity
        semantic_groups = defaultdict(list)
        
        for col in columns:
            col_lower = col.lower()
            # Remove common prefixes and suffixes for analysis
            cleaned_col = self._clean_column_for_semantic_analysis(col_lower)
            
            # Try to match with base categories
            matched = False
            for category, keywords in base_categories.items():
                if any(keyword in cleaned_col for keyword in keywords):
                    semantic_groups[category].append(col_lower)
                    matched = True
                    break
            
            # If no match found, create a new category based on cleaned name
            if not matched and cleaned_col:
                semantic_groups[cleaned_col].append(col_lower)
        
        # Convert groups to semantic mappings format
        self.semantic_mappings = {}
        for category, terms in semantic_groups.items():
            if len(terms) >= 2 or category in base_categories:  # Only keep if multiple terms or base category
                self.semantic_mappings[tuple(set(terms))] = category
    
    def _clean_column_for_semantic_analysis(self, col_name):
        """
        Clean column name for semantic analysis by removing common patterns.
        
        Args:
            col_name (str): Column name to clean
            
        Returns:
            str: Cleaned column name
        """
        cleaned = col_name.lower()
        
        # Remove numbers and special characters except underscores
        cleaned = re.sub(r'[0-9]+', '', cleaned)
        cleaned = re.sub(r'[^a-z_]', '', cleaned)
        
        # Remove known prefixes and suffixes (using current lists if available)
        if self.common_prefixes:
            for prefix in self.common_prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    break
        
        if self.common_suffixes:
            for suffix in self.common_suffixes:
                if cleaned.endswith(suffix):
                    cleaned = cleaned[:-len(suffix)]
                    break
        
        # Remove underscores and get the main semantic part
        cleaned = cleaned.strip('_')
        if '_' in cleaned:
            # Take the longest part if split by underscores
            parts = cleaned.split('_')
            cleaned = max(parts, key=len) if parts else cleaned
        
        return cleaned
    
    def get_semantic_label(self, col1, col2):
        """
        Generate a semantic label for two column names using derived patterns.
        
        Args:
            col1, col2: Column names to generate label for
            
        Returns:
            str: Semantically cleaner column label
        """
        # Convert to lowercase for comparison
        c1_lower = col1.lower()
        c2_lower = col2.lower()
        
        # Check if either column matches a derived semantic category
        for terms, clean_label in self.semantic_mappings.items():
            if any(term in c1_lower for term in terms) or any(term in c2_lower for term in terms):
                return clean_label
        
        # If no semantic mapping found, try to extract the most descriptive parts
        def clean_column_name(name):
            cleaned = name.lower()
            for prefix in self.common_prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    break
            for suffix in self.common_suffixes:
                if cleaned.endswith(suffix):
                    cleaned = cleaned[:-len(suffix)]
                    break
            return cleaned.strip('_')
        
        clean_c1 = clean_column_name(col1)
        clean_c2 = clean_column_name(col2)
        
        # If one is a subset of the other, use the shorter one
        if clean_c1 and clean_c2:
            if clean_c1 in clean_c2:
                return clean_c1
            elif clean_c2 in clean_c1:
                return clean_c2
        
        # If they share common words, use the shared portion
        words1 = set(clean_c1.split('_')) if clean_c1 else set()
        words2 = set(clean_c2.split('_')) if clean_c2 else set()
        common_words = words1.intersection(words2)
        
        if common_words:
            return '_'.join(sorted(common_words))
        
        # Fall back to the shorter original name
        return col1 if len(col1) <= len(col2) else col2


class EntityAssignment:
    """
    A class that handles the consolidation and assignment of entities based on 
    column matching results from entity resolution processes.
    
    This class implements intelligent decision-making for how to combine datasets:
    - JOIN: When column matches are strong (high confidence)
    - CONCATENATE: When column matches are weak (low confidence)
    
    The class provides semantic column renaming and maintains metadata about
    the consolidation process for transparency and debugging.
    """
    
    def __init__(self, match_threshold=0.7, semantic_consolidation=None):
        """
        Initialize the EntityAssignment with a match threshold and optional semantic consolidation.
        
        Args:
            match_threshold (float): Threshold for average match score (default 0.7)
            semantic_consolidation (SemanticConsolidation): Optional pre-configured semantic analyzer
            
        Raises:
            TypeError: If match_threshold is not a number or semantic_consolidation is invalid
            ValueError: If match_threshold is not between 0 and 1
        """
        if not isinstance(match_threshold, (int, float)):
            raise TypeError("match_threshold must be a number")
        if not (0 <= match_threshold <= 1):
            raise ValueError("match_threshold must be between 0 and 1")
        if semantic_consolidation is not None and not isinstance(semantic_consolidation, SemanticConsolidation):
            raise TypeError("semantic_consolidation must be a SemanticConsolidation instance or None")
            
        self.match_threshold = match_threshold
        self.semantic_consolidation = semantic_consolidation
    
    def consolidate_datasets_based_on_matches(self, df1, df2, column_matches, match_threshold=None):
        """
        Consolidates two datasets by either joining on matched columns or concatenating rows
        based on whether the average fuzzy match score of half the columns exceeds the threshold.
        
        Args:
            df1 (pd.DataFrame): First dataset
            df2 (pd.DataFrame): Second dataset  
            column_matches (list): List of tuples (col1, col2, score) from find_similar_columns
            match_threshold (float): Threshold for average match score (uses instance default if None)
            
        Returns:
            tuple: (consolidated_df, consolidation_method, metadata)
                - consolidated_df: The resulting consolidated dataset
                - consolidation_method: 'join' or 'concatenate'
                - metadata: Dictionary with consolidation details
                
        Raises:
            TypeError: If inputs are not of expected types
            ValueError: If DataFrames are empty or threshold is invalid
        """
        # Validate inputs
        if not isinstance(df1, pd.DataFrame):
            raise TypeError("df1 must be a pandas DataFrame")
        if not isinstance(df2, pd.DataFrame):
            raise TypeError("df2 must be a pandas DataFrame")
        if df1.empty:
            raise ValueError("df1 cannot be empty")
        if df2.empty:
            raise ValueError("df2 cannot be empty")
        if not isinstance(column_matches, list):
            raise TypeError("column_matches must be a list")
        
        if match_threshold is None:
            match_threshold = self.match_threshold
        else:
            if not isinstance(match_threshold, (int, float)):
                raise TypeError("match_threshold must be a number")
            if not (0 <= match_threshold <= 1):
                raise ValueError("match_threshold must be between 0 and 1")
            
        # Initialize or update semantic consolidation with current datasets
        if self.semantic_consolidation is None:
            self.semantic_consolidation = SemanticConsolidation()
        
        # Analyze datasets to derive semantic patterns
        self.semantic_consolidation.analyze_datasets(df1, df2)
            
        # Calculate the number of columns we need to consider (half of the smaller dataset)
        min_columns = min(len(df1.columns), len(df2.columns))
        required_matches = max(1, min_columns // 2)  # At least 1, but typically half
        
        # Sort matches by score (highest first) and take the top matches
        sorted_matches = sorted(column_matches, key=lambda x: x[2], reverse=True)
        top_matches = sorted_matches[:required_matches]
        
        # Calculate average score of top matches
        if top_matches:
            avg_score = np.mean([match[2] for match in top_matches])
            avg_score_normalized = avg_score / 100.0  # Convert from 0-100 to 0-1 scale
        else:
            avg_score_normalized = 0.0
        
        metadata = {
            'total_matches_found': len(column_matches),
            'matches_considered': len(top_matches), 
            'required_matches': required_matches,
            'average_match_score': avg_score_normalized,
            'threshold': match_threshold,
            'top_matches': top_matches
        }
        
        if avg_score_normalized >= match_threshold:
            # JOIN: High confidence matches, merge datasets on matched columns
            consolidated_df, join_metadata = self._join_datasets_on_matches(df1, df2, top_matches)
            metadata.update(join_metadata)
            return consolidated_df, 'join', metadata
        else:
            # CONCATENATE: Low confidence matches, stack datasets vertically
            consolidated_df, concat_metadata = self._concatenate_datasets_with_labels(df1, df2, top_matches)
            metadata.update(concat_metadata)
            return consolidated_df, 'concatenate', metadata
    
    def _join_datasets_on_matches(self, df1, df2, matches):
        """
        Joins two datasets on the matched columns with semantically cleaner labels.
        
        Args:
            df1, df2: DataFrames to join
            matches: List of (col1, col2, score) tuples
            
        Returns:
            tuple: (joined_df, metadata)
        """
        if not matches:
            # No matches to join on, return df1 as-is
            return df1.copy(), {'join_type': 'no_matches', 'join_columns': []}
        
        # Prepare datasets for joining
        df1_prep = df1.copy()
        df2_prep = df2.copy()
        
        # Create column mapping for cleaner labels
        column_mapping = {}
        join_columns = []
        
        for col1, col2, score in matches:
            # Generate semantically cleaner label for the matched columns
            clean_label = self._generate_semantic_label(col1, col2)
            
            # Rename columns to use the clean label
            df1_prep = df1_prep.rename(columns={col1: clean_label})
            df2_prep = df2_prep.rename(columns={col2: clean_label})
            
            column_mapping[f"{col1}|{col2}"] = clean_label
            join_columns.append(clean_label)
        
        # Perform the join on matched columns
        try:
            # Use outer join to preserve all data
            joined_df = pd.merge(df1_prep, df2_prep, on=join_columns, how='outer', suffixes=('_dataset1', '_dataset2'))
            
            metadata = {
                'join_type': 'outer_join',
                'join_columns': join_columns,
                'column_mapping': column_mapping,
                'original_df1_shape': df1.shape,
                'original_df2_shape': df2.shape,
                'result_shape': joined_df.shape
            }
            
            return joined_df, metadata
            
        except Exception as e:
            # If join fails, fall back to concatenation
            return self._concatenate_datasets_with_labels(df1, df2, matches)
    
    def _concatenate_datasets_with_labels(self, df1, df2, matches):
        """
        Concatenates two datasets vertically with dataset source labels and semantic column alignment.
        
        Args:
            df1, df2: DataFrames to concatenate
            matches: List of (col1, col2, score) tuples for column alignment
            
        Returns:
            tuple: (concatenated_df, metadata)
        """
        # Create copies to avoid modifying originals
        df1_prep = df1.copy()
        df2_prep = df2.copy()
        
        # Add source dataset identifier
        df1_prep['_dataset_source'] = 'dataset_1'
        df2_prep['_dataset_source'] = 'dataset_2'
        
        # Apply semantic column renaming based on matches
        column_mapping = {}
        
        for col1, col2, score in matches:
            clean_label = self._generate_semantic_label(col1, col2)
            
            # Rename columns to use semantic labels
            if col1 in df1_prep.columns:
                df1_prep = df1_prep.rename(columns={col1: clean_label})
                column_mapping[col1] = clean_label
                
            if col2 in df2_prep.columns:
                df2_prep = df2_prep.rename(columns={col2: clean_label})
                column_mapping[col2] = clean_label
        
        # Concatenate datasets
        concatenated_df = pd.concat([df1_prep, df2_prep], ignore_index=True, sort=False)
        
        metadata = {
            'concatenation_type': 'vertical_stack',
            'column_mapping': column_mapping,
            'original_df1_shape': df1.shape,
            'original_df2_shape': df2.shape,
            'result_shape': concatenated_df.shape,
            'matched_columns_aligned': len(matches)
        }
        
        return concatenated_df, metadata
    
    def _generate_semantic_label(self, col1, col2):
        """
        Generates a semantically cleaner label by combining information from both column names.
        Uses the SemanticConsolidation class to derive patterns from the datasets.
        
        Args:
            col1, col2: Column names to combine
            
        Returns:
            str: Semantically cleaner column label
        """
        # Use semantic consolidation if available, otherwise fall back to basic approach
        if self.semantic_consolidation is not None:
            return self.semantic_consolidation.get_semantic_label(col1, col2)
        
        # Fallback: basic approach when no semantic consolidation is available
        # This should rarely be reached after the constructor changes
        c1_lower = col1.lower()
        c2_lower = col2.lower()
        
        # Basic semantic categories (fallback only)
        basic_categories = {
            'identifier': ['id', 'identifier', 'number'],
            'name': ['name', 'title', 'label'], 
            'date': ['date', 'time', 'timestamp'],
            'amount': ['amount', 'value', 'price', 'cost'],
        }
        
        for category, terms in basic_categories.items():
            if any(term in c1_lower for term in terms) or any(term in c2_lower for term in terms):
                return category
        
        # Fall back to the shorter original name
        return col1 if len(col1) <= len(col2) else col2
