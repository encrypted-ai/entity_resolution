# from langchain.graphs import NodeGraph  # Commented out to avoid import issues
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from fuzzywuzzy import fuzz
import pandas as pd  # Assuming you'll work with Pandas DataFrames
import numpy as np
from typing import Dict, List, Tuple, Any, Union

class SimilarColumnFinder:
    """
    A comprehensive entity resolution tool that identifies semantically similar columns 
    between two datasets using a combination of Large Language Model (LLM) analysis 
    and fuzzy string matching techniques.
    
    Entity Resolution Context:
    -------------------------
    Entity resolution is the process of identifying when different records refer to 
    the same real-world entity. A critical first step is identifying which columns 
    in different datasets contain the same type of information, even when they have 
    different names (e.g., 'cust_id' vs 'customer_number').
    
    This class solves the column alignment problem by:
    1. Using an LLM to generate semantic descriptions of what each column represents
    2. Comparing these descriptions using fuzzy matching to find columns with similar meanings
    3. Providing similarity scores to help users make informed decisions about column mappings
    
    Methodology:
    -----------
    The two-stage approach (LLM + fuzzy matching) is crucial because:
    - Column names alone are often insufficient (e.g., 'id' could mean many things)
    - LLM descriptions capture semantic meaning beyond surface-level naming
    - Fuzzy matching handles variations in description phrasing while maintaining precision
    - The combination provides both accuracy and explainability for the matching process

    Args:
        llm: A Langchain LLM instance (e.g., OpenAI) used to generate semantic 
             descriptions of column purposes. Defaults to OpenAI with low temperature 
             for consistent, factual descriptions.
        column_description_prompt: A PromptTemplate that guides the LLM in generating 
                                 consistent, comparable column descriptions. If None, 
                                 uses a default template optimized for entity resolution.
    
    Example Use Cases:
        - Merging customer databases from different systems
        - Aligning product catalogs from multiple vendors
        - Preparing datasets for data integration or migration
        - Identifying duplicate or similar data sources
    """

    def __init__(self, llm=OpenAI(temperature=0.1), column_description_prompt=None):
        """
        Initializes the SimilarColumnFinder with essential components for entity resolution.
        
        Configuration Strategy:
        ----------------------
        The initialization sets up two critical components that work together to enable 
        accurate entity resolution:
        
        1. LLM CONFIGURATION: Uses a low temperature (0.1) to ensure consistent, factual 
           column descriptions rather than creative interpretations. This consistency 
           is crucial for reliable fuzzy matching.
           
        2. PROMPT ENGINEERING: Employs a carefully crafted prompt template that guides 
           the LLM to generate standardized, comparable descriptions optimized for 
           entity resolution tasks.
        
        Args:
            llm: Large Language Model instance for generating semantic descriptions.
                 Default uses OpenAI with temperature=0.1 for consistency.
            column_description_prompt: Custom prompt template for column analysis.
                                     If None, uses an optimized default template.
                                     
        Raises:
            TypeError: If llm is not a valid LLM instance
            ValueError: If column_description_prompt is not a valid PromptTemplate
        """
        # Validate LLM instance
        if llm is None:
            raise ValueError("LLM instance cannot be None")
        if not hasattr(llm, 'invoke'):
            raise TypeError("LLM must have an 'invoke' method")
            
        # Store the LLM instance for semantic column description generation
        self.llm = llm
        
        # Set up the prompt template for consistent column description generation
        if column_description_prompt is None:
            # Use our optimized default prompt template designed specifically for entity resolution
            # This prompt is engineered to generate consistent, comparable descriptions
             self.column_description_prompt = PromptTemplate.from_template(
                """
                You are an expert data engineer analyzing column schemas for entity resolution.
                Given a column name, provide a concise and descriptive explanation of what the 
                column likely represents, focusing on the data type and business purpose.
                
                Your description should be:
                - Consistent in format and terminology
                - Focused on semantic meaning rather than technical implementation
                - Suitable for comparison with other column descriptions
                - Clear about the likely data type and purpose
                
                Example:
                Column Name: 'cust_id'
                Explanation: Customer identifier, likely a unique integer value for each customer.

                Column Name: {column_name}
                Explanation: 
                """
            )
        else:
            # Validate custom prompt template
            if column_description_prompt is None or not hasattr(column_description_prompt, 'format'):
                raise TypeError("column_description_prompt must be a PromptTemplate with a 'format' method")
            # Use the custom prompt template provided by the user
            self.column_description_prompt = column_description_prompt

    def describe_columns(self, df, columns):
        """
        Generates semantic descriptions for a list of columns using the LLM to enable 
        intelligent column matching in entity resolution workflows.
        
        Purpose in Entity Resolution:
        ----------------------------
        This method is the foundation of semantic column matching. Raw column names 
        (e.g., 'id', 'name', 'date') are often ambiguous and can represent vastly 
        different concepts across datasets. By generating standardized, descriptive 
        explanations of what each column likely contains, we create a common semantic 
        vocabulary that can be compared across different datasets.
        
        How It Contributes to the Final Output:
        --------------------------------------
        1. STANDARDIZATION: Converts diverse column naming conventions into consistent, 
           comparable descriptions (e.g., 'cust_id' and 'customer_number' both become 
           descriptions about customer identifiers)
        
        2. CONTEXT ENRICHMENT: The LLM leverages its training to infer the likely 
           purpose and data type of columns based on naming patterns and conventions
        
        3. SEMANTIC FOUNDATION: These descriptions become the basis for fuzzy matching, 
           allowing the algorithm to identify columns with similar purposes even when 
           their names are completely different
        
        4. HUMAN INTERPRETABILITY: The generated descriptions help users understand 
           and validate the matching decisions made by the algorithm

        Args:
            df (pd.DataFrame): The Pandas DataFrame containing the columns. While not 
                             directly used in the current implementation, it's available 
                             for future enhancements that might analyze data types, 
                             sample values, or statistical properties to improve descriptions.
            columns (list): A list of column names to generate descriptions for. Each 
                          column name will be processed individually to create focused, 
                          specific descriptions.

        Returns:
            dict: A dictionary mapping each column name to its LLM-generated description.
                 These descriptions are designed to be:
                 - Consistent in format and style
                 - Semantically rich and descriptive
                 - Comparable across different datasets
                 - Human-readable for validation purposes
                 
        Raises:
            TypeError: If df is not a pandas DataFrame or columns is not a list
            ValueError: If df is empty or columns list is empty or contains invalid column names
        
        Example Flow:
            Input: ['cust_id', 'full_name', 'order_date']
            Processing: Each column name -> LLM prompt -> semantic description
            Output: {
                'cust_id': 'Customer identifier, likely a unique integer...',
                'full_name': 'Customer full name, likely a string containing...',
                'order_date': 'Order date, likely a date/datetime field...'
            }
        """
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        if not isinstance(columns, list):
            raise TypeError("columns must be a list")
        if not columns:
            raise ValueError("columns list cannot be empty")
        
        # Validate that all columns exist in the DataFrame
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are not found in the DataFrame: {missing_columns}")
        
        descriptions = {}
        
        # Process each column individually to ensure focused, accurate descriptions
        for col in columns:
            try:
                # Format the column name into our standardized prompt template
                # This ensures consistent questioning style across all columns
                prompt = self.column_description_prompt.format(column_name=col)
                
                # Invoke the LLM to generate a semantic description
                # The low temperature setting ensures consistent, factual responses
                description = self.llm.invoke(prompt)
                
                # Store the description for later comparison with other datasets
                descriptions[col] = description
                
            except Exception as e:
                raise RuntimeError(f"Failed to generate description for column '{col}': {str(e)}")
            
        return descriptions


    def find_similar_columns(self, df1, df2, threshold=80):
        """
        Executes the complete entity resolution column matching pipeline by combining 
        LLM-generated semantic descriptions with fuzzy string matching to identify 
        columns that likely represent the same type of information across datasets.
        
        Core Entity Resolution Strategy:
        -------------------------------
        This method implements a sophisticated two-phase approach that addresses the 
        fundamental challenge in entity resolution: determining when columns with 
        different names contain the same type of information.
        
        Phase 1 - Semantic Understanding:
        - Generate rich, contextual descriptions of what each column represents
        - Transform column names into standardized semantic descriptions
        - Create a common vocabulary for comparison across datasets
        
        Phase 2 - Intelligent Matching:
        - Use fuzzy string matching on semantic descriptions (not column names)
        - Apply configurable similarity thresholds for precision control
        - Generate confidence scores to support human decision-making
        
        Why This Approach Works for Entity Resolution:
        ---------------------------------------------
        1. HANDLES NAMING VARIATIONS: Columns like 'cust_id', 'customer_number', 
           'client_identifier' all generate similar descriptions about customer IDs
        
        2. CAPTURES SEMANTIC MEANING: Goes beyond surface-level string similarity 
           to understand what data the columns actually contain
        
        3. PROVIDES CONFIDENCE METRICS: Similarity scores help users decide which 
           matches to trust and which need manual review
        
        4. SCALES ACROSS DOMAINS: Works for any type of data (customer, product, 
           transaction, etc.) without domain-specific rules

        Args:
            df1 (pd.DataFrame): The first dataset to analyze. All columns will be 
                              described and compared against df2's columns.
            df2 (pd.DataFrame): The second dataset to analyze. All columns will be 
                              described and compared against df1's columns.
            threshold (int): Minimum fuzzy matching score (0-100) to consider columns 
                           similar. Higher values = more conservative matching.
                           - 90+: Very conservative, only near-identical descriptions
                           - 80-89: Balanced, catches most relevant matches
                           - 70-79: Liberal, may include false positives
                           - <70: Very liberal, high false positive risk

        Returns:
            list: A list of tuples (col1, col2, score) representing potential column 
                 matches, where:
                 - col1: Column name from df1
                 - col2: Column name from df2  
                 - score: Similarity score (0-100) indicating confidence
                 
                 Results are sorted by similarity score (highest first) to prioritize 
                 the most confident matches for user review.
                 
        Raises:
            TypeError: If df1 or df2 are not pandas DataFrames
            ValueError: If df1 or df2 are empty, or threshold is not within valid range
        
        Entity Resolution Workflow:
            1. Extract column lists from both datasets
            2. Generate semantic descriptions for all columns using LLM
            3. Perform exhaustive pairwise comparison of descriptions
            4. Apply fuzzy matching with configurable threshold
            5. Return ranked list of potential matches with confidence scores
        
        Example:
            df1 columns: ['cust_id', 'name', 'signup_date']
            df2 columns: ['customer_number', 'full_name', 'registration_date']
            
            Result: [
                ('cust_id', 'customer_number', 92),
                ('name', 'full_name', 88),
                ('signup_date', 'registration_date', 85)
            ]
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
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number")
        if not (0 <= threshold <= 100):
            raise ValueError("threshold must be between 0 and 100")
        
        # STEP 1: Extract column metadata from both datasets
        # This creates the foundation for our comparison matrix
        columns1 = df1.columns.tolist()
        columns2 = df2.columns.tolist()

        # STEP 2: Generate semantic descriptions for all columns
        # This is the key innovation - we compare meanings, not names
        print(f"Generating semantic descriptions for {len(columns1)} columns in dataset 1...")
        descriptions1 = self.describe_columns(df1, columns1)
        
        print(f"Generating semantic descriptions for {len(columns2)} columns in dataset 2...")
        descriptions2 = self.describe_columns(df2, columns2)

        # Initialize container for potential matches
        similar_columns = []

        # STEP 3: Exhaustive pairwise comparison of semantic descriptions
        # We compare every column in df1 against every column in df2
        # This ensures we don't miss any potential matches due to ordering
        print(f"Comparing {len(columns1)} x {len(columns2)} = {len(columns1) * len(columns2)} column pairs...")
        
        for col1 in columns1:
            for col2 in columns2:
                # Retrieve the semantic descriptions for this column pair
                desc1 = descriptions1.get(col1)
                desc2 = descriptions2.get(col2)

                # Only proceed if both descriptions were successfully generated
                if desc1 and desc2:
                    # STEP 4: Apply fuzzy string matching to semantic descriptions
                    # We strip whitespace to ensure clean comparison and avoid 
                    # spurious differences due to formatting variations
                    score = fuzz.ratio(desc1.strip(), desc2.strip())
                    
                    # STEP 5: Apply threshold filter to control match quality
                    # Only matches above the threshold are considered viable
                    if score >= threshold:
                        similar_columns.append((col1, col2, score))

        # STEP 6: Sort results by confidence score (highest first)
        # This helps users focus on the most confident matches first
        similar_columns.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(similar_columns)} potential column matches above threshold {threshold}")
        return similar_columns

    def _analyze_data_type_compatibility(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """
        Analyzes data type compatibility between two pandas Series.
        
        This helper method performs deep data type analysis to determine if two columns
        contain compatible data types, which is crucial for validating column matches
        in entity resolution workflows.
        
        Args:
            series1 (pd.Series): First column data
            series2 (pd.Series): Second column data
            
        Returns:
            dict: Detailed compatibility analysis containing:
                - 'compatible': Boolean indicating if types are compatible
                - 'series1_type': Detected data type of first series
                - 'series2_type': Detected data type of second series
                - 'compatibility_score': Numeric score (0-100) of type compatibility
                - 'type_conversion_possible': Whether automatic conversion is possible
                - 'analysis_details': Additional technical details
        """
        def get_enhanced_dtype(series):
            """Enhanced data type detection that goes beyond pandas dtypes"""
            # Handle null/empty series
            if series.empty or series.isna().all():
                return 'empty'
            
            # Remove nulls for analysis
            clean_series = series.dropna()
            
            # Check for numeric types (including strings that could be numbers)
            try:
                pd.to_numeric(clean_series, errors='raise')
                if clean_series.dtype in ['int64', 'int32', 'float64', 'float32']:
                    return 'numeric'
                else:
                    return 'numeric_string'  # Numbers stored as strings
            except (ValueError, TypeError):
                pass
            
            # Check for datetime types
            try:
                pd.to_datetime(clean_series, errors='raise')
                return 'datetime'
            except (ValueError, TypeError):
                pass
            
            # Check for boolean types
            if clean_series.dtype == 'bool':
                return 'boolean'
            try:
                unique_str_vals = [str(val).lower() for val in clean_series.unique()]
                if set(unique_str_vals) <= {'true', 'false', '1', '0', 'yes', 'no'}:
                    return 'boolean'
            except:
                pass
            
            # Check for categorical/enum types (limited unique values)
            unique_ratio = len(clean_series.unique()) / len(clean_series)
            if unique_ratio < 0.1 and len(clean_series.unique()) < 50:
                return 'categorical'
            
            # Default to string/text
            return 'text'
        
        # Analyze both series
        type1 = get_enhanced_dtype(series1)
        type2 = get_enhanced_dtype(series2)
        
        # Define compatibility matrix
        compatibility_matrix = {
            ('numeric', 'numeric'): 100,
            ('numeric', 'numeric_string'): 90,
            ('numeric_string', 'numeric'): 90,
            ('numeric_string', 'numeric_string'): 95,
            ('datetime', 'datetime'): 100,
            ('datetime', 'text'): 60,  # Might be parseable
            ('text', 'datetime'): 60,
            ('boolean', 'boolean'): 100,
            ('boolean', 'categorical'): 70,
            ('categorical', 'categorical'): 85,
            ('categorical', 'text'): 75,
            ('text', 'text'): 90,
            ('empty', 'empty'): 0,
        }
        
        # Calculate compatibility score
        type_pair = (type1, type2)
        reverse_pair = (type2, type1)
        
        if type_pair in compatibility_matrix:
            score = compatibility_matrix[type_pair]
        elif reverse_pair in compatibility_matrix:
            score = compatibility_matrix[reverse_pair]
        else:
            # Default low compatibility for unmatched types
            score = 30
        
        # Determine if conversion is possible
        convertible_pairs = {
            ('numeric_string', 'numeric'),
            ('numeric', 'numeric_string'),
            ('text', 'datetime'),
            ('datetime', 'text'),
            ('boolean', 'categorical'),
            ('categorical', 'boolean')
        }
        
        conversion_possible = (type_pair in convertible_pairs or 
                             reverse_pair in convertible_pairs or 
                             score >= 80)
        
        return {
            'compatible': score >= 70,
            'series1_type': type1,
            'series2_type': type2,
            'compatibility_score': score,
            'type_conversion_possible': conversion_possible,
            'analysis_details': {
                'series1_pandas_dtype': str(series1.dtype),
                'series2_pandas_dtype': str(series2.dtype),
                'series1_unique_count': series1.nunique(),
                'series2_unique_count': series2.nunique(),
                'series1_null_count': series1.isna().sum(),
                'series2_null_count': series2.isna().sum()
            }
        }

    def _calculate_value_overlap(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """
        Calculates the overlap of actual values between two pandas Series.
        
        This method analyzes the intersection of unique values between columns
        to determine if they contain similar data distributions, which helps
        validate whether semantically similar columns actually contain 
        compatible data.
        
        Args:
            series1 (pd.Series): First column data
            series2 (pd.Series): Second column data
            
        Returns:
            dict: Value overlap analysis containing:
                - 'overlap_percentage': Percentage of values that overlap
                - 'jaccard_similarity': Jaccard coefficient of the two sets
                - 'common_values': Set of values present in both columns
                - 'unique_to_series1': Values only in first series
                - 'unique_to_series2': Values only in second series
                - 'overlap_details': Additional statistical information
        """
        # Handle empty series
        if series1.empty and series2.empty:
            return {
                'overlap_percentage': 0,
                'jaccard_similarity': 0,
                'common_values': set(),
                'unique_to_series1': set(),
                'unique_to_series2': set(),
                'overlap_details': {'both_empty': True}
            }
        
        # Get unique values, handling NaN appropriately
        values1 = set(series1.dropna().astype(str).unique())
        values2 = set(series2.dropna().astype(str).unique())
        
        # Calculate intersections and differences
        common_values = values1.intersection(values2)
        unique_to_1 = values1 - values2
        unique_to_2 = values2 - values1
        total_unique = values1.union(values2)
        
        # Calculate similarity metrics
        if len(total_unique) > 0:
            jaccard_similarity = len(common_values) / len(total_unique)
            overlap_percentage = (len(common_values) / max(len(values1), len(values2))) * 100
        else:
            jaccard_similarity = 0
            overlap_percentage = 0
        
        # Additional overlap analysis
        overlap_details = {
            'series1_unique_count': len(values1),
            'series2_unique_count': len(values2),
            'common_count': len(common_values),
            'total_unique_count': len(total_unique),
            'series1_only_count': len(unique_to_1),
            'series2_only_count': len(unique_to_2)
        }
        
        # For large sets, only return sample of values to avoid memory issues
        max_sample_size = 100
        if len(common_values) > max_sample_size:
            common_values_sample = set(list(common_values)[:max_sample_size])
        else:
            common_values_sample = common_values
            
        if len(unique_to_1) > max_sample_size:
            unique_to_1_sample = set(list(unique_to_1)[:max_sample_size])
        else:
            unique_to_1_sample = unique_to_1
            
        if len(unique_to_2) > max_sample_size:
            unique_to_2_sample = set(list(unique_to_2)[:max_sample_size])
        else:
            unique_to_2_sample = unique_to_2
        
        return {
            'overlap_percentage': round(overlap_percentage, 2),
            'jaccard_similarity': round(jaccard_similarity, 4),
            'common_values': common_values_sample,
            'unique_to_series1': unique_to_1_sample,
            'unique_to_series2': unique_to_2_sample,
            'overlap_details': overlap_details
        }

    def validate_column_matches(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              column_matches: List[Tuple[str, str, float]], 
                              sample_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Validates column matches by analyzing data types and value overlaps between 
        the actual column data. This provides additional validation beyond semantic 
        similarity to ensure matched columns are truly compatible for entity resolution.
        
        Purpose in Entity Resolution:
        ----------------------------
        While semantic analysis identifies columns that conceptually represent the 
        same information, this validation step ensures the actual data is compatible:
        
        1. DATA TYPE VALIDATION: Confirms that matched columns contain compatible 
           data types (e.g., both numeric, both dates, etc.)
           
        2. VALUE OVERLAP ANALYSIS: Measures how much actual data overlap exists 
           between columns, which indicates data quality and compatibility
           
        3. COMPATIBILITY SCORING: Provides quantitative measures to help users 
           prioritize which matches to trust for integration
           
        4. INTEGRATION READINESS: Identifies potential data transformation needs 
           before attempting to merge datasets
        
        Why This Validation is Critical:
        ------------------------------
        - PREVENTS DATA CORRUPTION: Catching type mismatches before integration
        - IMPROVES MATCH QUALITY: Additional evidence for semantic matches
        - REDUCES FALSE POSITIVES: Eliminates semantically similar but data-incompatible columns
        - GUIDES DATA PREPARATION: Identifies what preprocessing is needed
        
        Args:
            df1 (pd.DataFrame): First dataset containing the source columns
            df2 (pd.DataFrame): Second dataset containing the target columns  
            column_matches (List[Tuple[str, str, float]]): Output from find_similar_columns()
                                                         containing (col1, col2, similarity_score) tuples
            sample_size (int): Number of rows to sample for analysis (default 1000).
                             Larger samples provide more accurate results but take longer.
                             
        Returns:
            List[Dict[str, Any]]: Detailed validation results for each column match, containing:
                - 'col1': Column name from first dataset
                - 'col2': Column name from second dataset  
                - 'semantic_similarity': Original similarity score from find_similar_columns
                - 'data_type_analysis': Detailed type compatibility analysis
                - 'value_overlap_analysis': Value intersection and overlap metrics
                - 'overall_validation_score': Combined score considering all factors
                - 'recommendation': Human-readable recommendation for this match
                - 'validation_flags': List of potential issues or concerns
                
        Example Usage:
            # First find semantic matches
            matches = finder.find_similar_columns(df1, df2, threshold=80)
            
            # Then validate with actual data
            validated_matches = finder.validate_column_matches(df1, df2, matches)
            
            # Review results
            for match in validated_matches:
                print(f"Match: {match['col1']} <-> {match['col2']}")
                print(f"Overall Score: {match['overall_validation_score']}")
                print(f"Recommendation: {match['recommendation']}")
        
        Validation Workflow:
            1. Sample data from both columns for performance
            2. Analyze data type compatibility and conversion requirements
            3. Calculate value overlap and intersection statistics  
            4. Generate combined validation score factoring all evidence
            5. Provide actionable recommendations for each match
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
        if not isinstance(sample_size, int):
            raise TypeError("sample_size must be an integer")
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        
        # Validate column_matches structure
        for i, match in enumerate(column_matches):
            if not isinstance(match, tuple) or len(match) != 3:
                raise ValueError(f"column_matches[{i}] must be a tuple of (col1, col2, score)")
            col1, col2, score = match
            if not isinstance(col1, str) or not isinstance(col2, str):
                raise ValueError(f"column_matches[{i}] must contain string column names")
            if not isinstance(score, (int, float)):
                raise ValueError(f"column_matches[{i}] must contain a numeric score")
        
        print(f"Validating {len(column_matches)} column matches with data analysis...")
        
        validated_results = []
        
        for col1, col2, semantic_score in column_matches:
            print(f"  Validating match: {col1} <-> {col2}")
            
            # Sample data for performance (handle small datasets gracefully)
            sample_size_actual = min(sample_size, len(df1), len(df2))
            
            if sample_size_actual < len(df1):
                df1_sample = df1.sample(n=sample_size_actual, random_state=42)
                df2_sample = df2.sample(n=sample_size_actual, random_state=42)
            else:
                df1_sample = df1.copy()
                df2_sample = df2.copy()
            
            # Extract the specific columns for analysis
            try:
                series1 = df1_sample[col1]
                series2 = df2_sample[col2]
            except KeyError as e:
                # Handle case where column doesn't exist
                validation_result = {
                    'col1': col1,
                    'col2': col2,
                    'semantic_similarity': semantic_score,
                    'data_type_analysis': {'error': f'Column not found: {str(e)}'},
                    'value_overlap_analysis': {'error': f'Column not found: {str(e)}'},
                    'overall_validation_score': 0,
                    'recommendation': 'ERROR: Column not found in dataset',
                    'validation_flags': ['COLUMN_NOT_FOUND']
                }
                validated_results.append(validation_result)
                continue
            
            # Perform data type analysis
            type_analysis = self._analyze_data_type_compatibility(series1, series2)
            
            # Perform value overlap analysis  
            overlap_analysis = self._calculate_value_overlap(series1, series2)
            
            # Calculate overall validation score
            # Weighted combination of semantic similarity, type compatibility, and value overlap
            weights = {
                'semantic': 0.4,     # Semantic similarity weight
                'type_compat': 0.35, # Data type compatibility weight  
                'value_overlap': 0.25 # Value overlap weight
            }
            
            overall_score = (
                weights['semantic'] * semantic_score +
                weights['type_compat'] * type_analysis['compatibility_score'] +
                weights['value_overlap'] * overlap_analysis['overlap_percentage']
            )
            
            # Generate recommendation based on overall score and specific criteria
            validation_flags = []
            
            # Check for potential issues
            if not type_analysis['compatible']:
                validation_flags.append('INCOMPATIBLE_DATA_TYPES')
            
            if overlap_analysis['overlap_percentage'] < 10:
                validation_flags.append('LOW_VALUE_OVERLAP')
                
            if type_analysis['analysis_details']['series1_null_count'] / len(series1) > 0.5:
                validation_flags.append('HIGH_NULL_RATE_SERIES1')
                
            if type_analysis['analysis_details']['series2_null_count'] / len(series2) > 0.5:
                validation_flags.append('HIGH_NULL_RATE_SERIES2')
            
            # Generate human-readable recommendation
            if overall_score >= 85:
                recommendation = "EXCELLENT MATCH: High confidence for data integration"
            elif overall_score >= 70:
                recommendation = "GOOD MATCH: Suitable for integration with minor validation"
            elif overall_score >= 55:
                recommendation = "FAIR MATCH: Requires careful review and potential data transformation"
            elif overall_score >= 40:
                recommendation = "POOR MATCH: Significant compatibility issues, manual review needed"
            else:
                recommendation = "REJECTED: Major incompatibilities, not recommended for integration"
            
            # Add specific guidance based on flags
            if 'INCOMPATIBLE_DATA_TYPES' in validation_flags:
                recommendation += " (Data type conversion required)"
            if 'LOW_VALUE_OVERLAP' in validation_flags:
                recommendation += " (Consider manual value mapping)"
            
            # Compile complete validation result
            validation_result = {
                'col1': col1,
                'col2': col2,
                'semantic_similarity': semantic_score,
                'data_type_analysis': type_analysis,
                'value_overlap_analysis': overlap_analysis,
                'overall_validation_score': round(overall_score, 2),
                'recommendation': recommendation,
                'validation_flags': validation_flags,
                'sample_size_used': sample_size_actual
            }
            
            validated_results.append(validation_result)
        
        # Sort results by overall validation score (highest first)
        validated_results.sort(key=lambda x: x['overall_validation_score'], reverse=True)
        
        print(f"Validation complete. {len([r for r in validated_results if r['overall_validation_score'] >= 70])} high-quality matches found.")
        
        return validated_results
