# Comprehensive Code Explanation: Entity Resolution System

This document provides a detailed, line-by-line explanation of every function in the entity resolution system, explaining how each piece of code contributes to the overall goal of intelligent column matching and dataset consolidation.

## Table of Contents

1. [Entity Resolution Core (`entity_resolution.py`)](#entity-resolution-core)
2. [Entity Assignment (`entity_assignment.py`)](#entity-assignment)
3. [Workflow Management (`workflow.py`)](#workflow-management)
4. [Example Usage (`example_usage.py`)](#example-usage)

---

## Entity Resolution Core (`entity_resolution.py`)

### Class: `SimilarColumnFinder`

The `SimilarColumnFinder` class is the heart of the entity resolution system. It uses a two-stage approach: LLM-based semantic analysis followed by fuzzy string matching to identify columns that represent the same information across different datasets.

#### Constructor: `__init__(self, llm=OpenAI(temperature=0.1), column_description_prompt=None)`

**Purpose**: Initialize the entity resolution system with LLM and prompt configuration.

**Line-by-line breakdown**:

```python
def __init__(self, llm=OpenAI(temperature=0.1), column_description_prompt=None):
```
- **Line 50**: Method signature defines two optional parameters
  - `llm`: Defaults to OpenAI with low temperature (0.1) for consistent, factual responses
  - `column_description_prompt`: Allows custom prompt engineering for specific domains

```python
self.llm = llm
```
- **Line 74**: Store the LLM instance as an instance variable for later use in semantic analysis
- This composition pattern allows flexibility in LLM choice while maintaining consistent interface

```python
if column_description_prompt is None:
```
- **Line 77**: Check if user provided custom prompt template
- If None, system will use optimized default prompt designed for entity resolution

```python
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
```
- **Lines 80-99**: Create default prompt template optimized for entity resolution
- The prompt is carefully engineered to:
  - Establish expert persona for domain knowledge
  - Request consistent formatting for reliable fuzzy matching
  - Focus on semantic meaning over technical details
  - Provide clear example to guide response format
  - Use placeholder `{column_name}` for dynamic column insertion

```python
else:
    self.column_description_prompt = column_description_prompt
```
- **Lines 100-102**: Use custom prompt if provided, allowing domain-specific customization

#### Method: `describe_columns(self, df, columns)`

**Purpose**: Generate semantic descriptions for a list of columns using LLM analysis to create comparable descriptions for fuzzy matching.

**Line-by-line breakdown**:

```python
def describe_columns(self, df, columns):
```
- **Line 104**: Method signature accepts DataFrame (for potential future analysis) and column list
- `df` parameter currently unused but available for future enhancements like data type analysis

```python
descriptions = {}
```
- **Line 159**: Initialize empty dictionary to store column name → description mappings
- Dictionary structure allows efficient lookup during comparison phase

```python
for col in columns:
```
- **Line 162**: Iterate through each column name individually
- Individual processing ensures focused, specific descriptions for each column

```python
prompt = self.column_description_prompt.format(column_name=col)
```
- **Line 165**: Insert current column name into prompt template
- The `.format()` method replaces `{column_name}` placeholder with actual column name
- Creates standardized prompt structure while customizing content for each column

```python
description = self.llm.invoke(prompt)
```
- **Line 169**: Send formatted prompt to LLM and get semantic description
- The low temperature setting (0.1) ensures consistent, factual responses
- `.invoke()` method handles the API call and returns the generated text

```python
descriptions[col] = description
```
- **Line 172**: Store the generated description with column name as key
- Creates mapping that will be used in fuzzy matching comparison

```python
return descriptions
```
- **Line 174**: Return complete dictionary of column descriptions
- Output format: `{'column_name': 'semantic_description', ...}`

#### Method: `find_similar_columns(self, df1, df2, threshold=80)`

**Purpose**: Execute the complete entity resolution pipeline by combining semantic descriptions with fuzzy matching to identify similar columns across datasets.

**Line-by-line breakdown**:

```python
def find_similar_columns(self, df1, df2, threshold=80):
```
- **Line 177**: Method signature with configurable similarity threshold
- `threshold=80` provides balanced matching (not too strict, not too loose)

```python
columns1 = df1.columns.tolist()
columns2 = df2.columns.tolist()
```
- **Lines 255-256**: Extract column names from both DataFrames
- `.tolist()` converts pandas Index to Python list for easier iteration
- Creates foundation for exhaustive pairwise comparison

```python
print(f"Generating semantic descriptions for {len(columns1)} columns in dataset 1...")
descriptions1 = self.describe_columns(df1, columns1)
```
- **Lines 260-261**: Generate descriptions for first dataset
- Progress message provides user feedback during potentially long LLM operations
- Calls previously defined `describe_columns` method

```python
print(f"Generating semantic descriptions for {len(columns2)} columns in dataset 2...")
descriptions2 = self.describe_columns(df2, columns2)
```
- **Lines 263-264**: Generate descriptions for second dataset
- Parallel structure to first dataset processing
- Two separate calls allow for different processing if needed

```python
similar_columns = []
```
- **Line 267**: Initialize list to store potential matches
- List structure preserves order and allows sorting by similarity score

```python
print(f"Comparing {len(columns1)} x {len(columns2)} = {len(columns1) * len(columns2)} column pairs...")
```
- **Line 272**: Inform user about comparison complexity
- Shows total number of pairwise comparisons to be performed
- Helps users understand processing time for large datasets

```python
for col1 in columns1:
    for col2 in columns2:
```
- **Lines 274-275**: Nested loops for exhaustive pairwise comparison
- Ensures no potential matches are missed due to column ordering
- O(n*m) complexity where n and m are column counts

```python
desc1 = descriptions1.get(col1)
desc2 = descriptions2.get(col2)
```
- **Lines 277-278**: Retrieve semantic descriptions for current column pair
- `.get()` method safely handles missing keys (returns None if not found)
- Defensive programming prevents KeyError exceptions

```python
if desc1 and desc2:
```
- **Line 281**: Verify both descriptions exist before comparison
- Prevents errors from missing or failed LLM generation
- Ensures quality control in the matching process

```python
score = fuzz.ratio(desc1.strip(), desc2.strip())
```
- **Line 285**: Apply fuzzy string matching to semantic descriptions
- `fuzz.ratio()` calculates Levenshtein distance-based similarity (0-100 scale)
- `.strip()` removes whitespace to avoid spurious differences
- Compares meanings, not column names - this is the key innovation

```python
if score >= threshold:
    similar_columns.append((col1, col2, score))
```
- **Lines 289-290**: Filter matches by threshold and store results
- Only high-confidence matches are retained
- Tuple format: (dataset1_column, dataset2_column, similarity_score)

```python
similar_columns.sort(key=lambda x: x[2], reverse=True)
```
- **Line 294**: Sort results by similarity score (highest first)
- `key=lambda x: x[2]` sorts by third element (score)
- `reverse=True` puts highest scores first for user prioritization

```python
print(f"Found {len(similar_columns)} potential column matches above threshold {threshold}")
return similar_columns
```
- **Lines 296-297**: Report results and return matches
- Provides user feedback on matching success
- Returns sorted list of tuples ready for further processing

#### Method: `_analyze_data_type_compatibility(self, series1, series2)`

**Purpose**: Perform deep data type analysis to determine compatibility between two pandas Series for validation.

**Line-by-line breakdown**:

```python
def _analyze_data_type_compatibility(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
```
- **Line 299**: Private method (prefix `_`) for internal data analysis
- Type hints specify pandas Series input and dictionary output
- Returns comprehensive compatibility analysis

```python
def get_enhanced_dtype(series):
    """Enhanced data type detection that goes beyond pandas dtypes"""
```
- **Lines 320-321**: Nested helper function for sophisticated type detection
- Goes beyond basic pandas dtypes to understand semantic data types

```python
if series.empty or series.isna().all():
    return 'empty'
```
- **Lines 323-324**: Handle edge case of empty or all-null series
- Prevents errors in subsequent analysis steps
- Returns special 'empty' type for compatibility matrix lookup

```python
clean_series = series.dropna()
```
- **Line 327**: Remove null values for accurate type analysis
- Null values would interfere with numeric and datetime detection
- Creates clean dataset for pattern analysis

```python
try:
    pd.to_numeric(clean_series, errors='raise')
    if clean_series.dtype in ['int64', 'int32', 'float64', 'float32']:
        return 'numeric'
    else:
        return 'numeric_string'  # Numbers stored as strings
except (ValueError, TypeError):
    pass
```
- **Lines 330-337**: Detect numeric types including numbers stored as strings
- `pd.to_numeric(errors='raise')` will throw exception if not convertible
- Distinguishes between true numeric types and string representations
- Exception handling allows fallthrough to other type checks

```python
try:
    pd.to_datetime(clean_series, errors='raise', infer_datetime_format=True)
    return 'datetime'
except (ValueError, TypeError):
    pass
```
- **Lines 340-344**: Detect datetime types using pandas inference
- `infer_datetime_format=True` helps recognize various date formats
- Catches both obvious and subtle datetime representations

```python
if clean_series.dtype == 'bool' or set(clean_series.unique().astype(str).str.lower()) <= {'true', 'false', '1', '0', 'yes', 'no'}:
    return 'boolean'
```
- **Lines 347-348**: Detect boolean types including text representations
- Checks both native boolean dtype and common text boolean values
- `<=` operator checks if unique values are subset of boolean indicators

```python
unique_ratio = len(clean_series.unique()) / len(clean_series)
if unique_ratio < 0.1 and len(clean_series.unique()) < 50:
    return 'categorical'
```
- **Lines 351-353**: Detect categorical data by uniqueness ratio
- Low unique ratio (< 10%) suggests categorical/enum data
- Additional constraint (< 50 unique values) prevents large text fields being categorized

```python
return 'text'
```
- **Line 356**: Default fallback for all other data types
- Covers general string/text data that doesn't fit other patterns

```python
type1 = get_enhanced_dtype(series1)
type2 = get_enhanced_dtype(series2)
```
- **Lines 359-360**: Apply enhanced type detection to both series
- Creates foundation for compatibility matrix lookup

```python
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
```
- **Lines 363-377**: Define compatibility scoring matrix
- Scores range from 0-100 based on how easily types can be converted
- Perfect matches (e.g., numeric-numeric) score 100
- Potential conversions (e.g., numeric_string-numeric) score 90
- Difficult conversions (e.g., datetime-text) score lower
- Domain knowledge encoded in scoring decisions

```python
type_pair = (type1, type2)
reverse_pair = (type2, type1)

if type_pair in compatibility_matrix:
    score = compatibility_matrix[type_pair]
elif reverse_pair in compatibility_matrix:
    score = compatibility_matrix[reverse_pair]
else:
    # Default low compatibility for unmatched types
    score = 30
```
- **Lines 380-389**: Look up compatibility score handling order independence
- Checks both (type1, type2) and (type2, type1) pairs
- Handles symmetric compatibility (numeric-text same as text-numeric)
- Default low score (30) for unknown type combinations

```python
convertible_pairs = {
    ('numeric_string', 'numeric'),
    ('numeric', 'numeric_string'),
    ('text', 'datetime'),
    ('datetime', 'text'),
    ('boolean', 'categorical'),
    ('categorical', 'boolean')
}
```
- **Lines 392-399**: Define which type pairs support automatic conversion
- Set structure allows efficient membership testing
- Includes bidirectional pairs for comprehensive coverage

```python
conversion_possible = (type_pair in convertible_pairs or 
                     reverse_pair in convertible_pairs or 
                     score >= 80)
```
- **Lines 401-403**: Determine if automatic conversion is feasible
- Checks explicit convertible pairs or high compatibility score
- Score >= 80 threshold indicates strong compatibility

```python
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
```
- **Lines 405-419**: Return comprehensive analysis results
- `compatible` boolean based on 70% threshold for practical compatibility
- Includes all derived types and scores for transparency
- `analysis_details` provides raw pandas information for debugging
- Structure supports both programmatic and human interpretation

#### Method: `_calculate_value_overlap(self, series1, series2)`

**Purpose**: Calculate intersection and overlap statistics between two data series to validate whether semantically similar columns contain compatible data.

**Line-by-line breakdown**:

```python
def _calculate_value_overlap(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
```
- **Line 421**: Private method for analyzing actual data value overlap
- Type hints ensure correct pandas Series input and structured output

```python
if series1.empty and series2.empty:
    return {
        'overlap_percentage': 0,
        'jaccard_similarity': 0,
        'common_values': set(),
        'unique_to_series1': set(),
        'unique_to_series2': set(),
        'overlap_details': {'both_empty': True}
    }
```
- **Lines 444-452**: Handle edge case where both series are empty
- Returns zero overlap with special flag for empty condition
- Prevents division by zero in subsequent calculations

```python
values1 = set(series1.dropna().astype(str).unique())
values2 = set(series2.dropna().astype(str).unique())
```
- **Lines 455-456**: Extract unique values as string sets for comparison
- `.dropna()` removes null values that would interfere with comparison
- `.astype(str)` normalizes all values to strings for consistent comparison
- `set()` creates unordered collection optimized for intersection operations

```python
common_values = values1.intersection(values2)
unique_to_1 = values1 - values2
unique_to_2 = values2 - values1
total_unique = values1.union(values2)
```
- **Lines 459-462**: Calculate set relationships for comprehensive analysis
- `intersection()` finds values present in both datasets
- Set subtraction (`-`) finds values unique to each dataset
- `union()` finds all unique values across both datasets

```python
if len(total_unique) > 0:
    jaccard_similarity = len(common_values) / len(total_unique)
    overlap_percentage = (len(common_values) / max(len(values1), len(values2))) * 100
else:
    jaccard_similarity = 0
    overlap_percentage = 0
```
- **Lines 465-470**: Calculate similarity metrics with division-by-zero protection
- Jaccard similarity: |intersection| / |union| (standard set similarity measure)
- Overlap percentage: intersection size relative to larger dataset
- Zero values for edge case where no unique values exist

```python
overlap_details = {
    'series1_unique_count': len(values1),
    'series2_unique_count': len(values2),
    'common_count': len(common_values),
    'total_unique_count': len(total_unique),
    'series1_only_count': len(unique_to_1),
    'series2_only_count': len(unique_to_2)
}
```
- **Lines 473-480**: Compile detailed statistics for transparency
- Provides raw counts for each set operation result
- Enables users to understand the basis for similarity calculations

```python
max_sample_size = 100
if len(common_values) > max_sample_size:
    common_values_sample = set(list(common_values)[:max_sample_size])
else:
    common_values_sample = common_values
```
- **Lines 483-487**: Limit returned values to prevent memory issues
- Large datasets could have thousands of unique values
- Sampling preserves representative examples while controlling memory usage
- Full counts still available in `overlap_details`

```python
return {
    'overlap_percentage': round(overlap_percentage, 2),
    'jaccard_similarity': round(jaccard_similarity, 4),
    'common_values': common_values_sample,
    'unique_to_series1': unique_to_1_sample,
    'unique_to_series2': unique_to_2_sample,
    'overlap_details': overlap_details
}
```
- **Lines 499-506**: Return comprehensive overlap analysis
- Rounded values for clean display while preserving precision
- Includes both summary metrics and sample data
- Structure supports both programmatic analysis and human review

#### Method: `validate_column_matches(self, df1, df2, column_matches, sample_size=1000)`

**Purpose**: Validate semantic column matches using actual data analysis to provide comprehensive compatibility assessment.

**Line-by-line breakdown**:

```python
def validate_column_matches(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                          column_matches: List[Tuple[str, str, float]], 
                          sample_size: int = 1000) -> List[Dict[str, Any]]:
```
- **Lines 508-510**: Method signature with comprehensive type hints
- Takes semantic matches from `find_similar_columns()` as input
- `sample_size=1000` balances analysis accuracy with performance

```python
print(f"Validating {len(column_matches)} column matches with data analysis...")
validated_results = []
```
- **Lines 580-582**: Initialize validation process with user feedback
- Empty list to collect comprehensive validation results

```python
for col1, col2, semantic_score in column_matches:
    print(f"  Validating match: {col1} <-> {col2}")
```
- **Lines 584-585**: Iterate through each semantic match for validation
- Progress feedback helps users track validation of large match sets

```python
sample_size_actual = min(sample_size, len(df1), len(df2))

if sample_size_actual < len(df1):
    df1_sample = df1.sample(n=sample_size_actual, random_state=42)
    df2_sample = df2.sample(n=sample_size_actual, random_state=42)
else:
    df1_sample = df1.copy()
    df2_sample = df2.copy()
```
- **Lines 588-595**: Handle sampling for performance while ensuring data integrity
- `min()` ensures sample size doesn't exceed dataset size
- `random_state=42` ensures reproducible sampling across runs
- Creates copies for small datasets to avoid sampling artifacts

```python
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
```
- **Lines 598-614**: Robust error handling for missing columns
- Defensive programming prevents crashes from inconsistent data
- Creates structured error result maintaining output format consistency
- `continue` skips to next match without breaking validation loop

```python
type_analysis = self._analyze_data_type_compatibility(series1, series2)
overlap_analysis = self._calculate_value_overlap(series1, series2)
```
- **Lines 617-620**: Perform comprehensive data analysis using helper methods
- Leverages previously defined type and overlap analysis functions
- Modular approach allows independent testing and validation

```python
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
```
- **Lines 624-634**: Calculate weighted overall validation score
- Semantic similarity weighted highest (40%) as primary indicator
- Type compatibility important (35%) for integration feasibility
- Value overlap provides additional evidence (25%)
- Weighted combination provides balanced assessment

```python
validation_flags = []

if not type_analysis['compatible']:
    validation_flags.append('INCOMPATIBLE_DATA_TYPES')

if overlap_analysis['overlap_percentage'] < 10:
    validation_flags.append('LOW_VALUE_OVERLAP')
    
if type_analysis['analysis_details']['series1_null_count'] / len(series1) > 0.5:
    validation_flags.append('HIGH_NULL_RATE_SERIES1')
    
if type_analysis['analysis_details']['series2_null_count'] / len(series2) > 0.5:
    validation_flags.append('HIGH_NULL_RATE_SERIES2')
```
- **Lines 637-650**: Identify potential data quality issues
- Flags provide specific warnings about compatibility problems
- Threshold-based detection (10% overlap, 50% null rate) based on practical experience
- List structure allows multiple flags per match

```python
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
```
- **Lines 653-662**: Generate human-readable recommendations
- Score ranges based on practical integration experience
- Provides clear guidance for data integration decisions

```python
if 'INCOMPATIBLE_DATA_TYPES' in validation_flags:
    recommendation += " (Data type conversion required)"
if 'LOW_VALUE_OVERLAP' in validation_flags:
    recommendation += " (Consider manual value mapping)"
```
- **Lines 665-668**: Append specific guidance based on detected issues
- Contextual advice helps users understand what actions are needed

```python
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
```
- **Lines 671-683**: Compile comprehensive validation result
- Includes all analysis components for transparency
- Structured format supports both programmatic and human consumption

```python
validated_results.sort(key=lambda x: x['overall_validation_score'], reverse=True)

print(f"Validation complete. {len([r for r in validated_results if r['overall_validation_score'] >= 70])} high-quality matches found.")

return validated_results
```
- **Lines 686-690**: Finalize and return validation results
- Sorts by overall score to prioritize best matches
- Summary feedback on high-quality matches (>= 70% score)
- Returns complete validation dataset for further processing

---

## Entity Assignment (`entity_assignment.py`)

### Class: `SemanticConsolidation`

The `SemanticConsolidation` class analyzes datasets to derive semantic patterns, common prefixes/suffixes, and semantic mappings for intelligent column name consolidation.

#### Constructor: `__init__(self, min_frequency_threshold=0.1)`

**Purpose**: Initialize the semantic analyzer with configurable frequency threshold for pattern recognition.

**Line-by-line breakdown**:

```python
def __init__(self, min_frequency_threshold=0.1):
```
- **Line 35**: Constructor with configurable minimum frequency threshold
- Default 0.1 means patterns must appear in at least 10% of columns to be considered significant

```python
self.min_frequency_threshold = min_frequency_threshold
self.semantic_mappings = {}
self.common_prefixes = []
self.common_suffixes = []
self._analyzed_columns = set()
```
- **Lines 43-47**: Initialize instance variables for pattern storage
- `semantic_mappings`: Dictionary storing semantic category relationships
- `common_prefixes/suffixes`: Lists of frequently occurring naming patterns
- `_analyzed_columns`: Set tracking which columns have been processed (private variable)

#### Method: `analyze_datasets(self, *datasets)`

**Purpose**: Analyze one or more datasets to derive semantic patterns and naming conventions.

**Line-by-line breakdown**:

```python
def analyze_datasets(self, *datasets):
```
- **Line 49**: Method accepts variable number of datasets using `*args` syntax
- Flexible interface allows analysis of 2+ datasets simultaneously

```python
all_columns = []

for df in datasets:
    if isinstance(df, pd.DataFrame):
        all_columns.extend(df.columns.tolist())
```
- **Lines 59-64**: Collect all column names from all provided datasets
- `isinstance()` check ensures only DataFrames are processed
- `.extend()` flattens column lists from multiple datasets into single list

```python
unique_columns = list(dict.fromkeys(all_columns))
self._analyzed_columns.update(unique_columns)
```
- **Lines 67-68**: Remove duplicates while preserving order
- `dict.fromkeys()` trick removes duplicates faster than set() while maintaining order
- Updates internal tracking of analyzed columns

```python
self._derive_common_prefixes(unique_columns)
self._derive_common_suffixes(unique_columns)
self._derive_semantic_mappings(unique_columns)
```
- **Lines 71-73**: Execute the three core pattern analysis functions
- Modular approach allows independent testing of each pattern type
- Order matters: prefixes and suffixes derived before semantic mappings

```python
return {
    'total_columns_analyzed': len(unique_columns),
    'semantic_mappings': self.semantic_mappings,
    'common_prefixes': self.common_prefixes,
    'common_suffixes': self.common_suffixes
}
```
- **Lines 75-80**: Return analysis summary for transparency and debugging
- Includes both statistics and derived patterns
- Enables validation of pattern extraction quality

#### Method: `_derive_common_prefixes(self, columns)`

**Purpose**: Extract and identify common prefixes from column names based on frequency analysis.

**Line-by-line breakdown**:

```python
def _derive_common_prefixes(self, columns):
```
- **Line 82**: Private method focusing on prefix pattern extraction

```python
prefix_counter = Counter()
```
- **Line 89**: Initialize Counter for frequency tracking
- Counter class provides efficient counting and frequency analysis

```python
for col in columns:
    col_lower = col.lower()
```
- **Lines 91-92**: Process each column with case normalization
- Lowercase conversion ensures case-insensitive pattern matching

```python
if '_' in col_lower:
    prefix = col_lower.split('_')[0] + '_'
    prefix_counter[prefix] += 1
```
- **Lines 94-96**: Extract prefixes from underscore-delimited names
- Takes everything before first underscore as potential prefix
- Adds underscore back to maintain delimiter in pattern

```python
elif len(col_lower) > 4:
    for length in [3, 4, 5, 6]:
        if length < len(col_lower):
            prefix = col_lower[:length] + '_'
            if re.match(r'^[a-z]+_$', prefix):
                prefix_counter[prefix] += 1
```
- **Lines 97-104**: Try different prefix lengths for non-delimited names
- Tests common prefix lengths (3-6 characters) systematically
- Regex `^[a-z]+_$` ensures prefix contains only letters plus underscore
- Prevents numeric or special character prefixes from being counted

```python
total_columns = len(columns)
min_occurrences = max(1, int(total_columns * self.min_frequency_threshold))
```
- **Lines 107-108**: Calculate minimum frequency threshold
- Converts percentage threshold to absolute count
- `max(1, ...)` ensures at least 1 occurrence required even for small datasets

```python
self.common_prefixes = [
    prefix for prefix, count in prefix_counter.items() 
    if count >= min_occurrences
]
```
- **Lines 110-113**: Filter prefixes by frequency threshold
- List comprehension creates clean list of significant prefixes
- Only patterns meeting threshold are retained

```python
self.common_prefixes.sort(
    key=lambda x: prefix_counter[x], reverse=True
)
```
- **Lines 116-118**: Sort prefixes by frequency (most common first)
- Enables prioritized processing where most common patterns are tried first
- Improves efficiency in later semantic label generation

#### Method: `_derive_common_suffixes(self, columns)`

**Purpose**: Extract and identify common suffixes from column names based on frequency analysis.

**Line-by-line breakdown**:

```python
def _derive_common_suffixes(self, columns):
```
- **Line 120**: Private method focusing on suffix pattern extraction
- Parallel structure to prefix derivation for consistency

```python
suffix_counter = Counter()
```
- **Line 127**: Initialize Counter for suffix frequency tracking

```python
for col in columns:
    col_lower = col.lower()
```
- **Lines 129-130**: Process each column with case normalization
- Identical preprocessing to prefix analysis

```python
if '_' in col_lower:
    suffix = '_' + col_lower.split('_')[-1]
    suffix_counter[suffix] += 1
```
- **Lines 132-134**: Extract suffixes from underscore-delimited names
- Takes everything after last underscore as potential suffix
- Prepends underscore to maintain delimiter in pattern

```python
elif len(col_lower) > 4:
    for length in [2, 3, 4, 5]:
        if length < len(col_lower):
            suffix = '_' + col_lower[-length:]
            if re.match(r'^_[a-z]+$', suffix):
                suffix_counter[suffix] += 1
```
- **Lines 135-142**: Try different suffix lengths for non-delimited names
- Tests common suffix lengths (2-5 characters) systematically
- Regex `^_[a-z]+$` ensures suffix contains only underscore plus letters
- Negative indexing `[-length:]` extracts from end of string

```python
total_columns = len(columns)
min_occurrences = max(1, int(total_columns * self.min_frequency_threshold))

self.common_suffixes = [
    suffix for suffix, count in suffix_counter.items() 
    if count >= min_occurrences
]

self.common_suffixes.sort(
    key=lambda x: suffix_counter[x], reverse=True
)
```
- **Lines 145-156**: Apply same filtering and sorting logic as prefixes
- Maintains consistency between prefix and suffix processing
- Results in frequency-ordered list of significant suffixes

#### Method: `_derive_semantic_mappings(self, columns)`

**Purpose**: Create semantic categories by clustering similar column names based on meaning.

**Line-by-line breakdown**:

```python
def _derive_semantic_mappings(self, columns):
```
- **Line 158**: Private method for semantic category derivation

```python
base_categories = {
    'identifier': ['id', 'identifier', 'number', 'num', 'key', 'pk', 'uid'],
    'name': ['name', 'title', 'label', 'description', 'desc'],
    'date': ['date', 'time', 'timestamp', 'created', 'updated', 'modified'],
    'amount': ['amount', 'value', 'price', 'cost', 'total', 'sum', 'balance'],
    'address': ['address', 'location', 'addr', 'street', 'city', 'state'],
    'contact': ['phone', 'telephone', 'mobile', 'email', 'mail'],
    'status': ['status', 'state', 'condition', 'flag', 'active', 'enabled']
}
```
- **Lines 166-174**: Define common semantic categories with keyword variations
- Categories based on common business data types across domains
- Keywords chosen to catch most common naming variations
- Extensible structure allows easy addition of domain-specific categories

```python
semantic_groups = defaultdict(list)
```
- **Line 177**: Initialize grouping structure with defaultdict
- Automatically creates new lists for new categories
- Simplifies grouping logic by avoiding key existence checks

```python
for col in columns:
    col_lower = col.lower()
    cleaned_col = self._clean_column_for_semantic_analysis(col_lower)
```
- **Lines 179-182**: Process each column for semantic analysis
- Case normalization for consistent comparison
- Clean column name to remove prefixes/suffixes that might obscure semantic meaning

```python
matched = False
for category, keywords in base_categories.items():
    if any(keyword in cleaned_col for keyword in keywords):
        semantic_groups[category].append(col_lower)
        matched = True
        break
```
- **Lines 185-190**: Attempt to match column with predefined categories
- `any()` returns True if any keyword found in cleaned column name
- `break` ensures column only assigned to first matching category
- Prevents duplicate categorization

```python
if not matched and cleaned_col:
    semantic_groups[cleaned_col].append(col_lower)
```
- **Lines 193-194**: Create new category if no predefined match found
- Uses cleaned column name as category label
- Enables discovery of domain-specific patterns not in base categories

```python
self.semantic_mappings = {}
for category, terms in semantic_groups.items():
    if len(terms) >= 2 or category in base_categories:  # Only keep if multiple terms or base category
        self.semantic_mappings[tuple(set(terms))] = category
```
- **Lines 197-200**: Convert groups to final semantic mappings format
- Requires multiple terms OR base category to reduce noise
- `tuple(set(terms))` creates hashable key from unique terms
- Maps term collections to semantic category labels

#### Method: `_clean_column_for_semantic_analysis(self, col_name)`

**Purpose**: Clean column names by removing patterns that obscure semantic meaning.

**Line-by-line breakdown**:

```python
def _clean_column_for_semantic_analysis(self, col_name):
```
- **Line 202**: Private helper method for column name cleaning

```python
cleaned = col_name.lower()
```
- **Line 212**: Start with lowercase normalization

```python
cleaned = re.sub(r'[0-9]+', '', cleaned)
cleaned = re.sub(r'[^a-z_]', '', cleaned)
```
- **Lines 215-216**: Remove numbers and special characters
- First regex removes all digits that don't contribute to semantic meaning
- Second regex keeps only letters and underscores for clean analysis

```python
if self.common_prefixes:
    for prefix in self.common_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
```
- **Lines 219-223**: Remove common prefixes if they've been identified
- Only runs if prefixes have been derived (prevents errors on first run)
- `break` ensures only one prefix removed to avoid over-cleaning

```python
if self.common_suffixes:
    for suffix in self.common_suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
```
- **Lines 225-229**: Remove common suffixes if they've been identified
- Parallel logic to prefix removal
- Negative slicing `[:-len(suffix)]` removes suffix from end

```python
cleaned = cleaned.strip('_')
if '_' in cleaned:
    parts = cleaned.split('_')
    cleaned = max(parts, key=len) if parts else cleaned
```
- **Lines 232-236**: Final cleaning and main semantic part extraction
- Remove leading/trailing underscores from prefix/suffix removal
- If multiple parts remain, take longest as most semantically meaningful
- `max(parts, key=len)` finds longest string in list

```python
return cleaned
```
- **Line 238**: Return cleaned column name ready for semantic analysis

#### Method: `get_semantic_label(self, col1, col2)`

**Purpose**: Generate semantically cleaner labels by combining information from both column names.

**Line-by-line breakdown**:

```python
def get_semantic_label(self, col1, col2):
```
- **Line 240**: Public method for generating semantic labels from column pairs

```python
c1_lower = col1.lower()
c2_lower = col2.lower()
```
- **Lines 252-253**: Normalize case for consistent comparison

```python
for terms, clean_label in self.semantic_mappings.items():
    if any(term in c1_lower for term in terms) or any(term in c2_lower for term in terms):
        return clean_label
```
- **Lines 255-257**: Check if either column matches derived semantic categories
- Uses previously derived semantic mappings for consistent labeling
- Returns category label if match found

```python
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
```
- **Lines 260-270**: Local function for column name cleaning
- Applies same cleaning logic as semantic analysis method
- Removes common prefixes and suffixes to find core semantic meaning

```python
clean_c1 = clean_column_name(col1)
clean_c2 = clean_column_name(col2)
```
- **Lines 272-273**: Apply cleaning to both column names

```python
if clean_c1 and clean_c2:
    if clean_c1 in clean_c2:
        return clean_c1
    elif clean_c2 in clean_c1:
        return clean_c2
```
- **Lines 276-280**: Check if one cleaned name is subset of the other
- Substring relationship suggests one is abbreviation of the other
- Return shorter name as more concise label

```python
words1 = set(clean_c1.split('_')) if clean_c1 else set()
words2 = set(clean_c2.split('_')) if clean_c2 else set()
common_words = words1.intersection(words2)

if common_words:
    return '_'.join(sorted(common_words))
```
- **Lines 283-288**: Find common words between column names
- Split on underscores to find word-level overlap
- Return sorted combination of common words if found

```python
return col1 if len(col1) <= len(col2) else col2
```
- **Line 291**: Fallback to shorter original name
- Simple heuristic when no semantic patterns found

### Class: `EntityAssignment`

The `EntityAssignment` class handles dataset consolidation and assignment of entities based on column matching results.

#### Constructor: `__init__(self, match_threshold=0.7, semantic_consolidation=None)`

**Purpose**: Initialize entity assignment system with decision threshold and optional semantic analyzer.

**Line-by-line breakdown**:

```python
def __init__(self, match_threshold=0.7, semantic_consolidation=None):
```
- **Line 307**: Constructor with configurable threshold and optional semantic analyzer
- Default threshold 0.7 (70%) balances precision and recall for join decisions

```python
self.match_threshold = match_threshold
self.semantic_consolidation = semantic_consolidation
```
- **Lines 315-316**: Store configuration parameters as instance variables
- Allows customization per use case while maintaining defaults

#### Method: `consolidate_datasets_based_on_matches(self, df1, df2, column_matches, match_threshold=None)`

**Purpose**: Intelligently consolidate datasets by joining on strong matches or concatenating on weak matches.

**Line-by-line breakdown**:

```python
def consolidate_datasets_based_on_matches(self, df1, df2, column_matches, match_threshold=None):
```
- **Line 318**: Main consolidation method with optional threshold override

```python
if match_threshold is None:
    match_threshold = self.match_threshold
```
- **Lines 335-336**: Use instance default if no threshold provided
- Allows per-call customization while maintaining instance configuration

```python
if self.semantic_consolidation is None:
    self.semantic_consolidation = SemanticConsolidation()

self.semantic_consolidation.analyze_datasets(df1, df2)
```
- **Lines 339-343**: Initialize or update semantic consolidation
- Lazy initialization creates analyzer only when needed
- Always analyze current datasets to ensure patterns are up-to-date

```python
min_columns = min(len(df1.columns), len(df2.columns))
required_matches = max(1, min_columns // 2)  # At least 1, but typically half
```
- **Lines 346-347**: Calculate number of matches needed for join decision
- Uses half of smaller dataset's columns as threshold
- `max(1, ...)` ensures at least one match required even for single-column datasets

```python
sorted_matches = sorted(column_matches, key=lambda x: x[2], reverse=True)
top_matches = sorted_matches[:required_matches]
```
- **Lines 350-351**: Select highest-quality matches for decision
- Sorts by similarity score (third element in tuple)
- Takes only top N matches where N is calculated threshold

```python
if top_matches:
    avg_score = np.mean([match[2] for match in top_matches])
    avg_score_normalized = avg_score / 100.0  # Convert from 0-100 to 0-1 scale
else:
    avg_score_normalized = 0.0
```
- **Lines 354-358**: Calculate average quality of top matches
- Extract scores (third element) from match tuples
- Normalize to 0-1 scale for threshold comparison
- Handle empty matches gracefully

```python
metadata = {
    'total_matches_found': len(column_matches),
    'matches_considered': len(top_matches), 
    'required_matches': required_matches,
    'average_match_score': avg_score_normalized,
    'threshold': match_threshold,
    'top_matches': top_matches
}
```
- **Lines 360-367**: Compile metadata for transparency and debugging
- Provides full decision audit trail
- Helps users understand why join vs concatenate was chosen

```python
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
```
- **Lines 369-378**: Make join vs concatenate decision based on match quality
- Join when average score meets threshold (high confidence in column alignment)
- Concatenate when scores below threshold (low confidence, safer to stack)
- Return method name and metadata for transparency

#### Method: `_join_datasets_on_matches(self, df1, df2, matches)`

**Purpose**: Join datasets on matched columns with semantic column renaming.

**Line-by-line breakdown**:

```python
def _join_datasets_on_matches(self, df1, df2, matches):
```
- **Line 380**: Private method for dataset joining logic

```python
if not matches:
    return df1.copy(), {'join_type': 'no_matches', 'join_columns': []}
```
- **Lines 391-393**: Handle edge case of no matches
- Returns first dataset unchanged with appropriate metadata
- Prevents errors in downstream processing

```python
df1_prep = df1.copy()
df2_prep = df2.copy()
```
- **Lines 396-397**: Create copies to avoid modifying original datasets
- Defensive programming ensures original data remains unchanged

```python
column_mapping = {}
join_columns = []
```
- **Lines 400-401**: Initialize tracking for column transformations

```python
for col1, col2, score in matches:
    clean_label = self._generate_semantic_label(col1, col2)
    
    df1_prep = df1_prep.rename(columns={col1: clean_label})
    df2_prep = df2_prep.rename(columns={col2: clean_label})
    
    column_mapping[f"{col1}|{col2}"] = clean_label
    join_columns.append(clean_label)
```
- **Lines 403-412**: Rename matched columns to consistent labels
- Generate semantic label that captures meaning of both columns
- Apply same label to both datasets for join compatibility
- Track all transformations for metadata and debugging

```python
try:
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
    return self._concatenate_datasets_with_labels(df1, df2, matches)
```
- **Lines 415-432**: Perform pandas merge with error handling
- Outer join preserves all data from both datasets
- Suffixes handle non-matched columns with same names
- Comprehensive metadata tracking for audit trail
- Fallback to concatenation if join fails

#### Method: `_concatenate_datasets_with_labels(self, df1, df2, matches)`

**Purpose**: Concatenate datasets vertically with source tracking and semantic column alignment.

**Line-by-line breakdown**:

```python
def _concatenate_datasets_with_labels(self, df1, df2, matches):
```
- **Line 434**: Private method for dataset concatenation logic

```python
df1_prep = df1.copy()
df2_prep = df2.copy()
```
- **Lines 445-446**: Create copies to avoid modifying originals

```python
df1_prep['_dataset_source'] = 'dataset_1'
df2_prep['_dataset_source'] = 'dataset_2'
```
- **Lines 449-450**: Add source tracking columns
- Enables identification of data origin after concatenation
- Critical for debugging and data lineage

```python
column_mapping = {}

for col1, col2, score in matches:
    clean_label = self._generate_semantic_label(col1, col2)
    
    if col1 in df1_prep.columns:
        df1_prep = df1_prep.rename(columns={col1: clean_label})
        column_mapping[col1] = clean_label
        
    if col2 in df2_prep.columns:
        df2_prep = df2_prep.rename(columns={col2: clean_label})
        column_mapping[col2] = clean_label
```
- **Lines 454-466**: Apply semantic renaming to matched columns
- Aligns similar columns under same name for cleaner concatenation
- Conditional checks prevent errors if columns don't exist
- Track all renames for metadata

```python
concatenated_df = pd.concat([df1_prep, df2_prep], ignore_index=True, sort=False)
```
- **Line 469**: Perform vertical concatenation
- `ignore_index=True` creates continuous index
- `sort=False` preserves column order for better readability

```python
metadata = {
    'concatenation_type': 'vertical_stack',
    'column_mapping': column_mapping,
    'original_df1_shape': df1.shape,
    'original_df2_shape': df2.shape,
    'result_shape': concatenated_df.shape,
    'matched_columns_aligned': len(matches)
}

return concatenated_df, metadata
```
- **Lines 471-480**: Compile metadata and return results
- Comprehensive tracking of transformation applied
- Includes statistics on data size changes

#### Method: `_generate_semantic_label(self, col1, col2)`

**Purpose**: Generate semantically meaningful labels for column pairs using learned patterns.

**Line-by-line breakdown**:

```python
def _generate_semantic_label(self, col1, col2):
```
- **Line 482**: Private method for semantic label generation

```python
if self.semantic_consolidation is not None:
    return self.semantic_consolidation.get_semantic_label(col1, col2)
```
- **Lines 494-495**: Use semantic consolidation if available
- Leverages learned patterns from dataset analysis
- Preferred approach for consistent, intelligent labeling

```python
c1_lower = col1.lower()
c2_lower = col2.lower()

basic_categories = {
    'identifier': ['id', 'identifier', 'number'],
    'name': ['name', 'title', 'label'], 
    'date': ['date', 'time', 'timestamp'],
    'amount': ['amount', 'value', 'price', 'cost'],
}

for category, terms in basic_categories.items():
    if any(term in c1_lower for term in terms) or any(term in c2_lower for term in terms):
        return category

return col1 if len(col1) <= len(col2) else col2
```
- **Lines 499-515**: Fallback approach when semantic consolidation unavailable
- Basic pattern matching for common semantic categories
- Returns shorter original name if no patterns found
- Ensures method always returns valid label

---

## Workflow Management (`workflow.py`)

### Class: `LangGraphWorkflow`

The `LangGraphWorkflow` class creates and manages LangGraph computational workflows for entity resolution tasks, separating orchestration from implementation.

#### Constructor: `__init__(self, column_finder, match_threshold=0.7)`

**Purpose**: Initialize workflow orchestrator with entity resolution engine and decision threshold.

**Line-by-line breakdown**:

```python
def __init__(self, column_finder, match_threshold=0.7):
```
- **Line 70**: Constructor with required column finder and optional threshold

```python
self.column_finder = column_finder
self.entity_assignment = EntityAssignment(match_threshold)
```
- **Lines 84-85**: Store dependencies using composition pattern
- Creates entity assignment system with provided threshold
- Separates concerns between workflow orchestration and implementation

#### Method: `create_column_matching_graph(self, df1, df2)`

**Purpose**: Create a visual computational workflow that orchestrates the complete entity resolution pipeline.

**Line-by-line breakdown**:

```python
def create_column_matching_graph(self, df1, df2):
```
- **Line 87**: Method creates LangGraph representation of workflow

```python
graph = NodeGraph()
```
- **Line 184**: Initialize empty LangGraph workflow container
- Provides framework for adding nodes and edges

```python
describe_col1_node = graph.add_node(
    name="Describe columns in Dataset 1",
    function=self.column_finder.describe_columns,
    inputs={"df": df1, "columns": df1.columns.tolist()},
    output_key="descriptions1",
)
```
- **Lines 189-194**: Create node for dataset 1 column description
- `name`: Human-readable identifier for visualization
- `function`: Points to actual implementation method
- `inputs`: Provides required parameters for function execution
- `output_key`: Labels output for downstream node consumption

```python
describe_col2_node = graph.add_node(
    name="Describe columns in Dataset 2", 
    function=self.column_finder.describe_columns,
    inputs={"df": df2, "columns": df2.columns.tolist()},
    output_key="descriptions2",
)
```
- **Lines 198-204**: Create parallel node for dataset 2 column description
- Identical structure to first node enables parallel execution
- Different input dataset and output key for proper data flow

```python
compare_node = graph.add_node(
    name="Compare columns and find matches",
    function=self.column_finder.find_similar_columns,
    inputs={"df1": df1, "df2": df2},
    output_key="similar_columns",
)
```
- **Lines 209-214**: Create node for column comparison and matching
- Depends on both description nodes completing first
- Uses original DataFrames since find_similar_columns handles description internally

```python
consolidate_node = graph.add_node(
    name="Consolidate datasets based on matches",
    function=lambda similar_columns: self.entity_assignment.consolidate_datasets_based_on_matches(df1, df2, similar_columns),
    inputs={"similar_columns": "similar_columns"},  # Uses output from compare_node
    output_key="consolidated_result",
)
```
- **Lines 218-223**: Create node for dataset consolidation
- Lambda function enables method call with fixed parameters
- Input references output from previous node by key name
- Completes the workflow with final dataset production

```python
graph.add_edges([
    ("Describe columns in Dataset 1", "Compare columns and find matches"),
    ("Describe columns in Dataset 2", "Compare columns and find matches"),
    ("Compare columns and find matches", "Consolidate datasets based on matches"),
])
```
- **Lines 228-232**: Define execution dependencies between nodes
- Both description nodes must complete before comparison
- Comparison must complete before consolidation
- Creates proper directed acyclic graph (DAG) structure

```python
return graph
```
- **Line 234**: Return complete workflow graph
- Graph can be executed, visualized, or modified by caller

---

## Example Usage (`example_usage.py`)

### Function: `main()`

**Purpose**: Demonstrate complete entity resolution workflow with realistic business scenario.

**Line-by-line breakdown**:

```python
def main():
```
- **Line 30**: Main demonstration function entry point

```python
print("=" * 80)
print("ENTITY RESOLUTION DEMO: Customer Database Integration")
print("=" * 80)
```
- **Lines 34-36**: Create visual banner for demo presentation
- Clear marking of demo boundaries for user experience

```python
data1 = {
    'cust_id': [1, 2, 3],  # Customer identifier (technical)
    'name': ['Alice', 'Bob', 'Charlie'],  # Customer name (abbreviated)
    'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']  # Purchase date (technical)
}

data2 = {
    'customer_number': [1, 2, 3],  # Customer identifier (business-friendly)
    'full_name': ['Alice', 'Bob', 'Charlie'],  # Customer name (descriptive)
    'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03']  # Purchase date (business-friendly)
}
```
- **Lines 44-55**: Create representative datasets with different naming conventions
- Dataset 1 uses technical abbreviations common in legacy systems
- Dataset 2 uses business-friendly names common in modern systems
- Same data with different column names demonstrates entity resolution challenge

```python
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
```
- **Lines 57-58**: Convert dictionaries to pandas DataFrames
- Required format for entity resolution system input

```python
finder = SimilarColumnFinder()  
print("CHECKMARK SimilarColumnFinder initialized with OpenAI LLM")
print("CHECKMARK Using semantic analysis + fuzzy matching approach")
```
- **Lines 70-72**: Initialize entity resolution system
- Uses default configuration (OpenAI LLM, optimized prompts)
- Confirmation messages provide user feedback

```python
similar_cols = finder.find_similar_columns(df1, df2, threshold=75)
```
- **Line 83**: Execute core entity resolution pipeline
- Threshold 75 provides balanced precision/recall for demonstration
- Returns list of potential column matches with confidence scores

```python
if similar_cols:
    print(f"SUCCESS Found {len(similar_cols)} potential column matches:")
    print()
    
    for col1, col2, score in similar_cols:
        if score >= 90:
            confidence = "Very High"
        elif score >= 80:
            confidence = "High" 
        elif score >= 70:
            confidence = "Medium"
        else:
            confidence = "Low"
            
        print(f"  LINK MATCH FOUND:")
        print(f"     Company A: '{col1}' <-> Company B: '{col2}'")
        print(f"     Similarity Score: {score}/100 ({confidence} Confidence)")
        print(f"     -> These columns likely contain the same type of information")
        print()
```
- **Lines 89-108**: Display results with confidence level interpretation
- Iterates through each match found by the system
- Converts numeric scores to human-readable confidence levels
- Provides clear explanation of what each match means

```python
validated_matches = finder.validate_column_matches(df1, df2, similar_cols)

print("VALIDATION RESULTS:")
print("=" * 50)

for i, match in enumerate(validated_matches, 1):
    print(f"\nMATCH #{i}: {match['col1']} <-> {match['col2']}")
    print(f"  Semantic Similarity: {match['semantic_similarity']}/100")
    print(f"  Data Type Compatibility: {match['data_type_analysis']['compatibility_score']}/100")
    print(f"  Value Overlap: {match['value_overlap_analysis']['overlap_percentage']}%")
    print(f"  OVERALL SCORE: {match['overall_validation_score']}/100")
    print(f"  RECOMMENDATION: {match['recommendation']}")
```
- **Lines 124-136**: Demonstrate enhanced validation with data analysis
- Shows how semantic matches are validated against actual data
- Displays comprehensive scoring across multiple dimensions
- Provides actionable recommendations for each match

```python
workflow = LangGraphWorkflow(finder)
consolidated_df, method, metadata = workflow.consolidate_datasets_based_on_matches(
    df1, df2, similar_cols, match_threshold=0.7
)

print(f"CONSOLIDATION METHOD: {method.upper()}")
print(f"REASON: Average match score of top {metadata['matches_considered']} matches = {metadata['average_match_score']:.3f}")
```
- **Lines 203-209**: Demonstrate automatic dataset consolidation
- Creates workflow and applies intelligent join/concatenate decision
- Shows reasoning behind consolidation method choice
- Provides transparency in automated decision-making

```python
if method == 'join':
    print("CHART_BAR JOINED DATASETS:")
    print(f"  - Original df1 shape: {metadata['original_df1_shape']}")
    print(f"  - Original df2 shape: {metadata['original_df2_shape']}")
    print(f"  - Consolidated shape: {metadata['result_shape']}")
    print(f"  - Join columns: {metadata['join_columns']}")
    print(f"  - Column mappings applied: {len(metadata['column_mapping'])}")
    
elif method == 'concatenate':
    print("STACK CONCATENATED DATASETS:")
    print(f"  - Original df1 shape: {metadata['original_df1_shape']}")
    print(f"  - Original df2 shape: {metadata['original_df2_shape']}")
    print(f"  - Consolidated shape: {metadata['result_shape']}")
    print(f"  - Matched columns aligned: {metadata['matched_columns_aligned']}")
    print(f"  - Dataset source column added for tracking")
```
- **Lines 214-227**: Display consolidation results based on method chosen
- Different output format for join vs concatenate results
- Shows impact on data shape and structure
- Explains what transformations were applied

### Function: `demonstrate_advanced_features()`

**Purpose**: Show advanced customization options and validation scenarios.

**Line-by-line breakdown**:

```python
def demonstrate_advanced_features():
```
- **Line 245**: Secondary demonstration function for advanced features

```python
for threshold in [90, 80, 70]:
    print(f"\nThreshold {threshold}:")
    matches = finder.find_similar_columns(df1, df2, threshold=threshold)
    print(f"  Found {len(matches)} matches")
    for col1, col2, score in matches:
        print(f"    {col1} <-> {col2} (Score: {score})")
```
- **Lines 266-271**: Demonstrate impact of different similarity thresholds
- Shows how threshold affects number and quality of matches
- Helps users understand threshold selection trade-offs

```python
good_data1 = {
    'user_id': [1, 2, 3, 4, 5],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
}

good_data2 = {
    'customer_id': [1, 2, 3, 4, 5],  # Same numeric values
    'years_old': [25, 30, 35, 40, 45],  # Same numeric values
    'location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']  # Same text values
}

poor_data2 = {
    'customer_id': ['A', 'B', 'C', 'D', 'E'],  # String vs numeric
    'years_old': ['young', 'middle', 'old', 'senior', 'adult'],  # Categorical vs numeric
    'location': ['NY', 'LA', 'CHI', 'HOU', 'PHX']  # Different text values
}
```
- **Lines 281-298**: Create datasets with different compatibility scenarios
- Good compatibility: same data types and values
- Poor compatibility: different data types and value formats
- Demonstrates validation system's ability to distinguish quality

```python
print("\nScenario A: High Compatibility Datasets")
matches = finder.find_similar_columns(good_df1, good_df2, threshold=70)
if matches:
    validation_results = finder.validate_column_matches(good_df1, good_df2, matches)
    print(f"Found {len([r for r in validation_results if r['overall_validation_score'] >= 70])} high-quality validated matches")

print("\nScenario B: Poor Compatibility Datasets")
matches = finder.find_similar_columns(good_df1, poor_df2, threshold=70)
if matches:
    validation_results = finder.validate_column_matches(good_df1, poor_df2, matches)
    print(f"Found {len([r for r in validation_results if r['overall_validation_score'] >= 70])} high-quality validated matches")
```
- **Lines 304-320**: Compare validation results across compatibility scenarios
- Shows how validation system distinguishes between good and poor data matches
- Demonstrates value of data-based validation beyond semantic similarity

```python
if __name__ == '__main__':
    try:
        main()
        demonstrate_advanced_features()
    except Exception as e:
        print(f"Error running demonstration: {e}")
        print("Make sure you have installed the required dependencies:")
        print("  pip install langchain openai fuzzywuzzy pandas")
```
- **Lines 328-338**: Main execution block with error handling
- Runs both demonstration functions sequentially
- Provides helpful error messages and dependency information
- Graceful failure handling for missing dependencies

---

This comprehensive explanation covers every function and method in the entity resolution system, explaining the purpose of each line of code and how it contributes to the overall goal of intelligent column matching and dataset consolidation. The system combines semantic analysis, fuzzy matching, data validation, and intelligent consolidation to provide a robust solution for entity resolution challenges.
