# Entity Resolution: Intelligent Column Matching

A sophisticated entity resolution system that uses Large Language Models (LLMs) and fuzzy matching to identify semantically similar columns across different datasets, even when they have completely different naming conventions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-82%20passed-green.svg)](./tests/)

## 🚀 Quick Start

```bash
# Install the package
pip install -e .

# Install with all dependencies  
pip install -e .[dev,test,docs,notebook]

# Run the demo
entity-resolution-demo

# Run tests
entity-resolution-test
```

```python
from entity_resolution import SimilarColumnFinder
import pandas as pd

# Create sample datasets with different naming conventions
df1 = pd.DataFrame({'cust_id': [1,2,3], 'name': ['Alice','Bob','Charlie']})
df2 = pd.DataFrame({'customer_num': [1,2,3], 'full_name': ['Alice','Bob','Charlie']})

# Find similar columns automatically
finder = SimilarColumnFinder()
matches = finder.find_similar_columns(df1, df2, threshold=80)

# Review results
for col1, col2, score in matches:
    print(f"{col1} <-> {col2} (Confidence: {score}%)")
```

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Our Solution](#our-solution)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Package Components](#package-components)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Configuration](#configuration)
- [Performance](#performance)
- [Use Cases](#use-cases)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Problem Statement

When integrating data from multiple sources, one of the biggest challenges is determining which columns contain the same type of information. Traditional approaches rely on exact name matching or simple string similarity, which fails when:

- Different systems use different naming conventions (`cust_id` vs `customer_number`)
- Column names are abbreviated differently (`desc` vs `description` vs `product_info`)
- Business terminology varies across organizations (`order_date` vs `transaction_date` vs `purchase_timestamp`)

## 💡 Our Solution: Semantic Column Matching

This system solves the column alignment problem using a two-stage approach:

1. **Semantic Understanding**: Generate rich, contextual descriptions of what each column represents using an LLM
2. **Intelligent Matching**: Use fuzzy string matching on these descriptions to find columns with similar meanings

## 🏗️ System Architecture

```
+-----------------+    +-----------------+
|    Dataset 1    |    |    Dataset 2    |
|   ['cust_id',   |    |['customer_num', |
|    'name',      |    | 'full_name',   |
|    'order_date']|    | 'trans_date']  |
+---------+-------+    +---------+-------+
          |                      |
          v                      v
+-----------------+    +-----------------+
| LLM Description |    | LLM Description |
|   Generation    |    |   Generation    |
+---------+-------+    +---------+-------+
          |                      |
          v                      v
+-----------------+    +-----------------+
|"Customer ID,    |    |"Customer number,|
| unique integer  |    | unique ID for   |
| identifier..."  |    | each customer..." |
+---------+-------+    +---------+-------+
          |                      |
          +----------+-----------+
                     v
           +-----------------+
           | Fuzzy String    |
           |    Matching     |
           |   (fuzzywuzzy)  |
           +---------+-------+
                     v
           +-----------------+
           | Dataset         |
           | Consolidation   |
           | (Join vs Concat)|
           +---------+-------+
                     v
           +-----------------+
           | Similarity      |
           | Score & Match   |
           | Recommendations |
           +-----------------+
```

## 📦 Installation

### Method 1: Install from Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/zacharyking/entity-resolution.git
cd entity-resolution

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e .[dev,test,docs,notebook]
```

### Method 2: Install from Built Package

```bash
# Build the package
python setup.py sdist bdist_wheel

# Install the built package
pip install dist/entity-resolution-1.0.0.tar.gz
```

### Method 3: Direct Installation (Future PyPI)

```bash
# Once published to PyPI
pip install entity-resolution
```

### Dependencies

#### Required Dependencies

These are automatically installed with the package:

- `langchain>=0.1.0` - LLM integration and prompt management
- `fuzzywuzzy>=0.18.0` - Fuzzy string matching
- `pandas>=2.0.0` - Data structure handling  
- `openai>=1.0.0` - OpenAI LLM provider
- `python-levenshtein>=0.12.2` - Fast string distance calculations

#### LangChain Community (Required for Full Functionality)

```bash
pip install langchain-community
```

#### Optional Dependencies

```bash
# Development tools
pip install entity-resolution[dev]    # pytest, black, flake8, mypy, pre-commit

# Testing tools  
pip install entity-resolution[test]   # pytest, pytest-cov, unittest-xml-reporting

# Documentation tools
pip install entity-resolution[docs]   # sphinx, sphinx-rtd-theme, myst-parser

# Notebook support
pip install entity-resolution[notebook]  # jupyter, matplotlib, seaborn
```

### Installation Verification

```bash
# Test basic package functionality
python -c "import entity_resolution; print(entity_resolution.get_version())"

# Check if all dependencies are available
python -c "import entity_resolution; entity_resolution.check_dependencies()"

# Run quick start guide
python -c "import entity_resolution; entity_resolution.quick_start()"

# Run the complete demonstration
entity-resolution-demo

# Run the test suite
entity-resolution-test
```

## 🧩 Package Components

### Core Classes

- **`SimilarColumnFinder`** - Main entity resolution engine
  - Orchestrates the complete matching pipeline
  - Configurable LLM and matching parameters
  - Supports both direct usage and workflow visualization

- **`SemanticConsolidation`** - Analyzes datasets to derive semantic patterns
  - Learns common prefixes/suffixes from data
  - Creates semantic mappings for cleaner column labels
  - Adapts to different domain-specific naming conventions

- **`EntityAssignment`** - Handles dataset consolidation logic
  - Automatically decides whether to join or concatenate datasets
  - Applies semantic column labels based on matched pairs
  - Preserves data integrity regardless of consolidation method

- **`LangGraphWorkflow`** - Workflow visualization and orchestration
  - Creates computational workflow graphs
  - Enables parallel processing of column descriptions
  - Provides workflow visualization for debugging

### Key Methods

- **`describe_columns()`** - Generate semantic descriptions using LLM
- **`find_similar_columns()`** - Execute complete matching pipeline
- **`validate_column_matches()`** - Deep validation with data analysis
- **`consolidate_datasets_based_on_matches()`** - Intelligent dataset merging
- **`create_column_matching_graph()`** - Create visualizable workflow graph

### Console Scripts

The package includes convenient console scripts:

```bash
# Run the complete demonstration
entity-resolution-demo

# Run the comprehensive test suite
entity-resolution-test
```

### Algorithm Workflow

#### Phase 1: Semantic Description Generation

```
For each dataset:
  For each column:
    1. Extract column name
    2. Send to LLM with standardized prompt
    3. Generate semantic description
    4. Store description for comparison
```

#### Phase 2: Intelligent Matching

```
For each column in Dataset 1:
  For each column in Dataset 2:
    1. Retrieve semantic descriptions
    2. Apply fuzzy string matching
    3. Calculate similarity score (0-100)
    4. If score >= threshold:
       - Record as potential match
       - Include confidence score
    5. Sort results by confidence
```

#### Phase 3: Dataset Consolidation

```
1. Calculate average fuzzy match score for top half of columns
2. If average score >= 0.7:
   - JOIN: Merge datasets on matched columns
   - Apply semantic column labels
   - Use outer join to preserve all data
3. If average score < 0.7:
   - CONCATENATE: Stack datasets vertically
   - Apply semantic column labels to matched columns
   - Add dataset source tracking column
4. Return consolidated dataset with metadata
```

**Benefits of Intelligent Consolidation:**
- **Automatic Decision Making**: No manual decision required for join vs concatenate
- **Semantic Labels**: Cleaner, more descriptive column names based on matched pairs
- **Data Integrity**: Preserves all data regardless of consolidation method
- **Source Tracking**: Maintains dataset origin for concatenated results
- **Quality Thresholds**: Configurable match quality requirements (default 0.7)

## 💻 Usage Examples

### Basic Column Matching

```python
from entity_resolution import SimilarColumnFinder
import pandas as pd

# Create sample datasets
df1 = pd.DataFrame({
    'cust_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']
})

df2 = pd.DataFrame({
    'customer_number': [1, 2, 3],
    'full_name': ['Alice', 'Bob', 'Charlie'],
    'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03']
})

# Initialize the system
finder = SimilarColumnFinder()

# Find similar columns
matches = finder.find_similar_columns(df1, df2, threshold=80)

# Review results
for col1, col2, score in matches:
    print(f"{col1} <-> {col2} (Confidence: {score}%)")
```

### Advanced Usage with Validation

```python
from entity_resolution import SimilarColumnFinder
from langchain.llms import OpenAI

# Custom LLM configuration
custom_llm = OpenAI(temperature=0.0, model_name="gpt-4")
finder = SimilarColumnFinder(llm=custom_llm)

# Find matches
matches = finder.find_similar_columns(df1, df2, threshold=80)

# Validate matches with data analysis
validated_matches = finder.validate_column_matches(df1, df2, matches)

# Review validation results
for match in validated_matches:
    print(f"Match: {match['col1']} <-> {match['col2']}")
    print(f"Overall Score: {match['overall_validation_score']}/100")
    print(f"Recommendation: {match['recommendation']}")
    print(f"Flags: {match['validation_flags']}")
    print()
```

### Workflow with Dataset Consolidation

```python
from entity_resolution import SimilarColumnFinder, LangGraphWorkflow

# Initialize components
finder = SimilarColumnFinder()
workflow = LangGraphWorkflow(finder)

# Find matches
matches = finder.find_similar_columns(df1, df2, threshold=80)

# Automatic consolidation based on match quality
consolidated_df, method, metadata = workflow.entity_assignment.consolidate_datasets_based_on_matches(
    df1, df2, matches, match_threshold=0.7
)

print(f"Consolidation method: {method}")  # 'join' or 'concatenate'
print(f"Result shape: {consolidated_df.shape}")
print(f"Average match score: {metadata['average_match_score']}")
print(f"Join columns: {metadata.get('join_columns', 'N/A')}")
```

### Example: Customer Database Integration

**Company A (Technical Naming)**
```python
company_a = pd.DataFrame({
    'cust_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']
})
```

**Company B (Business Naming)**
```python
company_b = pd.DataFrame({
    'customer_number': [1, 2, 3],
    'full_name': ['Alice Smith', 'Bob Jones', 'Charlie Brown'],
    'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03']
})
```

**Expected Results:**
```
MATCHES FOUND:
✓ cust_id <-> customer_number (Score: 92/100)
✓ name <-> full_name (Score: 88/100)  
✓ order_date <-> transaction_date (Score: 85/100)
```

## 🧪 Testing

The package includes a comprehensive test suite with **82 unit tests** covering all functions:

### Running Tests

```bash
# Run all tests using console script
entity-resolution-test

# Run all tests directly
python -m entity_resolution.run_tests

# Run individual test files
cd tests/
python test_entity_resolution.py
python test_entity_assignment.py  
python test_workflow.py

# Run with pytest (if installed)
pytest tests/ -v
```

### Test Coverage

- **✅ 82 total tests** - 100% pass rate
- **✅ 3+ tests per function** - Edge cases, expected outputs, exception handling
- **✅ Exception handling validation** - All invalid inputs properly handled
- **✅ Data compatibility testing** - Real pandas DataFrame operations
- **✅ Mock external dependencies** - Tests work without LangChain/OpenAI setup

### Test Categories

1. **Edge Cases**: Empty DataFrames, invalid inputs, boundary conditions
2. **Expected Outputs**: Known input/output pairs, normal operations
3. **Exception Handling**: Type validation, range validation, error conditions

See [TEST_SUMMARY.md](./TEST_SUMMARY.md) for detailed test documentation.

## ⚙️ Configuration

### OpenAI API Key (Required for LLM functionality)

```bash
# On Windows:
set OPENAI_API_KEY=your_api_key_here

# On macOS/Linux:
export OPENAI_API_KEY=your_api_key_here

# Or in .env file:
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Similarity Thresholds

- **90+**: Very conservative, only near-identical descriptions
- **80-89**: Balanced, catches most relevant matches (recommended)
- **70-79**: Liberal, may include false positives
- **<70**: Very liberal, high false positive risk

### LLM Settings

- **Temperature 0.1**: Consistent, factual descriptions (recommended)
- **Temperature 0.0**: Most deterministic output
- **Temperature 0.3+**: More creative but less consistent

### Custom LLM Configuration

```python
from entity_resolution import SimilarColumnFinder
from langchain.llms import OpenAI

# Custom LLM configuration
custom_llm = OpenAI(temperature=0.0, model_name="gpt-4")
finder = SimilarColumnFinder(llm=custom_llm)
```

### Environment Setup

#### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv entity-resolution-env

# Activate virtual environment
# On Windows:
entity-resolution-env\Scripts\activate
# On macOS/Linux:
source entity-resolution-env/bin/activate

# Install the package
pip install -e .
```

#### Conda Environment

```bash
# Create conda environment
conda create -n entity-resolution python=3.9
conda activate entity-resolution

# Install the package
pip install -e .
```

#### Docker Installation

```dockerfile
FROM python:3.9-slim

# Install the package
RUN pip install entity-resolution

# Set environment variables
ENV OPENAI_API_KEY=your_api_key_here

# Copy your data and scripts
COPY . /app
WORKDIR /app

# Run your entity resolution scripts
CMD ["python", "your_script.py"]
```

## 🚀 Performance

### Accuracy Characteristics

- **High Precision**: Few false positives due to semantic filtering
- **High Recall**: Catches variations that string matching misses
- **Confidence Scoring**: Enables human validation of uncertain matches

### Efficiency Features

- **Parallel Processing**: Column descriptions generated independently
- **Caching Friendly**: Descriptions can be reused across comparisons
- **Threshold Filtering**: Reduces unnecessary comparisons
- **Memory Optimization**: Configurable sample sizes for large datasets

### Performance Optimization

```bash
# Install with performance optimizations
pip install entity-resolution python-levenshtein[speedups]

# For GPU acceleration (if available)
pip install torch  # For neural LLMs
```

### Scalability

- Works with datasets of any size
- Parallel processing capability  
- Domain-agnostic approach
- Memory usage optimization for large datasets

## 🎯 Use Cases

### Data Integration
- Merging databases from acquired companies
- Consolidating data from multiple systems
- Preparing datasets for analytics platforms

### Data Migration
- Moving from legacy systems to modern platforms
- Cloud migration with schema transformation
- Database modernization projects

### Data Quality
- Identifying duplicate or redundant columns
- Validating data mapping assumptions
- Auditing data integration processes

## 🔧 Troubleshooting

### Common Issues

#### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'langchain_community'`

**Solution**:
```bash
pip install langchain-community
```

**Issue**: `ModuleNotFoundError: No module named 'fuzzywuzzy'`

**Solution**:
```bash
pip install fuzzywuzzy python-levenshtein
```

#### OpenAI API Issues

**Issue**: OpenAI API key not found

**Solution**: Set the OPENAI_API_KEY environment variable as shown above.

**Issue**: OpenAI rate limits

**Solution**: Use a lower temperature setting or implement retry logic.

#### Memory Issues with Large Datasets

**Solution**: Use the `sample_size` parameter in validation functions:
```python
finder.validate_column_matches(df1, df2, matches, sample_size=500)
```

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Create a clean environment
pip install --force-reinstall entity-resolution

# Or use conda for dependency management
conda install -c conda-forge pandas numpy
pip install entity-resolution
```

### Package Structure Issues

If imports fail, verify the package structure:

```bash
python -c "import sys; print('\n'.join(sys.path))"
python -c "import entity_resolution; print(entity_resolution.__file__)"
```

### Dependency Check

```python
import entity_resolution
entity_resolution.check_dependencies()
```

## 🤝 Contributing

We welcome contributions! For contributing to the package:

```bash
# Clone and install in development mode
git clone https://github.com/zacharyking/entity-resolution.git
cd entity-resolution

# Install with development dependencies
pip install -e .[dev,test]

# Install pre-commit hooks
pre-commit install

# Run tests
entity-resolution-test

# Run linting
black src/ tests/
flake8 src/ tests/
```

### Development Guidelines

- Code style and standards (Black, Flake8)
- Testing requirements (3+ tests per function)
- Documentation expectations
- Pull request process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🗺️ Roadmap

### Planned Features
- Support for additional LLM providers (Anthropic, Cohere, etc.)
- Statistical column analysis integration
- Web-based visualization interface
- Batch processing for large datasets
- Column type inference and validation

### Performance Improvements
- Async processing for faster execution
- Caching layer for repeated analyses
- Optimized fuzzy matching algorithms
- Memory usage optimization for large datasets

## 📞 Support

- **Documentation**: [GitHub README](https://github.com/zacharyking/entity-resolution)
- **Issues**: [GitHub Issues](https://github.com/zacharyking/entity-resolution/issues)
- **Email**: contact@entity-resolution.com

## 🏆 Why This Approach Works

### 1. Semantic Understanding
- Captures meaning beyond surface-level naming
- Handles abbreviations and terminology differences
- Works across different domains and industries

### 2. Informtion Theory Robustness
- Handles minor phrasing variations in descriptions
- Provides confidence scores for decision-making
- Filters out clearly unrelated columns

### 3. Scalability
- Works with datasets of any size
- Parallel processing capability
- Domain-agnostic approach

### 4. Explainability
- Human-readable descriptions
- Clear confidence scores
- Traceable decision process

---

*Built with care for the data integration community* 🎯

*Happy entity resolving!* 🚀