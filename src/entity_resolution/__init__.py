"""
Entity Resolution Package

A sophisticated entity resolution system that uses Large Language Models (LLMs) 
and fuzzy matching to identify semantically similar columns across different datasets.

Main Classes:
    SimilarColumnFinder: Core entity resolution engine
    SemanticConsolidation: Semantic pattern analysis
    EntityAssignment: Dataset consolidation logic
    LangGraphWorkflow: Workflow visualization and orchestration

Example Usage:
    from entity_resolution import SimilarColumnFinder
    import pandas as pd
    
    # Create sample datasets
    df1 = pd.DataFrame({'cust_id': [1,2,3], 'name': ['A','B','C']})
    df2 = pd.DataFrame({'customer_num': [1,2,3], 'full_name': ['A','B','C']})
    
    # Initialize the system
    finder = SimilarColumnFinder()
    
    # Find similar columns
    matches = finder.find_similar_columns(df1, df2, threshold=80)
    
    # Review results
    for col1, col2, score in matches:
        print(f"{col1} <-> {col2} (Confidence: {score}%)")
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Zachary King"
__email__ = "contact@entity-resolution.com"
__license__ = "MIT"
__description__ = "Intelligent column matching using LLMs and fuzzy matching for entity resolution"

# Main imports - conditional to handle missing dependencies gracefully
_IMPORTS_AVAILABLE = True
_IMPORT_ERRORS = []

try:
    from .entity_assignment import SemanticConsolidation, EntityAssignment
except ImportError as e:
    _IMPORTS_AVAILABLE = False
    _IMPORT_ERRORS.append(f"entity_assignment: {e}")
    SemanticConsolidation = None
    EntityAssignment = None

try:
    from .entity_resolution import SimilarColumnFinder
except ImportError as e:
    _IMPORTS_AVAILABLE = False
    _IMPORT_ERRORS.append(f"entity_resolution: {e}")
    SimilarColumnFinder = None

try:
    from .workflow import LangGraphWorkflow
except ImportError as e:
    _IMPORTS_AVAILABLE = False
    _IMPORT_ERRORS.append(f"workflow: {e}")
    LangGraphWorkflow = None

# Make main classes available at package level
__all__ = [
    "SimilarColumnFinder",
    "SemanticConsolidation", 
    "EntityAssignment",
    "LangGraphWorkflow",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    # Utility functions
    "get_version",
    "get_info", 
    "quick_start",
    "check_dependencies"
]

# Package-level configuration
DEFAULT_SIMILARITY_THRESHOLD = 80
DEFAULT_MATCH_THRESHOLD = 0.7
DEFAULT_SAMPLE_SIZE = 1000

# Utility functions
def get_version():
    """Get package version"""
    return __version__

def get_info():
    """Get package information"""
    return {
        "name": "entity-resolution",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": __description__
    }

def check_dependencies():
    """Check if all dependencies are available and provide helpful error messages"""
    if _IMPORTS_AVAILABLE:
        print("✅ All dependencies are available!")
        return True
    else:
        print("❌ Some dependencies are missing:")
        for error in _IMPORT_ERRORS:
            print(f"   - {error}")
        print("\nTo install missing dependencies, run:")
        print("   pip install langchain openai fuzzywuzzy pandas python-levenshtein")
        print("   pip install langchain-community  # For LangChain community features")
        return False

def quick_start():
    """Print quick start guide"""
    if not _IMPORTS_AVAILABLE:
        print(f"Entity Resolution Package v{__version__}")
        print("=" * 50)
        print("⚠️  Warning: Some dependencies are missing!")
        check_dependencies()
        print("\nPlease install dependencies before using the package.")
        return
    
    print(f"""
Entity Resolution Package v{__version__}
=======================================

Quick Start:

1. Import the package:
   from entity_resolution import SimilarColumnFinder

2. Create sample data:
   import pandas as pd
   df1 = pd.DataFrame({{'cust_id': [1,2,3], 'name': ['A','B','C']}})
   df2 = pd.DataFrame({{'customer_num': [1,2,3], 'full_name': ['A','B','C']}})

3. Find similar columns:
   finder = SimilarColumnFinder()
   matches = finder.find_similar_columns(df1, df2, threshold=80)

4. Review results:
   for col1, col2, score in matches:
       print(f"{{col1}} <-> {{col2}} ({{score}}% confidence)")

For more examples, run: python -m entity_resolution.example_usage
For full documentation, visit: https://github.com/zacharyking/entity-resolution
""")
