"""
Setup configuration for entity-resolution package

A sophisticated entity resolution system that uses Large Language Models (LLMs) 
and fuzzy matching to identify semantically similar columns across different datasets.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md file for long description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A sophisticated entity resolution system for intelligent column matching."

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Package metadata
setup(
    # Basic package information
    name="entity-resolution",
    version="1.0.0",
    
    # Author information
    author="Zachary King",
    author_email="contact@entity-resolution.com",
    
    # Package description
    description="Intelligent column matching using LLMs and fuzzy matching for entity resolution",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/zacharyking/entity-resolution",
    project_urls={
        "Bug Reports": "https://github.com/zacharyking/entity-resolution/issues",
        "Source": "https://github.com/zacharyking/entity-resolution",
        "Documentation": "https://github.com/zacharyking/entity-resolution/blob/main/README.md",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "entity_resolution": [
            "*.md",
            "*.txt",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies for development and testing
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "unittest-xml-reporting>=3.2.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "entity-resolution-demo=entity_resolution.example_usage:main",
            "entity-resolution-test=entity_resolution.run_tests:main",
        ],
    },
    
    # Package classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Data Analysts",
        "Intended Audience :: Science/Research",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Framework
        "Framework :: Jupyter",
        
        # Natural Language
        "Natural Language :: English",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "entity-resolution",
        "data-integration", 
        "column-matching",
        "data-science",
        "machine-learning",
        "nlp",
        "fuzzy-matching",
        "llm",
        "langchain",
        "pandas",
        "data-quality",
        "schema-matching",
        "data-mapping",
        "artificial-intelligence",
        "data-engineering",
    ],
    
    # License
    license="MIT",
    
    # Zip safe
    zip_safe=False,
    
    # Test suite
    test_suite="tests",
    
    # Additional metadata for setuptools
    platforms=["any"],
)
