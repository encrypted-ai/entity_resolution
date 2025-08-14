"""
Example Usage - Complete Entity Resolution Column Matching Demo
==============================================================

This example demonstrates how to use the SimilarColumnFinder for a realistic 
entity resolution scenario where two customer databases from different systems 
need to be integrated, but their column names follow different conventions.

Business Context:
----------------
Company A (e-commerce platform) is acquiring Company B (subscription service).
Both companies have customer databases, but they use different naming conventions:
- Company A uses technical abbreviations: 'cust_id', 'name', 'order_date'
- Company B uses business-friendly names: 'customer_number', 'full_name', 'transaction_date'

The Challenge:
-------------
Before merging these databases, we need to identify which columns contain 
the same type of information so we can:
1. Map columns correctly during data integration
2. Avoid duplicate or misaligned data 
3. Ensure referential integrity across systems
4. Enable unified customer analytics and reporting
"""

def main():
    """Main entry point that handles dependencies gracefully"""
    try:
        import pandas as pd
        from .entity_resolution import SimilarColumnFinder
        from .workflow import LangGraphWorkflow
        
        # Run the actual demo
        _run_demo(pd, SimilarColumnFinder, LangGraphWorkflow)
        
    except ImportError as e:
        print("❌ Error: Missing dependencies for entity-resolution demo")
        print(f"   {e}")
        print("\nTo install missing dependencies, run:")
        print("   pip install langchain openai fuzzywuzzy pandas python-levenshtein")
        print("   pip install langchain-community  # For LangChain community features")
        return 1
    
    return 0

def _run_demo(pd, SimilarColumnFinder, LangGraphWorkflow):
    """
    Main demonstration function showing complete entity resolution workflow
    """
    print("=" * 80)
    print("ENTITY RESOLUTION DEMO: Customer Database Integration")
    print("=" * 80)
    
    # STEP 1: Create representative sample datasets
    # These represent the different naming conventions used by each company
    print("\n1. SETTING UP SAMPLE DATASETS")
    print("-" * 40)
    
    # Company A Dataset - Technical naming convention
    data1 = {
        'cust_id': [1, 2, 3],  # Customer identifier (technical)
        'name': ['Alice', 'Bob', 'Charlie'],  # Customer name (abbreviated)
        'order_date': ['2023-01-01', '2023-01-02', '2023-01-03']  # Purchase date (technical)
    }
    
    # Company B Dataset - Business-friendly naming convention  
    data2 = {
        'customer_number': [1, 2, 3],  # Customer identifier (business-friendly)
        'full_name': ['Alice', 'Bob', 'Charlie'],  # Customer name (descriptive)
        'transaction_date': ['2023-01-01', '2023-01-02', '2023-01-03']  # Purchase date (business-friendly)
    }

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    print(f"Company A Dataset (df1): {list(df1.columns)}")
    print(f"Company B Dataset (df2): {list(df2.columns)}")
    print(f"Challenge: Column names are completely different, but they contain the same information!")

    # STEP 2: Initialize the Entity Resolution System
    print("\n2. INITIALIZING ENTITY RESOLUTION SYSTEM")
    print("-" * 40)
    
    # Create the SimilarColumnFinder with default OpenAI model
    # The low temperature (0.1) ensures consistent, factual column descriptions
    finder = SimilarColumnFinder()  
    print("CHECKMARK SimilarColumnFinder initialized with OpenAI LLM")
    print("CHECKMARK Using semantic analysis + fuzzy matching approach")

    # STEP 3: Execute the Entity Resolution Pipeline
    print("\n3. EXECUTING ENTITY RESOLUTION PIPELINE")
    print("-" * 40)
    
    # This is where the magic happens - the system will:
    # a) Generate semantic descriptions for all columns using the LLM
    # b) Compare descriptions using fuzzy string matching  
    # c) Identify columns that likely contain the same information
    # d) Return confidence scores for each potential match
    similar_cols = finder.find_similar_columns(df1, df2, threshold=75)

    # STEP 4: Analyze and Present Results
    print("\n4. ENTITY RESOLUTION RESULTS")
    print("-" * 40)
    
    if similar_cols:
        print(f"SUCCESS Found {len(similar_cols)} potential column matches:")
        print()
        
        for col1, col2, score in similar_cols:
            # Determine confidence level based on score
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
    else:
        print("ERROR No column matches found above the similarity threshold")
        print("   Consider lowering the threshold or checking data quality")

    # STEP 4.5: Enhanced Validation with Data Analysis
    if similar_cols:
        print("\n4.5. ENHANCED VALIDATION WITH DATA ANALYSIS")
        print("-" * 40)
        print("MICROSCOPE Performing deep validation of column matches...")
        print("   - Analyzing data types compatibility")
        print("   - Checking value overlaps and distributions")
        print("   - Generating comprehensive compatibility scores")
        print()
        
        # Validate the matches using actual data content
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
            
            if match['validation_flags']:
                print(f"  WARNING FLAGS: {', '.join(match['validation_flags'])}")
            
            # Show some data type details
            type_analysis = match['data_type_analysis']
            print(f"  Data Types: {type_analysis['series1_type']} vs {type_analysis['series2_type']}")
            
            # Show value overlap details if there's overlap
            overlap_analysis = match['value_overlap_analysis']
            if overlap_analysis['common_values']:
                sample_common = list(overlap_analysis['common_values'])[:5]  # Show first 5
                print(f"  Common Values (sample): {sample_common}")
        
        print("\n" + "=" * 50)

    # STEP 5: Explain the Business Impact
    print("\n5. BUSINESS IMPACT & NEXT STEPS")
    print("-" * 40)
    
    if similar_cols:
        print("SUCCESS SUCCESSFUL ENTITY RESOLUTION:")
        print("   - Column mappings identified for data integration")
        print("   - Data compatibility validated through comprehensive analysis")
        print("   - Can proceed with confidence to merge customer databases") 
        print("   - Reduced risk of data misalignment or duplication")
        print("   - Foundation established for unified customer analytics")
        print()
        print("CLIPBOARD RECOMMENDED ACTIONS:")
        print("   1. Focus on 'EXCELLENT' and 'GOOD' validated matches for immediate integration")
        print("   2. Review 'FAIR' matches for potential data transformation needs")
        print("   3. Investigate 'POOR' or 'REJECTED' matches manually")
        print("   4. Address validation flags (data type conversion, value mapping)")
        print("   5. Use these mappings to create data transformation pipelines")
    else:
        print("WARNING ENTITY RESOLUTION CHALLENGES:")
        print("   - No clear column mappings found automatically")
        print("   - May need manual analysis or domain expertise")
        print("   - Consider adjusting similarity threshold or improving data quality")

    # STEP 6: Optional Workflow Visualization (commented for demo)
    print("\n6. OPTIONAL: WORKFLOW VISUALIZATION")
    print("-" * 40)
    print("LIGHTBULB TIP: Uncomment the lines below to visualize the entity resolution workflow:")
    print("   workflow = LangGraphWorkflow(finder)")
    print("   graph = workflow.create_column_matching_graph(df1, df2)")
    print("   graph.visualize()  # Shows the processing pipeline as a graph")
    print()
    print("   This helps with:")
    print("   - Understanding the step-by-step process")
    print("   - Debugging issues in the pipeline")
    print("   - Monitoring performance of each stage")
    print("   - Extending the workflow with additional steps")
    
    # Uncomment these lines to see the workflow visualization:
    # workflow = LangGraphWorkflow(finder)
    # graph = workflow.create_column_matching_graph(df1, df2)
    # graph.visualize()
    
    # STEP 7: Demonstrate Dataset Consolidation
    print("\n7. DATASET CONSOLIDATION DEMONSTRATION")
    print("-" * 40)
    
    if similar_cols:
        print("GEAR Demonstrating automatic dataset consolidation...")
        
        # Create workflow and consolidate datasets
        workflow = LangGraphWorkflow(finder)
        consolidated_df, method, metadata = workflow.consolidate_datasets_based_on_matches(
            df1, df2, similar_cols, match_threshold=0.7
        )
        
        print(f"CONSOLIDATION METHOD: {method.upper()}")
        print(f"REASON: Average match score of top {metadata['matches_considered']} matches = {metadata['average_match_score']:.3f}")
        print(f"Threshold: {metadata['threshold']}")
        print()
        
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
            
        print()
        print("PREVIEW OF CONSOLIDATED DATASET:")
        print(consolidated_df.head())
        
        print()
        print("CHECKMARK CONSOLIDATION BENEFITS:")
        print("  - Semantically cleaner column labels applied")
        print("  - Data integrity preserved")
        print("  - Source tracking maintained (for concatenation)")
        print("  - Ready for downstream analytics and reporting")
    
    print("\n" + "=" * 80)
    print("ENTITY RESOLUTION DEMO COMPLETE")
    print("=" * 80)


def demonstrate_advanced_features():
    """
    Additional examples showing advanced features and customization options
    """
    try:
        import pandas as pd
        from .entity_resolution import SimilarColumnFinder
        
        _run_advanced_demo(pd, SimilarColumnFinder)
        
    except ImportError as e:
        print("❌ Error: Missing dependencies for advanced features demo")
        print(f"   {e}")
        return

def _run_advanced_demo(pd, SimilarColumnFinder):
    """
    Run the advanced features demonstration
    """
    print("\n" + "=" * 80)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 80)
    
    # Example with custom threshold
    print("\n1. CUSTOM SIMILARITY THRESHOLD")
    print("-" * 40)
    
    data1 = {'prod_id': [1, 2], 'description': ['Widget A', 'Widget B']}
    data2 = {'product_code': [1, 2], 'desc': ['Widget A', 'Widget B']}
    
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    
    finder = SimilarColumnFinder()
    
    # Try different thresholds
    for threshold in [90, 80, 70]:
        print(f"\nThreshold {threshold}:")
        matches = finder.find_similar_columns(df1, df2, threshold=threshold)
        print(f"  Found {len(matches)} matches")
        for col1, col2, score in matches:
            print(f"    {col1} <-> {col2} (Score: {score})")
    
    print("\nCONCLUSION: Different thresholds affect match sensitivity")


if __name__ == '__main__':
    """
    Run the complete demonstration
    """
    exit_code = main()
    if exit_code == 0:
        demonstrate_advanced_features()
    exit(exit_code)
