"""
Console script entry point for running entity resolution tests
"""

import sys
import os
import unittest

def main():
    """Main entry point for running tests via console script"""
    
    # Add the parent directory to the path so we can import the test modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    tests_dir = os.path.join(parent_dir, 'tests')
    
    # Check for missing dependencies
    try:
        import pandas as pd
        # Test basic imports without the full dependency chain
        print("✅ Basic dependencies available")
    except ImportError as e:
        print("❌ Error: Missing basic dependencies for running tests")
        print(f"   {e}")
        print("\nTo install missing dependencies, run:")
        print("   pip install pandas")
        return 1
    
    if os.path.exists(tests_dir):
        sys.path.insert(0, tests_dir)
        sys.path.insert(0, os.path.join(parent_dir, 'src'))
        
        print("Entity Resolution Test Suite")
        print("=" * 40)
        
        # Discover and run tests
        loader = unittest.TestLoader()
        start_dir = tests_dir
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
        result = runner.run(suite)
        
        # Print summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.failures:
            print(f"\nFAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"- {test}")
        
        if result.errors:
            print(f"\nERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"- {test}")
        
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        return 0 if result.wasSuccessful() else 1
    else:
        print(f"Tests directory not found: {tests_dir}")
        print("Please run this command from the entity-resolution package root directory.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
