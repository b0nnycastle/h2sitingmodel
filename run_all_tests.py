#!/usr/bin/env python3
"""
Master test runner for all H2 Station Siting Model components
"""

import unittest
import sys
import os
import time
from datetime import datetime
import traceback

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all test modules
try:
    from test_h2_station_model import TestH2StationSitingModel, TestH2StationModelComponents, TestDataValidation
    print("✓ Imported H2 Station Model tests")
except ImportError as e:
    print(f"✗ Failed to import H2 Station Model tests: {e}")
    TestH2StationSitingModel = None
    TestH2StationModelComponents = None
    TestDataValidation = None

try:
    from test_loi_route_matching import TestLOIRouteMatchingWorkflow
    print("✓ Imported LOI Route Matching tests")
except ImportError as e:
    print(f"✗ Failed to import LOI Route Matching tests: {e}")
    TestLOIRouteMatchingWorkflow = None

try:
    from test_quick_setup import TestQuickSetupFunctions, TestQuickSetupIntegration
    print("✓ Imported Quick Setup tests")
except ImportError as e:
    print(f"✗ Failed to import Quick Setup tests: {e}")
    TestQuickSetupFunctions = None
    TestQuickSetupIntegration = None

try:
    from test_fixtures import TestDataGenerator, TestFileManager, MockModelComponents
    print("✓ Imported Test Fixtures")
except ImportError as e:
    print(f"✗ Failed to import Test Fixtures: {e}")
    TestDataGenerator = None


class TestResults:
    """Track and report test results"""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_suites = []
        self.total_tests = 0
        self.total_failures = 0
        self.total_errors = 0
        self.failed_test_cases = []
        self.error_test_cases = []
        
    def add_suite_result(self, suite_name, result):
        """Add results from a test suite"""
        suite_info = {
            'name': suite_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
        }
        
        self.test_suites.append(suite_info)
        self.total_tests += result.testsRun
        self.total_failures += len(result.failures)
        self.total_errors += len(result.errors)
        
        # Track specific failed/error cases
        for test, traceback in result.failures:
            self.failed_test_cases.append((suite_name, str(test), traceback))
            
        for test, traceback in result.errors:
            self.error_test_cases.append((suite_name, str(test), traceback))
            
    def print_summary(self):
        """Print comprehensive test results summary"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*80)
        print(f"Test execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print()
        
        # Overall summary
        success_rate = (self.total_tests - self.total_failures - self.total_errors) / max(self.total_tests, 1) * 100
        print(f"OVERALL RESULTS:")
        print(f"  Total tests run: {self.total_tests}")
        print(f"  Successful: {self.total_tests - self.total_failures - self.total_errors}")
        print(f"  Failures: {self.total_failures}")
        print(f"  Errors: {self.total_errors}")
        print(f"  Success rate: {success_rate:.1f}%")
        print()
        
        # Per-suite breakdown
        print("PER-SUITE BREAKDOWN:")
        print("-" * 60)
        for suite in self.test_suites:
            status = "PASS" if suite['failures'] == 0 and suite['errors'] == 0 else "FAIL"
            print(f"  {suite['name']:<35} {status:>6} ({suite['success_rate']:.1f}%)")
            print(f"    Tests: {suite['tests_run']}, Failures: {suite['failures']}, Errors: {suite['errors']}")
        print()
        
        # Detailed failure/error reporting
        if self.failed_test_cases:
            print("DETAILED FAILURE REPORTS:")
            print("-" * 60)
            for i, (suite, test, tb) in enumerate(self.failed_test_cases[:5], 1):  # Limit to first 5
                print(f"  {i}. {suite} - {test}")
                print(f"     {tb.split('AssertionError:')[-1].strip() if 'AssertionError:' in tb else 'See full traceback'}")
                print()
                
        if self.error_test_cases:
            print("DETAILED ERROR REPORTS:")
            print("-" * 60)
            for i, (suite, test, tb) in enumerate(self.error_test_cases[:5], 1):  # Limit to first 5
                print(f"  {i}. {suite} - {test}")
                # Extract just the error message
                lines = tb.split('\n')
                error_line = next((line for line in lines if any(err in line for err in ['Error:', 'Exception:'])), lines[-2] if len(lines) > 1 else tb)
                print(f"     {error_line.strip()}")
                print()
                
        # Recommendations
        print("RECOMMENDATIONS:")
        print("-" * 40)
        if success_rate >= 90:
            print("  ✓ Excellent test coverage and reliability")
        elif success_rate >= 75:
            print("  ⚠ Good test coverage, some issues to address")
            print("  • Review failed test cases for potential bugs")
        elif success_rate >= 50:
            print("  ⚠ Moderate test coverage, significant issues present")
            print("  • Priority: Fix error test cases")
            print("  • Review and update failing tests")
        else:
            print("  ✗ Poor test coverage, major issues detected")
            print("  • Immediate action required")
            print("  • Review dependencies and environment setup")
            print("  • Consider debugging individual test modules")
            
        if self.total_errors > 0:
            print("  • Check import dependencies and environment setup")
        if self.total_failures > 0:
            print("  • Review test assertions and expected behavior")
            
        print()
        print("="*80)


def create_test_suite():
    """Create comprehensive test suite from all available test classes"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        # Core model tests
        (TestH2StationSitingModel, "H2 Station Siting Model - Core"),
        (TestH2StationModelComponents, "H2 Station Siting Model - Components"),
        (TestDataValidation, "Data Validation"),
        
        # Workflow tests
        (TestLOIRouteMatchingWorkflow, "LOI Route Matching Workflow"),
        
        # Setup script tests
        (TestQuickSetupFunctions, "Quick Setup Functions"),
        (TestQuickSetupIntegration, "Quick Setup Integration"),
    ]
    
    available_suites = []
    
    for test_class, description in test_classes:
        if test_class is not None:
            try:
                test_suite = loader.loadTestsFromTestCase(test_class)
                suite.addTest(test_suite)
                available_suites.append(description)
                print(f"✓ Added {description} tests")
            except Exception as e:
                print(f"✗ Failed to load {description} tests: {e}")
        else:
            print(f"⚠ Skipping {description} tests (import failed)")
            
    print(f"\nTotal test suites loaded: {len(available_suites)}")
    return suite, available_suites


def run_individual_test_suites():
    """Run each test suite individually for better error isolation"""
    results = TestResults()
    
    test_modules = [
        ('test_h2_station_model', [TestH2StationSitingModel, TestH2StationModelComponents, TestDataValidation]),
        ('test_loi_route_matching', [TestLOIRouteMatchingWorkflow]),
        ('test_quick_setup', [TestQuickSetupFunctions, TestQuickSetupIntegration]),
    ]
    
    for module_name, test_classes in test_modules:
        print(f"\n{'='*60}")
        print(f"RUNNING {module_name.upper()} TESTS")
        print(f"{'='*60}")
        
        module_suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        
        for test_class in test_classes:
            if test_class is not None:
                try:
                    class_suite = loader.loadTestsFromTestCase(test_class)
                    module_suite.addTest(class_suite)
                except Exception as e:
                    print(f"Error loading {test_class.__name__}: {e}")
                    
        if module_suite.countTestCases() > 0:
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            try:
                result = runner.run(module_suite)
                results.add_suite_result(module_name, result)
            except Exception as e:
                print(f"Error running {module_name} tests: {e}")
                traceback.print_exc()
        else:
            print(f"No tests found for {module_name}")
            
    return results


def run_quick_smoke_tests():
    """Run a quick subset of critical tests for smoke testing"""
    print("\n" + "="*60)
    print("RUNNING QUICK SMOKE TESTS")
    print("="*60)
    
    # Test basic imports and initialization
    smoke_tests = []
    
    try:
        from h2_station_model import H2StationSitingModel
        model = H2StationSitingModel()
        print("✓ H2StationSitingModel initialization")
        smoke_tests.append(True)
    except Exception as e:
        print(f"✗ H2StationSitingModel initialization failed: {e}")
        smoke_tests.append(False)
        
    try:
        import loi_route_matching_workflow as workflow
        print("✓ LOI route matching workflow import")
        smoke_tests.append(True)
    except Exception as e:
        print(f"✗ LOI route matching workflow import failed: {e}")
        smoke_tests.append(False)
        
    try:
        import h2_station_quick_setup as quick_setup
        print("✓ Quick setup script import")
        smoke_tests.append(True)
    except Exception as e:
        print(f"✗ Quick setup script import failed: {e}")
        smoke_tests.append(False)
        
    try:
        from test_fixtures import TestDataGenerator
        generator = TestDataGenerator()
        routes = generator.generate_california_routes(n_routes=5)
        print(f"✓ Test data generation ({len(routes)} routes created)")
        smoke_tests.append(True)
    except Exception as e:
        print(f"✗ Test data generation failed: {e}")
        smoke_tests.append(False)
        
    success_rate = sum(smoke_tests) / len(smoke_tests) * 100
    print(f"\nSmoke test success rate: {success_rate:.1f}%")
    
    return success_rate >= 75


def main():
    """Main test execution function"""
    print("H2 STATION SITING MODEL - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if we should run smoke tests first
    if len(sys.argv) > 1 and sys.argv[1] == '--smoke':
        smoke_success = run_quick_smoke_tests()
        if not smoke_success:
            print("\n⚠ Smoke tests failed. Full test suite may have issues.")
            return 1
        else:
            print("\n✓ Smoke tests passed. Proceeding with full test suite.")
    
    # Run comprehensive tests
    try:
        results = run_individual_test_suites()
        results.print_summary()
        
        # Determine exit code
        if results.total_tests == 0:
            print("⚠ No tests were executed")
            return 2
        elif results.total_errors > 0:
            print("✗ Test suite completed with errors")
            return 3
        elif results.total_failures > 0:
            print("⚠ Test suite completed with failures")
            return 1
        else:
            print("✓ All tests passed successfully")
            return 0
            
    except Exception as e:
        print(f"✗ Critical error during test execution: {e}")
        traceback.print_exc()
        return 4


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)