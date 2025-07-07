#!/usr/bin/env python3
"""
Simple test runner that tests individual components safely
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import Mock, patch
import traceback

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic module imports"""
    print("Testing basic imports...")
    results = []
    
    # Test H2 station model import
    try:
        from h2_station_model import H2StationSitingModel
        model = H2StationSitingModel()
        print("✓ H2StationSitingModel import and initialization")
        results.append(("H2StationSitingModel", True, None))
    except Exception as e:
        print(f"✗ H2StationSitingModel import failed: {e}")
        results.append(("H2StationSitingModel", False, str(e)))
    
    # Test quick setup import (without execution)
    try:
        import h2_station_quick_setup
        print("✓ h2_station_quick_setup import")
        results.append(("h2_station_quick_setup", True, None))
    except Exception as e:
        print(f"✗ h2_station_quick_setup import failed: {e}")
        results.append(("h2_station_quick_setup", False, str(e)))
    
    # Test test fixtures
    try:
        from test_fixtures import TestDataGenerator, TestFileManager
        generator = TestDataGenerator()
        print("✓ test_fixtures import")
        results.append(("test_fixtures", True, None))
    except Exception as e:
        print(f"✗ test_fixtures import failed: {e}")
        results.append(("test_fixtures", False, str(e)))
    
    return results

def test_data_generation():
    """Test data generation capabilities"""
    print("\nTesting data generation...")
    results = []
    
    try:
        from test_fixtures import TestDataGenerator, TestFileManager
        
        generator = TestDataGenerator()
        file_manager = TestFileManager()
        
        # Test route generation
        routes = generator.generate_california_routes(n_routes=5)
        print(f"✓ Generated {len(routes)} test routes")
        results.append(("route_generation", True, f"{len(routes)} routes"))
        
        # Test LOI generation
        lois = generator.generate_california_lois(n_lois=3)
        print(f"✓ Generated {len(lois)} test LOIs")
        results.append(("loi_generation", True, f"{len(lois)} LOIs"))
        
        # Test file management
        temp_dir = file_manager.create_temp_dir()
        test_file = file_manager.save_test_data(routes, "test_routes.geojson", temp_dir)
        print(f"✓ Saved test data to {test_file}")
        results.append(("file_management", True, "Files saved"))
        
        # Cleanup
        file_manager.cleanup()
        print("✓ Cleanup completed")
        results.append(("cleanup", True, "Completed"))
        
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        results.append(("data_generation", False, str(e)))
        
    return results

def test_model_initialization():
    """Test model initialization with different configurations"""
    print("\nTesting model initialization...")
    results = []
    
    try:
        from h2_station_model import H2StationSitingModel
        
        # Test default initialization
        model1 = H2StationSitingModel()
        print("✓ Default model initialization")
        results.append(("default_init", True, "Success"))
        
        # Test custom configuration
        custom_config = {
            'model_name': 'TestModel',
            'service_radius_miles': 3.0,
            'station_capacity_kg_per_day': 1500,
            'h2_price_per_kg': 30.0
        }
        model2 = H2StationSitingModel(custom_config)
        print("✓ Custom model initialization")
        results.append(("custom_init", True, "Success"))
        
        # Test configuration validation
        assert model2.config['model_name'] == 'TestModel'
        assert model2.config['service_radius_miles'] == 3.0
        print("✓ Configuration validation")
        results.append(("config_validation", True, "Success"))
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        results.append(("model_initialization", False, str(e)))
        
    return results

def test_spatial_operations():
    """Test spatial operations with mock data"""
    print("\nTesting spatial operations...")
    results = []
    
    try:
        import geopandas as gpd
        import pandas as pd
        from shapely.geometry import Point, LineString
        import numpy as np
        
        # Create test geometries
        points = [Point(i, i) for i in range(5)]
        lines = [LineString([(i, i), (i+1, i+1)]) for i in range(3)]
        
        # Test GeoDataFrame creation
        gdf_points = gpd.GeoDataFrame({'id': range(5)}, geometry=points, crs="EPSG:4326")
        gdf_lines = gpd.GeoDataFrame({'id': range(3)}, geometry=lines, crs="EPSG:4326")
        
        print(f"✓ Created {len(gdf_points)} point geometries")
        print(f"✓ Created {len(gdf_lines)} line geometries")
        results.append(("geometry_creation", True, f"{len(gdf_points)} points, {len(gdf_lines)} lines"))
        
        # Test coordinate system conversion
        gdf_projected = gdf_points.to_crs("EPSG:3310")
        print("✓ Coordinate system conversion")
        results.append(("crs_conversion", True, "EPSG:4326 to EPSG:3310"))
        
        # Test spatial operations
        distances = gdf_points.geometry.distance(Point(2, 2))
        print(f"✓ Distance calculations (max: {distances.max():.2f})")
        results.append(("distance_calc", True, f"Max distance: {distances.max():.2f}"))
        
    except Exception as e:
        print(f"✗ Spatial operations failed: {e}")
        results.append(("spatial_operations", False, str(e)))
        
    return results

def test_loi_workflow_functions():
    """Test LOI workflow functions without full execution"""
    print("\nTesting LOI workflow functions...")
    results = []
    
    try:
        # Import workflow functions (avoiding main execution)
        import loi_route_matching_workflow as workflow
        
        # Test helper functions
        test_columns = ['tot_truck_aadt', 'truck_aadt_ab', 'truck_aadt_ba', 'other']
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(test_columns)
        
        assert tot_col == 'tot_truck_aadt'
        assert ab_col == 'truck_aadt_ab'
        assert ba_col == 'truck_aadt_ba'
        print("✓ Flow column detection")
        results.append(("flow_detection", True, "Columns detected correctly"))
        
        # Test geometry bounds
        from shapely.geometry import Point
        point = Point(0, 0)
        bounds = workflow.get_geometry_bounds(point)
        assert len(bounds) == 4
        print("✓ Geometry bounds calculation")
        results.append(("geometry_bounds", True, "Bounds calculated"))
        
    except Exception as e:
        print(f"✗ LOI workflow functions failed: {e}")
        results.append(("loi_workflow", False, str(e)))
        
    return results

def test_quick_setup_functions():
    """Test quick setup utility functions"""
    print("\nTesting quick setup functions...")
    results = []
    
    try:
        import h2_station_quick_setup as quick_setup
        
        # Test version directory function
        test_base = "/tmp/test_results"
        next_version = quick_setup.get_next_version_dir(test_base)
        expected = f"{test_base}_v1"
        assert next_version == expected
        print("✓ Version directory generation")
        results.append(("version_dir", True, f"Generated: {next_version}"))
        
        # Test mock model creation for save_iteration_results
        mock_model = Mock()
        mock_model.iteration_history = {
            0: {'station_id': 1, 'base_demand': 1000, 'npv': 500000}
        }
        mock_model.demand_evolution_matrix = pd.DataFrame({'iter_0': [1000]})
        
        temp_dir = tempfile.mkdtemp()
        result_dir = quick_setup.save_iteration_results(mock_model, temp_dir)
        print("✓ Iteration results saving")
        results.append(("iteration_results", True, f"Saved to: {result_dir}"))
        
    except Exception as e:
        print(f"✗ Quick setup functions failed: {e}")
        results.append(("quick_setup", False, str(e)))
        
    return results

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("="*60)
    print("COMPREHENSIVE COMPONENT TESTS")
    print("="*60)
    
    all_results = []
    
    # Run all test suites
    test_suites = [
        ("Basic Imports", test_basic_imports),
        ("Data Generation", test_data_generation),
        ("Model Initialization", test_model_initialization),
        ("Spatial Operations", test_spatial_operations),
        ("LOI Workflow Functions", test_loi_workflow_functions),
        ("Quick Setup Functions", test_quick_setup_functions),
    ]
    
    for suite_name, test_func in test_suites:
        print(f"\n{'-'*40}")
        print(f"Running {suite_name} tests...")
        print(f"{'-'*40}")
        
        try:
            suite_results = test_func()
            all_results.extend(suite_results)
        except Exception as e:
            print(f"✗ {suite_name} test suite failed: {e}")
            all_results.append((suite_name, False, str(e)))
    
    return all_results

def print_summary(results):
    """Print test results summary"""
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success, _ in results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    
    print(f"\nDETAILED RESULTS:")
    print("-" * 40)
    for test_name, success, details in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<25} {status:>6}")
        if details and not success:
            print(f"  Error: {details[:100]}...")
        elif details and success:
            print(f"  Details: {details}")
    
    print("\n" + "="*60)
    
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED!")
        return 0
    elif passed_tests / total_tests >= 0.75:
        print("⚠️  Most tests passed, some issues to address")
        return 1
    else:
        print("❌ Significant test failures detected")
        return 2

def main():
    """Main test execution"""
    try:
        results = run_comprehensive_tests()
        return print_summary(results)
    except Exception as e:
        print(f"Critical error during test execution: {e}")
        traceback.print_exc()
        return 3

if __name__ == '__main__':
    import pandas as pd  # Import here to avoid early import issues
    exit_code = main()
    sys.exit(exit_code)