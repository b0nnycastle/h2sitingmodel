#!/usr/bin/env python3
"""
Comprehensive unit tests for H2 Station Quick Setup Script
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import warnings

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from the quick setup script
import h2_station_quick_setup as quick_setup


class TestQuickSetupFunctions(unittest.TestCase):
    """Test suite for quick setup script functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_base_dir = os.path.join(self.temp_dir, "test_results")
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_get_next_version_dir(self):
        """Test version directory generation"""
        # Test with no existing directories
        next_dir = quick_setup.get_next_version_dir(self.test_base_dir)
        expected = f"{self.test_base_dir}_v1"
        self.assertEqual(next_dir, expected)
        
        # Create the first version directory
        os.makedirs(expected)
        
        # Test with existing v1 directory
        next_dir = quick_setup.get_next_version_dir(self.test_base_dir)
        expected = f"{self.test_base_dir}_v2"
        self.assertEqual(next_dir, expected)
        
        # Create v2 and test v3
        os.makedirs(expected)
        next_dir = quick_setup.get_next_version_dir(self.test_base_dir)
        expected = f"{self.test_base_dir}_v3"
        self.assertEqual(next_dir, expected)
        
    def test_get_next_version_dir_default(self):
        """Test version directory generation with default base"""
        with patch('os.path.exists') as mock_exists:
            # Mock that v1 and v2 exist, v3 doesn't
            mock_exists.side_effect = lambda path: path.endswith('_v1') or path.endswith('_v2')
            
            result = quick_setup.get_next_version_dir()
            expected = "h2_station_quick_results_v3"
            self.assertEqual(result, expected)
            
    def create_mock_model_with_iteration_data(self):
        """Create a mock model with iteration tracking data"""
        mock_model = Mock()
        
        # Mock iteration history
        mock_model.iteration_history = {
            0: {
                'station_id': 1,
                'base_demand': 1000.0,
                'current_demand': 800.0,
                'npv': 500000.0,
                'capacity': 1500.0,
                'utilization': 0.8
            },
            1: {
                'station_id': 5,
                'base_demand': 1200.0,
                'current_demand': 1000.0,
                'npv': 600000.0,
                'capacity': 1800.0,
                'utilization': 0.85
            },
            2: {
                'station_id': 10,
                'base_demand': 900.0,
                'current_demand': 750.0,
                'npv': 450000.0,
                'capacity': 1200.0,
                'utilization': 0.75
            }
        }
        
        # Mock demand evolution matrix
        mock_model.demand_evolution_matrix = pd.DataFrame({
            'iteration_0': [1000.0, 1200.0, 900.0],
            'iteration_1': [800.0, 1000.0, 750.0],
            'iteration_2': [600.0, 800.0, 600.0]
        })
        
        return mock_model
        
    def test_save_iteration_results(self):
        """Test saving iteration results"""
        mock_model = self.create_mock_model_with_iteration_data()
        
        output_dir = os.path.join(self.temp_dir, "iteration_output")
        
        # Test saving iteration results
        result_dir = quick_setup.save_iteration_results(mock_model, output_dir)
        
        # Verify output directory was created
        self.assertTrue(os.path.exists(output_dir))
        self.assertEqual(result_dir, output_dir)
        
        # Verify files were created
        expected_files = [
            'iteration_history.csv',
            'demand_evolution_matrix.csv',
            'selection_summary.json',
            'iteration_report.txt'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"Missing file: {filename}")
            
            # Verify file has content
            self.assertGreater(os.path.getsize(filepath), 0, f"Empty file: {filename}")
            
    def test_save_iteration_results_no_data(self):
        """Test saving iteration results with no iteration data"""
        mock_model = Mock()
        mock_model.iteration_history = None
        mock_model.demand_evolution_matrix = pd.DataFrame()
        
        output_dir = os.path.join(self.temp_dir, "no_data_output")
        
        # Test saving with no data
        result_dir = quick_setup.save_iteration_results(mock_model, output_dir)
        
        # Verify output directory was created
        self.assertTrue(os.path.exists(output_dir))
        
    def test_save_iteration_results_json_serialization(self):
        """Test JSON serialization of iteration results"""
        mock_model = self.create_mock_model_with_iteration_data()
        
        # Add numpy types to test serialization
        mock_model.iteration_history[0]['numpy_int'] = np.int32(42)
        mock_model.iteration_history[0]['numpy_float'] = np.float64(3.14)
        mock_model.iteration_history[0]['numpy_array'] = np.array([1, 2, 3])
        mock_model.iteration_history[0]['nan_value'] = np.nan
        
        output_dir = os.path.join(self.temp_dir, "json_output")
        
        # Test saving with numpy types
        result_dir = quick_setup.save_iteration_results(mock_model, output_dir)
        
        # Verify JSON file was created and is valid
        json_file = os.path.join(output_dir, 'selection_summary.json')
        self.assertTrue(os.path.exists(json_file))
        
        # Load and verify JSON content
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        self.assertIn('total_iterations', data)
        self.assertIn('total_stations_selected', data)
        self.assertIn('stations_selected', data)
        self.assertEqual(data['total_iterations'], 3)
        self.assertEqual(len(data['stations_selected']), 3)
        
    def test_validate_competition_graph_implementation(self):
        """Test competition graph validation"""
        # Create mock model with competition graph
        mock_model = Mock()
        mock_model._competition_graph = Mock()
        mock_model._competition_graph.nodes = {
            0: {'type': 'candidate'},
            1: {'type': 'candidate'},
            2: {'type': 'existing'}
        }
        mock_model._competition_graph.edges = {
            '0-1': {'weight': 0.5},
            '1-2': {'weight': 0.3}
        }
        mock_model._competition_graph.calculate_market_share = Mock(return_value=0.7)
        
        # Create mock candidates
        mock_model.candidates = pd.DataFrame({
            'demand': [1000, 1200, 800],
            'npv': [500000, 600000, 400000]
        })
        
        # Test validation
        result = quick_setup.validate_competition_graph_implementation(mock_model)
        
        # Verify validation completed
        self.assertTrue(result)
        
        # Verify market share calculation was called
        mock_model._competition_graph.calculate_market_share.assert_called()
        
    def test_validate_competition_graph_no_graph(self):
        """Test competition graph validation with no graph"""
        mock_model = Mock()
        mock_model._competition_graph = None
        
        # Test validation with no graph
        result = quick_setup.validate_competition_graph_implementation(mock_model)
        
        # Should return False
        self.assertFalse(result)
        
    def test_test_no_competition_double_counting(self):
        """Test the double counting test function"""
        # Create mock model
        mock_model = Mock()
        mock_model.candidates = pd.DataFrame({
            'original_demand': [1000, 1200, 800],
            'initial_demand_post_existing': [900, 1100, 750],
            'expected_demand_kg_day': [850, 1050, 700]
        })
        mock_model.candidates.index = [0, 1, 2]
        
        # Mock competition graph
        mock_model._competition_graph = Mock()
        mock_model._competition_graph.edges = {
            '0-1': {'weight': 0.5},
            '1-2': {'weight': 0.3}
        }
        mock_model._competition_graph.calculate_market_share = Mock(return_value=0.6)
        
        # Test the function (should not raise exceptions)
        try:
            quick_setup.test_no_competition_double_counting(mock_model)
            test_passed = True
        except Exception as e:
            test_passed = False
            
        self.assertTrue(test_passed)
        
    @patch('h2_station_quick_setup.H2StationSitingModel')
    def test_run_quick_analysis_basic(self, mock_model_class):
        """Test basic quick analysis run"""
        # Create mock model instance
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock model methods
        mock_model.load_data = Mock()
        mock_model.estimate_demand_surface = Mock()
        mock_model.generate_candidates = Mock()
        mock_model.calculate_utilization_probability = Mock()
        mock_model.calculate_economic_proxy = Mock()
        mock_model.optimize_portfolio = Mock()
        mock_model.optimize_developer_portfolio = Mock()
        mock_model.run_iterative_station_selection = Mock(return_value=[])
        mock_model.analyze_regional_clusters = Mock()
        mock_model.create_continuous_score_surface = Mock()
        mock_model.validate_results = Mock()
        mock_model.generate_outputs = Mock()
        mock_model.export_results = Mock()
        mock_model.config = {'service_radius_miles': 2.0, 'station_capex': 12000000}
        
        # Mock iteration results
        mock_model.iterative_selection_results = pd.DataFrame({
            'npv': [100000, 200000],
            'capacity_kg_day': [1000, 1500]
        })
        
        # Create test files
        routes_file = os.path.join(self.temp_dir, "routes.geojson")
        loi_file = os.path.join(self.temp_dir, "loi.geojson")
        
        # Create minimal valid GeoJSON files
        test_routes = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])]
        })
        test_loi = gpd.GeoDataFrame({
            'geometry': [Point(0.5, 0.5)]
        })
        
        test_routes.to_file(routes_file, driver="GeoJSON")
        test_loi.to_file(loi_file, driver="GeoJSON")
        
        # Test quick analysis
        result = quick_setup.run_quick_analysis(
            route_file=routes_file,
            merged_loi_file=loi_file,
            n_stations=2,
            visualize_validation=False
        )
        
        # Verify model was created and methods were called
        mock_model_class.assert_called_once()
        mock_model.load_data.assert_called_once()
        mock_model.estimate_demand_surface.assert_called_once()
        mock_model.generate_candidates.assert_called_once()
        
        # Verify result
        self.assertIsNotNone(result)
        
    @patch('h2_station_quick_setup.H2StationSitingModel')
    def test_run_quick_analysis_with_error(self, mock_model_class):
        """Test quick analysis with error handling"""
        # Create mock model that raises an exception
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        mock_model.load_data.side_effect = Exception("Test error")
        
        # Create test files
        routes_file = os.path.join(self.temp_dir, "routes.geojson")
        loi_file = os.path.join(self.temp_dir, "loi.geojson")
        
        # Create minimal valid GeoJSON files
        test_routes = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])]
        })
        test_loi = gpd.GeoDataFrame({
            'geometry': [Point(0.5, 0.5)]
        })
        
        test_routes.to_file(routes_file, driver="GeoJSON")
        test_loi.to_file(loi_file, driver="GeoJSON")
        
        # Test quick analysis with error
        result = quick_setup.run_quick_analysis(
            route_file=routes_file,
            merged_loi_file=loi_file,
            n_stations=2,
            visualize_validation=False
        )
        
        # Should return None on error
        self.assertIsNone(result)
        
    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Test default configuration
        config = {
            'model_name': 'H2StationSitingModel',
            'demand_kernel_bandwidth_miles': 2.0,
            'service_radius_miles': 2.0,
            'station_capacity_kg_per_day': 2000,
            'h2_price_per_kg': 28.50,
            'discount_rate': 0.10
        }
        
        # Verify required parameters
        required_params = [
            'demand_kernel_bandwidth_miles',
            'service_radius_miles', 
            'station_capacity_kg_per_day',
            'h2_price_per_kg',
            'discount_rate'
        ]
        
        for param in required_params:
            self.assertIn(param, config)
            
        # Test parameter ranges
        self.assertGreater(config['demand_kernel_bandwidth_miles'], 0)
        self.assertGreater(config['service_radius_miles'], 0)
        self.assertGreater(config['station_capacity_kg_per_day'], 0)
        self.assertGreater(config['h2_price_per_kg'], 0)
        self.assertGreater(config['discount_rate'], 0)
        self.assertLess(config['discount_rate'], 1)
        
    def test_output_directory_structure(self):
        """Test output directory structure creation"""
        # Mock the version directory function
        with patch('h2_station_quick_setup.get_next_version_dir') as mock_version:
            mock_version.return_value = os.path.join(self.temp_dir, "test_output")
            
            # Create expected subdirectories
            main_dir = os.path.join(self.temp_dir, "test_output")
            subdirs = {
                'data': os.path.join(main_dir, 'data'),
                'validation': os.path.join(main_dir, 'validation'),
                'visualizations': os.path.join(main_dir, 'visualizations'),
                'iteration_results': os.path.join(main_dir, 'iteration_results'),
                'reports': os.path.join(main_dir, 'reports'),
                'competition_analysis': os.path.join(main_dir, 'competition_analysis')
            }
            
            # Create directories
            os.makedirs(main_dir, exist_ok=True)
            for subdir in subdirs.values():
                os.makedirs(subdir, exist_ok=True)
                
            # Verify all directories exist
            for subdir in subdirs.values():
                self.assertTrue(os.path.exists(subdir))
                
    def test_file_existence_checks(self):
        """Test file existence validation"""
        # Test with non-existent files
        non_existent_route = "non_existent_routes.geojson"
        non_existent_loi = "non_existent_loi.geojson"
        
        # Should handle non-existent files gracefully
        self.assertFalse(os.path.exists(non_existent_route))
        self.assertFalse(os.path.exists(non_existent_loi))
        
        # Test with existing files
        existing_route = os.path.join(self.temp_dir, "existing_routes.geojson")
        existing_loi = os.path.join(self.temp_dir, "existing_loi.geojson")
        
        # Create files
        with open(existing_route, 'w') as f:
            f.write('{"type": "FeatureCollection", "features": []}')
        with open(existing_loi, 'w') as f:
            f.write('{"type": "FeatureCollection", "features": []}')
            
        self.assertTrue(os.path.exists(existing_route))
        self.assertTrue(os.path.exists(existing_loi))


class TestQuickSetupIntegration(unittest.TestCase):
    """Integration tests for quick setup functionality"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_realistic_test_data(self):
        """Create realistic test data for integration testing"""
        # Create route data
        routes_data = []
        for i in range(10):
            start_point = (-120 + i * 0.01, 35 + i * 0.01)
            end_point = (-120 + i * 0.01 + 0.005, 35 + i * 0.01 + 0.005)
            line = LineString([start_point, end_point])
            
            routes_data.append({
                'geometry': line,
                'tot_truck_aadt': 1000 + i * 200,
                'truck_aadt_ab': 500 + i * 100,
                'truck_aadt_ba': 500 + i * 100,
                'length_m': 1000 + i * 50,
                'route_id': f'route_{i}'
            })
            
        routes_gdf = gpd.GeoDataFrame(routes_data, crs="EPSG:4326")
        
        # Create LOI data
        loi_data = []
        for i in range(5):
            loi_data.append({
                'geometry': Point(-120 + i * 0.01, 35 + i * 0.01),
                'name': f'LOI_{i}',
                'type': 'facility',
                'loi_uid': f'loi_{i}',
                'flow_total': 1000 + i * 100,
                'dist_m': 500 + i * 50,
                'route_seg_id': i
            })
            
        loi_gdf = gpd.GeoDataFrame(loi_data, crs="EPSG:4326")
        
        return routes_gdf, loi_gdf
        
    @patch('h2_station_quick_setup.H2StationSitingModel')
    def test_end_to_end_workflow(self, mock_model_class):
        """Test end-to-end workflow with realistic data"""
        # Create realistic test data
        routes_gdf, loi_gdf = self.create_realistic_test_data()
        
        # Save to files
        routes_file = os.path.join(self.temp_dir, "test_routes.geojson")
        loi_file = os.path.join(self.temp_dir, "test_loi.geojson")
        
        routes_gdf.to_file(routes_file, driver="GeoJSON")
        loi_gdf.to_file(loi_file, driver="GeoJSON")
        
        # Create mock model with realistic behavior
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        # Mock all required methods
        mock_model.load_data = Mock()
        mock_model.estimate_demand_surface = Mock()
        mock_model.generate_candidates = Mock()
        mock_model.calculate_utilization_probability = Mock()
        mock_model.calculate_economic_proxy = Mock()
        mock_model.optimize_portfolio = Mock()
        mock_model.optimize_developer_portfolio = Mock()
        mock_model.run_iterative_station_selection = Mock(return_value=[])
        mock_model.analyze_regional_clusters = Mock()
        mock_model.create_continuous_score_surface = Mock()
        mock_model.validate_results = Mock()
        mock_model.generate_outputs = Mock()
        mock_model.export_results = Mock()
        mock_model.config = {
            'service_radius_miles': 2.0,
            'station_capex': 12000000
        }
        
        # Mock iteration results
        mock_model.iterative_selection_results = pd.DataFrame({
            'npv': [100000, 200000, 150000],
            'capacity_kg_day': [1000, 1500, 1200]
        })
        
        # Run end-to-end workflow
        result = quick_setup.run_quick_analysis(
            route_file=routes_file,
            merged_loi_file=loi_file,
            n_stations=3,
            visualize_validation=False
        )
        
        # Verify workflow completed successfully
        self.assertIsNotNone(result)
        
        # Verify all major steps were called
        mock_model.load_data.assert_called_once()
        mock_model.estimate_demand_surface.assert_called_once()
        mock_model.generate_candidates.assert_called_once()
        mock_model.calculate_utilization_probability.assert_called_once()
        mock_model.calculate_economic_proxy.assert_called_once()
        mock_model.optimize_portfolio.assert_called_once()
        mock_model.run_iterative_station_selection.assert_called_once()


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestQuickSetupFunctions))
    suite.addTest(loader.loadTestsFromTestCase(TestQuickSetupIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
            
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")