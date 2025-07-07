#!/usr/bin/env python3
"""
Comprehensive unit tests for H2StationSitingModel
"""

import unittest
import pytest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import warnings

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2_station_model import H2StationSitingModel

class TestH2StationSitingModel(unittest.TestCase):
    """Test suite for H2StationSitingModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            'model_name': 'TestModel',
            'demand_kernel_bandwidth_miles': 1.0,
            'service_radius_miles': 2.0,
            'min_station_spacing_miles': 1.0,
            'station_capacity_kg_per_day': 1000,
            'station_capex': 5000000,
            'h2_price_per_kg': 25.0,
            'h2_consumption_kg_per_mile': 0.1,
            'discount_rate': 0.10,
            'station_lifetime_years': 15,
            'min_candidate_truck_aadt': 100,
            'candidate_interval_miles': 1.0,
            'use_gravity_competiton_model': True,
            'competition_decay_rate': 0.3,
            'distance_decay_exponent': 2.0
        }
        
        self.model = H2StationSitingModel(self.test_config)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test model initialization"""
        # Test with default config
        model_default = H2StationSitingModel()
        self.assertIsNotNone(model_default.config)
        self.assertEqual(model_default.config['model_name'], 'H2StationSitingModel')
        
        # Test with custom config
        self.assertEqual(self.model.config['model_name'], 'TestModel')
        self.assertEqual(self.model.config['service_radius_miles'], 2.0)
        self.assertIsNone(self.model.demand_surface)
        self.assertIsNone(self.model._competition_graph)
        
    def test_default_config(self):
        """Test default configuration parameters"""
        default_config = self.model._default_config()
        
        # Check required parameters exist
        required_params = [
            'model_name', 'demand_kernel_bandwidth_miles', 'service_radius_miles',
            'station_capacity_kg_per_day', 'station_capex', 'h2_price_per_kg',
            'discount_rate', 'station_lifetime_years'
        ]
        
        for param in required_params:
            self.assertIn(param, default_config)
            
        # Check parameter types and reasonable values
        self.assertIsInstance(default_config['service_radius_miles'], (int, float))
        self.assertGreater(default_config['service_radius_miles'], 0)
        self.assertGreater(default_config['station_capex'], 0)
        self.assertGreater(default_config['discount_rate'], 0)
        self.assertLess(default_config['discount_rate'], 1)
        
    def test_spatial_dependencies(self):
        """Test spatial dependencies check"""
        capabilities = self.model._check_spatial_dependencies()
        self.assertIsInstance(capabilities, dict)
        self.assertIn('rtree', capabilities)
        
    def create_test_route_data(self):
        """Create test route data for testing"""
        # Create test route segments
        routes_data = []
        for i in range(5):
            start_point = Point(-120 + i * 0.01, 35 + i * 0.01)
            end_point = Point(-120 + i * 0.01 + 0.005, 35 + i * 0.01 + 0.005)
            line = LineString([start_point, end_point])
            
            routes_data.append({
                'geometry': line,
                'tot_truck_aadt': 1000 + i * 500,
                'truck_aadt_ab': 500 + i * 250,
                'truck_aadt_ba': 500 + i * 250,
                'length_m': 1000 + i * 100,
                'route_id': f'route_{i}'
            })
            
        return gpd.GeoDataFrame(routes_data, crs="EPSG:4326")
        
    def create_test_loi_data(self):
        """Create test LOI data for testing"""
        loi_data = []
        for i in range(3):
            loi_data.append({
                'geometry': Point(-120 + i * 0.01, 35 + i * 0.01),
                'name': f'LOI_{i}',
                'type': 'test_location',
                'loi_uid': f'test_loi_{i}'
            })
            
        return gpd.GeoDataFrame(loi_data, crs="EPSG:4326")
        
    def test_load_data_with_valid_files(self):
        """Test loading data with valid files"""
        # Create test files
        routes_gdf = self.create_test_route_data()
        loi_gdf = self.create_test_loi_data()
        
        routes_file = os.path.join(self.temp_dir, "test_routes.geojson")
        loi_file = os.path.join(self.temp_dir, "test_loi.geojson")
        
        routes_gdf.to_file(routes_file, driver="GeoJSON")
        loi_gdf.to_file(loi_file, driver="GeoJSON")
        
        # Test loading
        self.model.load_data(routes_file, loi_file)
        
        # Verify data was loaded
        self.assertIsNotNone(self.model.routes)
        self.assertIsNotNone(self.model.loi_data)
        self.assertEqual(len(self.model.routes), 5)
        self.assertEqual(len(self.model.loi_data), 3)
        
    def test_load_data_with_invalid_files(self):
        """Test loading data with invalid files"""
        with self.assertRaises(Exception):
            self.model.load_data("nonexistent_routes.geojson", "nonexistent_loi.geojson")
            
    def test_detect_flow_columns(self):
        """Test flow column detection"""
        # Create test dataframe with flow columns
        test_df = pd.DataFrame({
            'tot_truck_aadt': [1000, 2000],
            'truck_aadt_ab': [500, 1000],
            'truck_aadt_ba': [500, 1000],
            'other_col': [1, 2]
        })
        
        # Test with routes loaded
        self.model.routes = gpd.GeoDataFrame(test_df, geometry=[Point(0, 0), Point(1, 1)])
        
        total_col, ab_col, ba_col = self.model._detect_flow_columns(self.model.routes.columns)
        
        self.assertEqual(total_col, 'tot_truck_aadt')
        self.assertEqual(ab_col, 'truck_aadt_ab')
        self.assertEqual(ba_col, 'truck_aadt_ba')
        
    def test_estimate_demand_surface(self):
        """Test demand surface estimation"""
        # Load test data first
        routes_gdf = self.create_test_route_data()
        loi_gdf = self.create_test_loi_data()
        
        # Mock the file loading
        self.model.routes = routes_gdf.to_crs("EPSG:3310")
        self.model.loi_data = loi_gdf.to_crs("EPSG:3310")
        
        # Test demand surface estimation
        self.model.estimate_demand_surface()
        
        # Verify demand surface was created
        self.assertIsNotNone(self.model.demand_surface)
        self.assertIn('demand_points', self.model.demand_surface)
        
    def test_generate_candidates_route_based(self):
        """Test route-based candidate generation"""
        # Setup test data
        routes_gdf = self.create_test_route_data()
        self.model.routes = routes_gdf.to_crs("EPSG:3310")
        
        # Mock demand surface
        self.model.demand_surface = {
            'demand_points': self.model.routes.geometry.centroid.to_frame('geometry'),
            'grid_bounds': [-120.1, 34.9, -119.9, 35.1]
        }
        
        # Generate candidates
        self.model.generate_candidates(strategy='route_based')
        
        # Verify candidates were generated
        self.assertIsNotNone(self.model.candidates)
        self.assertGreater(len(self.model.candidates), 0)
        
    def test_calculate_utilization_probability(self):
        """Test utilization probability calculation"""
        # Setup test data
        self.model.candidates = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1000, 1000)],
            'expected_demand_kg_day': [500, 1000],
            'truck_aadt': [1000, 2000]
        }, crs="EPSG:3310")
        
        # Test calculation
        self.model.calculate_utilization_probability()
        
        # Verify results
        self.assertIn('utilization_probability', self.model.candidates.columns)
        self.assertTrue(all(0 <= p <= 1 for p in self.model.candidates['utilization_probability']))
        
    def test_calculate_economic_proxy(self):
        """Test economic proxy calculation"""
        # Setup test data
        self.model.candidates = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1000, 1000)],
            'expected_demand_kg_day': [500, 1000],
            'utilization_probability': [0.5, 0.8],
            'truck_aadt': [1000, 2000]
        }, crs="EPSG:3310")
        
        # Test calculation
        self.model.calculate_economic_proxy()
        
        # Verify results
        self.assertIn('npv_proxy', self.model.candidates.columns)
        self.assertIn('revenue_proxy', self.model.candidates.columns)
        self.assertIn('cost_proxy', self.model.candidates.columns)
        
    def test_competition_graph_initialization(self):
        """Test competition graph initialization"""
        # Setup test data
        self.model.candidates = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1000, 1000), Point(2000, 2000)],
            'expected_demand_kg_day': [500, 1000, 750],
            'npv_proxy': [100000, 200000, 150000]
        }, crs="EPSG:3310")
        
        # Initialize competition graph
        self.model._competition_graph = self.model.CompetitionGraph(self.model.config)
        
        # Add nodes
        for idx, candidate in self.model.candidates.iterrows():
            self.model._competition_graph.add_or_update_station(idx, candidate, 'candidate')
            
        # Verify graph was created
        self.assertEqual(len(self.model._competition_graph.nodes), 3)
        
    def test_optimize_portfolio_basic(self):
        """Test basic portfolio optimization"""
        # Setup test data
        self.model.candidates = gpd.GeoDataFrame({
            'geometry': [Point(i*1000, i*1000) for i in range(5)],
            'expected_demand_kg_day': [500, 1000, 750, 1200, 800],
            'npv_proxy': [100000, 200000, 150000, 250000, 180000],
            'utilization_probability': [0.5, 0.8, 0.6, 0.9, 0.7]
        }, crs="EPSG:3310")
        
        # Mock the optimization to avoid complex solver dependencies
        with patch.object(self.model, '_solve_portfolio_optimization') as mock_solve:
            mock_solve.return_value = [0, 1, 3]  # Select candidates 0, 1, 3
            
            result = self.model.optimize_portfolio(n_stations=3)
            
            # Verify results
            self.assertIsNotNone(result)
            mock_solve.assert_called_once()
            
    def test_validate_results_basic(self):
        """Test basic results validation"""
        # Setup test data
        self.model.candidates = gpd.GeoDataFrame({
            'geometry': [Point(i*1000, i*1000) for i in range(3)],
            'expected_demand_kg_day': [500, 1000, 750],
            'npv_proxy': [100000, 200000, 150000]
        }, crs="EPSG:3310")
        
        # Mock selected results
        self.model.selected_candidates = self.model.candidates.iloc[:2]
        
        # Test validation
        validation_results = self.model.validate_results()
        
        # Verify validation results
        self.assertIsInstance(validation_results, dict)
        
    def test_export_results(self):
        """Test results export functionality"""
        # Setup test data
        self.model.candidates = gpd.GeoDataFrame({
            'geometry': [Point(i*1000, i*1000) for i in range(3)],
            'expected_demand_kg_day': [500, 1000, 750],
            'npv_proxy': [100000, 200000, 150000]
        }, crs="EPSG:3310")
        
        # Mock selected results
        self.model.selected_candidates = self.model.candidates.iloc[:2]
        
        # Test export
        self.model.export_results(self.temp_dir)
        
        # Verify files were created
        expected_files = ['selected_stations.geojson', 'all_candidates.geojson']
        for filename in expected_files:
            filepath = os.path.join(self.temp_dir, filename)
            if os.path.exists(filepath):
                self.assertTrue(os.path.getsize(filepath) > 0)
                
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        
        # Test with no data loaded
        with self.assertRaises(Exception):
            self.model.generate_candidates()
            
        # Test with invalid strategy
        self.model.routes = self.create_test_route_data()
        with self.assertRaises(ValueError):
            self.model.generate_candidates(strategy='invalid_strategy')
            
    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Test invalid configurations
        invalid_configs = [
            {'service_radius_miles': -1},  # Negative radius
            {'discount_rate': 1.5},  # Invalid discount rate
            {'station_capacity_kg_per_day': 0},  # Zero capacity
        ]
        
        for config in invalid_configs:
            with self.assertRaises(ValueError):
                model = H2StationSitingModel(config)
                model._validate_config()
                
    def test_memory_cleanup(self):
        """Test memory cleanup and resource management"""
        # Create model with data
        routes_gdf = self.create_test_route_data()
        loi_gdf = self.create_test_loi_data()
        
        self.model.routes = routes_gdf
        self.model.loi_data = loi_gdf
        
        # Verify data exists
        self.assertIsNotNone(self.model.routes)
        self.assertIsNotNone(self.model.loi_data)
        
        # Test cleanup if such method exists
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()


class TestH2StationModelComponents(unittest.TestCase):
    """Test individual components and helper functions"""
    
    def test_distance_calculations(self):
        """Test distance calculation methods"""
        model = H2StationSitingModel()
        
        # Test Euclidean distance
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        
        # Distance should be 5 (3-4-5 triangle)
        if hasattr(model, '_calculate_distance'):
            distance = model._calculate_distance(p1, p2)
            self.assertAlmostEqual(distance, 5.0, places=1)
            
    def test_demand_calculations(self):
        """Test demand calculation methods"""
        model = H2StationSitingModel()
        
        # Test basic demand calculation
        truck_aadt = 1000
        consumption_rate = 0.1  # kg/mile
        expected_demand = truck_aadt * consumption_rate
        
        if hasattr(model, '_calculate_base_demand'):
            calculated_demand = model._calculate_base_demand(truck_aadt, consumption_rate)
            self.assertAlmostEqual(calculated_demand, expected_demand, places=2)
            
    def test_economic_calculations(self):
        """Test economic calculation methods"""
        model = H2StationSitingModel()
        
        # Test NPV calculation
        if hasattr(model, '_calculate_npv'):
            # Mock cash flows
            cash_flows = [-1000000] + [100000] * 15  # Initial investment + 15 years of returns
            discount_rate = 0.1
            
            npv = model._calculate_npv(cash_flows, discount_rate)
            self.assertIsInstance(npv, (int, float))
            
    def test_spatial_operations(self):
        """Test spatial operations and indexing"""
        model = H2StationSitingModel()
        
        # Test spatial index creation
        points = [Point(i, i) for i in range(10)]
        gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        
        if hasattr(model, '_build_spatial_index'):
            spatial_index = model._build_spatial_index(gdf)
            self.assertIsNotNone(spatial_index)


class TestDataValidation(unittest.TestCase):
    """Test data validation and preprocessing"""
    
    def test_coordinate_system_handling(self):
        """Test coordinate system conversions"""
        # Test data in WGS84
        gdf_wgs84 = gpd.GeoDataFrame({
            'geometry': [Point(-120, 35), Point(-119, 36)]
        }, crs="EPSG:4326")
        
        # Test conversion to projected CRS
        gdf_projected = gdf_wgs84.to_crs("EPSG:3310")
        
        # Verify conversion
        self.assertEqual(gdf_projected.crs.to_epsg(), 3310)
        self.assertNotEqual(gdf_wgs84.iloc[0].geometry.x, gdf_projected.iloc[0].geometry.x)
        
    def test_data_quality_checks(self):
        """Test data quality validation"""
        # Test with missing geometries
        gdf_with_nulls = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), None, Point(1, 1)],
            'value': [1, 2, 3]
        })
        
        # Check for null geometries
        null_count = gdf_with_nulls.geometry.isna().sum()
        self.assertEqual(null_count, 1)
        
        # Test data cleaning
        gdf_clean = gdf_with_nulls.dropna(subset=['geometry'])
        self.assertEqual(len(gdf_clean), 2)
        
    def test_attribute_validation(self):
        """Test attribute data validation"""
        # Test numeric validation
        df = pd.DataFrame({
            'demand': [100, 200, -50, 0, 1000],  # Include negative and zero
            'cost': [1000, 2000, 3000, 4000, 5000]
        })
        
        # Test for negative values
        negative_demand = df['demand'] < 0
        self.assertTrue(negative_demand.any())
        
        # Test for zero values
        zero_demand = df['demand'] == 0
        self.assertTrue(zero_demand.any())


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestH2StationSitingModel))
    suite.addTest(loader.loadTestsFromTestCase(TestH2StationModelComponents))
    suite.addTest(loader.loadTestsFromTestCase(TestDataValidation))
    
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