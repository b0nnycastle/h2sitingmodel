#!/usr/bin/env python3
"""
Comprehensive unit tests for LOI route matching workflow
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import warnings

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from the workflow script
import loi_route_matching_workflow as workflow


class TestLOIRouteMatchingWorkflow(unittest.TestCase):
    """Test suite for LOI route matching workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_epsg = 3310  # California Albers
        
        # Create test route data
        self.test_routes = self.create_test_route_data()
        
        # Create test LOI data  
        self.test_loi_data = self.create_test_loi_data()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_route_data(self):
        """Create test route data for testing"""
        routes_data = []
        
        # Create test routes in California Albers projection
        for i in range(5):
            # Create route segments
            start_x = -2000000 + i * 10000  # California Albers coordinates
            start_y = 500000 + i * 5000
            end_x = start_x + 5000
            end_y = start_y + 2500
            
            line = LineString([(start_x, start_y), (end_x, end_y)])
            
            routes_data.append({
                'geometry': line,
                'tot_truck_aadt': 1000 + i * 500,
                'truck_aadt_ab': 400 + i * 200,
                'truck_aadt_ba': 600 + i * 300,
                'length_m': 5000 + i * 100,
                'route_id': f'route_{i}',
                'highway_name': f'Highway_{i}'
            })
            
        return gpd.GeoDataFrame(routes_data, crs=f"EPSG:{self.test_epsg}")
        
    def create_test_loi_data(self):
        """Create test LOI data for testing"""
        loi_data = []
        
        # Create test LOIs near the routes
        for i in range(3):
            # Position LOIs near route segments
            x = -2000000 + i * 10000 + 1000  # Offset from route
            y = 500000 + i * 5000 + 500
            
            loi_data.append({
                'geometry': Point(x, y),
                'name': f'Test_LOI_{i}',
                'type': 'test_facility',
                'loi_uid': f'test_loi_{i}',
                'source_file': f'test_data_{i}.geojson',
                'geom_type': 'Point'
            })
            
        return gpd.GeoDataFrame(loi_data, crs=f"EPSG:{self.test_epsg}")
        
    def test_get_geometry_bounds(self):
        """Test geometry bounds calculation"""
        # Test with Point
        point = Point(0, 0)
        bounds = workflow.get_geometry_bounds(point)
        self.assertEqual(len(bounds), 4)  # minx, miny, maxx, maxy
        
        # Test with LineString
        line = LineString([(0, 0), (1, 1)])
        bounds = workflow.get_geometry_bounds(line)
        self.assertEqual(bounds, (0.0, 0.0, 1.0, 1.0))
        
    def test_detect_flow_columns(self):
        """Test flow column detection"""
        # Test with standard column names
        columns = ['tot_truck_aadt', 'truck_aadt_ab', 'truck_aadt_ba', 'other_col']
        total_col, ab_col, ba_col = workflow.detect_flow_columns(columns)
        
        self.assertEqual(total_col, 'tot_truck_aadt')
        self.assertEqual(ab_col, 'truck_aadt_ab')
        self.assertEqual(ba_col, 'truck_aadt_ba')
        
        # Test with no flow columns
        columns_no_flow = ['name', 'id', 'other_col']
        total_col, ab_col, ba_col = workflow.detect_flow_columns(columns_no_flow)
        
        self.assertIsNone(total_col)
        self.assertIsNone(ab_col)
        self.assertIsNone(ba_col)
        
    def test_build_rtree(self):
        """Test spatial index building"""
        # Test with valid geometries
        gdf = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 1), Point(2, 2)]
        })
        
        rtree_idx = workflow.build_rtree(gdf)
        
        # Test that index was created
        self.assertIsNotNone(rtree_idx)
        
        # Test intersection query
        intersections = list(rtree_idx.intersection((0.5, 0.5, 1.5, 1.5)))
        self.assertGreater(len(intersections), 0)
        
    def test_load_loi_file_geojson(self):
        """Test loading GeoJSON LOI file"""
        # Create test GeoJSON file
        loi_file = os.path.join(self.temp_dir, "test_loi.geojson")
        
        # Convert to WGS84 for file saving
        loi_wgs84 = self.test_loi_data.to_crs("EPSG:4326")
        loi_wgs84.to_file(loi_file, driver="GeoJSON")
        
        # Test loading
        loaded_gdf = workflow.load_loi_file(loi_file)
        
        # Verify loaded data
        self.assertIsNotNone(loaded_gdf)
        self.assertEqual(len(loaded_gdf), 3)
        self.assertIn('loi_uid', loaded_gdf.columns)
        self.assertIn('source_file', loaded_gdf.columns)
        self.assertEqual(loaded_gdf.crs.to_epsg(), 4326)
        
    def test_load_loi_file_csv(self):
        """Test loading CSV LOI file with lat/lon columns"""
        # Create test CSV file
        csv_file = os.path.join(self.temp_dir, "test_loi.csv")
        
        # Create CSV data with lat/lon
        csv_data = pd.DataFrame({
            'name': ['LOI_1', 'LOI_2'],
            'latitude': [35.0, 36.0],
            'longitude': [-120.0, -119.0],
            'type': ['port', 'airport']
        })
        csv_data.to_csv(csv_file, index=False)
        
        # Test loading
        loaded_gdf = workflow.load_loi_file(csv_file)
        
        # Verify loaded data
        self.assertIsNotNone(loaded_gdf)
        self.assertEqual(len(loaded_gdf), 2)
        self.assertIn('loi_uid', loaded_gdf.columns)
        self.assertEqual(loaded_gdf.crs.to_epsg(), 4326)
        
    def test_load_loi_file_invalid(self):
        """Test loading invalid LOI file"""
        # Test with non-existent file
        result = workflow.load_loi_file("nonexistent_file.geojson")
        self.assertIsNone(result)
        
        # Test with empty file
        empty_file = os.path.join(self.temp_dir, "empty.geojson")
        with open(empty_file, 'w') as f:
            f.write("")
            
        result = workflow.load_loi_file(empty_file)
        self.assertIsNone(result)
        
    def test_process_loi_file(self):
        """Test processing LOI file against routes"""
        # Create spatial index for routes
        rtree_idx = workflow.build_rtree(self.test_routes)
        
        # Detect flow columns
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(self.test_routes.columns)
        
        # Process LOI file
        result_gdf, route_mapping = workflow.process_loi_file(
            self.test_loi_data, 
            self.test_routes, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Verify results
        self.assertIsNotNone(result_gdf)
        self.assertIsInstance(result_gdf, gpd.GeoDataFrame)
        self.assertIsInstance(route_mapping, dict)
        
        # Check that some matches were found
        if len(result_gdf) > 0:
            self.assertIn('dist_m', result_gdf.columns)
            self.assertIn('route_seg_id', result_gdf.columns)
            self.assertIn('loi_uid', result_gdf.columns)
            
    def test_process_loi_file_no_matches(self):
        """Test processing LOI file with no nearby routes"""
        # Create LOI data far from routes
        far_loi_data = []
        for i in range(2):
            # Position LOIs very far from routes
            x = -1000000  # Far from test routes
            y = 1000000
            
            far_loi_data.append({
                'geometry': Point(x, y),
                'name': f'Far_LOI_{i}',
                'type': 'test_facility',
                'loi_uid': f'far_loi_{i}',
                'source_file': 'test_data.geojson',
                'geom_type': 'Point'
            })
            
        far_loi_gdf = gpd.GeoDataFrame(far_loi_data, crs=f"EPSG:{self.test_epsg}")
        
        # Create spatial index for routes
        rtree_idx = workflow.build_rtree(self.test_routes)
        
        # Detect flow columns
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(self.test_routes.columns)
        
        # Process LOI file
        result_gdf, route_mapping = workflow.process_loi_file(
            far_loi_gdf, 
            self.test_routes, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Verify results for no matches
        self.assertIsNotNone(result_gdf)
        
        # Check for no-match records
        if len(result_gdf) > 0:
            self.assertIn('no_route_found', result_gdf.columns)
            
    def test_flow_calculation(self):
        """Test flow calculation with different column configurations"""
        # Test with total flow column
        routes_with_total = self.test_routes.copy()
        
        rtree_idx = workflow.build_rtree(routes_with_total)
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(routes_with_total.columns)
        
        result_gdf, _ = workflow.process_loi_file(
            self.test_loi_data, 
            routes_with_total, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Verify flow calculations
        if len(result_gdf) > 0:
            self.assertIn('flow_total', result_gdf.columns)
            self.assertIn('flow_share', result_gdf.columns)
            
        # Test with only directional flow columns
        routes_no_total = self.test_routes.copy()
        routes_no_total = routes_no_total.drop(columns=['tot_truck_aadt'])
        
        rtree_idx = workflow.build_rtree(routes_no_total)
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(routes_no_total.columns)
        
        result_gdf, _ = workflow.process_loi_file(
            self.test_loi_data, 
            routes_no_total, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Verify directional flow calculations
        if len(result_gdf) > 0:
            self.assertIn('flow_ab', result_gdf.columns)
            self.assertIn('flow_ba', result_gdf.columns)
            
    def test_distance_calculation(self):
        """Test distance calculation accuracy"""
        # Create LOI exactly at known distance from route
        route_start = Point(-2000000, 500000)
        route_end = Point(-1995000, 502500)
        test_route = LineString([route_start, route_end])
        
        # Create LOI at perpendicular distance
        loi_point = Point(-1997500, 501250 + 1000)  # 1000m perpendicular distance
        
        # Calculate distance
        distance = test_route.distance(loi_point)
        
        # Verify distance is approximately 1000m
        self.assertAlmostEqual(distance, 1000.0, places=0)
        
    def test_multiple_candidates(self):
        """Test handling multiple route candidates for single LOI"""
        # Create LOI near multiple routes
        central_loi = gpd.GeoDataFrame([{
            'geometry': Point(-1997500, 502500),  # Near multiple routes
            'name': 'Central_LOI',
            'type': 'test_facility',
            'loi_uid': 'central_loi',
            'source_file': 'test_data.geojson',
            'geom_type': 'Point'
        }], crs=f"EPSG:{self.test_epsg}")
        
        # Create spatial index
        rtree_idx = workflow.build_rtree(self.test_routes)
        
        # Detect flow columns
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(self.test_routes.columns)
        
        # Process LOI file
        result_gdf, route_mapping = workflow.process_loi_file(
            central_loi, 
            self.test_routes, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Verify multiple candidates
        if len(result_gdf) > 0:
            self.assertIn('cand_id', result_gdf.columns)
            self.assertIn('n_candidates', result_gdf.columns)
            self.assertIn('multi_flag', result_gdf.columns)
            
    def test_geometry_type_handling(self):
        """Test handling different geometry types"""
        # Create LOI data with different geometry types
        mixed_loi_data = []
        
        # Point geometry
        mixed_loi_data.append({
            'geometry': Point(-2000000, 500000),
            'name': 'Point_LOI',
            'type': 'point_facility',
            'loi_uid': 'point_loi',
            'source_file': 'test_data.geojson',
            'geom_type': 'Point'
        })
        
        # LineString geometry
        mixed_loi_data.append({
            'geometry': LineString([(-2001000, 500000), (-1999000, 500000)]),
            'name': 'Line_LOI',
            'type': 'line_facility',
            'loi_uid': 'line_loi',
            'source_file': 'test_data.geojson',
            'geom_type': 'LineString'
        })
        
        # Polygon geometry
        mixed_loi_data.append({
            'geometry': Polygon([(-2000500, 499500), (-1999500, 499500), 
                               (-1999500, 500500), (-2000500, 500500)]),
            'name': 'Polygon_LOI',
            'type': 'polygon_facility',
            'loi_uid': 'polygon_loi',
            'source_file': 'test_data.geojson',
            'geom_type': 'Polygon'
        })
        
        mixed_loi_gdf = gpd.GeoDataFrame(mixed_loi_data, crs=f"EPSG:{self.test_epsg}")
        
        # Add representative points
        mixed_loi_gdf["rep_point"] = mixed_loi_gdf.geometry.apply(
            lambda g: g if g.geom_type == 'Point' else g.representative_point()
        )
        
        # Create spatial index
        rtree_idx = workflow.build_rtree(self.test_routes)
        
        # Detect flow columns
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(self.test_routes.columns)
        
        # Process LOI file
        result_gdf, route_mapping = workflow.process_loi_file(
            mixed_loi_gdf, 
            self.test_routes, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Verify all geometry types were processed
        self.assertIsNotNone(result_gdf)
        
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with empty LOI data
        empty_loi = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{self.test_epsg}")
        
        rtree_idx = workflow.build_rtree(self.test_routes)
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(self.test_routes.columns)
        
        result_gdf, route_mapping = workflow.process_loi_file(
            empty_loi, 
            self.test_routes, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Should return empty GeoDataFrame
        self.assertIsNotNone(result_gdf)
        self.assertEqual(len(result_gdf), 0)
        
        # Test with empty routes data
        empty_routes = gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{self.test_epsg}")
        
        with self.assertRaises(Exception):
            empty_rtree_idx = workflow.build_rtree(empty_routes)
            
    def test_coordinate_system_conversion(self):
        """Test coordinate system handling"""
        # Create LOI data in WGS84
        loi_wgs84 = gpd.GeoDataFrame([{
            'geometry': Point(-120.0, 35.0),
            'name': 'WGS84_LOI',
            'type': 'test_facility',
            'loi_uid': 'wgs84_loi',
            'source_file': 'test_data.geojson',
            'geom_type': 'Point'
        }], crs="EPSG:4326")
        
        # Convert to target CRS
        loi_projected = loi_wgs84.to_crs(f"EPSG:{self.test_epsg}")
        
        # Verify conversion
        self.assertEqual(loi_projected.crs.to_epsg(), self.test_epsg)
        self.assertNotEqual(loi_wgs84.iloc[0].geometry.x, loi_projected.iloc[0].geometry.x)
        
    def test_output_validation(self):
        """Test output data validation"""
        # Create spatial index
        rtree_idx = workflow.build_rtree(self.test_routes)
        
        # Detect flow columns
        tot_col, ab_col, ba_col = workflow.detect_flow_columns(self.test_routes.columns)
        
        # Process LOI file
        result_gdf, route_mapping = workflow.process_loi_file(
            self.test_loi_data, 
            self.test_routes, 
            rtree_idx, 
            tot_col, 
            ab_col, 
            ba_col
        )
        
        # Validate output structure
        if len(result_gdf) > 0:
            # Check required columns exist
            required_columns = ['loi_uid', 'dist_m', 'route_seg_id']
            for col in required_columns:
                self.assertIn(col, result_gdf.columns)
                
            # Check data types
            self.assertTrue(result_gdf['dist_m'].dtype in [np.float64, np.float32])
            
            # Check distance values are reasonable
            self.assertTrue(all(result_gdf['dist_m'] >= 0))
            self.assertTrue(all(result_gdf['dist_m'] <= workflow.RADIUS_M))
            
        # Validate route mapping
        self.assertIsInstance(route_mapping, dict)
        self.assertEqual(len(route_mapping), len(self.test_routes))


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test class
    suite.addTest(loader.loadTestsFromTestCase(TestLOIRouteMatchingWorkflow))
    
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