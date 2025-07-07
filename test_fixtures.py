#!/usr/bin/env python3
"""
Test fixtures and data generators for H2 Station Siting Model tests
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime, timedelta


class TestDataGenerator:
    """Generate realistic test data for H2 station siting model tests"""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducible test data"""
        np.random.seed(seed)
        self.california_bounds = {
            'west': -124.4,
            'east': -114.1,
            'south': 32.5,
            'north': 42.0
        }
        
    def generate_california_routes(self, n_routes=50, crs="EPSG:4326"):
        """Generate realistic California highway route segments"""
        routes_data = []
        
        for i in range(n_routes):
            # Generate highway segments roughly following California's geography
            if i < 20:  # I-5 corridor (north-south)
                start_lon = -122.0 + np.random.normal(0, 0.1)
                start_lat = 34.0 + i * 0.4 + np.random.normal(0, 0.05)
                end_lon = start_lon + np.random.normal(0, 0.02)
                end_lat = start_lat + 0.3 + np.random.normal(0, 0.05)
                highway_type = "Interstate"
                base_aadt = np.random.randint(2000, 8000)
                
            elif i < 35:  # I-10 corridor (east-west)
                start_lon = -118.0 + (i-20) * 0.2 + np.random.normal(0, 0.1)
                start_lat = 34.0 + np.random.normal(0, 0.1)
                end_lon = start_lon + 0.2 + np.random.normal(0, 0.05)
                end_lat = start_lat + np.random.normal(0, 0.02)
                highway_type = "Interstate"
                base_aadt = np.random.randint(1500, 6000)
                
            else:  # State highways and arterials
                start_lon = np.random.uniform(-122.0, -117.0)
                start_lat = np.random.uniform(33.0, 38.0)
                end_lon = start_lon + np.random.normal(0, 0.1)
                end_lat = start_lat + np.random.normal(0, 0.1)
                highway_type = "State Highway"
                base_aadt = np.random.randint(500, 3000)
            
            # Create LineString geometry
            geometry = LineString([(start_lon, start_lat), (end_lon, end_lat)])
            
            # Calculate truck traffic (typically 5-15% of total traffic)
            truck_percentage = np.random.uniform(0.05, 0.15)
            tot_truck_aadt = int(base_aadt * truck_percentage)
            
            # Split directional traffic
            directional_split = np.random.uniform(0.4, 0.6)
            truck_aadt_ab = int(tot_truck_aadt * directional_split)
            truck_aadt_ba = tot_truck_aadt - truck_aadt_ab
            
            routes_data.append({
                'geometry': geometry,
                'route_id': f'CA_HWY_{i:03d}',
                'highway_name': f'{highway_type} {i}',
                'highway_type': highway_type,
                'tot_truck_aadt': tot_truck_aadt,
                'truck_aadt_ab': truck_aadt_ab,
                'truck_aadt_ba': truck_aadt_ba,
                'total_aadt': base_aadt,
                'length_m': geometry.length * 111320,  # Rough conversion to meters
                'functional_class': np.random.choice(['Interstate', 'Principal Arterial', 'Minor Arterial']),
                'speed_limit': np.random.choice([55, 65, 70, 75, 80]),
                'lane_count': np.random.choice([2, 4, 6, 8]),
                'median_type': np.random.choice(['None', 'Painted', 'Barrier', 'Grass'])
            })
            
        return gpd.GeoDataFrame(routes_data, crs=crs)
        
    def generate_california_lois(self, n_lois=30, crs="EPSG:4326"):
        """Generate realistic California locations of interest"""
        loi_data = []
        
        # Define LOI types and their typical characteristics
        loi_types = {
            'port': {
                'locations': [(-118.2, 33.7), (-121.9, 37.8), (-117.2, 32.7)],  # LA, SF, SD ports
                'count': 5,
                'attraction_weight': 0.3
            },
            'airport': {
                'locations': [(-118.4, 33.9), (-121.8, 37.6), (-117.1, 32.8)],  # LAX, SFO, SAN
                'count': 8,
                'attraction_weight': 0.2
            },
            'intermodal_facility': {
                'locations': [(-118.0, 34.0), (-121.5, 38.0), (-117.8, 33.8)],
                'count': 6,
                'attraction_weight': 0.25
            },
            'truck_stop': {
                'locations': None,  # Distributed along highways
                'count': 8,
                'attraction_weight': 0.15
            },
            'distribution_center': {
                'locations': None,  # Near urban areas
                'count': 5,
                'attraction_weight': 0.2
            }
        }
        
        loi_counter = 0
        
        for loi_type, config in loi_types.items():
            for i in range(config['count']):
                if config['locations']:
                    # Use predefined locations with some variation
                    base_loc = config['locations'][i % len(config['locations'])]
                    lon = base_loc[0] + np.random.normal(0, 0.2)
                    lat = base_loc[1] + np.random.normal(0, 0.15)
                else:
                    # Random distribution
                    if loi_type == 'truck_stop':
                        # Truck stops along major highways
                        lon = np.random.uniform(-122.0, -117.0)
                        lat = np.random.uniform(33.0, 38.0)
                    else:
                        # Distribution centers near urban areas
                        urban_centers = [(-118.2, 34.0), (-121.8, 37.4), (-117.2, 32.7)]
                        center = urban_centers[np.random.randint(0, len(urban_centers))]
                        lon = center[0] + np.random.normal(0, 0.3)
                        lat = center[1] + np.random.normal(0, 0.2)
                
                # Ensure coordinates are within California bounds
                lon = np.clip(lon, self.california_bounds['west'], self.california_bounds['east'])
                lat = np.clip(lat, self.california_bounds['south'], self.california_bounds['north'])
                
                loi_data.append({
                    'geometry': Point(lon, lat),
                    'loi_uid': f'{loi_type}_{loi_counter:03d}',
                    'name': f'{loi_type.replace("_", " ").title()} {i+1}',
                    'type': loi_type,
                    'source_file': f'{loi_type}_data.geojson',
                    'geom_type': 'Point',
                    'capacity': np.random.randint(100, 2000),
                    'daily_truck_visits': np.random.randint(10, 500),
                    'attraction_weight': config['attraction_weight'],
                    'operational_hours': np.random.choice(['24/7', 'Business Hours', 'Extended Hours']),
                    'fuel_services': np.random.choice([True, False]),
                    'parking_capacity': np.random.randint(20, 200)
                })
                
                loi_counter += 1
                
        return gpd.GeoDataFrame(loi_data[:n_lois], crs=crs)
        
    def generate_existing_h2_stations(self, n_stations=10, crs="EPSG:4326"):
        """Generate existing hydrogen stations in California"""
        stations_data = []
        
        # Existing H2 stations are primarily in urban areas
        urban_centers = [
            (-118.2, 34.0, "Los Angeles"),    # LA
            (-121.8, 37.4, "San Francisco"), # SF
            (-117.2, 32.7, "San Diego"),     # SD
            (-121.5, 38.5, "Sacramento"),    # Sacramento
            (-119.8, 36.7, "Fresno")         # Fresno
        ]
        
        for i in range(n_stations):
            # Select urban center
            center = urban_centers[i % len(urban_centers)]
            
            # Add variation around urban center
            lon = center[0] + np.random.normal(0, 0.1)
            lat = center[1] + np.random.normal(0, 0.08)
            
            # Station characteristics
            capacity = np.random.choice([500, 1000, 1500, 2000])  # kg/day
            utilization = np.random.uniform(0.1, 0.8)
            
            stations_data.append({
                'geometry': Point(lon, lat),
                'station_id': f'H2_EXIST_{i:03d}',
                'name': f'H2 Station {center[2]} {i+1}',
                'operator': np.random.choice(['Shell', 'Chevron', 'Air Liquide', 'FirstElement', 'True Zero']),
                'capacity_kg_day': capacity,
                'current_utilization': utilization,
                'daily_demand_kg': capacity * utilization,
                'station_type': np.random.choice(['Retail', 'Fleet', 'Mixed']),
                'operational_date': datetime.now() - timedelta(days=np.random.randint(30, 1000)),
                'h2_price_per_kg': np.random.uniform(12.0, 16.0),
                'dispensing_pressure': np.random.choice([350, 700]),  # bar
                'accessibility': np.random.choice(['Public', 'Private', 'Fleet Only']),
                'payment_methods': np.random.choice(['Credit Card', 'Fleet Card', 'App Only']),
                'amenities': np.random.choice(['None', 'Convenience Store', 'Restaurant', 'Rest Area'])
            })
            
        return gpd.GeoDataFrame(stations_data, crs=crs)
        
    def generate_gas_stations(self, n_stations=100, crs="EPSG:4326"):
        """Generate gas station locations for validation"""
        stations_data = []
        
        for i in range(n_stations):
            # Distribute gas stations across California
            lon = np.random.uniform(self.california_bounds['west'], self.california_bounds['east'])
            lat = np.random.uniform(self.california_bounds['south'], self.california_bounds['north'])
            
            # Gas station characteristics
            stations_data.append({
                'geometry': Point(lon, lat),
                'station_id': f'GAS_{i:04d}',
                'name': f'Gas Station {i+1}',
                'brand': np.random.choice(['Shell', 'Chevron', 'Arco', '76', 'Valero', 'Mobil']),
                'address': f'{np.random.randint(100, 9999)} Test St',
                'city': np.random.choice(['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento', 'Fresno']),
                'zip_code': f'{np.random.randint(90000, 96999)}',
                'diesel_available': np.random.choice([True, False]),
                'truck_accessible': np.random.choice([True, False]),
                'amenities': np.random.choice(['None', 'Store', 'Restaurant', 'Car Wash']),
                'latitude': lat,
                'longitude': lon
            })
            
        return gpd.GeoDataFrame(stations_data, crs=crs)
        
    def generate_truck_volumes(self, routes_gdf):
        """Generate truck volume segments along routes"""
        segments_data = []
        
        for idx, route in routes_gdf.iterrows():
            # Create volume segments along each route
            n_segments = np.random.randint(3, 8)
            route_length = route.geometry.length
            
            for seg_i in range(n_segments):
                # Interpolate along route
                start_ratio = seg_i / n_segments
                end_ratio = (seg_i + 1) / n_segments
                
                start_point = route.geometry.interpolate(start_ratio, normalized=True)
                end_point = route.geometry.interpolate(end_ratio, normalized=True)
                
                # Volume varies along route
                volume_variation = np.random.uniform(0.7, 1.3)
                segment_volume = int(route['tot_truck_aadt'] * volume_variation)
                
                segments_data.append({
                    'geometry': LineString([start_point, end_point]),
                    'parent_route_id': route['route_id'],
                    'segment_id': f'{route["route_id"]}_SEG_{seg_i}',
                    'truck_volume': segment_volume,
                    'segment_length_m': start_point.distance(end_point) * 111320,
                    'volume_confidence': np.random.uniform(0.6, 0.95),
                    'data_source': np.random.choice(['Caltrans', 'FHWA', 'Local DOT']),
                    'collection_year': np.random.choice([2020, 2021, 2022, 2023])
                })
                
        return gpd.GeoDataFrame(segments_data, crs=routes_gdf.crs)


class TestFileManager:
    """Manage test files and temporary directories"""
    
    def __init__(self):
        self.temp_dirs = []
        self.temp_files = []
        
    def create_temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
        
    def create_temp_file(self, suffix='.geojson', prefix='test_'):
        """Create temporary file"""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=suffix, 
            prefix=prefix, 
            delete=False
        )
        self.temp_files.append(temp_file.name)
        temp_file.close()
        return temp_file.name
        
    def save_test_data(self, gdf, filename, temp_dir=None):
        """Save GeoDataFrame to file in temp directory"""
        if temp_dir is None:
            temp_dir = self.create_temp_dir()
            
        filepath = os.path.join(temp_dir, filename)
        
        if filename.endswith('.geojson'):
            gdf.to_file(filepath, driver="GeoJSON")
        elif filename.endswith('.csv'):
            if isinstance(gdf, gpd.GeoDataFrame):
                # Convert to regular DataFrame with lat/lon columns
                df = gdf.copy()
                df['latitude'] = df.geometry.y
                df['longitude'] = df.geometry.x
                df = df.drop(columns=['geometry'])
                df.to_csv(filepath, index=False)
            else:
                gdf.to_csv(filepath, index=False)
        elif filename.endswith('.gpkg'):
            gdf.to_file(filepath, driver="GPKG")
            
        return filepath
        
    def create_test_config(self, temp_dir=None):
        """Create test configuration file"""
        if temp_dir is None:
            temp_dir = self.create_temp_dir()
            
        config = {
            "model_name": "TestH2Model",
            "demand_kernel_bandwidth_miles": 1.5,
            "service_radius_miles": 2.0,
            "min_station_spacing_miles": 1.0,
            "station_capacity_kg_per_day": 1000,
            "station_capex": 8000000,
            "h2_price_per_kg": 25.0,
            "h2_consumption_kg_per_mile": 0.1,
            "discount_rate": 0.08,
            "station_lifetime_years": 20,
            "min_candidate_truck_aadt": 150,
            "candidate_interval_miles": 1.5,
            "use_gravity_competiton_model": True,
            "competition_decay_rate": 0.25
        }
        
        config_file = os.path.join(temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
        return config_file
        
    def cleanup(self):
        """Clean up all temporary files and directories"""
        import shutil
        
        # Remove temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass
                
        # Remove temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass
                
        self.temp_files = []
        self.temp_dirs = []


class MockModelComponents:
    """Mock components for testing model functionality"""
    
    @staticmethod
    def create_mock_demand_surface():
        """Create mock demand surface data"""
        x = np.linspace(-122, -117, 50)
        y = np.linspace(33, 38, 40)
        X, Y = np.meshgrid(x, y)
        
        # Create realistic demand pattern (higher near urban centers)
        urban_centers = [(-118.2, 34.0), (-121.8, 37.4), (-117.2, 32.7)]
        demand = np.zeros_like(X)
        
        for ux, uy in urban_centers:
            distance = np.sqrt((X - ux)**2 + (Y - uy)**2)
            demand += 1000 * np.exp(-distance * 10)
            
        # Add noise
        demand += np.random.normal(0, 50, demand.shape)
        demand = np.maximum(demand, 0)  # Ensure non-negative
        
        # Create demand points
        points = []
        demands = []
        for i in range(len(x)):
            for j in range(len(y)):
                if demand[j, i] > 100:  # Only include significant demand
                    points.append(Point(x[i], y[j]))
                    demands.append(demand[j, i])
                    
        demand_gdf = gpd.GeoDataFrame({
            'geometry': points,
            'demand_kg_day': demands
        }, crs="EPSG:4326")
        
        return {
            'demand_points': demand_gdf,
            'grid_x': x,
            'grid_y': y,
            'demand_grid': demand,
            'grid_bounds': [x.min(), y.min(), x.max(), y.max()],
            'total_demand': np.sum(demand),
            'max_demand': np.max(demand),
            'validation': {
                'points_generated': len(points),
                'total_demand_kg_day': np.sum(demands)
            }
        }
        
    @staticmethod
    def create_mock_candidates(n_candidates=50):
        """Create mock candidate locations"""
        candidates_data = []
        
        for i in range(n_candidates):
            lon = np.random.uniform(-122, -117)
            lat = np.random.uniform(33, 38)
            
            # Simulate realistic candidate metrics
            truck_aadt = np.random.randint(200, 5000)
            expected_demand = truck_aadt * 0.1 * np.random.uniform(0.5, 1.5)
            utilization_prob = np.random.beta(2, 5)  # Skewed toward lower values
            
            # Economic metrics
            revenue = expected_demand * 25 * 365 * utilization_prob
            cost = 5000 * 365 + expected_demand * 6 * 365
            npv = (revenue - cost) * 8 - 8000000  # Simplified NPV
            
            candidates_data.append({
                'geometry': Point(lon, lat),
                'candidate_id': f'CAND_{i:03d}',
                'truck_aadt': truck_aadt,
                'expected_demand_kg_day': expected_demand,
                'utilization_probability': utilization_prob,
                'revenue_proxy': revenue,
                'cost_proxy': cost,
                'npv_proxy': npv,
                'location_score': np.random.uniform(0.3, 0.9),
                'is_interchange': np.random.choice([True, False]),
                'has_rest_area': np.random.choice([True, False]),
                'dist_to_port_miles': np.random.uniform(5, 100),
                'dist_to_gas_miles': np.random.uniform(0.5, 10),
                'dist_to_rest_miles': np.random.uniform(2, 50),
                'highway_access_score': np.random.uniform(0.2, 1.0),
                'competition_factor': np.random.uniform(0.6, 1.0)
            })
            
        return gpd.GeoDataFrame(candidates_data, crs="EPSG:4326")
        
    @staticmethod
    def create_mock_optimization_results(candidates_gdf, n_selected=10):
        """Create mock portfolio optimization results"""
        # Select top candidates by NPV
        selected_candidates = candidates_gdf.nlargest(n_selected, 'npv_proxy')
        
        # Add portfolio-specific metrics
        selected_candidates = selected_candidates.copy()
        selected_candidates['portfolio_rank'] = range(1, n_selected + 1)
        selected_candidates['selection_iteration'] = np.random.randint(1, 6, n_selected)
        selected_candidates['capacity_kg_day'] = np.random.choice([1000, 1500, 2000], n_selected)
        selected_candidates['investment_cost'] = selected_candidates['capacity_kg_day'] * 4000
        
        return selected_candidates


def create_complete_test_dataset(temp_dir=None):
    """Create a complete test dataset with all components"""
    generator = TestDataGenerator()
    file_manager = TestFileManager()
    
    if temp_dir is None:
        temp_dir = file_manager.create_temp_dir()
    
    # Generate all data components
    routes = generator.generate_california_routes(n_routes=25)
    lois = generator.generate_california_lois(n_lois=15)
    existing_stations = generator.generate_existing_h2_stations(n_stations=5)
    gas_stations = generator.generate_gas_stations(n_stations=50)
    truck_volumes = generator.generate_truck_volumes(routes)
    
    # Save all datasets
    files = {
        'routes': file_manager.save_test_data(routes, 'test_routes.geojson', temp_dir),
        'lois': file_manager.save_test_data(lois, 'test_lois.geojson', temp_dir),
        'existing_stations': file_manager.save_test_data(existing_stations, 'existing_h2.geojson', temp_dir),
        'gas_stations': file_manager.save_test_data(gas_stations, 'gas_stations.csv', temp_dir),
        'truck_volumes': file_manager.save_test_data(truck_volumes, 'truck_volumes.geojson', temp_dir),
        'config': file_manager.create_test_config(temp_dir)
    }
    
    return {
        'files': files,
        'data': {
            'routes': routes,
            'lois': lois,
            'existing_stations': existing_stations,
            'gas_stations': gas_stations,
            'truck_volumes': truck_volumes
        },
        'temp_dir': temp_dir,
        'file_manager': file_manager
    }


if __name__ == '__main__':
    # Test the fixture generation
    print("Generating test dataset...")
    
    dataset = create_complete_test_dataset()
    
    print(f"Test dataset created in: {dataset['temp_dir']}")
    print("\nDataset summary:")
    print(f"  Routes: {len(dataset['data']['routes'])} segments")
    print(f"  LOIs: {len(dataset['data']['lois'])} locations")
    print(f"  Existing H2 stations: {len(dataset['data']['existing_stations'])} stations")
    print(f"  Gas stations: {len(dataset['data']['gas_stations'])} stations")
    print(f"  Truck volume segments: {len(dataset['data']['truck_volumes'])} segments")
    
    print("\nFiles created:")
    for name, filepath in dataset['files'].items():
        print(f"  {name}: {filepath}")
        
    # Clean up
    dataset['file_manager'].cleanup()
    print("\nTest dataset cleaned up.")