#!/usr/bin/env python3
"""
Probabilistic Hydrogen Refueling Station Siting Model for California
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from datetime import datetime
from scipy import stats
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import contextily as ctx
from pulp import *  

try:
    from rtree import index as rtree_index
    RTREE_AVAILABLE = True
except ImportError:
    RTREE_AVAILABLE = False
    print("Warning: rtree not available. Spatial operations will use fallback methods.")

if RTREE_AVAILABLE:
    index = rtree_index
    idx = rtree_index.Index()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class H2StationSitingModel:
    """
    Comprehensive model for siting hydrogen refueling stations to maximize
    infrastructure developer returns under data constraints.
    """
    
    def __init__(self, config=None):
        """Initialize model with configuration parameters."""
        self.config = config or self._default_config()
        
        # Validate configuration
        try:
            self._validate_config()
        except ValueError as e:
            print(f"Warning: Configuration validation failed: {e}")
            print("Proceeding with potentially invalid configuration...")
        
        self.results = {}
        self.demand_surface = None
        # Check spatial dependencies
        self.spatial_capabilities = self._check_spatial_dependencies()
    
        # Initialize spatial index as None
        self._route_spatial_index = None
        self.utilization_probs = None
        self.location_score = None
        self._competition_graph = None 
        self.iteration_history = None   
        self.demand_evolution_matrix = None
        
        # Cache structures
        self._kdtree_cache = {}
        self._interchange_cache = None
        self._demand_grid_cache = None
        
    def _default_config(self):
        """Default configuration parameters - uses comprehensive transportation model config."""
        try:
            from comprehensive_config import get_comprehensive_config
            return get_comprehensive_config()
        except ImportError:
            # Fallback configuration if comprehensive_config is not available
            return self._fallback_config()
            
    def _fallback_config(self):
        """Fallback configuration with all essential parameters."""
        return {
            # Model parameters
            'model_name': 'H2StationSitingModel',
            'demand_kernel_bandwidth_miles': 2.0,
            'save_diagnostic_plots': True,
            'create_enhanced_plots': True,
            'validate_demand_surface': True,
            'candidate_interval_miles': 2.0,
            'min_candidate_truck_aadt': 200,
            'min_truck_aadt_highway': 1000,
            'multiple_developer_mode': True,
            
            # Fuel consumption parameters
            'h2_consumption_kg_per_mile': 0.1,
            'avg_refuel_amount_kg': 60.0,
            'refuel_threshold': 0.25,
            'typical_range_miles': 450.0,
            'tank_capacity_kg': 80.0,
            
            # Economic parameters - COMPLETE SET
            'h2_price_per_kg': 28.50,
            'station_capacity_kg_per_day': 2000,
            'station_capex': 12000000,
            'base_opex_daily': 5000,
            'variable_opex_per_kg': 6.0,
            'station_lifetime_years': 15.0,
            'discount_rate': 0.10,
            'operating_margin': 0.20,
            'base_utilization_rate': 0.05,
            'min_daily_visits': 5.0,
            'demand_blending_factor': 0.7,
            'base_capex_multiplier': 1.4,
            'min_iteration_capacity_kg_per_day': 500,
            'max_iteration_capacity_kg_per_day': 12000,
            'capacity_step_size_kg_per_day': 500,
            'min_viable_npv': 0.0,
            'maintenance_cost_annual': 300000,
            'insurance_cost_annual': 120000,
            'property_tax_rate': 0.012,
            'utilities_cost_monthly': 15000,
            'labor_cost_annual': 180000,
            'wholesale_h2_cost_per_kg': 8.50,
            'delivery_cost_per_kg': 2.00,
            'inflation_rate': 0.025,
            'tax_rate': 0.28,
            
            # Spatial parameters
            'service_radius_miles': 2.0,
            'min_station_spacing_miles': 1.0,
            'demand_kernel_bandwidth_miles': 2.0,
            'max_loi_route_distance_miles': 2.0,
            'interchange_detection_radius_m': 1000.0,
            'target_crs': 'EPSG:3310',
            'spatial_grid_resolution_m': 1000,
            'demand_grid_resolution_m': 500,
            
            # Behavioral parameters
            'rest_area_attraction_weight': 0.25,
            'port_proximity_weight': 0.20,
            'highway_access_weight': 0.20,
            'existing_station_weight': 0.10,
            'existing_station_utilization': 0.7,
            'gas_station_proximity_weight': 0.15,
            'amenity_weight': 0.10,
            
            # Competition parameters
            'capacity_saturation_factor': 0.7,
            'competition_decay_rate': 0.3,
            'competition_adjustment_factor': 0.8,
            'distance_decay_exponent': 2.0,
            'existing_station_competition_weight': 1.0,
            'potential_station_competition_weight': 0.7,
            'competition_distance_offset': 100,
            'use_gravity_competition_model': True,
            
            # Performance parameters
            'max_optimization_iterations': 1000,
            'convergence_tolerance': 1e-6,
            'optimization_timeout_seconds': 3600,
            'memory_cleanup_frequency': 100,
            
            # Validation parameters
            'cv_folds': 5,
            'validation_holdout_fraction': 0.2,
            'min_model_r_squared': 0.7,
            'max_cv_rmse': 0.3,
        }
    
    def _validate_config(self):
        """Validate configuration parameters for consistency and feasibility."""
        errors = []
        
        # Economic parameter validation
        if self.config['discount_rate'] <= 0 or self.config['discount_rate'] >= 1:
            errors.append("discount_rate must be between 0 and 1")
        
        if self.config.get('h2_price_per_kg', 0) <= self.config.get('wholesale_h2_cost_per_kg', 0):
            errors.append("h2_price_per_kg must be greater than wholesale_h2_cost_per_kg")
        
        # Spatial parameter validation
        if self.config['service_radius_miles'] <= 0:
            errors.append("service_radius_miles must be positive")
        
        if self.config['min_station_spacing_miles'] > self.config['service_radius_miles']:
            errors.append("min_station_spacing_miles cannot exceed service_radius_miles")
        
        # Capacity parameter validation
        if self.config['min_iteration_capacity_kg_per_day'] >= self.config['max_iteration_capacity_kg_per_day']:
            errors.append("min_iteration_capacity_kg_per_day must be less than max_iteration_capacity_kg_per_day")
        
        # Station economics validation
        if self.config['station_capex'] <= 0:
            errors.append("station_capex must be positive")
        
        if self.config['base_opex_daily'] <= 0:
            errors.append("base_opex_daily must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    """
    Deprecated methods
    
    
    def load_data(self, route_file, loi_files, existing_stations=None):
        
       # Load and preprocess all input data.
        
        Args:
            route_file: Path to truck route network with AADT
            loi_files: List of paths to locations of interest
            existing_stations: Path to existing/planned station data
        
        print("Loading route network...")
        self.routes = gpd.read_file(route_file)
        if self.routes.crs is None:
            self.routes.set_crs("EPSG:4326", inplace=True)
        self.routes = self.routes.to_crs("EPSG:3310")  # California Albers
        
        # Extract truck AADT
        self._extract_truck_flows()
        
        print("Loading locations of interest...")
        self.lois = self._load_multiple_lois(loi_files)
        
        if existing_stations:
            print("Loading existing infrastructure...")
            self.existing_stations = gpd.read_file(existing_stations).to_crs("EPSG:3310")
        else:
            self.existing_stations = gpd.GeoDataFrame(geometry=[], crs="EPSG:3310")
    
        """
        
    def load_data(self, route_file, merged_loi_file, gas_stations_csv=None, existing_stations=None):
        """
        Load route network and extract LOI types from merged LOI-route file.
        INTEGRATED: Directly processes merged LOI file instead of intermediate extractions.
        
        Args:
            route_file: Path to truck route network with AADT
            merged_loi_file: Path to merged LOI-route matching results (from loi_route_matching_workflow)
            gas_stations_csv: Optional path to gas stations CSV
            existing_stations: Path to existing/planned station data
        """
        print("Loading route network...")
        self.routes = gpd.read_file(route_file)
        if self.routes.crs is None:
            self.routes.set_crs("EPSG:4326", inplace=True)
        self.routes = self.routes.to_crs("EPSG:3310")  # California Albers
        
        # Extract truck AADT
        self._extract_truck_flows()
        
        print("Loading and processing merged LOI data...")
        self.lois = self._process_merged_loi_data(merged_loi_file, gas_stations_csv)
        
        if existing_stations:
            print("Loading existing infrastructure...")
            self.existing_stations = gpd.read_file(existing_stations).to_crs("EPSG:3310")
             # Convert station_capacity to expected_demand_kg_day
            if 'station_capacity' in self.existing_stations.columns:
                # Assume 70% utilization for existing stations as a reasonable estimate per config
                self.existing_stations['initial_demand_post_existing'] = self.existing_stations['station_capacity']
                utilization_rate = self.config.get('existing_station_utilization', 0.7)
                self.existing_stations['competition_agnostic_adjusted_demand'] = (
                    self.existing_stations['station_capacity'] * utilization_rate
                )
                self.existing_stations['expected_demand_kg_day'] = (
                    self.existing_stations['station_capacity'] * utilization_rate
                )
                print(("Successfully loaded existing stations with capacity data in preference loop. "))
            else:
                # Fallback if no capacity data
                default_capacity = self.config.get('station_capacity_kg_per_day', 2000)
                self.existing_stations['initial_demand_post_existing'] = default_capacity
                self.existing_stations['competition_agnostic_adjusted_demand'] = self.existing_stations['initial_demand_post_existing'] * self.config.get('existing_station_utilization', 0.7)
                self.existing_stations['expected_demand_kg_day'] = self.existing_stations['initial_demand_post_existing'] * self.config.get('existing_station_utilization', 0.7)
                print(("Successfully loaded existing stations with capacity data in fallback loop. "))
                
        else:
            print("No existing infrastructure data provided, initializing empty GeoDataFrame.")
            self.existing_stations = gpd.GeoDataFrame(geometry=[], crs="EPSG:3310")


    """""
    Deprecated methods
    def _load_and_extract_lois_from_merged(self, merged_file):
        
        #Load merged LOI-route file and extract different LOI types.
        INTEGRATED: Based on extract_loi_types_from_merged but processes in memory.
        
        print(f"  Loading merged LOI data from {merged_file}...")
        
        try:
            # Load the merged LOI-route matching results
            merged_gdf = gpd.read_file(merged_file)
            print(f"  Loaded {len(merged_gdf)} LOI-route matches")
            
            # Ensure proper CRS
            if merged_gdf.crs is None:
                merged_gdf.set_crs("EPSG:4326", inplace=True)
            if merged_gdf.crs != "EPSG:3310":
                merged_gdf = merged_gdf.to_crs("EPSG:3310")
            
            # Extract unique LOIs by grouping by source file and deduplicating
            all_lois = []
            
            print("  Extracting LOI types from merged data:")
            
            for source_file, group in merged_gdf.groupby('source_file'):
                if pd.notna(source_file):
                    # Get unique LOIs (since routes create duplicates due to multiple matches)
                    unique_lois = group.drop_duplicates(subset=['loi_uid'])
                    
                    # Determine LOI type from source file name (same logic as original)
                    loi_type = self._determine_loi_type(source_file)
                    
                    # Add LOI type and source information
                    unique_lois = unique_lois.copy()
                    unique_lois['loi_type'] = loi_type
                    unique_lois['source'] = f"{loi_type}_from_{os.path.basename(source_file)}"
                    
                    all_lois.append(unique_lois)
                    print(f"    Extracted {len(unique_lois)} {loi_type} from {os.path.basename(source_file)}")
            
            # Combine all LOI types
            if all_lois:
                combined_lois = pd.concat(all_lois, ignore_index=True)
                combined_gdf = gpd.GeoDataFrame(combined_lois, geometry='geometry', crs="EPSG:3310")
                
                print(f"  Total LOIs extracted: {len(combined_gdf)}")
                
                # Optional: Add gas stations from CSV if available
                gas_stations_csv = "gas_stations_from_CA_Energy_Commission_Gas_Stations.csv"
                if os.path.exists(gas_stations_csv):
                    gas_gdf = self._load_gas_stations_csv(gas_stations_csv)
                    if gas_gdf is not None and len(gas_gdf) > 0:
                        combined_gdf = pd.concat([combined_gdf, gas_gdf], ignore_index=True)
                        print(f"  Added {len(gas_gdf)} gas stations from CSV")
                
                return combined_gdf
            else:
                print("  Warning: No LOI data found in merged file")
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:3310")
                
        except Exception as e:
            print(f"  Error loading merged LOI file: {e}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:3310")
    """




    def _determine_loi_type_from_source(self, source_file):
        """
        Determine LOI type from source file name.
        EXTRACTED: Same logic as extract_loi_types_from_merged.
        """
        source_lower = source_file.lower()
        
        if 'airport' in source_lower:
            return 'airports'
        elif 'port' in source_lower or 'seaport' in source_lower:
            return 'ports'
        elif 'rest' in source_lower or 'parking' in source_lower or 'stop' in source_lower or 'truck stop' in source_lower:
            return 'rest_areas'
        elif 'traffic' in source_lower or 'generator' in source_lower:
            return 'traffic_generators'
        elif 'mhd' in source_lower or 'ze' in source_lower or 'infrastructure' in source_lower:
            return 'existing_ze_infrastructure'
        elif 'gas' in source_lower or 'energy' in source_lower:
            return 'gas_stations'
        else:
            return 'unknown'

    
    def _load_gas_stations_from_csv(self, csv_file):
        
        """
        Load gas stations from CSV file if available.
        INTEGRATED: From original extract_loi_types_from_merged logic.
        """
        try:
            print(f"    Processing gas stations from {os.path.basename(csv_file)}...")
            
            gas_df = pd.read_csv(csv_file)

            if gas_df.empty:
                print(f"    Warning: Gas stations CSV is empty, skipping...")
                return None
            
            print(f" CSV loaded: {len(gas_df)} rows")
            
            # Find coordinate columns
            lat_col = lon_col = None
            
            
            # Direct exact matches first  
            if 'latitude' in gas_df.columns:
                lat_col = 'latitude'
            if 'longitude' in gas_df.columns:
                lon_col = 'longitude'  
                
             
                
                """              
                coordinate_patterns = {
                    'latitude': ['lat', 'latitude', 'y', 'lat_dd', 'latitude_dd'],
                    'longitude': ['lon', 'long', 'longitude', 'x', 'lng', 'lon_dd', 'longitude_dd']
                }         

                for col in gas_df.columns:
                    col_lower = str(col).lower().strip()

                    
                    # Then partial matches:
                    #elif lat_col is None and any(p in col_lower for p in ['lat']):
                        #lat_col = col
                    #elif lon_col is None and any(p in col_lower for p in ['lon', 'lng']):
                        #lon_col = col 
                        
                                    
                    if lat_col is None:
                        for pattern in coordinate_patterns['latitude']:
                            if pattern in col_lower:
                                lat_col = col
                                break
                        
                    if lon_col is None:
                        for pattern in coordinate_patterns['longitude']:
                            if pattern in col_lower:
                                lon_col = col
                                break
            """  
            if not lat_col or not lon_col:
                print(f" No co-ordinate columns found. Available columns: {list(gas_df.columns)}")
                return None
            
            print(f" Found co-ordinates: {lat_col}, {lon_col}")
            
            # Clean and validate co-ordinates
            original_count = len(gas_df)
            
            # Remove rows with missing co-ordinates
            gas_df = gas_df.dropna(subset=[lat_col, lon_col])
            print(f" After removing NaN: {len(gas_df)} rows")
            
            # Convert to numeric
            gas_df[lat_col] = pd.to_numeric(gas_df[lat_col], errors='coerce')
            gas_df[lon_col] = pd.to_numeric(gas_df[lon_col], errors='coerce')
            
            # Remove rows that couldn't be converted
            gas_df = gas_df.dropna(subset=[lat_col, lon_col])
            print(f"    After numeric conversion: {len(gas_df)} rows")
            
            # Remove clearly invalid coordinates
            # Valid lat/lon ranges for California: lat 32-42, lon -125 to -114
            valid_coords = (
                (gas_df[lat_col] >= 32) & (gas_df[lat_col] <= 42) &
                (gas_df[lon_col] >= -125) & (gas_df[lon_col] <= -114) &
                (gas_df[lat_col] != 0) & (gas_df[lon_col] != 0)
            )
            
            gas_df = gas_df[valid_coords]
            print(f"    After coordinate validation: {len(gas_df)} rows")
            
            if gas_df.empty:
                print(f" No valid co-ordinates remaining")
                return None
            
            # Create GeoDataFrame
            
            try:
                geometry = [Point(x, y) for x, y in zip(gas_df[lon_col], gas_df[lat_col])]
                gas_gdf = gpd.GeoDataFrame(gas_df, geometry=geometry, crs="EPSG:4326")
                gas_gdf = gas_gdf.to_crs("EPSG:3310")
                
                # Add LOI metadata
                gas_gdf['loi_type'] = 'gas_stations'
                gas_gdf['source'] = 'gas_stations_from_csv'
                gas_gdf['loi_uid'] = [f"gas_station_csv_{i}" for i in range(len(gas_gdf))]
                
                print(f"    Successfully created {len(gas_gdf)} gas station locations")
                return gas_gdf
            
            except Exception as geom_error:
                print(f" Error creating GeoDataFrame: {geom_error}")
                import traceback
                traceback.print_exc()     
                return None              
                
        except Exception as e:
            print(f"    Error loading gas stations CSV: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    """
    Depracated method
    def _load_gas_stations_from_csv(self, csv_file):
        
        #DIAGNOSTIC VERSION: Find exactly what's happening
    
        try:
            print(f"🔍 DEBUG: _load_gas_stations_from_csv called with {csv_file}")
            print(f"🔍 DEBUG: This is the method in h2_station_model.py")
            
            gas_df = pd.read_csv(csv_file)
            print(f"🔍 DEBUG: CSV loaded: {len(gas_df)} rows")
            print(f"🔍 DEBUG: Column types: {dict(gas_df.dtypes)}")
            print(f"🔍 DEBUG: First 3 column names: {list(gas_df.columns[:3])}")
            print(f"🔍 DEBUG: Last 3 column names: {list(gas_df.columns[-3:])}")
            
            # Test coordinate detection step by step
            lat_col = lon_col = None
            
            print(f"🔍 DEBUG: Looking for coordinate columns...")
            print(f"🔍 DEBUG: 'latitude' in columns: {'latitude' in gas_df.columns}")
            print(f"🔍 DEBUG: 'longitude' in columns: {'longitude' in gas_df.columns}")
            
            # Direct exact match test
            if 'latitude' in gas_df.columns:
                lat_col = 'latitude'
                print(f"🔍 DEBUG: Found latitude column: {lat_col}")
                
            if 'longitude' in gas_df.columns:
                lon_col = 'longitude'  
                print(f"🔍 DEBUG: Found longitude column: {lon_col}")
            
            if lat_col and lon_col:
                print(f"🔍 DEBUG: SUCCESS - Both coordinate columns found")
                return None  # Don't process further, just debug
            else:
                print(f"🔍 DEBUG: FAILED - Coordinate detection failed")
                print(f"🔍 DEBUG: lat_col = {lat_col}, lon_col = {lon_col}")
                return None
                
        except Exception as e:
            print(f"🔍 DEBUG: Exception in _load_gas_stations_from_csv: {e}")
            import traceback
            traceback.print_exc()
            return None
    """



    def _process_merged_loi_data(self, merged_file, gas_stations_csv=None):
        """
        Process merged LOI-route file and extract different LOI types.
        INTEGRATED: Handles all LOI processing in memory without file extraction.
        """
        all_lois = []
        
        try:
            # Load merged LOI-route matching results
            print(f"  Loading merged LOI data from {os.path.basename(merged_file)}...")
            merged_gdf = gpd.read_file(merged_file)
            print(f"  Loaded {len(merged_gdf)} LOI-route matches")
            
            # Ensure proper CRS
            if merged_gdf.crs is None:
                merged_gdf.set_crs("EPSG:4326", inplace=True)
            if merged_gdf.crs != "EPSG:3310":
                merged_gdf = merged_gdf.to_crs("EPSG:3310")
            
            # Extract unique LOIs by source file
            loi_type_counts = {}
            
            for source_file, group in merged_gdf.groupby('source_file'):
                if pd.notna(source_file):
                    # Get unique LOIs (routes create duplicates)
                    unique_lois = group.drop_duplicates(subset=['loi_uid'])
                    
                    # Determine LOI type
                    loi_type = self._determine_loi_type_from_source(source_file)
                    
                    # Add metadata
                    unique_lois = unique_lois.copy()
                    unique_lois['loi_type'] = loi_type
                    unique_lois['source'] = f"{loi_type}_from_{os.path.basename(source_file)}"
                    
                    all_lois.append(unique_lois)
                    loi_type_counts[loi_type] = len(unique_lois)
            
            # Load additional gas stations from CSV if provided
            if gas_stations_csv and os.path.exists(gas_stations_csv):
                print(f"  Loading additional gas stations from CSV...")
                gas_gdf = self._load_gas_stations_from_csv(gas_stations_csv)
                
                if gas_gdf is not None and len(gas_gdf) > 0:
                    all_lois.append(gas_gdf)
                    loi_type_counts['gas_stations_csv'] = len(gas_gdf)
                    print(f" Added {len(gas_gdf)} gas stations from CSV")
                else:
                    print(f" Failed to load gas stations from CSV")
            
            # Combine all LOI types
            if all_lois:
                combined_lois = pd.concat(all_lois, ignore_index=True)
                combined_gdf = gpd.GeoDataFrame(combined_lois, geometry='geometry', crs="EPSG:3310")
                
                # Print summary
                print(f"  LOI types extracted:")
                for loi_type, count in loi_type_counts.items():
                    print(f"    {count:>5} {loi_type}")
                print(f"  Total LOIs: {len(combined_gdf)}")
                
                return combined_gdf
            else:
                print("  Warning: No LOI data found")
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:3310")
                
        except Exception as e:
            print(f"  Error processing merged LOI file: {e}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:3310")


    def _extract_truck_flows(self):
        """Extract truck AADT from route attributes."""
        # Look for truck-specific AADT columns
        truck_cols = [c for c in self.routes.columns if 'trk' in c.lower() and 'aadt' in c.lower()]
        
        if truck_cols:
            # Use total truck AADT if available
            total_truck_col = next((c for c in truck_cols if 'tot' in c.lower()), truck_cols[0])
            self.routes['truck_aadt'] = self.routes[total_truck_col]
        else:
            # Estimate truck percentage of total AADT
            aadt_cols = [c for c in self.routes.columns if 'aadt' in c.lower()]
            if aadt_cols:
                self.routes['truck_aadt'] = self.routes[aadt_cols[0]] * 0.15  # Assume 15% trucks
            else:
                raise ValueError("No AADT columns found in route data")
        
        # Calculate segment lengths
        self.routes['length_miles'] = self.routes.geometry.length / 1609.34
        
    def _load_multiple_lois(self, loi_files):
        """Load and combine multiple LOI files.
        Detects file format by content, not extension.
        """
        all_lois = []
        
        for file in loi_files:
            try:
                print(f"  Loading {os.path.basename(file)}...")
                
                # Detect file format by content
                file_format = self._detect_file_format(file)
                
                gdf = None
                
                #if file_format == 'geojson':
                # file if GeoJSON regardless of extension
                # Always try to read as GeoJson/GeoDataFrame first
                try:
                    gdf = gpd.read_file(file)
                    
                    # Verify it's actually a GeoDataFrame wit h geometry
                    if isinstance(gdf, gpd.GeoDataFrame) and hasattr(gdf, 'geometry') and not gdf.geometry.empty:
                        print(f"    Successfully loaded {len(gdf)} features from {os.path.basename(file)}")
                    else:
                        print(f"    GeoPandas read succeeded but no valid geometry, trying CSV approach...")
                        gdf = None  # Reset gdf to None to force CSV processing
                        
                except Exception as e:
                    print(f"    GeoPandas loading failed: {e}, trying as CSV...")
                    gdf = None  # Reset gdf to None to force CSV processing
                    
                    #if not isinstance(gdf, gpd.GeoDataFrame):
                        #print(f"    Warning: gpd.read_file returned DataFrame, converting...")
                        # Try to convert if it has geometry column
                        #if 'geometry' in gdf.columns:
                            #gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
                        #else:
                            #print(f"    No geometry column found, trying coordinate extraction...")
                            #raise ValueError("No geometry found")
                
                if gdf is None:
                    try:
                        # try reading as csv with more flexible matching        
                #except Exception as e:
                    #print(f"    GeoJSON loading failed: {e}, trying as CSV...")
                    #file_format = 'csv'  # Fall back to CSV processing        
                
                #if file_format == 'csv':
                    # file is actually CSV or GeoJson loading failed
                        df = pd.read_csv(file, 
                                         on_bad_lines='warn',
                                         quoting=3, # QUOTE_NONE - don't treat quotes as special
                                         sep=',',
                                         encoding='utf-8')
                    
                        if df.empty:
                            print(f"Warning: {file} read as CSV is empty, skipping...")
                            continue
                    
                        print(f"    Successfully loaded {len(df)} rows from {os.path.basename(file)}")
                        # Find lat/lon columns with more flexible matching
                        lat_col = None
                        lon_col = None
                    
                        # Check for coordinate columns in properties or direct columns
                        coordinate_patterns = {
                            'lat': ['lat', 'latitude', 'y_coord', 'y', 'LATITUDE', 'Latitude'],
                            'lon': ['lon', 'long', 'longitude', 'x_coord', 'x', 'LONGITUDE', 'Longitude']
                        }
                                    
                        for col in df.columns:
                            col_lower = col.lower()
                            if lat_col is None:
                                for pattern in coordinate_patterns['lat']:
                                    if pattern in col_lower:
                                        lat_col = col
                                        break
                                
                            if lon_col is None:
                                for pattern in coordinate_patterns['lon']:
                                    if pattern in col_lower:
                                        lon_col = col
                                        break
                                    
                                
                                
                        if lat_col and lon_col:
                            print(f"    Found coordinates: {lat_col}, {lon_col}")
                            # Clean the data - remove any rows with invalid coordinates
                            df = df.dropna(subset=[lat_col, lon_col])
                            
                            
                            # Convert to numeric, coercing errors to NaN
                            df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
                            df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
                            
                            # Remove any remaining NaN coordinates
                            df = df.dropna(subset=[lat_col, lon_col])
                            
                            if df.empty:
                                print(f"Warning: {file} has no valid coordinates, skipping...")
                                continue
                            
                            # Create GeoDataFrame
                            geometry = [Point(x, y) for x, y in zip(df[lon_col], df[lat_col])]
                            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
                            
                        else:
                            print(f"    Warning: No recognizable lat/lon columns in {file}. Could not create geometry")
                            print(f"    Available columns: {list(df.columns)}")
                            continue
                        
                        
                    except Exception as csv_error:
                        print(f"Warning: Could not read {file} as CSV: {csv_error}")
                        continue    
                    
                # If we still don't have a GeoDataFrame, try to read as shapefile or other formats
                if gdf is None:
                    
                    try:
                        print(f"    Attempting to read {file} as shapefile or other format...")
                        gdf = gpd.read_file(file)
                        
                    except:
                        print(f"Warning: Could not read {file} as any recognized format. Skipping...")
                        continue
                
                    
                # Ensure CRS consistency
                if not isinstance(gdf, gpd.GeoDataFrame):
                    print(f"Warning: Error: Could not create GeoDataFrame for {file}. Skipping...")
                    continue
                
                
                try:
                    if gdf.crs is None:
                        print(f"Warning: No CRS found for {file}, setting to EPSG:4326")
                        gdf.set_crs("EPSG:4326", inplace=True)
                        
                    if gdf.crs != "EPSG:3310":
                        print(f"Warning: CRS for {file} is not EPSG:3310, converting to EPSG:3310...")
                        gdf = gdf.to_crs("EPSG:3310")
                except Exception as crs_error:
                    print(f"Warning: Error setting CRS for {file}: {crs_error}")
                    
                    # Try to set a default CRS if possible
                    try:
                        gdf = gdf.set_crs("EPSG:4326").to_crs("EPSG:3310")
                    except:
                        print(f"Warning: Could not set CRS for {file}: {crs_error}. Skippping file...")
                        continue
                
                # Add source file information
                gdf['source'] = os.path.basename(file)
                all_lois.append(gdf)
                print(f"  Successfully loaded {len(gdf)} features from {os.path.basename(file)}")
                    
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                import traceback
                traceback.print_exc()
    
        
        if all_lois:
            combined = pd.concat(all_lois, ignore_index=True)
            print(f"Total LOIs loaded: {len(combined)}")
            return combined
        
        else:
            print("Warning: No LOI data could be loaded")  
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:3310")      
        
          
        
    def _detect_file_format(self, file_path):    
        """Detect file format by examining content, not extension.
        Returns 'geojson', 'csv', or 'unknown'.
        """      
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few characters to check format
                content_start = f.read(200)
                
            # Check if it contains GeoJSON structure
            if any(indicator in content_start for indicator in [
                '"type": "FeatureCollection"',
                '"type":"FeatureCollection"', 
                '"type": "Feature"',
                '"geometry"',
                '"coordinates"'
            ]):
                return 'geojson'
            
            # Check if it looks like CSV (has comma-separated headers)
            elif ',' in content_start and not '{' in content_start[:50]:
                
                return 'csv'
                ## Peek at a few more lines to be sure
                #with open(file_path, 'r', encoding='utf-8') as f:
                    #lines = [f.readline().strip() for _ in range(3)]
                
                # If multiple lines have commas and no JSON structure, likely CSV
                #if all(',' in line and not line.startswith('{') for line in lines if line):
                    #return 'csv'
            
            return 'unknown'
            
        except Exception as e:
            print(f"Warning: Could not detect format for {file_path}: {e}")
            return 'unknown'        
            

    """""
    #Depracated methods
    
    def estimate_demand_surface(self):
        
        #Create continuous hydrogen demand surface using kernel density estimation.
        
        print("Estimating hydrogen demand surface...")
        # Check if demand is already calculated
        if 'h2_demand_daily_kg' in self.routes.columns:
            print(" Using pre-calculated h2_demand_daily_kg")
            self.routes['h2_demand_kg_day'] = self.routes['h2_demand_daily_kg']
        else:
            print(" Using fallback calculation")
            # Calculate demand at each route segment
            self.routes['h2_demand_kg_day'] = (
                self.routes['truck_aadt'] * 
                self.routes['length_miles'] * 
                self.config['h2_consumption_kg_per_mile']
            )
            # Verify the values
        print(f"  Using demand - Max: {self.routes['h2_demand_kg_day'].max():.1f} kg/day")
        print(f"  Using demand - Total: {self.routes['h2_demand_kg_day'].sum():.0f} kg/day")
    
        # Create point grid for demand surface
        bounds = self.routes.total_bounds
        x_points = np.linspace(bounds[0], bounds[2], 100)
        y_points = np.linspace(bounds[1], bounds[3], 100)
        xx, yy = np.meshgrid(x_points, y_points)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Sample points along routes weighted by demand
        route_points = []
        route_weights = []
        
        print(f"Sampling route points from {len(self.routes)} routes...")

        
        for idx, route in self.routes.iterrows():
            if route['h2_demand_kg_day'] <= 0:
                continue
            
                # Cap sampling to reasonable levels
        
            n_samples = max(5, min(50, int(route['h2_demand_kg_day'] / 10000)))  # 5-50 samples per route
    
            # Sample points along route proportional to demand
            #n_samples = max(1, int(route['h2_demand_kg_day'] / 100))
            if route.geometry.geom_type == 'LineString':
                distances = np.linspace(0, route.geometry.length, n_samples)
                for d in distances:
                    pt = route.geometry.interpolate(d)
                    route_points.append([pt.x, pt.y])
                    route_weights.append(route['h2_demand_kg_day'] / n_samples)
        
        if not route_points:
            print(" No valid route points generated!")
            return
        
        route_points = np.array(route_points)
        route_weights = np.array(route_weights)
        
        print(f"Generated {len(route_points)} route points")
        print(f"Route weights - Min: {route_weights.min():.1f}, Max: {route_weights.max():.1f}")

        # Adaptive bandwidth based on actual coordinate system
        # Calculate typical distance between points for bandwidth estimation
        
        sample_distances = []
        if len(route_points) > 100:
            # Sample some distances to estimate scale
            for i in range(0, min(100, len(route_points)), 10):
                for j in range(i+1, min(i+10, len(route_points))):
                    dist = np.sqrt(((route_points[i] - route_points[j])**2).sum())
                    sample_distances.append(dist)

            if sample_distances:
                median_distance = np.median(sample_distances)
                # Set bandwidth as fraction of typical distance
                bandwidth = max(median_distance * 0.1, 1000)  # At least 1km
                print(f"Adaptive bandwidth: {bandwidth:.0f} meters")
            else:
                bandwidth = 5000  # 5km default
        else:
            bandwidth = 5000  # 5km default
            
            
        # Kernel density estimation
        print(f"Calculating KDE over {len(grid_points)} grid points...")
        
        #bandwidth = self.config['demand_kernel_bandwidth_miles'] * 1609.34  # Convert to meters
        
        # Calculate weighted KDE manually
        demand_values = np.zeros(len(grid_points))
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        
        for batch_start in range(0, len(grid_points), batch_size):
            batch_end = min(batch_start + batch_size, len(grid_points))
            batch_points = grid_points[batch_start:batch_end]
            
            # Calculate distances from batch grid points to all route points
        
            for i, grid_pt in enumerate(batch_points):
                distances = np.sqrt(((route_points - grid_pt)**2).sum(axis=1))
                
                # Only consider points within reasonable distance (5x bandwidth)
                nearby_mask = distances < (5 * bandwidth)
                
                if np.any(nearby_mask):
                    nearby_distances = distances[nearby_mask]
                    nearby_weights = route_weights[nearby_mask]
                    # Calculate kernel density value estimates
                    kernel_values = np.exp(-0.5 * (nearby_distances / bandwidth)**2) / (bandwidth * np.sqrt(2 * np.pi))
                    demand_values[batch_start + i] = np.sum(kernel_values * nearby_weights)
        
        print(f"KDE calculation complete")
        print(f"Demand values - Min: {demand_values.min():.3f}, Max: {demand_values.max():.1f}")

        self.demand_surface = {
            'x': xx,
            'y': yy,
            'demand': demand_values.reshape(xx.shape),
            'points': grid_points
        }
        
        print(f"  Max demand: {demand_values.max():.1f} kg/day")
        print(f"  Mean demand: {demand_values.mean():.1f} kg/day")
        
        
        print("=== SPATIAL COVERAGE DIAGNOSTIC ===")

        # Check how many grid points actually get non-zero values
        nonzero_points = np.sum(demand_values > 0)
        print(f"Grid points with non-zero demand: {nonzero_points} / {len(grid_points)}")
        print(f"Percentage coverage: {100 * nonzero_points / len(grid_points):.1f}%")

        # Check distance statistics
        sample_grid_pt = grid_points[len(grid_points)//2]  # Middle grid point
        distances_to_routes = np.sqrt(((route_points - sample_grid_pt)**2).sum(axis=1))
        print(f"Sample grid point distances to routes:")
        print(f"  Min distance: {distances_to_routes.min()/1000:.1f} km")
        print(f"  Median distance: {np.median(distances_to_routes)/1000:.1f} km")

        # Check route point spatial distribution
        print(f"Route points bounds:")
        print(f"  X: {route_points[:,0].min():.0f} to {route_points[:,0].max():.0f}")
        print(f"  Y: {route_points[:,1].min():.0f} to {route_points[:,1].max():.0f}")

        print(f"Grid bounds:")
        print(f"  X: {grid_points[:,0].min():.0f} to {grid_points[:,0].max():.0f}")  
        print(f"  Y: {grid_points[:,1].min():.0f} to {grid_points[:,1].max():.0f}")
    """
    
    class CompetitionGraph:
        """
        Network graph to track competitive relationships between stations.
        Prevents double counting and enables efficient market share calculations.
        """
        def __init__(self, config):
            self.nodes = {}  # {station_id: {'demand': float, 'utility': float, 'type': str, 'geometry': Point}}
            self.edges = {}  # {(station1, station2): {'weight': float, 'distance': float}}
            self.config = config
            
        def add_or_update_station(self, station_id, station_data, station_type='candidate'):
            """Add or update a station node in the graph."""
            # Calculate utility/attractiveness
            
            # Extract station geometry coordinates
            if hasattr(station_data, 'geometry'):
                x = station_data.geometry.x
                y = station_data.geometry.y
            elif 'geometry' in station_data:
                x = station_data['geometry'].x
                y = station_data['geometry'].y
            elif 'x' in station_data and 'y' in station_data:
                x = station_data['x']
                y = station_data['y']
            else:
                # Try to extract from other fields
                if 'longitude' in station_data and 'latitude' in station_data:
                    x = station_data['longitude']
                    y = station_data['latitude']
                else:
                    raise ValueError(f"Cannot extract coordinates for station {station_id}")
    
            
            utility = self._calculate_utility(station_data)
            competition_agnostic_station_utilization_factor = station_data['p_need_fuel'] * station_data['p_stop_given_need'] if 'stop_given_need' in station_data and 'p_need_fuel' in station_data else self.config.get('existing_station_utilization', 0.7)
            
            self.nodes[station_id] = {
                'demand': station_data.get('expected_demand_kg_day', 0),
                'base_demand': station_data.get('competition_agnostic_adjusted_demand', station_data['initial_demand_post_existing'] * competition_agnostic_station_utilization_factor),
                'utility': utility,
                'type': station_type,  # 'existing', 'selected', or 'candidate'
                'geometry': station_data['geometry'],
                'x': x,
                'y': y,
                'data': station_data  # Store full data for reference
            }
            
        def add_competition_edge(self, station1_id, station2_id):
            """Add competitive relationship if within service radius."""
            if station1_id == station2_id:
                return
            
            
            # Check that both nodes exist before creating edge
            if station1_id not in self.nodes:
                # Don't create edge to non-existent node
                return

            if station2_id not in self.nodes:
                # Don't create edge to non-existent node
                return
            
            # Create consistent edge key for indexing
            id1_str = str(station1_id)
            id2_str = str(station2_id)
            
            # Check if edge already exists
            edge_key = f"{min(id1_str, id2_str)}-{max(id1_str, id2_str)}"
            #edge_key = tuple(sorted([str(station1_id), str(station2_id)]))
            if edge_key in self.edges:
                return  # Edge already exists, don't recalculate
            
            # Calculate distance
            if hasattr(self.nodes[station1_id], 'geometry') and hasattr(self.nodes[station2_id], 'geometry'):
                geom1 = self.nodes[station1_id]['geometry']
                geom2 = self.nodes[station2_id]['geometry']
                distance = geom1.distance(geom2)
            else:
                node1 = self.nodes[station1_id]
                node2 = self.nodes[station2_id]
                
                distance = np.sqrt((node1['x'] - node2['x'])**2 + 
                                (node1['y'] - node2['y'])**2)
            
            # Only add edge if within service radius
            service_radius_m = self.config['service_radius_miles'] * 1609.34
            if distance < service_radius_m:
                # Calculate edge weight using gravity model
                offset = self.config.get('competition_distance_offset', 100)
                decay = self.config.get('distance_decay_exponent', 2.0)
                weight = (distance + offset) ** decay
                
                self.edges[edge_key] = {
                    'weight': weight,
                    'distance': distance
                }
                
        def calculate_market_share(self, station_id):
            """
            Calculate market share for a station based on graph structure.
            Returns the fraction of market this station captures.
            """
            if station_id not in self.nodes:
                print(f"Warning: Station {station_id} not found in competition graph")
                return 1.0
            
             
            station = self.nodes[station_id]
            
            # Convert station_id to string for edge key comparison
            station_id_str = str(station_id)
    
            station_utility = station['utility']
            total_utility = station_utility
            
            # Get competition weights based on station types
            type_weights = {
                'existing': self.config.get('existing_station_competition_weight', 1.0),
                'selected': 1.0,
                'candidate': self.config.get('potential_station_competition_weight', 0.7)
            }
            
            competitors = []
            # Sum competitive utilities from all connected stations
            for edge_key, edge_data in self.edges.items():
                # Parse edge key to check if station is involved
                edge_parts = edge_key.split('-')
                if len(edge_parts) == 2 and station_id_str in edge_parts:
                    # Find the competitor
                    if edge_parts[0] == station_id_str:
                        competitor_id_str = edge_parts[1]
                    else:
                        competitor_id_str = edge_parts[0]
                    
                    # Convert back to original type for node lookup
                    if competitor_id_str.startswith('existing_'):
                        competitor_id = competitor_id_str
                    elif competitor_id_str.isdigit():
                        competitor_id = int(competitor_id_str)
                    else:
                        competitor_id = competitor_id_str
                    
                    if competitor_id in self.nodes:
                        competitors.append(competitor_id)
                        competitor = self.nodes[competitor_id]    
                    
                    
                        # Calculate competitive utility
                        competitor_utility = competitor['utility']
                        competitor_weight = type_weights.get(competitor['type'], 1.0)
                        
                        # Add to total utility (distance decay through edge weight)
                        total_utility += (competitor_utility * competitor_weight) / edge_data['weight']
                        
                    #else:
                        #print(f"Warning: Competitor {competitor_id} not found in nodes. Skipping...")
            
            if not competitors:
                #print(f"Warning: No competitors found for station {station_id}")
                return 1.0
            
            # Market share with minimum retention
            market_share = station_utility / total_utility
            min_retention = self.config.get('min_demand_retention', 0.2)
            return max(market_share, min_retention)
        
        
            
        def update_network_demands(self):
            """Update demand for all nodes based on current market shares."""
            for station_id, node_data in self.nodes.items():
                market_share = self.calculate_market_share(station_id)
                base_demand = node_data['base_demand']
                node_data['demand'] = base_demand * market_share
                node_data['market_share'] = market_share
                
        def _calculate_utility(self, station_data):
            """Calculate station attractiveness/utility."""
            utility = 1.0
            
            # Location features
            if station_data.get('has_rest_area', 0):
                utility *= 1.2
            if station_data.get('is_interchange', 0):
                utility *= 1.1
                
            # Capacity effect
            capacity = station_data.get('capacity_kg_day', 
                                       self.config.get('station_capacity_kg_per_day', 2000))
            base_capacity = self.config.get('station_capacity_kg_per_day', 2000)
            utility *= (capacity / base_capacity) ** 0.5
            
            return utility
            
        def get_competitors(self, station_id):
            """Get all stations competing with this station."""
            competitors = []
            for edge_key in self.edges:
                if station_id in edge_key:
                    competitor_id = edge_key[0] if edge_key[1] == station_id else edge_key[1]
                    competitors.append(competitor_id)
            return competitors
    
    
    def estimate_demand_surface(self):
        """
        Estimate potential hydrogen demand surface using a theoretically grounded approach.
        
        This surface represents the MAXIMUM CAPTURABLE DEMAND at each location,
        considering:
        1. Proximity to multiple routes (network effects)
        2. Realistic detour behavior
        3. Demand decay with distance
        
        Key insight: This is NOT allocating route demand, but estimating the total
        addressable market at each location.
        """
        import time
        from scipy.spatial import cKDTree
        from scipy.sparse import csr_matrix
        from scipy.ndimage import gaussian_filter
        from shapely.geometry import Point, LineString
        from shapely.ops import unary_union
        
        start_time = time.time()
        
        print("\nEstimating hydrogen demand surface...")
        
        # Check cache first
        if hasattr(self, '_demand_grid_cache') and self._demand_grid_cache is not None:
            print("  Using cached demand surface")
            self.demand_surface = self._demand_grid_cache
            return
        
        # Validate input data
        if not hasattr(self, 'routes') or self.routes.empty:
            print("  ERROR: No route data available for demand surface")
            return
        
        # Step 1: Prepare demand data with proper column handling
        print("  Preparing route demand data...")
        
        # Handle different column names gracefully
        if 'h2_demand_daily_kg' in self.routes.columns:
            self.routes['h2_demand_kg_day'] = self.routes['h2_demand_daily_kg']
        elif 'h2_demand_kg_day' not in self.routes.columns:
            # Calculate from truck flows
            truck_col = 'truck_aadt' if 'truck_aadt' in self.routes.columns else 'TOT_TRK_AADT'
            length_col = 'length_miles' if 'length_miles' in self.routes.columns else 'length_mi'
            
            if truck_col in self.routes.columns and length_col in self.routes.columns:
                self.routes['h2_demand_kg_day'] = (
                    self.routes[truck_col] * 
                    self.routes[length_col] * 
                    self.config['h2_consumption_kg_per_mile']
                )
            else:
                print("  ERROR: Cannot calculate demand - missing truck flow or length data")
                return
        
        # Filter valid routes
        valid_routes = self.routes[
            (self.routes['h2_demand_kg_day'] > 0) & 
            (self.routes.geometry.notna())
        ].copy()
        
        if len(valid_routes) == 0:
            print("  ERROR: No valid routes with positive demand")
            return
        
        total_route_demand = valid_routes['h2_demand_kg_day'].sum()
        print(f"  Total route demand: {total_route_demand:,.0f} kg/day")
        print(f"  Processing {len(valid_routes)} valid routes")
        
        # Step 2: Define study area with intelligent bounds
        print("  Defining study area...")
        
        # Maximum detour distance (critical parameter)
        max_detour_miles = self.config.get('max_detour_miles', 5.0)
        max_detour_m = max_detour_miles * 1609.34
        
        # Create buffer around all routes
        try:
            all_geometries = []
            for geom in valid_routes.geometry:
                if geom.geom_type == 'LineString':
                    all_geometries.append(geom)
                elif geom.geom_type == 'MultiLineString':
                    all_geometries.extend(geom.geoms)
            
            unified_routes = unary_union(all_geometries)
            study_area = unified_routes.buffer(max_detour_m * 1.5)
            bounds = study_area.bounds
        except Exception as e:
            print(f"  Warning: Could not create unified geometry: {e}")
            bounds = valid_routes.total_bounds
        
        # Step 3: Create adaptive resolution grid
        area_km2 = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 1e6
        
        # Adaptive resolution based on area and route density
        if area_km2 < 1000:  # Small area - fine resolution
            grid_spacing = 1000  # 1 km
        elif area_km2 < 10000:  # Medium area
            grid_spacing = 2000  # 2 km
        else:  # Large area - coarser resolution
            grid_spacing = 5000  # 5 km
        
        # Ensure reasonable grid size
        n_x = int((bounds[2] - bounds[0]) / grid_spacing) + 1
        n_y = int((bounds[3] - bounds[1]) / grid_spacing) + 1
        
        # Cap grid size for memory efficiency
        if n_x * n_y > 10000:
            grid_spacing = np.sqrt(((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 10000)
            n_x = int((bounds[2] - bounds[0]) / grid_spacing) + 1
            n_y = int((bounds[3] - bounds[1]) / grid_spacing) + 1
        
        print(f"  Grid: {n_x}×{n_y} cells ({n_x * n_y:,} total)")
        print(f"  Resolution: {grid_spacing/1000:.1f} km")
        
        x_coords = np.linspace(bounds[0], bounds[2], n_x)
        y_coords = np.linspace(bounds[1], bounds[3], n_y)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Step 4: Create efficient route segment representation
        print("  Building route segment index...")
        
        # Segment routes for better spatial representation
        segment_length = min(grid_spacing / 2, 1000)  # Segments no longer than half grid size
        
        segments = []
        segment_demands = []
        segment_trucks = []
        
        for idx, route in valid_routes.iterrows():
            # Handle different geometry types
            if route.geometry.geom_type == 'MultiLineString':
                linestrings = list(route.geometry.geoms)
            else:
                linestrings = [route.geometry]
            
            route_length = route.geometry.length
            demand_per_meter = route['h2_demand_kg_day'] / route_length if route_length > 0 else 0
            
            # Get truck flow for this route
            truck_flow = route.get('truck_aadt', route.get('TOT_TRK_AADT', 0))
            trucks_per_meter = truck_flow / route_length if route_length > 0 else 0
            
            for linestring in linestrings:
                if linestring.length == 0:
                    continue
                    
                # Create segments
                n_segments = max(1, int(linestring.length / segment_length))
                
                for i in range(n_segments):
                    start_frac = i / n_segments
                    end_frac = (i + 1) / n_segments
                    
                    # Get segment endpoints
                    start_pt = linestring.interpolate(start_frac, normalized=True)
                    end_pt = linestring.interpolate(end_frac, normalized=True)
                    mid_pt = linestring.interpolate((start_frac + end_frac) / 2, normalized=True)
                    
                    # Store segment info
                    segments.append([mid_pt.x, mid_pt.y])
                    segment_demands.append(demand_per_meter * linestring.length / n_segments)
                    segment_trucks.append(trucks_per_meter * linestring.length / n_segments)
        
        segments = np.array(segments)
        segment_demands = np.array(segment_demands)
        segment_trucks = np.array(segment_trucks)
        
        print(f"  Created {len(segments):,} route segments")
        
        # Build KDTree for efficient spatial queries
        segment_tree = cKDTree(segments)
        
        # Step 5: Calculate demand surface using accessibility model
        print("  Calculating demand accessibility surface...")
        
        # Process in chunks for memory efficiency
        demand_grid = np.zeros((n_y, n_x))
        truck_grid = np.zeros((n_y, n_x))
        
        # Detour willingness decay parameter
        detour_decay = self.config.get('detour_decay_rate', 2.0)
        
        # Process each grid cell
        for i in range(n_x):
            if i % 10 == 0:
                print(f"    Processing column {i}/{n_x}...")
            
            for j in range(n_y):
                cell_location = np.array([xx[j, i], yy[j, i]])
                
                # Find all segments within maximum detour distance
                nearby_indices = segment_tree.query_ball_point(cell_location, max_detour_m)
                
                if not nearby_indices:
                    continue
                
                # Calculate accessibility-weighted demand
                total_accessible_demand = 0
                total_accessible_trucks = 0
                
                for seg_idx in nearby_indices:
                    # Distance from cell to segment
                    distance = np.linalg.norm(segments[seg_idx] - cell_location)
                    
                    # Detour probability decay function
                    # Based on transportation research: exponential decay with distance
                    detour_probability = np.exp(-detour_decay * distance / max_detour_m)
                    
                    # Account for competition from other potential locations
                    # (simplified - full model would consider actual alternatives)
                    competition_factor = 1.0
                    
                    # Capturable demand from this segment
                    capturable_demand = (
                        segment_demands[seg_idx] * 
                        detour_probability * 
                        competition_factor
                    )
                    
                    total_accessible_demand += capturable_demand
                    total_accessible_trucks += segment_trucks[seg_idx] * detour_probability
                
                demand_grid[j, i] = total_accessible_demand
                truck_grid[j, i] = total_accessible_trucks
        
        # Step 6: Apply spatial smoothing to account for uncertainty
        print("  Applying spatial smoothing...")
        
        # Smooth with Gaussian filter (sigma in grid cells)
        sigma = max(1, grid_spacing / 5000)  # More smoothing for coarser grids
        demand_grid_smooth = gaussian_filter(demand_grid, sigma=sigma, mode='constant')
        
        # Step 7: Create derived surfaces
        print("  Creating derived surfaces...")
        
        # Normalize surfaces for analysis
        max_demand = np.max(demand_grid_smooth)
        if max_demand > 0:
            demand_normalized = demand_grid_smooth / max_demand
        else:
            demand_normalized = demand_grid_smooth
        
        # Create capture efficiency surface (demand per truck)
        capture_efficiency = np.zeros_like(demand_grid)
        mask = truck_grid > 100  # Minimum trucks to avoid division issues
        capture_efficiency[mask] = demand_grid_smooth[mask] / truck_grid[mask]
        
        # Step 8: Validation and diagnostics
        print("\n  Validation metrics:")
        print(f"    Total route demand: {total_route_demand:,.0f} kg/day")
        print(f"    Max cell demand: {np.max(demand_grid_smooth):,.1f} kg/day")
        print(f"    Mean cell demand: {np.mean(demand_grid_smooth[demand_grid_smooth > 0]):,.1f} kg/day")
        print(f"    Non-zero cells: {np.sum(demand_grid_smooth > 0):,} ({np.sum(demand_grid_smooth > 0) / demand_grid_smooth.size:.1%})")
        
        # Check for reasonable values
        if np.max(demand_grid_smooth) > total_route_demand:
            print("  WARNING: Maximum cell demand exceeds total route demand")
            print("  This may indicate overlapping service areas (which is realistic)")
        
        # Step 9: Create interpolators for fast lookup
        from scipy.interpolate import RegularGridInterpolator
        
        self._demand_interpolator = RegularGridInterpolator(
            (y_coords, x_coords),
            demand_grid_smooth,
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        
        self._efficiency_interpolator = RegularGridInterpolator(
            (y_coords, x_coords),
            capture_efficiency,
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        
        # Step 10: Store comprehensive results
        self.demand_surface = {
            'x': xx,
            'y': yy,
            'points': grid_points,
            'demand': demand_grid_smooth,
            'demand_raw': demand_grid,
            'truck_flow': truck_grid,
            'capture_efficiency': capture_efficiency,
            'demand_normalized': demand_normalized,
            'bounds': bounds,
            'spacing': grid_spacing,
            'cell_area_km2': (grid_spacing / 1000) ** 2,
            'max_detour_miles': max_detour_miles,
            'total_route_demand': total_route_demand,
            'max_cell_demand': np.max(demand_grid_smooth),
            'n_segments': len(segments),
            'timestamp': time.time(),
            'method': 'accessibility_based_capture',
            'config': {
                'detour_decay_rate': detour_decay,
                'grid_spacing_m': grid_spacing,
                'smoothing_sigma': sigma
            }
        }
        
        # Cache for future use
        self._demand_grid_cache = self.demand_surface
        
        
        
        # Optional: Save diagnostic plots
        if self.config.get('save_diagnostic_plots', False):
            self._save_demand_surface_diagnostics()
           
            
        # Optional: Run validation if gas station data available
        if hasattr(self, 'lois') and 'source' in self.lois.columns:
            gas_stations = self.lois[self.lois['source'].str.contains('gas', case=False, na=False)]
            if len(gas_stations) > 0:
                validation_results = self._validate_demand_surface(gas_stations)
                self.demand_surface['validation'] = validation_results
        
        # Optional: Create enhanced visualizations
        if self.config.get('create_enhanced_plots', True):
            self._create_enhanced_demand_visualizations()
        
        elapsed = time.time() - start_time
        print(f"\n  Demand surface created in {elapsed:.1f} seconds")
        
        return self.demand_surface


    def _sample_demand_at_locations(self, locations):
        """
        Sample demand surface at candidate locations with proper error handling.
        """
        if not hasattr(self, '_demand_interpolator') or self._demand_interpolator is None:
            print("WARNING: No demand surface available, using fallback")
            # Fallback: estimate from nearby routes
            return self._estimate_demand_fallback(locations)
        
        locations = np.array(locations)
        
        # Ensure locations is 2D array
        if locations.ndim == 1:
            locations = locations.reshape(1, -1)
        
        # Sample demand at each location
        demands = np.zeros(len(locations))
        
        try:
            # RegularGridInterpolator expects (y, x) order
            points = np.c_[locations[:, 1], locations[:, 0]]
            demands = self._demand_interpolator(points)
            
            # Also get capture efficiency if available
            if hasattr(self, '_efficiency_interpolator'):
                efficiency = self._efficiency_interpolator(points)
                
                # Store for later use
                self._last_efficiency_sample = efficiency
        
        except Exception as e:
            print(f"WARNING: Interpolation failed: {e}")
            # Use nearest neighbor fallback
            demands = self._estimate_demand_fallback(locations)
        
        # Ensure minimum demand
        min_demand = self.config.get('min_expected_demand_kg_day', 10)
        demands = np.maximum(demands, min_demand)
        
        return demands


    def _estimate_demand_fallback(self, locations):
        """
        Fallback demand estimation when surface is not available.
        Estimates based on proximity to routes.
        """
        locations = np.array(locations).reshape(-1, 2)
        demands = np.zeros(len(locations))
        
        # Simple distance-based estimation
        for i, loc in enumerate(locations):
            point = Point(loc[0], loc[1])
            accessible_demand = 0
            
            for _, route in self.routes.iterrows():
                distance = point.distance(route.geometry)
                
                if distance < self.config.get('service_radius_miles', 2.0) * 1609.34:
                    # Simple decay function
                    decay = np.exp(-2 * distance / (self.config['service_radius_miles'] * 1609.34))
                    accessible_demand += route.get('h2_demand_kg_day', 0) * decay * 0.1
            
            demands[i] = accessible_demand
        
        return demands


    def _save_demand_surface_diagnostics(self):
        """
        Save comprehensive diagnostic plots for validation.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        
        fig = plt.figure(figsize=(20, 15))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Demand surface
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.contourf(
            self.demand_surface['x'] / 1609.34,  # Convert to miles
            self.demand_surface['y'] / 1609.34,
            self.demand_surface['demand'],
            levels=20,
            cmap='YlOrRd'
        )
        ax1.set_title('Demand Surface (kg/day)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (miles)')
        ax1.set_ylabel('Y (miles)')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Log-scale demand
        ax2 = fig.add_subplot(gs[0, 1])
        demand_log = self.demand_surface['demand'].copy()
        demand_log[demand_log <= 0] = 0.1
        im2 = ax2.contourf(
            self.demand_surface['x'] / 1609.34,
            self.demand_surface['y'] / 1609.34,
            demand_log,
            levels=20,
            cmap='YlOrRd',
            norm=LogNorm()
        )
        ax2.set_title('Demand Surface (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (miles)')
        plt.colorbar(im2, ax=ax2)
        
        # 3. Truck flow surface
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.contourf(
            self.demand_surface['x'] / 1609.34,
            self.demand_surface['y'] / 1609.34,
            self.demand_surface['truck_flow'],
            levels=20,
            cmap='Blues'
        )
        ax3.set_title('Accessible Truck Flow', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (miles)')
        plt.colorbar(im3, ax=ax3)
        
        # 4. Capture efficiency
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.contourf(
            self.demand_surface['x'] / 1609.34,
            self.demand_surface['y'] / 1609.34,
            self.demand_surface['capture_efficiency'] * 100,  # Convert to percentage
            levels=20,
            cmap='Greens'
        )
        ax4.set_title('Capture Efficiency (%)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (miles)')
        ax4.set_ylabel('Y (miles)')
        plt.colorbar(im4, ax=ax4)
        
        # 5. Demand with routes overlay
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.contourf(
            self.demand_surface['x'] / 1609.34,
            self.demand_surface['y'] / 1609.34,
            self.demand_surface['demand'],
            levels=20,
            cmap='YlOrRd',
            alpha=0.6
        )
        
        # Overlay routes
        for _, route in self.routes.iterrows():
            if route.geometry.geom_type == 'LineString':
                x, y = route.geometry.xy
                ax5.plot(np.array(x) / 1609.34, np.array(y) / 1609.34, 'b-', linewidth=0.5)
            elif route.geometry.geom_type == 'MultiLineString':
                for line in route.geometry.geoms:
                    x, y = line.xy
                    ax5.plot(np.array(x) / 1609.34, np.array(y) / 1609.34, 'b-', linewidth=0.5)
        
        ax5.set_title('Demand Surface with Route Network', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X (miles)')
        
        # 6. Demand histogram
        ax6 = fig.add_subplot(gs[1, 2])
        demand_flat = self.demand_surface['demand'].flatten()
        demand_nonzero = demand_flat[demand_flat > 0]
        ax6.hist(demand_nonzero, bins=50, edgecolor='black', alpha=0.7)
        ax6.axvline(np.mean(demand_nonzero), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(demand_nonzero):.1f}')
        ax6.axvline(np.median(demand_nonzero), color='green', linestyle='--',
                    label=f'Median: {np.median(demand_nonzero):.1f}')
        ax6.set_xlabel('Demand (kg/day)')
        ax6.set_ylabel('Number of Cells')
        ax6.set_title('Demand Distribution', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.set_yscale('log')
        
        # 7. Coverage analysis
        ax7 = fig.add_subplot(gs[2, 0])
        coverage_levels = [10, 50, 100, 500, 1000, 5000]
        coverage_pcts = []
        
        for level in coverage_levels:
            pct = np.sum(self.demand_surface['demand'] >= level) / self.demand_surface['demand'].size * 100
            coverage_pcts.append(pct)
        
        ax7.bar(range(len(coverage_levels)), coverage_pcts, tick_label=[str(l) for l in coverage_levels])
        ax7.set_xlabel('Demand Threshold (kg/day)')
        ax7.set_ylabel('Coverage (%)')
        ax7.set_title('Spatial Coverage by Demand Level', fontsize=12, fontweight='bold')
        
        # 8. Route demand distribution
        ax8 = fig.add_subplot(gs[2, 1])
        route_demands = self.routes['h2_demand_kg_day'].sort_values(ascending=False)
        top_20 = route_demands.head(20)
        
        ax8.bar(range(len(top_20)), top_20.values)
        ax8.set_xlabel('Route Rank')
        ax8.set_ylabel('Demand (kg/day)')
        ax8.set_title('Top 20 Route Demands', fontsize=12, fontweight='bold')
        
        # 9. Summary statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        stats_text = f"""Surface Statistics:
        
        Total Route Demand: {self.demand_surface['total_route_demand']:,.0f} kg/day
        Max Cell Demand: {self.demand_surface['max_cell_demand']:,.1f} kg/day
        Grid Resolution: {self.demand_surface['spacing']/1000:.1f} km
        Cell Area: {self.demand_surface['cell_area_km2']:.1f} km²
        Non-zero Cells: {np.sum(self.demand_surface['demand'] > 0):,}
        Coverage: {np.sum(self.demand_surface['demand'] > 0) / self.demand_surface['demand'].size:.1%}
        
        Method: {self.demand_surface['method']}
        Max Detour: {self.demand_surface['max_detour_miles']} miles
        Computation Time: {self.demand_surface.get('elapsed', 0):.1f} seconds
        """
        
        ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle('Hydrogen Demand Surface Diagnostics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        output_path = 'demand_surface_diagnostics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Diagnostic plots saved to {output_path}")
        
        

    def _validate_demand_surface(self, gas_stations_gdf=None):
        """
        Validate demand surface using available data.
        Returns validation metrics dictionary.
        """
        print("\n  Running demand surface validation...")
        validation_results = {}
        
        # 1. Internal consistency checks
        print("    - Internal consistency checks...")
        
        # Check demand magnitude reasonableness
        max_demand = np.max(self.demand_surface['demand'])
        total_demand = self.demand_surface['total_route_demand']
        max_demand_pct = (max_demand / total_demand) * 100
        
        validation_results['max_demand_percent'] = max_demand_pct
        validation_results['max_demand_reasonable'] = max_demand_pct < 2.0  # No cell should have >2% of total
        
        # Check spatial coverage
        nonzero_cells = np.sum(self.demand_surface['demand'] > 0)
        total_cells = self.demand_surface['demand'].size
        coverage_pct = (nonzero_cells / total_cells) * 100
        
        validation_results['spatial_coverage_percent'] = coverage_pct
        validation_results['coverage_reasonable'] = 5 < coverage_pct < 50  # Should cover 5-50% of area
        
        # 2. Gas station correlation (if available)
        if gas_stations_gdf is not None and len(gas_stations_gdf) > 0:
            print("    - Gas station density correlation...")
            
            try:
                # Create gas station density surface
                gas_density = self._create_point_density_surface(
                    gas_stations_gdf,
                    self.demand_surface['x'],
                    self.demand_surface['y'],
                    bandwidth_m=5000  # 5km bandwidth for gas stations
                )
                
                # Store gas density in demand_surface dict for persistence
                self.demand_surface['gas_density'] = gas_density
                self.demand_surface['gas_station_locations'] = gas_stations_gdf
                    
                # Calculate correlation
                demand_flat = self.demand_surface['demand'].flatten()
                gas_flat = gas_density.flatten()
                
                # Only correlate where both have data
                mask = (demand_flat > 0) & (gas_flat > 0)
                if np.sum(mask) > 10:
                    correlation = np.corrcoef(demand_flat[mask], gas_flat[mask])[0, 1]
                    validation_results['gas_station_correlation'] = correlation
                    validation_results['gas_correlation_reasonable'] = 0.3 < correlation < 0.7
                    
                    # Store additional statistics for visualization
                    validation_results['gas_density_max'] = np.max(gas_density)
                    validation_results['gas_density_mean'] = np.mean(gas_density[gas_density > 0])
                    validation_results['n_gas_stations'] = len(gas_stations_gdf)
                    
                    print(f"      Gas station correlation: {correlation:.3f}")
                    print(f"      Number of gas stations: {len(gas_stations_gdf)}")
                else:
                    validation_results['gas_station_correlation'] = None
                    print("      Insufficient overlapping data for correlation")
                    
            except Exception as e:
                print(f"      Gas station validation failed: {e}")
                validation_results['gas_station_correlation'] = None
        
        # 3. Route proximity validation
        print("    - Route proximity validation...")
        
        # Check that high demand areas are near routes
        high_demand_threshold = np.percentile(self.demand_surface['demand'][self.demand_surface['demand'] > 0], 90)
        high_demand_cells = self.demand_surface['demand'] > high_demand_threshold
        
        if hasattr(self, 'routes') and np.any(high_demand_cells):
            # Get high demand cell coordinates
            high_demand_coords = np.column_stack([
                self.demand_surface['x'][high_demand_cells].flatten(),
                self.demand_surface['y'][high_demand_cells].flatten()
            ])
            
            # Check minimum distance to routes
            min_distances = []
            for coord in high_demand_coords[:100]:  # Sample up to 100 points
                point = Point(coord)
                min_dist = min(point.distance(route.geometry) for _, route in self.routes.iterrows())
                min_distances.append(min_dist)
            
            avg_min_distance_miles = np.mean(min_distances) / 1609.34
            validation_results['avg_high_demand_route_distance_miles'] = avg_min_distance_miles
            validation_results['route_proximity_reasonable'] = avg_min_distance_miles < self.config.get('max_detour_miles', 5.0)
            
            print(f"      Avg distance from high-demand cells to routes: {avg_min_distance_miles:.1f} miles")
        
        # 4. Summary
        print("\n    Validation Summary:")
        for key, value in validation_results.items():
            if 'reasonable' in key and isinstance(value, bool):
                status = " PASS" if value else "✗ FAIL"
                print(f"      {key}: {status}")
        
        return validation_results


    def _create_point_density_surface(self, points_gdf, x_grid, y_grid, bandwidth_m=5000):
        """
        Create density surface from point locations (e.g., gas stations).
        """
        # Get point coordinates
        if hasattr(points_gdf, 'geometry'):
            coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])
        else:
            coords = np.array([[row.longitude, row.latitude] for _, row in points_gdf.iterrows()])
        
        # Create density grid
        density = np.zeros_like(x_grid)
        
        # Simple gaussian kernel density
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                grid_point = np.array([x_grid[i, j], y_grid[i, j]])
                distances = np.linalg.norm(coords - grid_point, axis=1)
                kernel_values = np.exp(-0.5 * (distances / bandwidth_m) ** 2)
                density[i, j] = np.sum(kernel_values)
        
        # Normalize
        if np.max(density) > 0:
            density = density / np.max(density)
        
        return density


    def _create_enhanced_demand_visualizations(self):
        """
        Create visualizations with geographic context.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Rectangle
        from matplotlib.colors import LogNorm
        
        print("\n  Creating enhanced visualizations...")
        
        # Calculate max demand for use in plots
        max_demand = np.max(self.demand_surface['demand'])
        
        # Check if we can use contextily for basemaps
        try:
            import contextily as ctx
            has_contextily = True
        except ImportError:
            has_contextily = False
            print("    Note: Install contextily for basemap support (pip install contextily)")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Convert coordinates to miles for display
        x_miles = self.demand_surface['x'] / 1609.34
        y_miles = self.demand_surface['y'] / 1609.34
        
        # Main demand surface with geographic context
        ax1 = fig.add_subplot(gs[:, 0:2])  # Span two columns
        
        # Create filled contour
        contour = ax1.contourf(x_miles, y_miles, self.demand_surface['demand'],
                            levels=20, cmap='YlOrRd', alpha=0.8)
        
        # Plot routes as connected network
        if hasattr(self, 'routes'):
            # Plot all routes together for connected appearance
            # Use thicker lines for major routes
            for _, route in self.routes.iterrows():
                if route.geometry is None:
                    continue
                    
                # Determine line width based on truck volume
                truck_vol = route.get('TOT_TRK_AADT', route.get('truck_aadt', 0))
                if truck_vol > 10000:
                    lw = 1.5
                    alpha = 0.8
                elif truck_vol > 5000:
                    lw = 1.0
                    alpha = 0.6
                else:
                    lw = 0.5
                    alpha = 0.4
                
                if route.geometry.geom_type == 'LineString':
                    x, y = route.geometry.xy
                    ax1.plot(np.array(x)/1609.34, np.array(y)/1609.34, 
                            'b-', linewidth=lw, alpha=alpha)
                elif route.geometry.geom_type == 'MultiLineString':
                    for line in route.geometry.geoms:
                        x, y = line.xy
                        ax1.plot(np.array(x)/1609.34, np.array(y)/1609.34, 
                                'b-', linewidth=lw, alpha=alpha)
        
        # City coordinates for California Albers (EPSG:3310)
        # These are more accurate positions based on typical CA maps
        cities = {
            'Los Angeles': (170000, -400000),      # LA Basin
            'San Francisco': (-180000, 90000),     # Bay Area
            'Sacramento': (-65000, 115000),        # Central Valley North
            'San Diego': (235000, -520000),        # Far South
            'Fresno': (40000, -85000),             # Central Valley
            'Bakersfield': (115000, -190000),      # Southern Central Valley
            'San Jose': (-155000, 10000),          # South Bay
            'Oakland': (-175000, 60000)            # East Bay
        }
        
        for city, (x, y) in cities.items():
            ax1.plot(x/1609.34, y/1609.34, 'k*', markersize=12)
            ax1.annotate(city, (x/1609.34, y/1609.34), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add major highway labels at better positions
        highway_labels = {
            'I-5': (0, -50),                        # Central Valley
            'I-10': (200, -300),                    # LA to Arizona  
            'US-101': (-160, 0),                    # Bay Area to LA
            'I-80': (-100, 100),                    # Bay to Sacramento
            'SR-99': (20, -100),                    # Central Valley parallel to I-5
        }
        
        for highway, (x, y) in highway_labels.items():
            ax1.text(x, y, highway, fontsize=9, color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
        
        # Colorbar
        cbar1 = plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Potential H2 Demand (kg/day)', fontsize=12)
        
        ax1.set_title('California Hydrogen Demand Surface with Geographic Context', 
                    fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('West-East (miles from origin)', fontsize=12)
        ax1.set_ylabel('South-North (miles from origin)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add scale reference
        ax1.text(0.02, 0.02, f'Grid Resolution: {self.demand_surface["spacing"]/1609.34:.1f} miles\n' + 
                f'Max Detour: {self.demand_surface["max_detour_miles"]} miles\n' +
                f'CRS: California Albers (EPSG:3310)',
                transform=ax1.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Add north arrow
        ax1.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.90),
                    xycoords='axes fraction', textcoords='axes fraction',
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', lw=2))
        
        # Log-scale visualization
        ax2 = fig.add_subplot(gs[0, 2])
        demand_log = self.demand_surface['demand'].copy()
        demand_log[demand_log <= 0] = 0.1
        
        im2 = ax2.contourf(x_miles, y_miles, demand_log,
                        levels=20, cmap='YlOrRd', norm=LogNorm(vmin=1, vmax=max_demand))
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Demand Surface (Log Scale)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('West-East (miles)')
        ax2.set_ylabel('South-North (miles)')
        
        # Validation scatter plot or regional comparison
        ax3 = fig.add_subplot(gs[1, 2])
        
        if 'validation' in self.demand_surface and self.demand_surface['validation'].get('gas_station_correlation') and 'gas_density' in self.demand_surface:
            print("    Creating gas station correlation plot...")
            # Get gas density from demand_surface
            gas_density = self.demand_surface['gas_density']
            # If we have gas station validation data, show correlation
            
            demand_sample = self.demand_surface['demand'].flatten()[::10]  # Subsample
            gas_sample = gas_density.flatten()[::10]
            
            # Only plot where both have data
            mask = (demand_sample > 0) & (gas_sample > 0)
            
            if np.sum(mask) > 1:
                ax3.scatter(gas_sample[mask], demand_sample[mask], alpha=0.5, s=1)
                ax3.set_xlabel('Gas Station Density (normalized)')
                ax3.set_ylabel('H2 Demand (kg/day)')
                ax3.set_title(f'Demand vs Gas Station Density\n' + 
                            f'Correlation: {self.demand_surface["validation"]["gas_station_correlation"]:.3f}',
                            fontsize=12, fontweight='bold')
                
                # Add trend line
                
                z = np.polyfit(gas_sample[mask], demand_sample[mask], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(0, np.max(gas_sample[mask]), 100)
                ax3.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Linear Fit')
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for correlation plot', 
                    transform=ax3.transAxes, ha='center', va='center')
        else:
            
            print("    No gas station validation data found, showing regional comparison...")
        
            # Show demand statistics by region with corrected boundaries
            regions = {
                'LA Basin': (50, 200, -450, -350),      # Adjusted for actual LA position
                'Central Valley': (-50, 100, -200, 100), # I-5 corridor
                'Bay Area': (-220, -120, 0, 120),       # SF Bay region
                'San Diego': (180, 280, -550, -480),    # Far south
                'North Coast': (-250, -150, 100, 200),  # North of Bay
                'Inland Empire': (100, 200, -350, -250)  # East of LA
            }
            
            region_demands = []
            region_names = []
            
            for region_name, bounds in regions.items():
                mask = ((x_miles >= bounds[0]) & (x_miles <= bounds[1]) &
                    (y_miles >= bounds[2]) & (y_miles <= bounds[3]))
                if np.any(mask):
                    region_demand = np.mean(self.demand_surface['demand'][mask])
                    region_demands.append(region_demand)
                    region_names.append(region_name)
            
            if region_demands:  # Only plot if we have data
                ax3.bar(range(len(region_names)), region_demands)
                ax3.set_xticks(range(len(region_names)))
                ax3.set_xticklabels(region_names, rotation=45, ha='right')
                ax3.set_ylabel('Average Demand (kg/day/cell)')
                ax3.set_title('Regional Demand Comparison', fontsize=12, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No regional data available', 
                        transform=ax3.transAxes, ha='center', va='center')
        
        # Add validation summary text
        if 'validation' in self.demand_surface:
            val = self.demand_surface['validation']
            summary_text = "Validation Results:\n"
            
            if 'max_demand_reasonable' in val:
                summary_text += f" Max demand check: {'PASS' if val['max_demand_reasonable'] else 'FAIL'}\n"
            if 'coverage_reasonable' in val:
                summary_text += f" Coverage check: {'PASS' if val['coverage_reasonable'] else 'FAIL'}\n"
            if 'route_proximity_reasonable' in val:
                summary_text += f" Route proximity: {'PASS' if val['route_proximity_reasonable'] else 'FAIL'}\n"
            if 'gas_correlation_reasonable' in val:
                summary_text += f" Gas correlation: {'PASS' if val['gas_correlation_reasonable'] else 'FAIL'}\n"
            
            fig.text(0.98, 0.02, summary_text, transform=fig.transFigure,
                    fontsize=10, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        plt.suptitle(f'Enhanced Hydrogen Demand Surface Analysis\n' + 
                    f'Total Demand: {self.demand_surface["total_route_demand"]:,.0f} kg/day | ' +
                    f'Max Cell: {max_demand:,.0f} kg/day',
                    fontsize=18, fontweight='bold')
        
        # Save figure
        
        # Detect current working directory
        import os
        cwd = os.getcwd()
        
        output_path = 'demand_surface_enhanced.png'
        # concatenate cwd with output path
        output_path = os.path.join(cwd, output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    Enhanced visualization saved to {output_path}")
        
        # Create interactive HTML version if plotly available
        try:
            self._create_interactive_demand_map()
        except ImportError:
            print("    Note: Install plotly for interactive maps (pip install plotly)")
    

    def _create_interactive_demand_map(self):
        """
        Create interactive HTML map using plotly.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        print("    Creating interactive map...")
        
        # Convert to DataFrame for plotly
        x_flat = self.demand_surface['x'].flatten() / 1609.34
        y_flat = self.demand_surface['y'].flatten() / 1609.34
        demand_flat = self.demand_surface['demand'].flatten()
        
        # Create figure
        fig = go.Figure()
        
        # Add demand surface as heatmap
        fig.add_trace(go.Heatmap(
            x=self.demand_surface['x'][0, :] / 1609.34,
            y=self.demand_surface['y'][:, 0] / 1609.34,
            z=self.demand_surface['demand'],
            colorscale='YlOrRd',
            name='Demand',
            hovertemplate='X: %{x:.0f} mi<br>Y: %{y:.0f} mi<br>Demand: %{z:.0f} kg/day<extra></extra>'
        ))
        
        # Add routes if available
        if hasattr(self, 'routes'):
            for _, route in self.routes.iterrows():
                if route.geometry.geom_type == 'LineString':
                    x, y = route.geometry.xy
                    fig.add_trace(go.Scatter(
                        x=np.array(x)/1609.34,
                        y=np.array(y)/1609.34,
                        mode='lines',
                        line=dict(color='blue', width=1),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Add city markers
        cities = {
            'Los Angeles': (368848/1609.34, -428984/1609.34),
            'San Francisco': (-177802/1609.34, 111251/1609.34),
            'Sacramento': (-103922/1609.34, 184131/1609.34),
            'San Diego': (479952/1609.34, -583959/1609.34)
        }
        
        city_x = [coord[0] for coord in cities.values()]
        city_y = [coord[1] for coord in cities.values()]
        city_names = list(cities.keys())
        
        fig.add_trace(go.Scatter(
            x=city_x,
            y=city_y,
            mode='markers+text',
            marker=dict(size=10, color='black', symbol='star'),
            text=city_names,
            textposition='top center',
            name='Cities'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='California H₂ Demand Surface (Interactive)',
                font=dict(size=20)
            ),
            xaxis_title='West-East (miles)',
            yaxis_title='South-North (miles)',
            hovermode='closest',
            width=1200,
            height=800
        )
        
        # Save as HTML
        # Detect current working directory
        import os
        cwd = os.getcwd()
        
        output_path = 'demand_surface_interactive.html'
        # concatenate cwd with output path
        output_path = os.path.join(cwd, output_path)
        
        fig.write_html(output_path)
        print(f"    Interactive map saved to {output_path}")

        

    def create_gas_station_validation_plots(self):
        """
        Create detailed visualizations of gas station validation results.
        """
        # Check if gas density exists in demand_surface
        if not hasattr(self, 'demand_surface') or 'gas_density' not in self.demand_surface:
            print("No gas station validation data available")
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import seaborn as sns
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Use gas density from demand_surface
        gas_density = self.demand_surface['gas_density']
        gas_locations = self.demand_surface.get('gas_station_locations', None)
            
        # Convert to miles
        x_miles = self.demand_surface['x'] / 1609.34
        y_miles = self.demand_surface['y'] / 1609.34
        
        # 1. Gas Station Density Surface
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.contourf(x_miles, y_miles, gas_density,
                        levels=20, cmap='Blues', alpha=0.8)
        
        # Overlay actual gas station locations
        if hasattr(self, '_gas_station_locations'):
            gas_x = [geom.x/1609.34 for geom in gas_locations]
            gas_y = [geom.y/1609.34 for geom in gas_locations]
            ax1.scatter(gas_x, gas_y, c='red', s=2, alpha=0.5, label='Gas Stations')
        
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('Gas Station Density Surface', fontsize=12, fontweight='bold')
        ax1.set_xlabel('West-East (miles)')
        ax1.set_ylabel('South-North (miles)')
        ax1.legend()
        
        # 2. H2 Demand vs Gas Density Scatter
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Sample points for scatter plot (every 10th point to avoid overcrowding)
        demand_sample = self.demand_surface['demand'].flatten()[::10]
        gas_sample = gas_density.flatten()[::10]
        
        # Only plot where both have data
        mask = (demand_sample > 0) & (gas_sample > 0)
        
        # Create 2D histogram for density
        h = ax2.hist2d(gas_sample[mask], demand_sample[mask], 
                    bins=50, cmap='YlOrRd', cmin=1)
        plt.colorbar(h[3], ax=ax2, label='Point Density')
        
        # Add correlation line
        if np.sum(mask) > 2:
            z = np.polyfit(gas_sample[mask], demand_sample[mask], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(gas_sample[mask].min(), gas_sample[mask].max(), 100)
            ax2.plot(x_trend, p(x_trend), "b-", linewidth=2, 
                    label=f'Linear Fit (R²={self.demand_surface["validation"]["gas_station_correlation"]**2:.3f})')
        
        ax2.set_xlabel('Gas Station Density (normalized)')
        ax2.set_ylabel('H₂ Demand (kg/day)')
        ax2.set_title('Demand vs Gas Station Density Correlation', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # 3. Residual Map
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Normalize both surfaces to 0-1 for comparison
        demand_norm = self.demand_surface['demand'] / np.max(self.demand_surface['demand'])
        gas_norm = gas_density / np.max(gas_density)
        
        # Calculate residuals (where demand exceeds gas station coverage)
        residuals = demand_norm - gas_norm
        
        im3 = ax3.contourf(x_miles, y_miles, residuals,
                        levels=20, cmap='RdBu_r', center=0)
        plt.colorbar(im3, ax=ax3)
        ax3.set_title('Demand-Gas Station Coverage Gap\n(Red = Underserved by Gas Stations)', 
                    fontsize=12, fontweight='bold')
        ax3.set_xlabel('West-East (miles)')
        
        # 4. Statistical Distribution Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Plot distributions
        demand_nonzero = self.demand_surface['demand'][self.demand_surface['demand'] > 0].flatten()
        gas_nonzero = gas_density[gas_density > 0].flatten()
        
        # Normalize for comparison
        ax4.hist(demand_nonzero / np.max(demand_nonzero), bins=50, alpha=0.5, 
                label='H₂ Demand', density=True, color='red')
        ax4.hist(gas_nonzero / np.max(gas_nonzero), bins=50, alpha=0.5, 
                label='Gas Density', density=True, color='blue')
        
        ax4.set_xlabel('Normalized Density')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.set_xlim(0, 1)
        
        # 5. Spatial Autocorrelation
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Calculate Moran's I or similar spatial statistic
        # For simplicity, show distance vs correlation decay
        from scipy.spatial.distance import pdist, squareform
        
        # Sample random points
        n_samples = min(100, self.demand_surface['demand'].size)
        sample_indices = np.random.choice(self.demand_surface['demand'].size, n_samples, replace=False)
        
        sample_coords = np.column_stack([
            self.demand_surface['x'].flatten()[sample_indices],
            self.demand_surface['y'].flatten()[sample_indices]
        ])
        
        sample_demand = self.demand_surface['demand'].flatten()[sample_indices]
        sample_gas = gas_density.flatten()[sample_indices]
        
        # Calculate pairwise distances and demand differences
        distances = squareform(pdist(sample_coords)) / 1609.34  # Convert to miles
        demand_diffs = np.abs(sample_demand[:, None] - sample_demand[None, :])
        gas_diffs = np.abs(sample_gas[:, None] - sample_gas[None, :])
        
        # Bin by distance
        dist_bins = np.linspace(0, 50, 20)
        demand_corr_by_dist = []
        gas_corr_by_dist = []
        
        for i in range(len(dist_bins)-1):
            mask = (distances >= dist_bins[i]) & (distances < dist_bins[i+1])
            if np.sum(mask) > 10:
                demand_corr_by_dist.append(np.mean(demand_diffs[mask]))
                gas_corr_by_dist.append(np.mean(gas_diffs[mask]))
        
        dist_centers = (dist_bins[:-1] + dist_bins[1:]) / 2
        dist_centers = dist_centers[:len(demand_corr_by_dist)]
        
        ax5.plot(dist_centers, demand_corr_by_dist, 'r-', label='H₂ Demand', linewidth=2)
        ax5.plot(dist_centers, gas_corr_by_dist, 'b-', label='Gas Stations', linewidth=2)
        ax5.set_xlabel('Distance (miles)')
        ax5.set_ylabel('Average Difference')
        ax5.set_title('Spatial Autocorrelation Decay', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        val = self.demand_surface['validation']
        stats_text = f"""Gas Station Validation Summary:
        
        Correlation: {val.get('gas_station_correlation', 'N/A'):.3f}
        Number of Gas Stations: {val.get('n_gas_stations', 'N/A')}
        
        Interpretation:
        • Correlation > 0.7: Very strong alignment
        • Correlation 0.5-0.7: Good alignment
        • Correlation 0.3-0.5: Moderate alignment
        • Correlation < 0.3: Weak alignment
        
        Current Status: {self._interpret_correlation(val.get('gas_station_correlation', 0))}
        
        Underserved Areas:
        {self._identify_underserved_areas()}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Gas Station Validation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Detect current directory and save plot
        import os
        current_dir = os.getcwd()
        print(f"Saving gas station validation plots to {current_dir}")
        
        output_path = 'gas_station_validation.png'
        
        # concatenate the current directory with the output path
        output_path = os.path.join(current_dir, output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gas station validation plots saved to {output_path}")


    def _interpret_correlation(self, corr):
        """Interpret correlation value."""
        if corr is None:
            return "No data"
        elif corr > 0.7:
            return "Very Strong - Excellent market signal alignment"
        elif corr > 0.5:
            return "Good - Strong market signal alignment"
        elif corr > 0.3:
            return "Moderate - Reasonable alignment"
        else:
            return "Weak - Poor alignment (may indicate new market)"


    def _identify_underserved_areas(self):
        """Identify areas with high H2 demand but low gas station coverage."""
        if not hasattr(self, '_last_gas_density'):
            return "No data available"
        
        # Normalize both surfaces
        demand_norm = self.demand_surface['demand'] / np.max(self.demand_surface['demand'])
        gas_norm = self._last_gas_density / np.max(self._last_gas_density)
        
        # Find areas where demand significantly exceeds gas coverage
        underserved = (demand_norm > 0.5) & (gas_norm < 0.2)
        
        if np.any(underserved):
            # Get coordinates of underserved areas
            underserved_coords = np.column_stack([
                self.demand_surface['x'][underserved].flatten() / 1609.34,
                self.demand_surface['y'][underserved].flatten() / 1609.34
            ])
            
            # Cluster to find main regions
            from sklearn.cluster import DBSCAN
            clusters = DBSCAN(eps=20, min_samples=5).fit(underserved_coords)
            
            n_clusters = len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
            return f"Found {n_clusters} underserved regions"
        else:
            return "No significant underserved areas identified"
            
        
    
    def estimate_demand_surface_old(self):
        """
        Optimized demand surface estimation using vectorized operations.
        ~10x faster than original implementation.
        """
        import time
        start_time = time.time()
        
        print("\nEstimating demand surface (optimized)...")
        
        if not hasattr(self, 'routes') or self.routes.empty:
            print("  No route data available for demand surface")
            return
        
        # Check cache first
        if self._demand_grid_cache is not None:
            print("  Using cached demand surface")
            self.demand_surface = self._demand_grid_cache
            return
        
        # Create coarser initial grid (5x5 km instead of 2x2 km)
        bounds = self.routes.total_bounds
        x_range = bounds[2] - bounds[0]
        y_range = bounds[3] - bounds[1]
        
        # Adaptive grid resolution based on area
        grid_spacing = max(5000, min(x_range, y_range) / 50)  # 50x50 grid max
        
        x_coords = np.arange(bounds[0], bounds[2], grid_spacing)
        y_coords = np.arange(bounds[1], bounds[3], grid_spacing)
        
        print(f"  Creating {len(x_coords)}x{len(y_coords)} grid (spacing: {grid_spacing/1000:.1f} km)")
        
        # Vectorized demand calculation
        xx, yy = np.meshgrid(x_coords, y_coords)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Pre-calculate route properties
        route_points = np.array([
            [geom.centroid.x, geom.centroid.y] 
            for geom in self.routes.geometry
        ])
        route_demands = self.routes['h2_demand_daily_kg'].values
        route_lengths = self.routes['length_m'].values
        
        # Use scipy's RBF interpolation for smooth surface
        from scipy.interpolate import RBFInterpolator
        
        # Sample points along routes for better coverage
        sample_points = []
        sample_demands = []
        
        for idx, route in self.routes.iterrows():
            if route.geometry.length > 0:
                # Sample 3 points per route
                for frac in [0.25, 0.5, 0.75]:
                    pt = route.geometry.interpolate(frac, normalized=True)
                    sample_points.append([pt.x, pt.y])
                    sample_demands.append(route['h2_demand_daily_kg'])
        
        sample_points = np.array(sample_points)
        sample_demands = np.array(sample_demands)
        
        # Remove duplicate points to avoid singular matrix
        unique_points, unique_indices = np.unique(
            np.round(sample_points / 100) * 100,  # Round to 100m grid
            axis=0, 
            return_index=True
        )
        
        sample_points = sample_points[unique_indices]
        sample_demands = sample_demands[unique_indices]
        
        # Create RBF interpolator (much faster than KDE)
        print(f"  Interpolating from {len(sample_points)} sample points...")
        rbf = RBFInterpolator(
            sample_points, 
            sample_demands,
            kernel='gaussian',
            epsilon=10000  # 10km smoothing radius - 
        )
        
        # Evaluate on grid
        demand_values = rbf(grid_points)
        demand_grid = demand_values.reshape(xx.shape)
        
        # Apply smoothing
        from scipy.ndimage import gaussian_filter
        demand_grid = gaussian_filter(demand_grid, sigma=1.5)
        
        # Store results
        self.demand_surface = {
            'x': xx,
            'y': yy,
            'demand': demand_grid,
            'points': grid_points,
            'values': demand_values,
            'bounds': bounds,
            'spacing': grid_spacing
        }
        
        # Cache for future use
        self._demand_grid_cache = self.demand_surface
        
        elapsed = time.time() - start_time
        print(f"  Demand surface created in {elapsed:.1f} seconds")
        print(f"  Total demand: {np.sum(demand_grid):.0f} kg/day")
        
    
    
    
    
    
    def estimate_demand_surface_older(self):
        """
        Create continuous hydrogen demand surface using kernel density estimation.
        Route-corridor focused approach with appropriate bandwidth.
        """
        print("Estimating hydrogen demand surface...")
        
        # Use pre-calculated demand if available
        if 'h2_demand_daily_kg' in self.routes.columns:
            print(" Using pre-calculated h2_demand_daily_kg")
            self.routes['h2_demand_kg_day'] = self.routes['h2_demand_daily_kg']
        else:
            print(" Using fallback calculation")
            self.routes['h2_demand_kg_day'] = (
                self.routes['truck_aadt'] * 
                self.routes['length_miles'] * 
                self.config['h2_consumption_kg_per_mile']
            )
        
        print(f"  Using demand - Max: {self.routes['h2_demand_kg_day'].max():.1f} kg/day")
        print(f"  Using demand - Total: {self.routes['h2_demand_kg_day'].sum():.0f} kg/day")
        
        # FIXED: Create route-corridor focused grid
        route_geometries = []
        route_demands = []
        
        # Collect all route geometries and their demands
        for idx, route in self.routes.iterrows():
            if route['h2_demand_kg_day'] > 0 and route.geometry.geom_type == 'LineString':
                route_geometries.append(route.geometry)
                route_demands.append(route['h2_demand_kg_day'])
        
        print(f"Processing {len(route_geometries)} routes with positive demand")
        
        # Create buffer around all routes to define study area
        from shapely.ops import unary_union
        all_routes = unary_union(route_geometries)
        study_area = all_routes.buffer(50000)  # 50km buffer around routes
        
        # Create focused grid only within study area
        bounds = study_area.bounds
        print(f"Study area bounds: {bounds}")
        
        # Create denser grid over smaller area
        x_points = np.linspace(bounds[0], bounds[2], 150)  # Increased resolution
        y_points = np.linspace(bounds[1], bounds[3], 150)
        xx, yy = np.meshgrid(x_points, y_points)
        
        # Filter grid points to only those within study area
        grid_points = []
        grid_indices = []
        
        for i, x in enumerate(x_points):
            for j, y in enumerate(y_points):
                point = Point(x, y)
                if study_area.contains(point) or study_area.distance(point) < 25000:  # 25km tolerance
                    grid_points.append([x, y])
                    grid_indices.append((i, j))
        
        grid_points = np.array(grid_points)
        print(f"Created focused grid with {len(grid_points)} points")
        
        # Sample points along routes with appropriate density
        route_points = []
        route_weights = []
        
        for route_geom, demand in zip(route_geometries, route_demands):
            # Sample points every 5km along route
            route_length = route_geom.length
            n_samples = max(5, int(route_length / 5000))  # Every 5km
            
            distances = np.linspace(0, route_length, n_samples)
            for d in distances:
                pt = route_geom.interpolate(d)
                route_points.append([pt.x, pt.y])
                route_weights.append(demand / n_samples)
        
        route_points = np.array(route_points)
        route_weights = np.array(route_weights)
        
        print(f"Generated {len(route_points)} route sample points")
        
        # FIXED: Use appropriate bandwidth for highway networks
        # Highways serve large areas - use 25km bandwidth
        bandwidth = 25000  # 25km - appropriate for highway service areas
        print(f"Using bandwidth: {bandwidth/1000:.0f} km")
        
        # Calculate demand surface using proper KDE
        demand_values = np.zeros(len(grid_points))
        
        for i, grid_pt in enumerate(grid_points):
            # Calculate distances to all route points
            distances = np.sqrt(((route_points - grid_pt)**2).sum(axis=1))
            
            # Apply Gaussian kernel with appropriate bandwidth
            kernel_values = np.exp(-0.5 * (distances / bandwidth)**2)
            
            # Weight by route demands (no bandwidth normalization for interpretability)
            demand_values[i] = np.sum(kernel_values * route_weights)
        
        # Create full grid for output (fill non-study areas with zero)
        full_demand = np.zeros((len(x_points), len(y_points)))
        for i, (grid_idx, demand_val) in enumerate(zip(grid_indices, demand_values)):
            full_demand[grid_idx[0], grid_idx[1]] = demand_val
        
        self.demand_surface = {
            'x': xx,
            'y': yy,
            'demand': full_demand,
            'points': grid_points
        }
        
        print(f"  Max demand: {demand_values.max():.1f} kg/day")
        print(f"  Mean demand: {demand_values.mean():.1f} kg/day")
        print(f"  Non-zero points: {np.sum(demand_values > 0)} / {len(demand_values)}")
        
        
        
        
        
    def generate_candidates(self, strategy='hybrid'):
        """
        Generate candidate locations using specified strategy.
        
        Parameters
        ----------
        strategy : str
            'route_based': Traditional along-route placement
            'loi_based': Use locations of interest (gas stations, rest areas)
            'hybrid': Combine both approaches (recommended)
        """
        print(f"\nGenerating candidate locations using {strategy} strategy...")
        
        if not hasattr(self, 'routes') or self.routes.empty:
            raise ValueError("No route data available. Call load_data() first.")
        
        if strategy == 'route_based':
            self.candidates = self._generate_route_based_candidates()
        elif strategy == 'loi_based':
            self.candidates = self._generate_loi_based_candidates()
        elif strategy == 'hybrid':
            self.candidates = self._generate_hybrid_candidates()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Add unique ID for each candidate
        self.candidates['candidate_id'] = range(len(self.candidates))
        
        print(f"  Generated {len(self.candidates)} candidate locations")
        
        # Validate candidates
        self._validate_candidates()
        
        return self.candidates


    def _generate_route_based_candidates(self):
        """Generate candidates at regular intervals along routes."""
        candidates = []
        
        # Configurable parameters
        min_truck_aadt = self.config.get('min_candidate_truck_aadt', 200)
        interval_miles = self.config.get('candidate_interval_miles', 2.0)
        interval_m = interval_miles * 1609.34
        
        for idx, route in self.routes.iterrows():
            # Skip low-traffic routes
            truck_aadt = route.get('truck_aadt', route.get('TOT_TRK_AADT', 0))
            if truck_aadt < min_truck_aadt:
                continue
            
            # Get total route length (same for LineString, sum for MultiLineString)
            total_route_length = route.geometry.length
            
            # Handle different geometry types
            if route.geometry.geom_type == 'LineString':
                geometries = [route.geometry]
            elif route.geometry.geom_type == 'MultiLineString':
                geometries = list(route.geometry.geoms)
            else:
                continue
            
            # Track cumulative distance for MultiLineString
            cumulative_distance = 0
            
            for geom in geometries:
                # Dynamic interval based on route importance
                route_interval = interval_m
                if truck_aadt > 5000:  # High-traffic routes
                    route_interval = interval_m * 0.5  # More dense placement
                
                n_candidates = max(1, int(geom.length / route_interval))
                
                for i in range(n_candidates):
                    # Stagger placement to avoid clustering at segment starts
                    offset = (i + 0.5) * route_interval
                    if offset < geom.length:
                        pt = geom.interpolate(offset)
                        
                        # Calculate position on overall route
                        # For LineString: simple ratio
                        # For MultiLineString: account for previous segments
                        if route.geometry.geom_type == 'LineString':
                            position_on_route = offset / geom.length
                            distance_along_route = offset
                        elif route.geometry.geom_type == 'MultiLineString':
                            # Position within this segment plus previous segments for MultiLineString
                            position_on_route = (cumulative_distance + offset) / total_route_length
                            distance_along_route = cumulative_distance + offset
                    
                        
                        candidates.append({
                            'geometry': pt,
                            'source': 'route_based',
                            'route_idx': idx,
                            'truck_aadt': truck_aadt,
                            'position_on_route': position_on_route,
                            'route_length_miles': total_route_length / 1609.34,
                            'distance_along_route_miles': distance_along_route / 1609.34,
                            'distance_to_route_miles': 0.0 # on the route
                        })
                
                # Update cumulative distance for next segment
                cumulative_distance += geom.length
        
        return gpd.GeoDataFrame(candidates, crs=self.routes.crs)


    def _generate_loi_based_candidates(self):
        """Generate candidates at locations of interest."""
        if not hasattr(self, 'lois') or self.lois.empty:
            print("  Warning: No LOIs available for candidate generation")
            return gpd.GeoDataFrame([], crs=self.routes.crs)
        
        
        # Ensure CRS match before any spatial operations
        if self.lois.crs != self.routes.crs:
            print(f"  Converting LOIs from {self.lois.crs} to {self.routes.crs}")
            lois_projected = self.lois.to_crs(self.routes.crs)
        else:
            lois_projected = self.lois
        
        # Filter suitable LOIs
        suitable_types = ['ports', 'airports', 'gas_stations', 'rest_areas', 'traffic_generators']
        type_col = 'loi_type' if 'loi_type' in self.lois.columns else 'type'
        candidates_df = lois_projected[
            lois_projected[type_col].str.lower().str.contains('|'.join(suitable_types), na=False)
        ].copy()
        
        # Add metadata
        candidates_df['source'] = 'loi_based'
        candidates_df['position_on_route'] = np.nan  # Calculated later
        
        # Find nearest route for each LOI
        print(f"  Finding nearest routes for {len(candidates_df)} LOIs...")
        
        for idx, candidate in candidates_df.iterrows():
            nearest_route_idx, nearest_dist = self._find_nearest_route(candidate.geometry)
            
            
            
            if nearest_route_idx is not None:
                
                candidates_df.loc[idx, 'route_idx'] = nearest_route_idx
                candidates_df.loc[idx, 'distance_to_route_miles'] = nearest_dist / 1609.34
                
                # Safely access the route geometry
                try:
                    
                    nearest_route_idx_int = int(nearest_route_idx)
                    route = self.routes.iloc[nearest_route_idx_int]
                    
                    # Get the route geometry
                    route_geom = route.geometry
                
                    # Find the closest point on the route and calculate position along the route
                    if route_geom.geom_type == 'LineString':
                        distance_along_route = route_geom.project(candidate.geometry)
                        position_on_route = distance_along_route / route_geom.length
                        candidates_df.loc[idx, 'position_on_route'] = position_on_route
                        candidates_df.loc[idx, 'distance_along_route_miles'] = distance_along_route / 1609.34
                        
                    
                    elif route_geom.geom_type == 'MultiLineString':
                        # Find the closest segment in MultiLineString
                        min_dist = float('inf')
                        best_segment = None
                        cumulative_dist = 0
                        best_cumulative = 0
                        
                        for segment in route_geom.geoms:
                            dist = candidate.geometry.distance(segment)
                            if dist < min_dist:
                                min_dist = dist
                                best_segment = segment
                                best_cumulative = cumulative_dist
                            cumulative_dist += segment.length
                        
                        # Project to the best segment
                        if best_segment:
                            distance_on_segment = best_segment.project(candidate.geometry)
                            distance_along_route = best_cumulative + distance_on_segment
                            position_on_route = distance_along_route / route_geom.length
                            candidates_df.loc[idx, 'position_on_route'] = position_on_route
                            candidates_df.loc[idx, 'distance_along_route_miles'] = distance_along_route
                            
                    # Add the route length in miles
                    candidates_df.loc[idx, 'route_length_miles'] = route_geom.length / 1609.34

                except (ValueError, TypeError) as e:
                    print(f"  Error processing route {nearest_route_idx}: {e}")
                    candidates_df.loc[idx, 'route_idx'] = np.nan
                    candidates_df.loc[idx, 'position_on_route'] = 0.5
                    candidates_df.loc[idx, 'distance_along_route_miles'] = 0.0
                    candidates_df.loc[idx, 'route_length_miles'] = 0.0
            else:
                # No nearest route found
                print(f"  No route found for LOI at index {idx}")
                candidates_df.loc[idx, 'route_idx'] = np.nan
                candidates_df.loc[idx, 'distance_to_route_miles'] = np.inf
                candidates_df.loc[idx, 'position_on_route'] = 0.5  # Default
                candidates_df.loc[idx, 'distance_along_route_miles'] = 0.0
                candidates_df.loc[idx, 'route_length_miles'] = 0.0
                
        
        # Filter out LOIs too far from routes
        max_distance_miles = self.config.get('max_loi_route_distance_miles', 2.0)
        
        candidates_df = candidates_df[
            candidates_df['distance_to_route_miles'] <= max_distance_miles
        ]
         
        # Filter out LOI-based candidates without valid route assignment
        valid_candidates = candidates_df[
            (candidates_df['route_idx'].notna()) 
        ]
        
        print(f"  After filtering: {len(valid_candidates)} valid LOI candidates")
        print(f"  Removed {len(candidates_df) - len(valid_candidates)} candidates (no route or too far)")
    
         
        return valid_candidates




    def _find_nearest_route(self, geometry):
        """
        Find the nearest route to a given geometry.
        
        Parameters:
        -----------
        geometry : shapely.geometry
            The geometry (usually Point) to find nearest route for
        
        Returns:
        --------
        nearest_route_idx : int
            Index of the nearest route
        min_distance : float
            Distance to nearest route in meters
                
        """
        if not hasattr(self, 'routes') or self.routes.empty:
            return None, float('inf')
        
        
        # Build spatial index if not exists
        if not hasattr(self, '_route_spatial_index') or self._route_spatial_index is None:
            self._build_route_spatial_index()
            
        # If spatial index is available, use it
        if self._route_spatial_index is not None:
            # Use spatial index for nearest neighbor search
            try:
                
                from rtree import index
                
                # Get potential matches within reasonable distance
                search_distance = 10000  # 10km search radius
                bounds = (geometry.x - search_distance, geometry.y - search_distance,
                        geometry.x + search_distance, geometry.y + search_distance)
                
                potential_matches = list(self._route_spatial_index.intersection(bounds))
                
                if not potential_matches:
                    # Expand search if no matches found
                    search_distance *= 5
                    bounds = (geometry.x - search_distance, geometry.y - search_distance,
                            geometry.x + search_distance, geometry.y + search_distance)
                    potential_matches = list(self._route_spatial_index.intersection(bounds))
                
                if not potential_matches:
                    # Fall back to brute force for this point
                    return self._find_nearest_route_brute_force(geometry)
                
                # Find actual nearest among candidates
                min_distance = float('inf')
                nearest_route_idx = None
            
                for idx in potential_matches:
                    
                    try:
                        route_geom = self.routes.iloc[idx].geometry
                        
                        if route_geom is not None:
                            
                            # Calculate distance to route
                            distance = geometry.distance(route_geom)
                            
                            if distance < min_distance:
                                min_distance = distance
                                nearest_route_idx = idx
                    except Exception as e:
                        print(f"Error processing route {idx}: {e}")
                        continue
                
                if nearest_route_idx is not None:
                    return nearest_route_idx, min_distance
                
                else:
                    # Fallback to brute force search if no nearest route found
                    return self._find_nearest_route_brute_force(geometry)
 
        
            except Exception as e:
                print(f"Error in spatial index search to find nearest route: {e}")
                # Fallback to brute force search if index fails
                return self._find_nearest_route_brute_force(geometry)
            
        else:
            # No spatial index available, use brute force search
            return self._find_nearest_route_brute_force(geometry)
            

    
    
    def _build_route_spatial_index(self):
        """Build spatial index for routes."""
        print("  Building spatial index for routes...")
        
        
        try:
            
            # Import rtree
            from rtree import index
            
            # Create new spatial index
            idx = index.Index()
            
            # Add each route to the index
            for i, route in self.routes.iterrows():
                if route.geometry is not None:
                    try:
                        # Get bounds and insert into index
                        bounds = route.geometry.bounds
                        idx.insert(i, bounds)
                    except Exception as e:
                        print(f"    Warning: Could not add route {i} to spatial index: {e}")
                        continue
            
            self._route_spatial_index = idx
            print(f"    Spatial index built with {len(list(idx.intersection(idx.bounds)))} routes")
            
        except ImportError:
            print("    Warning: rtree not available. Falling back to brute force search.")
            self._route_spatial_index = None
            
            
        except Exception as e:
            print(f"    Error building spatial index: {e}")
            self._route_spatial_index = None

        
    
    
    
    def _find_nearest_route_brute_force(self, geometry):
        """
        Brute force fallback for finding nearest route.
        More reliable but slower than spatial index.
        """
        if not hasattr(self, 'routes') or self.routes.empty:
            return None, float('inf')
        
        min_distance = float('inf')
        nearest_route_idx = None
        
        # Track search progress for large datasets
        total_routes = len(self.routes)
        progress_interval = max(1, total_routes // 10)
        
        print(f"  Using brute force search through {total_routes} routes...")
        
        for idx, route in self.routes.iterrows():
            # Progress indicator for long searches
            if idx % progress_interval == 0:
                progress = (idx / total_routes) * 100
                print(f"    Search progress: {progress:.0f}%", end='\r')
            
            if route.geometry is None:
                continue
            
            try:
                # Calculate distance with error handling
                distance = geometry.distance(route.geometry)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_route_idx = idx
                    
                    # Early exit if we find a very close match
                    if distance < 10:  # Within 10 meters
                        print(f"\n    Found exact match at {distance:.1f}m")
                        break
                        
            except Exception as e:
                # Handle geometry errors gracefully
                print(f"\n    Warning: Error calculating distance for route {idx}: {e}")
                continue
        
        print()  # Clear progress line
        
        # Validate result
        if nearest_route_idx is None:
            print(f"  ERROR: No valid route found for geometry at {geometry.coords[0]}")
            # Last resort - return the first valid route
            for idx, route in self.routes.iterrows():
                if route.geometry is not None:
                    print(f"  Using first valid route as fallback (idx={idx})")
                    return idx, float('inf')
            return None, float('inf')
        
        # Sanity check on distance
        if min_distance > 50000:  # More than 50km
            print(f"  Warning: Nearest route is {min_distance/1000:.1f}km away - may indicate CRS issue")
        
        return nearest_route_idx, min_distance


    def _find_routes_within_radius(self, geometry, radius_m=5000):
        """
        Find all routes within a given radius of a point.
        Used for averaging properties when exact match fails.
        """
        nearby_routes = []
        
        for idx, route in self.routes.iterrows():
            if route.geometry is None:
                continue
                
            try:
                distance = geometry.distance(route.geometry)
                if distance <= radius_m:
                    route_copy = route.copy()
                    route_copy['distance_m'] = distance
                    route_copy['route_idx'] = idx
                    nearby_routes.append(route_copy)
            except:
                continue
        
        if nearby_routes:
            # Convert to GeoDataFrame and sort by distance
            nearby_gdf = gpd.GeoDataFrame(nearby_routes, crs=self.routes.crs)
            nearby_gdf = nearby_gdf.sort_values('distance_m')
            return nearby_gdf
        
        return None


    def _validate_loi_route_matching(self, candidates_df):
        """
        Comprehensive validation of LOI-route matching results.
        """
        print("\n  LOI-Route Matching Validation:")
        
        # Check for unmatched LOIs
        total_lois = len(candidates_df)
        matched = candidates_df['route_idx'].notna().sum()
        unmatched = total_lois - matched
        
        print(f"    Total LOIs: {total_lois}")
        print(f"    Successfully matched: {matched} ({matched/total_lois*100:.1f}%)")
        print(f"    Unmatched: {unmatched} ({unmatched/total_lois*100:.1f}%)")
        
        if unmatched > 0:
            # Analyze unmatched LOIs
            unmatched_df = candidates_df[candidates_df['route_idx'].isna()]
            
            # Group by type
            if 'loi_type' in unmatched_df.columns:
                print("\n    Unmatched by type:")
                for loi_type, count in unmatched_df['loi_type'].value_counts().items():
                    print(f"      {loi_type}: {count}")
            
            # Check spatial distribution
            if unmatched > 5:
                print("\n    Spatial analysis of unmatched LOIs:")
                bounds = unmatched_df.total_bounds
                print(f"      Bounding box: ({bounds[0]:.0f}, {bounds[1]:.0f}) to ({bounds[2]:.0f}, {bounds[3]:.0f})")
                
                # Check if they're outside route coverage area
                route_bounds = self.routes.total_bounds
                if (bounds[0] < route_bounds[0] - 1000 or bounds[2] > route_bounds[2] + 1000 or
                    bounds[1] < route_bounds[1] - 1000 or bounds[3] > route_bounds[3] + 1000):
                    print("      WARNING: Some unmatched LOIs are outside route network coverage area")
        
        # Check distance distribution
        if 'distance_to_route_miles' in candidates_df.columns:
            distances = candidates_df['distance_to_route_miles'].dropna()
            if len(distances) > 0:
                print(f"\n    Distance to route statistics:")
                print(f"      Mean: {distances.mean():.2f} miles")
                print(f"      Median: {distances.median():.2f} miles")
                print(f"      Max: {distances.max():.2f} miles")
                
                # Warning for far matches
                far_matches = (distances > 5).sum()
                if far_matches > 0:
                    print(f"      WARNING: {far_matches} LOIs matched to routes >5 miles away")
        
        return candidates_df

    
    def _check_spatial_dependencies(self):
        """Check if spatial libraries are available."""
        dependencies = {
            'rtree': False,
            'shapely': False,
            'geopandas': True  # Assume this is available since it's required
        }
        
        try:
            import rtree
            dependencies['rtree'] = True
        except ImportError:
            print("  Note: rtree not available - spatial indexing will be slower")
        
        try:
            from shapely.geometry import Point
            dependencies['shapely'] = True
        except ImportError:
            print("  Note: shapely not fully available - some geometric operations may be limited")
        
        return dependencies
    

    def _interpolate_route_properties(self, geometry, nearby_routes_gdf):
        """
        Interpolate route properties from nearby routes when exact match fails.
        Uses inverse distance weighting.
        """
        if nearby_routes_gdf is None or nearby_routes_gdf.empty:
            return {
                'route_length_miles': 0,
                'truck_aadt': 0,
                'position_on_route': 0.5
            }
        
        # Use inverse distance weighting
        distances = nearby_routes_gdf['distance_m'].values
        weights = 1 / (distances + 1)  # Add 1 to avoid division by zero
        weights = weights / weights.sum()  # Normalize
        
        # Weighted average of properties
        interpolated = {
            'route_length_miles': (nearby_routes_gdf.geometry.length / 1609.34 * weights).sum(),
            'truck_aadt': 0,
            'position_on_route': 0.5  # Default to middle
        }
        
        # Handle truck AADT if available
        aadt_cols = ['truck_aadt', 'TOT_TRK_AADT', 'AADT']
        for col in aadt_cols:
            if col in nearby_routes_gdf.columns:
                interpolated['truck_aadt'] = (nearby_routes_gdf[col] * weights).sum()
                break
        
        return interpolated
    

    def _find_nearest_route_with_position(self, geometry):
        
        """
        Find the normalized position along the nearest route.
        Extended to return position along the route as well as nearest route index and distance.
        
        Parameters:
        -----------
        geometry : shapely.geometry
            The geometry to find nearest route for   
        
        Returns:
        --------
        nearest_route_idx : int
            Index of the nearest route
        min_distance : float
            Distance to nearest route in meters
        position : float
            Normalized position along route [0, 1]
            
        """
        nearest_route_idx, min_distance = self._find_nearest_route(geometry)
        
        if nearest_route_idx is None:
            return None, float('inf'), 0.5
        
        # Calculate position along the nearest route
        route_geom = self.routes.iloc[nearest_route_idx].geometry
        
        # Handle different geometry types
        if route_geom.geom_type == 'MultiLineString':
            # Find which part of the MultiLineString is closest
            min_dist = float('inf')
            best_geom = None
            for geom in route_geom.geoms:
                dist = geometry.distance(geom)
                if dist < min_dist:
                    min_dist = dist
                    best_geom = geom
            route_geom = best_geom
        
        # Calculate position
        if route_geom and route_geom.length > 0:
            closest_point_distance = route_geom.project(geometry)
            position = closest_point_distance / route_geom.length
        else:
            position = 0.5
        
        return nearest_route_idx, min_distance, position



    def _generate_hybrid_candidates(self):
        """Combine route-based and LOI-based strategies."""
        # Get both types
        route_candidates = self._generate_route_based_candidates()
        loi_candidates = self._generate_loi_based_candidates()
        
        print(f"  Route-based: {len(route_candidates)} candidates")
        print(f"  LOI-based: {len(loi_candidates)} candidates")
        
        # Combine and remove duplicates
        all_candidates = pd.concat([route_candidates, loi_candidates], ignore_index=True)
        
        # Remove candidates that are too close to each other
        min_spacing_m = self.config.get('min_candidate_spacing_miles', 0.5) * 1609.34
        
        # Use spatial index for efficient deduplication
        keep_indices = []
        candidate_tree = None
        
        for idx, candidate in all_candidates.iterrows():
            if candidate_tree is None:
                keep_indices.append(idx)
                candidate_tree = cKDTree([[candidate.geometry.x, candidate.geometry.y]])
            else:
                point = [candidate.geometry.x, candidate.geometry.y]
                dist, _ = candidate_tree.query(point)
                
                if dist > min_spacing_m:
                    keep_indices.append(idx)
                    # Rebuild tree with new point
                    points = candidate_tree.data.tolist()
                    points.append(point)
                    candidate_tree = cKDTree(points)
        
        filtered_candidates = all_candidates.loc[keep_indices]
        print(f"  After deduplication: {len(filtered_candidates)} candidates")
        
        return filtered_candidates
    


    def _validate_candidates(self):
        """Validate generated candidates."""
        issues = []
        
        # Check for required columns
        required_cols = ['geometry', 'source']
        for col in required_cols:
            if col not in self.candidates.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check for valid geometries
        invalid_geoms = self.candidates.geometry.isna().sum()
        if invalid_geoms > 0:
            issues.append(f"{invalid_geoms} candidates have invalid geometry")
        
        # Check CRS
        if self.candidates.crs != self.routes.crs:
            issues.append("CRS mismatch between candidates and routes")
        
        # Check for duplicates
        coords = np.c_[self.candidates.geometry.x, self.candidates.geometry.y]
        unique_coords = np.unique(coords, axis=0)
        if len(unique_coords) < len(self.candidates):
            issues.append(f"{len(self.candidates) - len(unique_coords)} duplicate locations")
        
        if issues:
            print("\n   Candidate Validation Issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(" Candidate validation passed")




    def calculate_utilization_probability(self):
        """
        Calculate refueling probability for existing candidates.
        
        This method should be called AFTER generate_candidates().
        It calculates location attributes and Bayesian probabilities.
        """
        print("\nCalculating utilization probabilities...")
        
        # Validate prerequisites
        if not hasattr(self, 'candidates') or self.candidates is None or self.candidates.empty:
            raise ValueError(
                "No candidates available. Call generate_candidates() first.\n"
                "Workflow: load_data() -> estimate_demand_surface() -> "
                "generate_candidates() -> calculate_utilization_probability()"
            )
        
        print(f"  Processing {len(self.candidates)} candidates")
        
        # Step 1: Calculate location attributes
        print("  Step 1/3: Calculating location attributes...")
        self._calculate_location_attributes()
        
        # Step 2: Calculate Bayesian probabilities
        print("  Step 2/3: Calculating Bayesian probabilities...")
        self._calculate_bayesian_probabilities()
        
        # Step 3: Validate results
        print("  Step 3/3: Validating probability calculations...")
        self._validate_probabilities()
        
        # Summary statistics
        print("\n  Utilization Probability Summary:")
        print(f"    Mean expected demand: {self.candidates['expected_demand_kg_day'].mean():.1f} kg/day")
        print(f"    Max expected demand: {self.candidates['expected_demand_kg_day'].max():.1f} kg/day")
        print(f"    Candidates with >100 kg/day: {(self.candidates['expected_demand_kg_day'] > 100).sum()}")
        
        return self.candidates
    
    
    def _validate_probabilities(self):
        """Validate probability calculations."""
        issues = []
        
        # Check for required columns
        prob_cols = ['p_need_fuel', 'p_stop_given_need', 'p_choose_given_alternatives']
        for col in prob_cols:
            if col not in self.candidates.columns:
                issues.append(f"Missing probability column: {col}")
            else:
                # Check bounds
                if (self.candidates[col] < 0).any() or (self.candidates[col] > 1).any():
                    issues.append(f"{col} has values outside [0,1]")
        
        # Check expected demand
        if 'expected_demand_kg_day' not in self.candidates.columns:
            issues.append("Missing expected_demand_kg_day")
        else:
            negative_demand = (self.candidates['expected_demand_kg_day'] < 0).sum()
            if negative_demand > 0:
                issues.append(f"{negative_demand} candidates have negative demand")
        
        if issues:
            print("\n   Probability Validation Issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("Probability validation passed") 
    
    
    
    
    
    def calculate_utilization_probability_old(self):
        """
        Calculate probability of refueling at each location using Bayesian framework.
        """
        print("Calculating utilization probabilities...")
        
        # Create candidate locations at regular intervals along routes
        candidates = []
        
        for idx, route in self.routes.iterrows():
            if route.geometry.geom_type == 'LineString':
                # Place candidates every 2 miles along high-traffic routes
                if route['truck_aadt'] > 500:
                    interval = 2 * 1609.34  # 2 miles in meters
                    n_candidates = int(route.geometry.length / interval)
                    for i in range(n_candidates):
                        dist = (i + 0.5) * interval
                        if dist < route.geometry.length:
                            pt = route.geometry.interpolate(dist)
                            candidates.append({
                                'geometry': pt,
                                'route_idx': idx,
                                'truck_aadt': route['truck_aadt'],
                                'position_on_route': dist / route.geometry.length
                            })
        
        self.candidates = gpd.GeoDataFrame(candidates, crs="EPSG:3310")
        
        # Calculate attributes for each candidate
        self._calculate_location_attributes()
        
        # Bayesian probability calculation
        self._calculate_bayesian_probabilities()
        
        
    """"
    Refined and depracated
    
    def _detect_highway_interchanges(self):
        
        #Detect highway interchanges based on route connectivity and geometric analysis
        #Handles multiple coordinate formats, including both geometry objects and coordinate columns properly.
        
        import numpy as np
        from shapely.geometry import Point
        from collections import defaultdict
        
        print("Detecting highway interchanges...")
        
        
        # Initialize interchange column
        self.candidates['is_interchange'] = 0
        
        if not hasattr(self, 'routes') or self.routes.empty:
            print("No routes available for interchange detection")
            return np.zeros(len(self.candidates), dtype=int)
        
        # Parameters for interchange detection
        intersection_buffer = 0.01  # degrees (~1km buffer for intersection detection)
        #intersection_buffer = 1000  # in meters (~1km buffer for intersection detection)
        min_routes_for_interchange = 2  # Minimum routes to qualify as interchange
        major_highway_threshold = 100  # Daily truck traffic threshold for major highways
        
        # Identify major highways
        major_highways = self.routes[
            self.routes['TOT_TRK_AADT'] >= major_highway_threshold
        ].copy()
        
        if major_highways.empty:
            print("No major highways found for interchange detection")
            return np.zeros(len(self.candidates), dtype=int)
        
        print(f"Found {len(major_highways)} major highway segments")
        
        # For each candidate, check proximity to route intersections
        interchange_scores = {}
        
        for idx, candidate in self.candidates.iterrows():
            # Get candidate point - handle both geometry and coordinate columns
            #candidate_point = Point(candidate['longitude'], candidate['latitude'])
            
            try:
                if hasattr(candidate.geometry, 'x') and hasattr(candidate.geometry, 'y'):
                    candidate_point = candidate.geometry  # Already a Point object
                else:
                    # Handle non-Point geometries
                    candidate_point = candidate.geometry.centroid
            except Exception as e:
                print(f"Warning: Could not extract coordinates for candidate {idx}: {e}")
                continue
                
            #comment out block start#   
            candidate_point = None
            
            if 'geometry' in candidate.index and candidate['geometry'] is not None:
                
                if isinstance(candidate['geometry'], Point):
                    candidate_point = candidate['geometry']  
                
                else: 
                    
                    # If it is some other geometry type, get its centroid
                    try:
                        candidate_point = candidate['geometry'].centroid
                        
                    except:
                        print(f" Warning: Could not get point for candidate {candidate.name} with index {idx}")
                        continue
                    
            else:
                # Try to find coordinates in columns using alternative column names
                lon = None
                lat = None
                
                # Try various common column names
                
                for lon_name in ['longitude', 'Longitude', 'lon', 'x']:
                    if lon_name in candidate.index:
                        lon = candidate[lon_name]
                        break
                    
                for lat_name in ['latitude', 'Latitude', 'lat', 'y']:
                    if lat_name in candidate.index:
                        lat = candidate[lat_name]
                        break
                
                if lon is not None and lat is not None:
                    
                    try:
                        candidate_point = Point(float(lon), float(lat))
                    except:
                        print(f" Warning: Invalid coordinates for candidate {candidate.name} with index {idx}")
                        continue
                else:
                    # Last resort: try to extract from geometry
                    if hasattr(candidate, 'geometry') and hasattr(candidate.geometry, 'x'):
                        candidate_point = Point(candidate.geometry.x, candidate.geometry.y)
                        
                    else:
                        print(f" Warning: No valid coordinates found for candidate {candidate.name} with index {idx}")
                        continue
                    
            if candidate_point is None:
                continue  # Skip if we couldn't get a valid point
            
            #comment out block end#
            
            # Find nearby routes within buffer
            nearby_routes = []
            route_types = set()
            total_traffic = 0
            
            for route_idx, route in self.routes.iterrows():
                if hasattr(route, 'geometry') and route.geometry is not None:
                    # Check if candidate is within buffer distance of route
                    try:
                        distance_to_route = candidate_point.distance(route.geometry)
                        
                        if distance_to_route <= intersection_buffer:
                            nearby_routes.append(route_idx)
                            #total_traffic += route['TOT_TRK_AADT']
                            total_traffic += route.get('truck_aadt', 0)
                            
                            # Classify route type based on designation
                            route_designation = str(route.get('DESIG', '')).upper()
                            if any(x in route_designation for x in ['I-', 'US-', 'SR-', 'CA-']):
                                if 'I-' in route_designation:
                                    route_types.add('Interstate')
                                elif 'US-' in route_designation:
                                    route_types.add('US Highway')
                                elif 'SR-' in route_designation or 'CA-' in route_designation:
                                    route_types.add('State Route')
                                else:
                                    route_types.add('Local')
                            else:
                                route_types.add('Unknown')
                                
                    except Exception as e:
                        # Skip routes with invalid geometries
                        
                        continue
            
            # Calculate interchange score
            interchange_score = 0
            
            if len(nearby_routes) >= min_routes_for_interchange:
                # Base score from number of intersecting routes
                route_count_score = min(len(nearby_routes) / 4.0, 1.0)  # Max score at 4+ routes
                
                # Traffic volume score
                traffic_score = min(total_traffic / 50000.0, 1.0)  # Max score at 50k+ daily trucks
                
                # Route type diversity score
                type_diversity_score = len(route_types) / 4.0  # Max score with all 4 types
                
                # Special bonus for interstate intersections
                interstate_bonus = 0.3 if 'Interstate' in route_types else 0
                
                # Combined score
                interchange_score = (
                    0.4 * route_count_score +
                    0.3 * traffic_score +
                    0.2 * type_diversity_score +
                    0.1 + interstate_bonus
                )
            
            interchange_scores[idx] = {
                'score': interchange_score,
                'nearby_routes': len(nearby_routes),
                'total_traffic': total_traffic,
                'route_types': list(route_types)
            }
        
        # Set interchange flag based on score threshold
        interchange_threshold = 0.6  # Threshold for qualifying as interchange
        
        for idx, score_data in interchange_scores.items():
            if score_data['score'] >= interchange_threshold:
                self.candidates.at[idx, 'is_interchange'] = 1
        
        # ToDo: Fix handling of geometry data input to ddetect geometric interchanges function ahead of pass
        # Alternative method using spatial clustering for geometric intersections
        #try:
            #self._detect_geometric_interchanges(intersection_buffer)
        
        #except Exception as e:
            #print(f"Error in geometric interchange detection: {e}")
        
        # Summary statistics
        num_interchanges = self.candidates['is_interchange'].sum()
        print(f"Detected {num_interchanges} highway interchanges from {len(self.candidates)} candidates")
        
        # Ensure we have the expected demand column (handle different naming)
        demand_column = None
        for col in ['expected_demand_kg_day', 'total_demand', 'demand_kg_day', 'h2_demand_kg_day']: # 'h2_demand_daily_kg']:
            if col in self.candidates.columns:
                demand_column = col
                break
        
        if num_interchanges > 0:
            avg_traffic_interchanges = self.candidates[
                self.candidates['is_interchange'] == 1
            ][demand_column].mean()
            
            if not np.isnan(avg_traffic_interchanges):
                print(f"Average daily demand at interchanges: {avg_traffic_interchanges:.0f} kg/day")
        
        """
    
    
    def _detect_highway_interchanges_old(self):
        """
        Optimized interchange detection using vectorized operations.
        ~20x faster than original implementation.
        """
        import time
        start_time = time.time()
        
        print("\nDetecting highway interchanges (optimized)...")
        
        # Initialize columns
        self.candidates['is_interchange'] = 0
        self.candidates['interchange_score'] = 0.0
        self.candidates['nearby_routes_count'] = 0
        
        # Check cache
        if self._interchange_cache is not None:
            print("  Using cached interchange data")
            for col in ['is_interchange', 'interchange_score', 'nearby_routes_count']:
                self.candidates[col] = self._interchange_cache[col]
            return
        
        # Build spatial index for routes
        route_bounds = []
        for idx, route in self.routes.iterrows():
            bounds = route.geometry.bounds
            # Expand bounds by 500m for intersection buffer
            route_bounds.append([
                bounds[0] - 500, bounds[1] - 500,
                bounds[2] + 500, bounds[3] + 500,
                idx
            ])
        
        route_bounds = np.array(route_bounds)
        
        # Vectorized intersection detection
        candidate_coords = np.c_[
            self.candidates.geometry.x,
            self.candidates.geometry.y
        ]
        
        # For each candidate, count nearby routes using bounding box test
        intersection_buffer = 500  # meters
        
        for i, (x, y) in enumerate(candidate_coords):
            # Quick bounding box test
            nearby_mask = (
                (route_bounds[:, 0] <= x + intersection_buffer) &
                (route_bounds[:, 2] >= x - intersection_buffer) &
                (route_bounds[:, 1] <= y + intersection_buffer) &
                (route_bounds[:, 3] >= y - intersection_buffer)
            )
            
            nearby_route_indices = route_bounds[nearby_mask, 4].astype(int)
            
            if len(nearby_route_indices) >= 2:
                # Calculate interchange score based on route properties
                nearby_routes = self.routes.iloc[nearby_route_indices]
                
                # Score factors
                unique_route_ids = nearby_routes['route_id'].nunique()
                total_traffic = nearby_routes['TOT_TRK_AADT'].sum()
                route_types = nearby_routes['DESIG'].nunique() if 'DESIG' in nearby_routes else 1
                
                # Calculate score (0-1 scale)
                connectivity_score = min(unique_route_ids / 4, 1.0)  # 4+ routes = max
                traffic_score = min(total_traffic / 20000, 1.0)  # 20k+ AADT = max
                diversity_score = min(route_types / 2, 1.0)  # 2+ route types = max
                
                score = 0.4 * connectivity_score + 0.4 * traffic_score + 0.2 * diversity_score
                
                self.candidates.at[self.candidates.index[i], 'interchange_score'] = score
                self.candidates.at[self.candidates.index[i], 'nearby_routes_count'] = len(nearby_route_indices)
                
                if score >= 0.6:  # Threshold for interchange
                    self.candidates.at[self.candidates.index[i], 'is_interchange'] = 1
        
        # Cache results
        self._interchange_cache = {
            'is_interchange': self.candidates['is_interchange'].copy(),
            'interchange_score': self.candidates['interchange_score'].copy(),
            'nearby_routes_count': self.candidates['nearby_routes_count'].copy()
        }
        
        num_interchanges = self.candidates['is_interchange'].sum()
        elapsed = time.time() - start_time
        print(f"  Detected {num_interchanges} interchanges in {elapsed:.1f} seconds")
        
    
    def _detect_highway_interchanges(self):
        """
        Detect and score highway interchanges based on connectivity and geometry.
        
        Returns both binary detection and continuous score for nuanced modeling.
        """
        n_candidates = len(self.candidates)
        if n_candidates == 0:
            print("No candidates available for interchange detection")
            return np.zeros(0, dtype=int)
        
        is_interchange = False * np.ones(n_candidates, dtype=int)
        interchange_score = np.zeros(n_candidates, dtype=float)
        
        if not hasattr(self, 'routes') or self.routes.empty:
            # Store as columns for later use
            self.candidates['is_interchange'] = is_interchange
            self.candidates['interchange_score'] = interchange_score
            return self.candidates['is_interchange']
        
        print("    Detecting highway interchanges...")
        
        # Define qualifying truck AADT for highways
        min_truck_aadt_for_highway = self.config.get('min_truck_aadt_highway', 1000)
        
        # Build spatial index 
        
        # Import rtree
        from rtree import index
        
        route_idx = index.Index()
        route_geometries = []   
        high_traffic_routes = []
        
        for i, route in self.routes.iterrows():
            truck_aadt = route.get('truck_aadt', route.get('TOT_TRK_AADT', 0))
            if truck_aadt >= min_truck_aadt_for_highway:
                if route.geometry is not None:
                    bounds = route.geometry.bounds
                    route_idx.insert(i, bounds)
                    route_geometries.append(route.geometry)
                    high_traffic_routes.append(i)
                    
        if not high_traffic_routes:
            print("No high-traffic routes found for interchange detection")
            return self.candidates['is_interchange']

        if not route_geometries:
            self.candidates['is_interchange'] = is_interchange
            self.candidates['interchange_score'] = interchange_score
            return self.candidates['is_interchange']
        
        print(f"  Indexed {len(high_traffic_routes)} high-traffic routes")
        
        # Analyze each candidate
        for pos, (idx, candidate) in enumerate(self.candidates.iterrows()): 
        #for i, candidate in self.candidates.iterrows():
            if candidate.geometry is None:
                continue
        
            if pos % 100 == 0:  # Progress indicator
                print(f"    Progress: {i}/{len(self.candidates)}", end='\r')
        
            # Find routes within influence radius
            search_radius = self.config.get('interchange_detection_radius_m', 1000.0)
            candidate_bounds = (
                candidate.geometry.x - search_radius,
                candidate.geometry.y - search_radius,
                candidate.geometry.x + search_radius,
                candidate.geometry.y + search_radius
            )
            
            nearby_route_indices = list(route_idx.intersection(candidate_bounds))
        
            # Detailed analysis of nearby routes
            close_routes = []
            route_bearings = []
            route_adt = []
            route_types = set()
            
            for nearby_idx in nearby_route_indices:
                route = self.routes.iloc[nearby_idx]
                route_geom = route.geometry
                
                # Determine route type for route designation
                if 'DESIG' in route.index and pd.notna(route['DESIG']):
                    desig = str(route['DESIG']).upper()
                    if 'I' in desig or 'INT' in desig:
                        route_type = 'Interstate'
                    elif 'US' in desig:
                        route_type = 'US Highway'
                    elif 'SR' in desig or 'CA' in desig:
                        route_type = 'State Route'
                    else:
                        route_type = 'Local'
                else:
                    route_type = 'Unknown'
                    
                route_types.add(route_type)
                
                
                # Check actual distance
                dist = candidate.geometry.distance(route_geom)
            
                if dist < search_radius:
                    close_routes.append(route_geom)
                    
                    # Calculate bearing at closest point
                    closest_pt = route_geom.interpolate(
                        route_geom.project(candidate.geometry)
                    )
                    bearing = self._calculate_bearing_at_point(route_geom, closest_pt)
                    route_bearings.append(bearing)
                    
                    # Get traffic volume
                    adt = route.get('truck_aadt', route.get('TOT_TRK_AADT', 0))
                    route_adt.append(adt)
            
            # Scoring logic
            n_routes = len(close_routes)
            
            if n_routes >= 2:
                # Calculate all pairwise angles
                angles = []
                for j in range(n_routes):
                    for k in range(j + 1, n_routes):
                        angle = self._calculate_intersection_angle(
                            route_bearings[j], 
                            route_bearings[k]
                        )
                        angles.append(angle)
                
                # Check for valid interchange geometry
                significant_angles = [a for a in angles if 30 < a < 150]
                
                if significant_angles:
                    is_interchange[pos] = 1

                    
                    # Calculate interchange score (0-1)
                    
                    # Components:
                    # 1. Number of routes (more is better)
                    route_score = min(n_routes / 4.0, 1.0)  # Max at 4 routes
                    
                    # 2. Angle quality (perpendicular is best)
                    best_angle = min(abs(a - 90) for a in significant_angles)
                    angle_score = 1.0 - (best_angle / 90.0)  # 1.0 at 90°, 0 at 0° or 180°
                    
                    # 3. Traffic volume (higher is better)
                    total_adt = sum(route_adt)
                    traffic_score = min(total_adt / 50000.0, 1.0)  # Max at 50k AADT
                    
                    # 4. Route diversity (different route types/directions)
                    diversity_score = min(len(route_types) / 4.0, 1.0)  # Max at 4 types
                    
                    # Special bonus for interstate intersections
                    interstate_bonus = 0.3 if 'Interstate' in route_types else 0
                    
                    # Weighted combination
                    interchange_score[pos] = (
                        0.2 * route_score +
                        0.1 * angle_score +
                        0.2 * traffic_score +
                        0.2 * diversity_score + 
                        0.1 * interstate_bonus
                    )
        
        print()
                
        # Store results in candidates DataFrame
        self.candidates['is_interchange'] = is_interchange
        self.candidates['interchange_score'] = interchange_score
        
        # Summary statistics
        n_interchanges = is_interchange.sum()
        if n_interchanges > 0:
            avg_score = interchange_score[is_interchange == 1].mean()
            print(f"      Found {n_interchanges} interchanges")
            print(f"      Average interchange score: {avg_score:.3f}")
            
            # Identify major interchanges
            major_interchanges = (interchange_score > 0.7).sum()
            print(f"      Major interchanges (score > 0.7): {major_interchanges}")
        
        return is_interchange
                

        

    def _detect_highway_interchanges_older(self):
        """
        Detect highway interchanges based on route connectivity and geometric analysis.
        Includes Proper coordinate handling for geometry-based candidates.
        """
        print("Detecting highway interchanges...")
        
        # Initialize interchange column
        self.candidates['is_interchange'] = 0
        
        if not hasattr(self, 'routes') or self.routes.empty:
            print("No routes available for interchange detection")
            return np.zeros(len(self.candidates), dtype=int)
        
        # Parameters for interchange detection
        intersection_buffer = 1000  # meters (~0.6 miles buffer for intersection detection)
        min_routes_for_interchange = 2  # Minimum routes to qualify as interchange
        major_highway_threshold = 1000  # Daily truck traffic threshold for major highways
        
        # Step 1: Identify major highways
        major_highways = self.routes[
            self.routes['truck_aadt'] >= major_highway_threshold
        ].copy()
        
        if major_highways.empty:
            print("No major highways found for interchange detection")
            return np.zeros(len(self.candidates), dtype=int)
        
        print(f"Found {len(major_highways)} major highway segments")
        
        # Step 2: For each candidate, check proximity to route intersections
        interchange_scores = {}
        
        for idx, candidate in self.candidates.iterrows():
            # Extract coordinates from geometry column
            try:
                if hasattr(candidate.geometry, 'x') and hasattr(candidate.geometry, 'y'):
                    candidate_point = candidate.geometry  # Already a Point object
                else:
                    # Handle non-Point geometries
                    candidate_point = candidate.geometry.centroid
            except Exception as e:
                print(f"Warning: Could not extract coordinates for candidate {idx}: {e}")
                continue
            
            # Find nearby routes within buffer
            nearby_routes = []
            route_types = set()
            total_traffic = 0
            
            for route_idx, route in self.routes.iterrows():
                if hasattr(route, 'geometry') and route.geometry is not None:
                    try:
                        # Check if candidate is within buffer distance of route
                        distance_to_route = candidate_point.distance(route.geometry)
                        
                        if distance_to_route <= intersection_buffer:
                            nearby_routes.append(route_idx)
                            total_traffic += route.get('truck_aadt', 0)
                            
                            # Determine route type from route designation
                            if 'DESIG' in route.index and pd.notna(route['DESIG']):
                                desig = str(route['DESIG']).upper()
                                if 'I' in desig or 'INT' in desig:
                                    route_types.add('Interstate')
                                elif 'US' in desig:
                                    route_types.add('US Highway')
                                elif 'SR' in desig or 'CA' in desig:
                                    route_types.add('State Route')
                                else:
                                    route_types.add('Local')
                            else:
                                route_types.add('Unknown')
                                
                    except Exception as e:
                        # Skip routes with geometry issues
                        continue
            
            # Step 3: Score interchange potential
            if len(nearby_routes) >= min_routes_for_interchange:
                # Calculate various scoring factors
                route_count_score = min(len(nearby_routes) / 4.0, 1.0)  # Max score at 4+ routes
                traffic_score = min(total_traffic / 50000.0, 1.0)  # Max score at 50K+ daily trucks
                type_diversity_score = len(route_types) / 4.0  # More route types = better
                
                # Special bonus for interstate intersections
                interstate_bonus = 0.3 if 'Interstate' in route_types else 0
                
                # Combined score
                interchange_score = (
                    0.4 * route_count_score +
                    0.3 * traffic_score +
                    0.2 * type_diversity_score +
                    0.1 + interstate_bonus
                )
                
                interchange_scores[idx] = {
                    'score': interchange_score,
                    'nearby_routes': len(nearby_routes),
                    'total_traffic': total_traffic,
                    'route_types': list(route_types)
                }
        
        # Step 4: Set interchange flag based on score threshold
        interchange_threshold = 0.6  # Threshold for qualifying as interchange
        
        for idx, score_data in interchange_scores.items():
            if score_data['score'] >= interchange_threshold:
                self.candidates.at[idx, 'is_interchange'] = 1
        
        # Summary statistics
        num_interchanges = self.candidates['is_interchange'].sum()
        print(f"Detected {num_interchanges} highway interchanges from {len(self.candidates)} candidates")
        
        if num_interchanges > 0:
            # Find a demand column that exists
            demand_columns = ['expected_demand_kg_day', 'total_demand', 'demand_kg_day', 'truck_aadt']
            demand_column = None
            
            for col in demand_columns:
                if col in self.candidates.columns:
                    demand_column = col
                    break
            
            if demand_column:
                avg_demand_interchanges = self.candidates[
                    self.candidates['is_interchange'] == 1
                ][demand_column].mean()
                
                if not np.isnan(avg_demand_interchanges):
                    print(f"Average {demand_column} at interchanges: {avg_demand_interchanges:.0f}")
        
        return self.candidates['is_interchange'].values




    def _detect_geometric_interchanges_old(self, buffer_distance):
        """
        Secondary method: detect interchanges using geometric intersection analysis
        """
        from shapely.geometry import Point
        import numpy as np
        
        if not hasattr(self.routes, 'geometry'):
            return
        
        # Create spatial index for efficient intersection detection
        try:
            from rtree import index
            spatial_idx = index.Index()
            
            # Build spatial index of route geometries
            for idx, route in self.routes.iterrows():
                if hasattr(route, 'geometry') and route.geometry is not None:
                    spatial_idx.insert(idx, route.geometry.bounds)
            
            # For each candidate, find potential geometric intersections
            for idx, candidate in self.candidates.iterrows():
                candidate_point = Point(candidate['longitude'], candidate['latitude'])
                candidate_buffer = candidate_point.buffer(buffer_distance)
                
                # Query spatial index for nearby routes
                possible_matches = list(spatial_idx.intersection(candidate_buffer.bounds))
                
                # Check actual intersections
                intersecting_routes = 0
                intersecting_geometries = []
                
                for route_idx in possible_matches:
                    route = self.routes.iloc[route_idx]
                    if hasattr(route, 'geometry') and route.geometry is not None:
                        try:
                            if candidate_buffer.intersects(route.geometry):
                                intersecting_routes += 1
                                intersecting_geometries.append(route.geometry)
                        except Exception:
                            continue
                
                # Enhanced interchange detection based on intersection angles
                if intersecting_routes >= 2 and len(intersecting_geometries) >= 2:
                    # Calculate intersection angles to confirm it's a real interchange
                    angles = self._calculate_intersection_angles(
                        intersecting_geometries, candidate_point
                    )
                    
                    # Look for perpendicular or near-perpendicular intersections
                    if any(abs(angle - 90) < 30 for angle in angles):  # Within 30 degrees of perpendicular
                        # Boost interchange score for geometric intersections
                        if self.candidates.at[idx, 'is_interchange'] == 0:
                            # Check if it meets minimum traffic threshold
                            nearby_traffic = sum(
                                self.routes.iloc[route_idx]['TOT_TRK_AADT'] 
                                for route_idx in possible_matches
                                if route_idx < len(self.routes)
                            )
                            
                            if nearby_traffic >= 5000:  # Minimum traffic for geometric interchange
                                self.candidates.at[idx, 'is_interchange'] = 1
        
        except ImportError:
            print("rtree not available - skipping advanced geometric interchange detection")
        except Exception as e:
            print(f"Error in geometric interchange detection: {e}")



    def _calculate_intersection_angles(self, geometries, point):
        """
        Calculate intersection angles between route geometries near a point.
        
        Parameters:
        -----------
        geometries : list of LineString
            Route geometries that intersect near the point
        point : Point
            The intersection point
        
        Returns:
        --------
        list of float : Intersection angles in degrees (0-180)
        """
        angles = []
        
        if len(geometries) < 2:
            return angles
        
        try:
            # Calculate bearing for each geometry at the intersection point
            bearings = []
            for geom in geometries:
                # Get bearing of this route at the intersection
                bearing = self._calculate_bearing_at_point(geom, point)
                bearings.append(bearing)
            
            # Calculate all pairwise intersection angles
            for i in range(len(bearings)):
                for j in range(i + 1, len(bearings)):
                    # Calculate acute angle between bearings
                    angle = self._calculate_intersection_angle(bearings[i], bearings[j])
                    angles.append(angle)
                    
        except Exception as e:
            print(f"Warning: Error calculating intersection angles: {e}")
            # Return empty list on error rather than crashing
            return []
        
        return angles


    def _calculate_intersection_angle(self, bearing1, bearing2):
        """
        Calculate the acute angle between two bearings.
        
        Parameters:
        -----------
        bearing1, bearing2 : float
            Bearings in degrees (0-360)
            
        Returns:
        --------
        float : Acute angle in degrees (0-180)
        """
        # Calculate difference
        diff = abs(bearing1 - bearing2)
        
        # Ensure we get the acute angle
        if diff > 180:
            diff = 360 - diff
            
        return diff



    def _calculate_intersection_angles_old(self, geometries, point):
        """
        Calculate intersection angles between route geometries near a point
        """
        import numpy as np
        from shapely.geometry import Point
        
        angles = []
        
        if len(geometries) < 2:
            return angles
        
        try:
            # For each pair of geometries, calculate intersection angle
            for i in range(len(geometries)):
                for j in range(i + 1, len(geometries)):
                    geom1 = geometries[i]
                    geom2 = geometries[j]
                    
                    # Get nearest points on each geometry to the intersection point
                    nearest1 = geom1.project(point)
                    nearest2 = geom2.project(point)
                    
                    # Get bearing of each geometry at the intersection
                    if hasattr(geom1, 'coords') and hasattr(geom2, 'coords'):
                        coords1 = list(geom1.coords)
                        coords2 = list(geom2.coords)
                        
                        if len(coords1) >= 2 and len(coords2) >= 2:
                            # Calculate bearings
                            bearing1 = self._calculate_bearing(coords1[0], coords1[1])
                            bearing2 = self._calculate_bearing(coords2[0], coords2[1])
                            
                            # Calculate intersection angle
                            angle_diff = abs(bearing1 - bearing2)
                            intersection_angle = min(angle_diff, 180 - angle_diff)
                            angles.append(intersection_angle)
        
        except Exception as e:
            print(f"Error calculating intersection angles: {e}")
        
        return angles



    def _calculate_bearing(self, point1, point2):
        """
        Calculate bearing between two points in degrees (0-360).
        
        For projected coordinates (e.g., EPSG:3310 California Albers).
        0° = Grid North, 90° = Grid East
        
        Parameters:
        -----------
        point1, point2 : shapely.Point or tuple
            Points to calculate bearing between
        
        Returns:
        --------
        float : Bearing in degrees (0=North, 90=East, etc.)
        """
        # Convert to coordinates if needed
        if hasattr(point1, 'x'):
            x1, y1 = point1.x, point1.y
        else:
            x1, y1 = point1
            
        if hasattr(point2, 'x'):
            x2, y2 = point2.x, point2.y
        else:
            x2, y2 = point2
        
        # Calculate bearing
        dx = x2 - x1
        dy = y2 - y1
        
        # atan2(dx, dy) gives bearing from north
        # atan2(y, x) would give bearing from east
        
        # Convert to degrees
        bearing = np.degrees(np.arctan2(dx, dy))
        
        # Normalize to 0-360
        return (bearing + 360) % 360


    def _calculate_bearing_at_point(self, line, point, segment_length=50):
        """
        Calculate the bearing of a line at a specific point.
        
        This method samples a small segment of the line around the point
        to determine the local bearing, which is crucial for accurate
        interchange detection.
        
        Parameters:
        -----------
        line : LineString
            The line geometry
        point : Point
            Point on or near the line
        segment_length : float
            Length in meters to sample for bearing calculation
        
        Returns:
        --------
        float : Bearing in degrees
        """
        # Project point onto line
        distance_along = line.project(point)
        
        # Determine sampling points based on position
        # Get two points for bearing calculation
        if distance_along <= segment_length:
            # Near start of line - sample forward
            pt1 = line.interpolate(0)
            pt2 = line.interpolate(min(segment_length * 2, line.length))
            
        elif distance_along >= line.length - segment_length:
            # Near end of line - sample backward
            pt1 = line.interpolate(max(0, line.length - segment_length * 2))
            pt2 = line.interpolate(line.length)
            
        else:
            # Middle of line - sample both directions
            pt1 = line.interpolate(distance_along - segment_length)
            pt2 = line.interpolate(distance_along + segment_length)
        
        return self._calculate_bearing(pt1, pt2)



    def _calculate_bearing_old(self, point1, point2):
        """
        Calculate bearing between two points in degrees (0-360).
        Calculates great circle bearing for geographic coordinates.
    
        For lat/lon coordinates only (e.g., EPSG:4326).
    
        
        Parameters:
        -----------
        point1, point2 : shapely.Point or tuple
            Points to calculate bearing between
        
        Returns:
        --------
        float : Bearing in degrees (0=North, 90=East, etc.)
        
        """
        import math
        
        
        lat1, lon1 = point1[1], point1[0]  # Note: coords are (x,y) = (lon,lat)
        lat2, lon2 = point2[1], point2[0]
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing




    # Additional helper method for route classification
    def _classify_route_importance(self, route):
        """
        Classify route importance based on designation and traffic
        """
        route_designation = str(route.get('DESIG', '')).upper()
        traffic = route.get('TOT_TRK_AADT', 0)
        
        # Classification based on route designation
        if 'I-' in route_designation:
            base_importance = 4  # Interstate
        elif 'US-' in route_designation:
            base_importance = 3  # US Highway
        elif 'SR-' in route_designation or 'CA-' in route_designation:
            base_importance = 2  # State Route
        else:
            base_importance = 1  # Local/Other
        
        # Adjust based on traffic volume
        if traffic >= 20000:
            traffic_multiplier = 1.5
        elif traffic >= 10000:
            traffic_multiplier = 1.2
        elif traffic >= 5000:
            traffic_multiplier = 1.0
        else:
            traffic_multiplier = 0.8
        
        return base_importance * traffic_multiplier


    def _calculate_location_attributes(self):
        """Calculate spatial attributes for each candidate location."""
        
        # Defensive check
        if not hasattr(self, 'candidates') or self.candidates.empty:
            raise ValueError("No candidates available for attribute calculation")
        
        
        # Standardize coordinate access for ALL candidates
        print("Standardizing candidate coordinates...")
        
        if 'longitude' not in self.candidates.columns or 'latitude' not in self.candidates.columns:
            # Extract coordinates from geometry column and store as standard columns
            try:
                self.candidates['longitude'] = self.candidates.geometry.x
                self.candidates['latitude'] = self.candidates.geometry.y
                print(f"  Extracted coordinates for {len(self.candidates)} candidates")
            except Exception as e:
                print(f"  Warning: Could not extract coordinates from geometry: {e}")
                # Fallback: create default coordinates
                self.candidates['longitude'] = 0.0
                self.candidates['latitude'] = 0.0
        
        # Verify coordinate extraction worked
        if self.candidates['longitude'].isna().any() or self.candidates['latitude'].isna().any():
            print("  Warning: Some candidates have NaN coordinates, filling with defaults")
            self.candidates['longitude'] = self.candidates['longitude'].fillna(0.0)
            self.candidates['latitude'] = self.candidates['latitude'].fillna(0.0)
        
        print(f"  Coordinate ranges: Lon [{self.candidates['longitude'].min():.0f}, {self.candidates['longitude'].max():.0f}], Lat [{self.candidates['latitude'].min():.0f}, {self.candidates['latitude'].max():.0f}]")
        
        
        print("    Calculating location attributes...")
        
        # Initialize all attributes
        self.candidates['has_rest_area_or_parking'] = 0
        self.candidates['is_traffic_generator'] = 0
        self.candidates['is_port'] = 0
        self.candidates['is_near_city'] = 0
        self.candidates['dist_to_port_miles'] = 999
        self.candidates['dist_to_rest_miles'] = 999
        self.candidates['dist_to_airport_miles'] = 999
        self.candidates['dist_to_gas_miles'] = 999
        self.candidates['dist_to_traffic_generator_miles'] = 999
        
        
        
        # Use vectorized operations 
        candidate_coords = np.c_[
            self.candidates.geometry.x,
            self.candidates.geometry.y
        ]
        
        
        
        # Distance to various POIs using KDTree
        if hasattr(self, 'lois') and not self.lois.empty:
            # Get LOI type name
            loi_type_name = 'loi_type' if 'loi_type' in self.lois.columns else 'type'
            
            # Ports
            ports = self.lois[self.lois[loi_type_name].str.contains('port|seaport', case=False, na=False)]
            if len(ports) > 0:
                port_tree = cKDTree(np.c_[ports.geometry.x, ports.geometry.y])
                distances, _ = port_tree.query(candidate_coords)
                self.candidates['dist_to_port_miles'] = distances / 1609.34
                
                # Mark candidates near ports
                self.candidates.loc[distances < self.config.get('service_radius_miles', 2.0), 'is_port'] = 1
            
            # Rest areas
            rest_areas = self.lois[
                self.lois[loi_type_name].str.contains('rest|parking|stop', case=False, na=False)
            ]
            if len(rest_areas) > 0:
                rest_tree = cKDTree(np.c_[rest_areas.geometry.x, rest_areas.geometry.y])
                distances, indices = rest_tree.query(candidate_coords)
                self.candidates['dist_to_rest_miles'] = distances / 1609.34
                
                # Mark candidates at rest areas
                self.candidates.loc[distances < self.config.get('service_radius_miles', 2.0), 'has_rest_area_or_parking'] = 1
                
            # Airports
            airports = self.lois[self.lois[loi_type_name].str.contains('airport|airstrip', case=False, na=False)]
            if len(airports) > 0:
                airport_tree = cKDTree(np.c_[airports.geometry.x, airports.geometry.y])
                distances, _ = airport_tree.query(candidate_coords)
                self.candidates['dist_to_airport_miles'] = distances / 1609.34
                
            # Gas stations
            gas_mask = self.lois[loi_type_name].str.contains('gas|fuel|fueling', case=False, na=False)
            gas_stations = self.lois[gas_mask]
            if len(gas_stations) > 0:
                gas_tree = cKDTree(np.c_[gas_stations.geometry.x, gas_stations.geometry.y])
                distances, _ = gas_tree.query(candidate_coords)
                self.candidates['dist_to_gas_miles'] = distances / 1609.34
            
            
            # Traffic generators (e.g. warehouses, distribution centers)
            traffic_gen_mask = self.lois[loi_type_name].str.contains('traffic_generator|warehouse|distribution', case=False, na=False)
            traffic_generators = self.lois[traffic_gen_mask]
            if len(traffic_generators) > 0:
                traffic_tree = cKDTree(np.c_[traffic_generators.geometry.x, traffic_generators.geometry.y])
                distances, _ = traffic_tree.query(candidate_coords)
                self.candidates['dist_to_traffic_generator_miles'] = distances / 1609.34    
                
                # Mark candidates near traffic generators
                self.candidates.loc[distances < self.config.get('service_radius_miles', 2.0), 'is_traffic_generator'] = 1  
                
        
        # Highway interchange detection
        print("    Detecting highway interchanges...")
        self.candidates['is_interchange'] = self._detect_highway_interchanges()
        
        # Calculate derived features
        self.candidates['trip_position_factor'] = np.sin(
            self.candidates.get('position_on_route', 0.5) * np.pi
        )
        
        print(f"    Found {self.candidates['is_interchange'].sum()} interchange locations")
        print(f"    Found {self.candidates['has_rest_area_or_parking'].sum()} rest area locations")
        print(f"    Found {self.candidates['is_traffic_generator'].sum()} traffic generator locations")




        
    def _calculate_location_attributes_old(self):
        """
        
        Calculate spatial attributes for each candidate location.
        Standardizes coordinate access and calculates distances to key infrastructure.
        
        """
        
        
        # Standardize coordinate access for ALL candidates
        print("Standardizing candidate coordinates...")
        
        if 'longitude' not in self.candidates.columns or 'latitude' not in self.candidates.columns:
            # Extract coordinates from geometry column and store as standard columns
            try:
                self.candidates['longitude'] = self.candidates.geometry.x
                self.candidates['latitude'] = self.candidates.geometry.y
                print(f"  Extracted coordinates for {len(self.candidates)} candidates")
            except Exception as e:
                print(f"  Warning: Could not extract coordinates from geometry: {e}")
                # Fallback: create default coordinates
                self.candidates['longitude'] = 0.0
                self.candidates['latitude'] = 0.0
        
        # Verify coordinate extraction worked
        if self.candidates['longitude'].isna().any() or self.candidates['latitude'].isna().any():
            print("  Warning: Some candidates have NaN coordinates, filling with defaults")
            self.candidates['longitude'] = self.candidates['longitude'].fillna(0.0)
            self.candidates['latitude'] = self.candidates['latitude'].fillna(0.0)
        
        print(f"  Coordinate ranges: Lon [{self.candidates['longitude'].min():.0f}, {self.candidates['longitude'].max():.0f}], Lat [{self.candidates['latitude'].min():.0f}, {self.candidates['latitude'].max():.0f}]")
        
        # Continue with existing location attribute calculations
        print("Calculating location attributes...")
        
        # Ensure candidate_coords is available for all distance calculations
        candidate_coords = np.c_[self.candidates.geometry.x, self.candidates.geometry.y]
        
        # Distance to ports/major traffic generators
        ports = self.lois[self.lois['source'].str.contains('port|seaport', case=False, na=False)]
        if len(ports) > 0:
            port_tree = cKDTree(np.c_[ports.geometry.x, ports.geometry.y])
            candidate_coords = np.c_[self.candidates.geometry.x, self.candidates.geometry.y]
            distances, _ = port_tree.query(candidate_coords)
            self.candidates['dist_to_port_miles'] = distances / 1609.34
        else:
            self.candidates['dist_to_port_miles'] = 100 # Default distance in miles
            print("No ports found in locations of interest")
        
        # Distance to rest areas
        rest_areas = self.lois[self.lois['source'].str.contains('rest|parking', case=False, na=False)]
        if len(rest_areas) > 0:
            rest_tree = cKDTree(np.c_[rest_areas.geometry.x, rest_areas.geometry.y])
            distances, _ = rest_tree.query(candidate_coords)
            self.candidates['dist_to_rest_miles'] = distances / 1609.34
        else:
            self.candidates['dist_to_rest_miles'] = 50 # Default distance in miles
            print("No rest areas found in locations of interest")
            
        
        # Distance to existing gas stations (market signal)
        #gas_stations = self.lois[self.lois['source'].str.contains('gas|station', case=False, na=False)]
        gas_mask = self.lois['source'].str.contains('gas|fuel|energy|station', case=False, na=False)
        gas_stations = self.lois[gas_mask]
        
         # Create the default dist_to_gas_miles column to avoid KeyError
        default_gas_distance = 10.0  # Default distance in miles
    

        if len(gas_stations) > 0:
            print(f"Found {len(gas_stations)} gas stations, validating coordinates...")
            
            try:
                # Extract coordinates directly (no additional validation needed - they're already processed)
                gas_coords = np.c_[gas_stations.geometry.x, gas_stations.geometry.y]
                
                # first step simple validation - check for finite values
                valid_mask = np.all(np.isfinite(gas_coords), axis=1)
                valid_coords = gas_coords[valid_mask]
            
                print(f"  After validation: {len(valid_coords)} valid gas stations")
            
                if len(valid_coords) > 0:
                    # Calculate distances
                    gas_tree = cKDTree(valid_coords)
                    distances, _ = gas_tree.query(candidate_coords)
                    self.candidates['dist_to_gas_miles'] = distances / 1609.34
                    print(f"  Successfully calculated gas station distances")
                else:
                    # No valid coordinates - use default
                    self.candidates['dist_to_gas_miles'] = default_gas_distance
                    print(f"  No valid gas station coordinates, using default distance: {default_gas_distance} miles")
                    
            except Exception as e:
                print(f"  Error processing gas station coordinates: {e}")
                self.candidates['dist_to_gas_miles'] = default_gas_distance
                print(f"  Using default gas station distance: {default_gas_distance} miles")
                
        else:
            # no gas stations found - use default
            self.candidates['dist_to_gas_miles'] = default_gas_distance
            print(F" No gas station found; using default distance: {default_gas_distance} miles")
            
            """""    
            # Clean and validate gas station coordinates
            gas_stations_clean = gas_stations.copy()
            
            # Remove rows with invalid geometry
            gas_stations_clean = gas_stations_clean[gas_stations_clean.geometry.notna()]
            
            # Extract coordinates and check for validity
            try:
                gas_coords_x = gas_stations_clean.geometry.x
                gas_coords_y = gas_stations_clean.geometry.y
                
                # Check for NaN, inf, or unreasonable values
                valid_mask = (
                    np.isfinite(gas_coords_x) & 
                    np.isfinite(gas_coords_y) &
                    (gas_coords_x != 0) & 
                    (gas_coords_y != 0) &
                    (gas_coords_x > -180) & (gas_coords_x < 180) &  # Reasonable longitude range
                    (gas_coords_y > -90) & (gas_coords_y < 90)      # Reasonable latitude range
                )
            
                gas_stations_valid = gas_stations_clean[valid_mask]
            
                print(f"  After validation: {len(gas_stations_valid)} valid gas stations")

                if len(gas_stations_valid) > 0:
                    gas_coords = np.c_[gas_stations_valid.geometry.x, gas_stations_valid.geometry.y]
                    
                    # Double-check for any remaining invalid values
                    if np.any(~np.isfinite(gas_coords)):
                        print("  Warning: Still found invalid coordinates, removing them...")
                        finite_mask = np.all(np.isfinite(gas_coords), axis=1)
                        gas_coords = gas_coords[finite_mask]
                        print(f"  Final count: {len(gas_coords)} gas stations with valid coordinates")
                
                    if len(gas_coords) > 0:
                        gas_tree = cKDTree(gas_coords)    
                
                        #gas_tree = cKDTree(np.c_[gas_stations.geometry.x, gas_stations.geometry.y])
                        distances, _ = gas_tree.query(candidate_coords)
                        self.candidates['dist_to_gas_miles'] = distances / 1609.34
                        print(f"Found {len(gas_stations)} gas stations for distance calculations")
                    else:
                        # If no gas stations in LOIs, set a moderate default distance
                        default_gas_distance_miles = 10 
                        self.candidates['dist_to_gas_miles'] = default_gas_distance_miles # Default to 10 miles if no gas stations found
            except Exception as e:
                print(f"Error processing gas station coordinates: {e}")
                # If error occurs, set a default distance
                default_gas_distance_miles = 10 
                self.candidates['dist_to_gas_miles'] = default_gas_distance_miles
                print("No gas stations found in locations of interest - using default distance of {default_gas_distance_miles} miles")
            
        else:
            # If no gas stations in LOIs, set a moderate default distance
            default_gas_distance_miles = 10 
            self.candidates['dist_to_gas_miles'] = default_gas_distance_miles
            print("No gas stations found in locations of interest - using default distance of {default_gas_distance_miles} miles")
            """""
            
        # Highway interchange detection (based on route connectivity)
        self._detect_highway_interchanges()
        
        # Trip position factor (higher probability mid-trip)
        if 'position_on_route' in self.candidates.columns:
            self.candidates['trip_position_factor'] = np.sin(self.candidates['position_on_route'] * np.pi)
        else:
            print("Warning: 'position_on_route' column not found, setting trip_position_factor to 0.25")
            self.candidates['trip_position_factor'] = 0.25
        
        
        
        # Summary of calculated attributes
        print(f"Location attributes calculated for {len(self.candidates)} candidates")
        print(f"  - Interchanges detected: {self.candidates['is_interchange'].sum()}")
        print(f"  - Average distance to port: {self.candidates['dist_to_port_miles'].mean():.1f} miles")
        print(f"  - Average distance to rest areas: {self.candidates['dist_to_rest_miles'].mean():.1f} miles") 
        print(f"  - Average distance to gas stations: {self.candidates['dist_to_gas_miles'].mean():.1f} miles")

        # Only try to access the column if we're sure it exists
        if 'dist_to_gas_miles' in self.candidates.columns:
            print(f"  - Average distance to gas stations: {self.candidates['dist_to_gas_miles'].mean():.1f} miles")
        else:
            print(f"  - Gas station distances: Column not created")
    

    def initialize_competition_graph(self):
        """
        Optimized competition graph initialization using spatial indexing.
       
        """
        import time
        start_time = time.time()
        
        print("\nInitializing competition graph (optimized)...")
        
        if self._competition_graph is None:
            print("Initializing competition graph with all stations...")
            self._competition_graph = self.CompetitionGraph(self.config)
        
        # Batch add all nodes first (more efficient than one-by-one)
        nodes_to_add = []
        
        # Add existing stations
        if hasattr(self, 'existing_stations') and len(self.existing_stations) > 0:
            for idx, station in self.existing_stations.iterrows():
                nodes_to_add.append({
                    'id': f"existing_{idx}",
                    'data': station,
                    'type': 'existing'
                })
        
        # Add candidates
        for idx, candidate in self.candidates.iterrows():
            nodes_to_add.append({
                'id': idx,
                'data': candidate,
                'type': 'candidate'
            })
        
        # Batch add nodes
        for node in nodes_to_add:
            self._competition_graph.add_or_update_station(
                node['id'], node['data'], node['type']
            )
        
        print(f"  Added {len(nodes_to_add)} nodes")
        
        # Efficient edge creation using KDTree
        all_coords = []
        all_ids = []
        
        for node_id, node_data in self._competition_graph.nodes.items():
            all_coords.append([node_data['x'], node_data['y']])
            all_ids.append(node_id)
        
        if len(all_coords) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.array(all_coords))
            
            # Find all pairs within competition radius
            competition_radius = self.config.get('service_radius_miles', 2.0) * 1609.34
            pairs = tree.query_pairs(competition_radius, output_type='ndarray')
            
            # Create edges for all pairs
            edge_count = 0
            for i, j in pairs:
                self._competition_graph.add_competition_edge(all_ids[i], all_ids[j])
                edge_count += 1
            
            print(f"  Created {edge_count} competition edges")
        
        elapsed = time.time() - start_time
        print(f"  Competition graph initialized in {elapsed:.1f} seconds")
        
       
    
    
    
    
    
    def calculate_competition_adjusted_demand(self, station, all_competitors, 
                                            include_existing=True, include_candidates=True):
        """
        Unified competition adjustment using consistent gravity model.
        Uses competition graph for demand calculations.
        This method handles two scenarios:
        1. Calculate station's demand given competitors (forward)
        2. Calculate how station affects competitors (reverse)
        Handles graph initialization on first call
        
        Parameters:
        - station: Station to calculate demand for
        - all_competitors: DataFrame of new competitors to consider
        - include_existing: Whether to include existing/selected stations in competition
        - include_candidates: Whether to include other candidate stations in competition
        
        """
 
        # Check graph exists
        # Graph should already exist from _calculate_bayesian_probabilities
        if self._competition_graph is None:
            # ToDo: re-add the initializatuon block - same as from calculate bayesian probabilities p(choose) method
            raise ValueError("Competition graph not initialized. Run calculate_bayesian_probabilities first.")
    
            
        graph = self._competition_graph
        
        # Determine station ID consistently
        if hasattr(station, 'name'):
            station_id = station.name  # Series from DataFrame
        elif hasattr(station, 'index'):
            station_id = station.index
        else:
            station_id = id(station)
        
        station_type = 'candidate'  # Default
        if hasattr(station, 'type'):
            if hasattr(station, 'get'):
                station_type = station.get('type', 'candidate') # Use 'type' if available, else default to 'candidate'
            else: 
                station_type = station['type']
        
        # Ensure this station is in the graph and only update it if it does (do not add)
        # Verify station exists in graph
        if station_id not in graph.nodes:
            print(f"Warning: Station {station_id} not in graph - adding it")
            graph.add_or_update_station(station_id, station, station_type)
        else:
            # Update station properties in graph 
            graph.add_or_update_station(station_id, station, station_type)
       
        
        # Determine station ID
        #station_id = station.name if hasattr(station, 'name') else (station.index if hasattr(station, 'index') else id(station))

        
        # Ensure station is not set to compete with itself
        if isinstance(all_competitors, pd.DataFrame) and station_id in all_competitors.index:
            all_competitors = all_competitors.drop(station_id)
        elif isinstance(all_competitors, list) and station_id in all_competitors:
            all_competitors = [comp for comp in all_competitors if comp != station_id]

        
        """""
        # Add specified competitors to graph
        competitors_added = []
        
        # Add competitors to graph based on parameters
        if include_existing and isinstance(all_competitors, pd.DataFrame):
            for idx, comp in all_competitors.iterrows():
                comp_id = f"existing_{idx}" if 'existing' in str(idx) else idx
                comp_type = comp.get('type', 'selected') if hasattr(comp, 'get') else 'selected' # Assuming these are selected stations
                graph.add_or_update_station(comp_id, comp, comp_type)
                graph.add_competition_edge(station_id, comp_id)
                competitors_added.append(comp_id)  
                
        if include_candidates and hasattr(self, 'candidates'):
            # Add other candidates
            # Only add candidates that are in the current working self
            if hasattr(self, '_current_working_candidates'):
                # Use the working subset if available (set by iterative selection)
                working_candidates = self._current_working_candidates
            elif isinstance(all_competitors, pd.DataFrame) and 'type' in all_competitors.columns:
                # Use candidates from all_competitors if they're included there
                working_candidates = all_competitors[all_competitors['type'] == 'candidate']
            else:
                # Fallback: no additional candidates
                working_candidates = pd.DataFrame()
            
            for idx, cand in working_candidates.iterrows():
                if idx != station_id:  # Don't add self
                    graph.add_or_update_station(idx, cand, 'candidate')
                    graph.add_competition_edge(station_id, idx) 
                    competitors_added.append(idx)   
                      

        
        # Calculate market share and adjusted demand
        market_share = graph.calculate_market_share(station_id)
        base_demand = station.get('initial_demand_post_existing',
                                 station.get('expected_demand_kg_day', 0))
        adjusted_demand = base_demand * market_share
        
        # Update node in graph with new demand
        if station_id in graph.nodes:
            graph.nodes[station_id]['demand'] = adjusted_demand
            graph.nodes[station_id]['market_share'] = market_share
            
        """
        
        # Update graph with NEW competitors from all_competitors
        if all_competitors is not None and len(all_competitors) > 0:
            if isinstance(all_competitors, pd.DataFrame):
                for idx, comp in all_competitors.iterrows():
                    comp_id = f"existing_{idx}" if 'existing' in str(idx) else idx
                    # Update competitor in graph (might change type, e.g., candidate → selected)
                    if comp_id in graph.nodes:
                        comp_type = comp.get('type', graph.nodes[comp_id]['type']) if hasattr(comp, 'get') else 'selected'
                        graph.add_or_update_station(comp_id, comp, comp_type)
                        # Edge should already exist if within service radius
                        
        # Calculate market share based on ALL competition in graph
        # The market share calculation considers ALL nodes connected to station_id
        # filtered by type (existing/selected vs candidates) based on parameters
        
        # Temporarily set competition weights based on parameters
        original_weights = {}
        if hasattr(graph, 'config'):
            if not include_existing:
                original_weights['existing_station_competition_weight'] = graph.config.get('existing_station_competition_weight', 1.0)
                graph.config['existing_station_competition_weight'] = 0.0
            if not include_candidates:
                original_weights['potential_station_competition_weight'] = graph.config.get('potential_station_competition_weight', 0.7)
                graph.config['potential_station_competition_weight'] = 0.0
        
        # Calculate market share with current graph state
        market_share = graph.calculate_market_share(station_id)
        
        # Restore original weights
        for key, value in original_weights.items():
            graph.config[key] = value
        
        competition_agnostic_station_utilization_factor = station['p_need_fuel'] * station['p_stop_given_need'] if 'stop_given_need' in station and 'p_need_fuel' in station else self.config.get('existing_station_utilization', 0.7)
           
        # Calculate adjusted demand
        base_demand = station.get('competition_agnostic_adjusted_demand',
                                station.get('initial_demand_post_existing', 0)*competition_agnostic_station_utilization_factor)
        adjusted_demand = base_demand * market_share
                    
        
        return adjusted_demand

    
    
    
    def _calculate_bayesian_probabilities(self):
        """
        Calculate refueling probability using Bayesian framework.
        
        P(refuel|location) = P(need_fuel) × P(stop|need_fuel, attributes) × P(choose|alternatives)
        """
        
        # First ensure we have basic demand data
        if 'truck_aadt' not in self.candidates.columns:
            print("Warning: truck_aadt not found in candidates, setting default values")
            self.candidates['truck_aadt'] = 1000  # Default value
            
            
        # Initialize expected_demand_kg_day early
        if hasattr(self, 'demand_surface') and self.demand_surface is not None:
            # Use demand surface to initialize
            from scipy.interpolate import RegularGridInterpolator
            
            x = self.demand_surface['x'][0, :]
            y = self.demand_surface['y'][:, 0]
            demand_grid = self.demand_surface['demand']
            
            interpolator = RegularGridInterpolator(
                (y, x), demand_grid,
                bounds_error=False,
                fill_value=0.0
            )
            
            # Sample at candidate locations
            sample_points = np.column_stack([
                self.candidates.geometry.y,
                self.candidates.geometry.x
            ])
            
            surface_demand = interpolator(sample_points)
            
            # Ensure minimum demand
            self.candidates['traffic_flow_demand_equivalent'] = (
                self.candidates['truck_aadt'] *  
                self.config.get('avg_refuel_amount_kg', 60.0)
            )
            min_demand = self.config.get('min_daily_visits', 5.0) * self.config.get('avg_refuel_amount_kg', 60.0)
            self.candidates['initial_demand_post_existing'] = np.maximum(surface_demand, min_demand)
            self.candidates['expected_demand_kg_day'] = self.candidates['initial_demand_post_existing']
            print(f"Initialized expected_demand_kg_day for candidates using demand surface with range [{self.candidates['expected_demand_kg_day'].min():.0f}, {self.candidates['expected_demand_kg_day'].max():.0f}] kg/day")
        
        else:
            # Fallback: Initialize based on truck traffic
            print("Warning: No demand surface available, initializing based on truck traffic")
            min_demand = self.config.get('min_daily_visits', 5.0) * self.config.get('avg_refuel_amount_kg', 60.0)
            self.candidates['traffic_flow_demand_equivalent'] = (
                self.candidates['truck_aadt'] *  
                self.config.get('avg_refuel_amount_kg', 60.0)
            )
            self.candidates['initial_demand_post_existing'] = np.maximum(self.candidates['traffic_flow_demand_equivalent'], min_demand)
            self.candidates['expected_demand_kg_day'] = self.candidates['initial_demand_post_existing']
    
        
    
        
        # P(need_fuel) - based on trip position and typical range
        
        refuel_window = self.config.get('refuel_threshold', 0.25)  # 25% of range where refueling is likely per specification
        typical_range_miles = self.config.get('typical_range_miles', 450.0)  # Fuel cell truck range
        h2_consumption_per_mile = self.config.get('h2_consumption_kg_per_mile', 0.1)  # Hydrogen consumption per mile
        tank_capacity_kg = self.config.get('tank_capacity_kg', 80.0)  # Hydrogen tank capacity
        
        # Calculate the distance at which refueling is likely  
        miles_per_tank = min(typical_range_miles, 0.8 * tank_capacity_kg / h2_consumption_per_mile)  # How far can we go on a full tank, assume 80% of tank capacity usable
        refuel_start_miles = miles_per_tank * (1.0 - refuel_window)  # Miles driven by truck at start of refuel window
        #refuel_urgent_miles = miles_per_tank * (1.0 - 0.5 * refuel_window)  # Miles driven by truck at urgent refuel point
        refuel_urgent_miles = miles_per_tank * 0.9 # 90% tank used 
        
        """""
        
        #### Start P(NEED_FUEL) CALCULATION ALTERNATIVE HERE####  
        if 'position_on_route' in self.candidates.columns and route_idx in self.candidates.columns:
            # Calculate effective miles traveled for each candidate
            miles_traveled = []
            for idx, candidate in self.candidates.iterrows():
                route_idx = candidate['route_idx']
                position = candidate['position_on_route']
                
            # Get route segment info for this candidate
            if route_idx in self.routes.index:
                route = self.routes.loc[route_idx]
                segment_length = route['length_miles']
                miles_from_start = segment_length * position  
                
                # Estimate total trip distance based on route characteristics
                # Higher traffic routes typically serve longer-distance trips
                truck_aadt = route.get('truck_aadt', 1000)  # Default to 1000 if not available
                
                # Estimate trip length multiplier based on route importance
                # Major highways: 300-500 mile trips
                # Secondary routes: 100-300 mile trips  
                # Local routes: 50-150 mile trips
                
                if truck_aadt > 5000:  # Major highway
                    base_trip_length = 400
                    trip_variation = 100
                elif truck_aadt > 2000:  # Secondary highway
                    base_trip_length = 200
                    trip_variation = 50
                else:  # Local/smaller routes
                    base_trip_length = 100
                    trip_variation = 30
                    
                # Add some randomness based on position (early/late in trip)
                trip_progress_factor = 0.3 + 1.4 * position  # 30% to 170% of base
                estimated_trip_length = base_trip_length * trip_progress_factor
                
                Estimate how far into the trip this candidate represents
                # Assume this segment is somewhere in the middle portion of longer trip
                estimated_miles_traveled = (
                    estimated_trip_length * 0.3 +  # 30% baseline progress
                    miles_from_segment_start +      # Plus distance along this segment
                    (estimated_trip_length * 0.4 * position)  # Plus trip-position factor
                )    
                
                miles_traveled.append(min(estimated_miles_traveled, estimated_trip_length))
            else:
                # Default to moderate distance if route not found
                miles_traveled.append(refuel_start_miles * 0.7)  # 70% of refuel threshold
        
        self.candidates['miles_traveled'] = miles_traveled
                        
         # More gentle sigmoid function for better probability distribution
        # Calculate probability of needing fuel based on estimated trip distance
        refuel_range_size = refuel_urgent_miles - refuel_start_miles  # ~56 miles 
        
        # Normalize distance relative to refuel range size
        normalized_distance = (
            np.array(self.candidates['miles_traveled']) - refuel_start_miles
        ) / refuel_window_size    
        
        # Use gentler sigmoid (coefficient of 3 instead of 10)
        self.candidates['p_need_fuel'] = 1 / (1 + np.exp(-3 * normalized_distance))

        # Add baseline probability - some trucks always need fuel regardless of distance
        baseline_fuel_need = 0.15  # 15% baseline probability
        self.candidates['p_need_fuel'] = (
            baseline_fuel_need + 
            (1 - baseline_fuel_need) * self.candidates['p_need_fuel']
        )    
        
        
        # Additional factor for very long estimated trips
        refuels_needed = np.array(self.candidates['miles_traveled']) / miles_per_tank
        long_trip_bonus = 0.1 * (refuels_needed - 0.8).clip(0, 2)  # Bonus for trips >80% of tank capacity
        self.candidates['p_need_fuel'] = np.minimum(
            1.0,
            self.candidates['p_need_fuel'] + long_trip_bonus
        )
        
        # Debug output
        print(f"  Miles traveled - Min: {min(miles_traveled):.0f}, Max: {max(miles_traveled):.0f}, Mean: {np.mean(miles_traveled):.0f}")
        
    else:     
    
        # Default to moderate probability if position unknown
        self.candidates['p_need_fuel'] = 0.4
        print("  Using default fuel need probability: 0.4")
                            
         
    #### END P(NEED_FUEL) CALCULATION ALTERNATIVE HERE####    
    ###Older version below ####
    
        # Simple model: highest probability at 60-80% of typical range
        # Assume bernoulli distribution for need based on trip position  
        if 'position_on_route' in self.candidates.columns:
            # Use beta distribution to model fuel need probability based on trip position            
            self.candidates['p_need_fuel'] = stats.beta.pdf(
                self.candidates['position_on_route'], 
                a=6, b=4
                ) # Shape parameters for beta distribution
        else:
            # Default to uniform distribution if position not available
            print("Warning: 'position_on_route' column not found, setting p_need_fuel to uniform distribution")
            self.candidates['p_need_fuel'] = 0.5
        
        """
        
        # P(need_fuel) based on distance traveled and tank capacity
        print("Calculating fuel need probabilities...")
        
        #if 'position_on_route' in self.candidates.columns and 'route_idx' in self.candidates.columns:
            
        if 'position_on_route' in self.candidates.columns:
            # Use position on route to estimate travel distance
            # position_on_route is a fraction of the route length (0.0 to 1.0) along the route segment
            
            # Estimate the total trip distance or miles traveled for each candidate based on the route they are on
            
            candidate_trip_distances = []
            #miles_traveled = []
            
            for idx, candidate in self.candidates.iterrows():
                route_idx = candidate.get('route_idx', 0)  #route_idx = candidate['route_idx']
                position = candidate.get('position_on_route', 0.25)  # default to 0.25  #candidate['position_on_route']
                
                
                # Get the route this candidate is on
                if route_idx in self.routes.index:
                    route = self.routes.loc[route_idx]
                    route_length_miles = route.get('length_miles', 50)  # Route segment length # default to 50 miles
                
                
                    # Estimate trip distance: assume trucks travel multiple segments
                    # Use a multiplier based on route importance (higher AADT = longer trips)
                    aadt = route.get('truck_aadt', 1000)
                    
                    trip_multiplier = 1 + (aadt / 10000)  # Busier routes = longer trip distances
                    
                    # Calculate estimated distance traveled when reaching this candidate
                    estimated_trip_distance = position * route_length_miles * trip_multiplier
                    candidate_trip_distances.append(estimated_trip_distance)
                
                else:
                    # Default to middle of refuel window if no route info
                    candidate_trip_distances.append((refuel_start_miles + refuel_urgent_miles) / 2)
                    
            candidate_trip_distances = np.array(candidate_trip_distances)                    
                    
                # Get total route length
                #if route_idx in self.routes.index:
                    #route_length = self.routes.loc[route_idx, 'length_miles']
                    #miles_from_start = route_length * position
                    #miles_traveled.append(miles_from_start)
                #else:
                    # Default to moderate distance if route not found
                    #miles_traveled.append(refuel_start_miles)
            
            #self.candidates['miles_traveled'] = miles_traveled
            
            # Calculate probability of needing fuel based on distance
            # Using a sigmoid function that ramps up between refuel_start and refuel_urgent
            # Calculate p(need_fuel) based on trip distance
            # Sigmoid function centered around refuel window
            refuel_center = (refuel_start_miles + refuel_urgent_miles) / 2  # ~365 miles
            refuel_width = refuel_urgent_miles - refuel_start_miles  # ~56 miles
        
            # Probability increases sigmoidally in refuel window
            # Center normalized makes sense for public refueling infrastructure with a slope factor k =1
            # Note implicit constraint that urgency factor of tank range (0.9) + refuel window (0.25) > 1.0
            normalized_distance = (candidate_trip_distances - refuel_center) / (refuel_width / 2)
            self.candidates['p_need_fuel'] = 1 / (1 + np.exp(-normalized_distance))
            
            # A start-anchored sigmoid function could also be used when we have a known refuel start point
            # This would be useful for trips where the truck has homebase refueling available at start
            # In which case the trucks also use public refueling infrastructure for long trips
            # Higher k values make the sigmoid steeper  e.g. k=10
            #self.candidates['p_need_fuel'] = 1 / (1 + np.exp(
                #-10 * (self.candidates['miles_traveled'] - refuel_start_miles) / 
                #(refuel_urgent_miles - refuel_start_miles)
            #))
            
            # Cap minimum probability (some trucks always need fuel)
            # Add baseline probability - some trucks always need fuel regardless of distance
            baseline_fuel_need = 0.10  # 10% baseline probability
            self.candidates['p_need_fuel'] = (
                baseline_fuel_need + 
                (1 - baseline_fuel_need) * self.candidates['p_need_fuel']
            )    
        
        
        else:
            # Fallback: assume moderate fuel need probability
            print("Warning: 'position_on_route' column not found, using default fuel need probability")
            self.candidates['p_need_fuel'] = 0.25  # 25% of trucks need fuel 
            #ToDO: make this a config parameter for tunability

            
            
            #To-Do: refine additional refuel need probability adjustment function
            # Additional factor for very long routes where multiple refuels needed
            #refuels_needed = self.candidates['miles_traveled'] / refuel_start_miles
            #self.candidates['p_need_fuel'] = np.minimum(
                #1.0,
                #self.candidates['p_need_fuel'] + 0.2 * (refuels_needed - 1).clip(0)
            #)
        self.candidates['fuel_need_adjusted_demand'] = self.candidates['initial_demand_post_existing'] * self.candidates['p_need_fuel']
        
        # P(stop|need_fuel, attributes) - willingness to stop based on location quality
        # weighted combination of a number of attraction factors
        # initialize weights based on config
        #location_multipliers = []
        #weights = self.config
        
        
        # Initialize scores with default values to avoid NaN
        #self.candidates['port_factor'] = 0.5  # Default moderate score
        #self.candidates['rest_factor'] = 0.5
        #self.candidates['gas_factor'] = 0.5
        
        # Update scores based on actual distances if available
        # Normalize distances to [0, 1] with decay
        
        # Port proximity (freight generators)
        if 'dist_to_port_miles' in self.candidates.columns:
            port_factor = np.exp(-self.candidates['dist_to_port_miles'] / 20)  # 20 miles as a reasonable distance decay
            self.candidates['port_factor'] = port_factor
            port_weight = self.config.get('port_proximity_weight', 0.20)  # Default to 0.20 if not specified
        else:
            port_factor = 0.5  # Default moderate score
            self.candidates['port_factor'] = port_factor
            port_weight = 0.2  # Default weight for ports
            
                     
        # Rest area proximity (closer = higher probability)
        if 'dist_to_rest_miles' in self.candidates.columns:
            rest_factor = np.exp(-self.candidates['dist_to_rest_miles'] / 2) # 2 miles as a reasonable distance decay
            self.candidates['rest_factor'] = rest_factor
            rest_weight = self.config.get('rest_area_attraction_weight', 0.2)  # Default to 0.2 if not specified
        else:
            print("Warning: 'dist_to_rest_miles' column not found, setting rest factor to default of 0.5")
            rest_factor = 0.5  # Default moderate score
            self.candidates['rest_factor'] = rest_factor
            rest_weight = 0.2  # Default weight for rest areas
            
        # Gas station proximity (market signal)    
        if 'dist_to_gas_miles' in self.candidates.columns:
            gas_factor = np.exp(-self.candidates['dist_to_gas_miles'] / 3)  # 3 miles as a reasonable distance decay
            self.candidates['gas_factor'] = gas_factor
            gas_weight = self.config.get('gas_station_proximity_weight', 0.15)  # Default to 0.15 if not specified
        else:
            gas_factor = 0.5  # Default moderate score
            self.candidates['gas_factor'] = gas_factor
            gas_weight = 0.2  # Default weight for gas stations
        
        
        # Highway interchange bonus
        if 'is_interchange' in self.candidates.columns:
            interchange_factor = 1.0 + 0.3 * self.candidates['is_interchange']  # Boost score for interchanges
            self.candidates['interchange_factor'] = interchange_factor
            interchange_weight = self.config.get('interchange_weight', 0.2)  # Default to 0.2 if not specified
        else:
            interchange_factor = 1.0
            self.candidates['interchange_factor'] = interchange_factor
            interchange_weight = 0.2  # Default weight for interchanges
        
        # Combine factors into a single stop probability 
        base_stop_probability = 0.01 # Base probability of stopping at any location, uniform across all locations
        
        self.candidates['p_stop_given_need'] = base_stop_probability + (
            port_weight * port_factor + 
            rest_weight * rest_factor + 
            gas_weight * gas_factor + 
            interchange_weight * (interchange_factor - 1)
        )
        
        # Cap at reasonable range
        self.candidates['p_stop_given_need'] = np.clip(self.candidates['p_stop_given_need'], 0.05, 0.95) # Cap between 5% and 95%

        
        # Specify weights for each factor from config
        
        #required_weights = [
            #'port_proximity_weight', 
            #'rest_area_attraction_weight', 
            #'gas_station_proximity_weight', 
            #'highway_access_weight'
        #]
        
        # store location multipliers in a dictionary
        #location_attribute_weights = {
            #'port_proximity_weight': port_weight,
            #'rest_area_attraction_weight': rest_weight,
            #'gas_station_proximity_weight': gas_weight,
            #'interchange_weight': interchange_weight
        #}
        
        #self.candidates['location_attribute_weights'] = location_attribute_weights
        
 
        # ensure weights sum to 1.0 and if not, normalize them
        #total_weight = sum(self.config[w] for w in required_weights)
        #if not np.isclose(total_weight, 1.0, atol=1e-2):
            #print(f"Warning: Weights do not sum to 1.0, weights sum to {total_weight:.2f}. Normalizing weights.")
            #for weight in required_weights:
                #weights[weight] /= total_weight
        
        # Normalize [0, 1]
        #max_score = self.candidates['p_stop_given_need'].max()
        #if max_score > 0:
        #    self.candidates['p_stop_given_need'] = (
        #        self.candidates['p_stop_given_need'] / max_score
        #    )
        #else:
        #    print("Warning: All p_stop_given_need scores are zero, setting to 0.5")
        #    self.candidates['p_stop_given_need'] = 0.5  # Default to 0.5 if all scores are zero
        
        self.candidates['stop_given_need_and_station_attribute_adjusted_demand'] = self.candidates['initial_demand_post_existing'] * self.candidates['p_stop_given_need']
        self.candidates['competition_agnostic_adjusted_demand'] = self.candidates['initial_demand_post_existing'] * self.candidates['p_need_fuel'] * self.candidates['p_stop_given_need']
        
        # P(choose|alternatives) - competition from existing stations
        # Add all candidates to graph 
        
        if hasattr(self, 'existing_stations') and len(self.existing_stations) > 0:
            
            # Initialize competition graph with all stations upfront
        
            if self._competition_graph is None:
                print("Initializing competition graph...")
                self._competition_graph = self.CompetitionGraph(self.config)
                
                # Add ALL existing stations first
                for idx, existing in self.existing_stations.iterrows():
                    existing_dict = existing.to_dict()
                    if 'geometry' in existing_dict:
                        existing_dict['geometry'] = existing.geometry  # Preserve geometry object
                    self._competition_graph.add_or_update_station(
                        f"existing_{idx}",  # Consistent ID format
                        existing_dict,
                        'existing'
                    )
                
                # Add ALL candidates
                for idx, candidate in self.candidates.iterrows():
                    candidate_dict = candidate.to_dict()
                    if 'geometry' in candidate_dict:
                        candidate_dict['geometry'] = candidate.geometry
                    self._competition_graph.add_or_update_station(
                        idx,  # Use actual index
                        candidate_dict,
                        'candidate'
                    )
                
                # Create all edges within service radius
                service_radius_m = self.config.get('service_radius_miles', 2.0) * 1609.34
                all_ids = list(self._competition_graph.nodes.keys())
                
                for i, id1 in enumerate(all_ids):
                    node1 = self._competition_graph.nodes[id1]
                    for j in range(i+1, len(all_ids)):
                        id2 = all_ids[j]
                        node2 = self._competition_graph.nodes[id2]
                        
                        dist = np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
                        if dist <= service_radius_m:
                            self._competition_graph.add_competition_edge(id1, id2)
                
                print(f"Competition graph initialized: {len(self._competition_graph.nodes)} nodes, {len(self._competition_graph.edges)} edges")

            # Calculate market share based on competition graph
            # Uses the graph-based competition adjustment using Gravity model
            # Updates graph without creating duplicate edges
            self.candidates['market_share'] = self.candidates.apply(
                lambda s: self.calculate_competition_adjusted_demand(
                    s, 
                    self.existing_stations,  # Pass existing stations
                    include_existing=True,   # Will use graph edges
                    include_candidates=False # Only existing competition for now
                ) / s['competition_agnostic_adjusted_demand'], 
                axis=1
            )
            
            self.candidates['p_choose'] = self.candidates['market_share']
            #self.candidates['initial_demand_post_existing'] = self.candidates['expected_demand_kg_day'] * self.candidates['market_share']
            
            """"
            else:
                # Use a simple competition model based on distance to existing stations
                
                from scipy.spatial import cKDTree
                existing_coords = np.c_[
                    self.existing_stations.geometry.x, 
                    self.existing_stations.geometry.y
                ]
                existing_tree = cKDTree(existing_coords)
                
                # Find competing stations within service radius
                service_radius_m = self.config['service_radius_miles'] * 1609.34 # in meters
                candidate_coords = np.c_[self.candidates.geometry.x, self.candidates.geometry.y]
                
                # ToDo: include existing ze_infrastructure stations in competition
                competition_scores = []
                
                for coord in candidate_coords:
                    indices = existing_tree.query_ball_point(coord, service_radius_m)
                    n_competitors = len(indices)
                    # Probability decreases with competition
                    competition_scores.append(np.exp(-self.config['competition_decay_rate'] * n_competitors))
                
                self.candidates['p_choose'] = competition_scores
            """
        else:
            self.candidates['p_choose'] = 1.0
            self.candidates['market_share'] = 1.0
            
            
        
        # Combined probability (Bayesian components)
        self.candidates['utilization_prob'] = (
            self.candidates['p_need_fuel'] * 
            self.candidates['p_stop_given_need'] * 
            self.candidates['p_choose']
        )
        
        self.candidates['expected_demand_kg_day'] = (
            self.candidates['utilization_prob'] *
            self.candidates['initial_demand_post_existing']
        )
        
        # Expected daily visits - ensure no NaN  values
        # already done earlier
        #min_daily_visits = self.config.get('min_daily_visits', 5.0)  # Minimum daily visits per candidate
        #self.candidates['expected_visits_day'] = np.maximum(
            #self.candidates['truck_aadt'] * self.candidates['utilization_prob'],
            #min_daily_visits
        #    #)
        
        # Expected demand - ensure no NaN values
        #avg_refuel_amount_kg = self.config.get('avg_refuel_amount_kg', 60.0)  # Average refuel amount in kg
        # Calculate refined demand based on Bayesian probabilities
        #refined_demand = (
                    #self.candidates['expected_visits_day'] * 
                    #avg_refuel_amount_kg
                #)
        #demand_blend_factor = self.config.get('demand_blend_factor', 0.7)  # Blend factor for demand weighting between surface and Bayesian model
        #self.candidates['expected_demand_kg_day'] = (
            #demand_blend_factor * self.candidates['expected_demand_kg_day'] + 
            #(1 - demand_blend_factor) * refined_demand
        #)
        
        # Add total_demand column if missing (needed for economic calculations)
        if 'total_demand' not in self.candidates.columns:
            self.candidates['total_demand'] = self.candidates['expected_demand_kg_day']
            
        
        self.candidates['bayesian_weights'] = {
            'rest_weight': rest_weight,
            'gas_weight': gas_weight,
            'interchange_weight': interchange_weight,
            'port_weight': port_weight
        }
        
        self.bayesian_weights = {
            'rest_weight': rest_weight,
            'gas_weight': gas_weight,
            'interchange_weight': interchange_weight,
            'port_weight': port_weight
        }
        
        self.candidates['location_multiplier'] = {
            'rest_factor': rest_factor,
            'gas_factor': gas_factor,
            'interchange_factor': interchange_factor,
            'port_factor': port_factor
        }
            
        print(f"Demand calculation complete:")
        print(f"  Refuel starts at: {refuel_start_miles:.0f} miles ({(1-refuel_window)*100:.0f}% tank used)")
        print(f"  Refuel urgent at: {refuel_urgent_miles:.0f} miles (90% tank used)")
        print(f"  Average p(need_fuel): {self.candidates['p_need_fuel'].mean():.3f}")
        print(f" Average p(stop|need_fuel): {self.candidates['p_stop_given_need'].mean():.3f}")
        print(f"  Average p(choose|alternatives): {self.candidates['p_choose'].mean():.3f}")
        print(f"  Average expected demand: {self.candidates['expected_demand_kg_day'].mean():.1f} kg/day")
        print(f"  Min demand: {self.candidates['expected_demand_kg_day'].min():.1f} kg/day")
        print(f"  Max demand: {self.candidates['expected_demand_kg_day'].max():.1f} kg/day")
        
    
    
    def calculate_economic_proxy(self):
        """
        Calculate proxy NPV for each candidate location.
        Integrates demand, costs, location quality, and interchange benefits.
        """
        print("Calculating economic proxy metrics...")
        
        # Ensure we have the expected demand column (handle different naming)
        demand_column = None
        for col in ['expected_demand_kg_day', 'total_demand', 'demand_kg_day', 'h2_demand_kg_day']: # 'h2_demand_daily_kg']:
            if col in self.candidates.columns:
                demand_column = col
                break
        
        if demand_column is None:
            raise ValueError("No demand column found in candidates. Expected 'expected_demand_kg_day', 'total_demand', or 'demand_kg_day'")
        
        # Standardize demand column name for consistency
        if demand_column != 'expected_demand_kg_day':
            self.candidates['expected_demand_kg_day'] = self.candidates[demand_column]
                
        
        # Base revenue calculation
        base_daily_revenue = np.minimum(
            self.candidates['expected_demand_kg_day'] * self.config['h2_price_per_kg'],
            self.config['station_capacity_kg_per_day'] * self.config['h2_price_per_kg']
        )
        
        self.candidates['base_daily_revenue'] = base_daily_revenue
        

        # Location quality multipliers
        # Orthogonal revenue multipliers (operational premiums, not demand-based)
        location_revenue_multiplier = self._calculate_location_revenue_multipliers()
        # Store component values for analysis
        self.candidates['location_revenue_multiplier'] = location_revenue_multiplier
        
        # Validate for revenue multiplier fpr double-counting
        self._validate_double_counting()
    
        # Apply location multipliers to revenue
        self.candidates['daily_revenue'] = base_daily_revenue * location_revenue_multiplier
    

        # Utilization rate
        self.candidates['utilization_rate'] = (
            self.candidates['expected_demand_kg_day'] / 
            self.config['station_capacity_kg_per_day']
        ).clip(0, 1)
        
        # Operating costs with location-based adjustments
        base_opex_daily = self.config['base_opex_daily']  # Fixed daily operating cost
        variable_opex_per_kg = self.config['variable_opex_per_kg']  # Variable cost per kg, including hydrogen delivery, electricity, maintenance, etc.
        
        # Adjust OPEX based on location quality
        # Location-based cost adjustments
        location_cost_multiplier = self._calculate_location_cost_multipliers()
        # Store component values for analysis
        self.candidates['location_cost_multiplier'] = location_cost_multiplier
    
        
        self.candidates['daily_opex'] = (
            (base_opex_daily * location_cost_multiplier) + 
            self.candidates['expected_demand_kg_day'].clip(0, self.config['station_capacity_kg_per_day']) * 
            variable_opex_per_kg
        )
        
        # Daily profit
        self.candidates['daily_profit'] = (
            self.candidates['daily_revenue'] - 
            self.candidates['daily_opex']
        )
        
        # Simple NPV proxy (assuming constant daily profit) with location premium adjustments
        years = self.config['station_lifetime_years']
        discount_rate = self.config['discount_rate']
        
        # Annual profit (asuming 365 days of operation)
        annual_profit = self.candidates['daily_profit'] * 365
        
        
        # NPV calculation with present value of annuity
        
        #npv_factors = [(1 / (1 + discount_rate)**t) for t in range(1, years + 1)]
        #total_npv_factor = sum(npv_factors)
        
        if discount_rate > 0:
            npv_factor = (1 - (1 + discount_rate)**-years) / discount_rate
        else:
            npv_factor = years  # If discount rate is 0
        
        # Base station CAPEX with location adjustments
        base_capex = self.config['station_capex']
        land_cost = base_capex * 0.15
        permitting_cost = base_capex * 0.05
        site_development_cost = base_capex * 0.20
        contingency = base_capex * 0.10
        
        variable_capex_costs = (land_cost + site_development_cost) * location_cost_multiplier
        fixed_capex_costs = base_capex + permitting_cost + contingency
        #adjusted_capex = self.config['station_capex'] * location_cost_multiplier
        
        # Total development costs
        total_adjusted_capex = (
            variable_capex_costs + 
            fixed_capex_costs
        )

        self.candidates['capex'] = total_adjusted_capex
    
        self.candidates['npv_proxy'] = (
            annual_profit * npv_factor - 
            total_adjusted_capex
        )
        
        # Annual OPEX
        annual_opex = self.candidates['daily_opex'] * 365
    
        # Present value of total OPEX
        opex_pv = annual_opex * npv_factor
        self.candidates['pv_opex'] = opex_pv
        
        # Total lifecycle cost, NPV of CAPEX + OPEX
        self.candidates['total_cost'] = total_adjusted_capex + opex_pv
    
        
        # Payback period (years)
        self.candidates['payback_years'] = np.where(
            annual_profit > 0,
            total_adjusted_capex / annual_profit,
            999
        )
        
        # IRR proxy (simplified)
        # Using rule of thumb: IRR ≈ (Total Return / Initial Investment)^(1/years) - 1
        total_return = annual_profit * years
        
        self.candidates['irr_proxy'] = np.where(
            total_return > 0,
            (total_return / total_adjusted_capex)**(1/years) - 1,
            -1
        )
        
        # Store component values for analysis
        self.candidates['adjusted_capex'] = total_adjusted_capex # same as 'capex' (which was already adjusted)
        
    
        # Summary statistics
        self._print_economic_summary()
        
        
    def _validate_budget_approach(self, budget):
        """Validate budget sufficiency under different approaches"""
        
        print(f"\n{'='*60}")
        print(f"BUDGET VALIDATION")
        print(f"{'='*60}")
        
        cost_column_names = ['capex', 'adjusted_capex', 'total_adjusted_capex']
        for col in cost_column_names:
            if col in self.candidates.columns:
                cost_column_name = col
                break

        cost_column = 'capex' if 'capex' in self.candidates.columns else cost_column_name
            
        if cost_column not in self.candidates.columns:
            print(f"No budget specified, using default budget assumption of {100.0*base_capex_multiplier:,.0f}% of configured ${self.config.get('station_capex', 12000000):,} base capex per station")
            base_capex_multiplier = self.config.get('base_capex_multiplier', 1.4)  # Default 40% buffer in multiplier for CAPEX
            costs = np.full(len(self.candidates), self.config['station_capex']*base_capex_multiplier)
        else:
            costs = self.candidates[cost_column]
        
        # Calculate feasible combinations
        sorted_costs = costs.sort_values()
        cumulative_costs = sorted_costs.cumsum() # cumulative costs of sorted costs
        max_stations = (cumulative_costs <= budget).sum()
        
        if max_stations > 0:
            cost_per_station = cumulative_costs.iloc[max_stations-1] / max_stations
            remaining_budget = budget - cumulative_costs.iloc[max_stations-1]
        else:
            cost_per_station = costs.min()
            remaining_budget = budget - cost_per_station
            max_stations = 0 if cost_per_station > budget else 1
        
        print(f"Budget available: ${budget:,.0f}") if budget else print("No budget constraint. Assuming infinite budget.")
        print(f"Max stations affordable: {max_stations}")
        print(f"Average cost per station: ${cost_per_station:,.0f}")
        print(f"Remaining budget: ${remaining_budget:,.0f}")
        
        # Business implications
        
        # Check if CAPEX budget covers total lifecycle costs
        total_investment_capital_needed = self.candidates['total_cost'].nsmallest(max_stations).sum()
        if total_investment_capital_needed > budget:
            shortfall = total_investment_capital_needed - budget
            print(f"    WARNING: CAPEX budget insufficient for operations")
            print(f"   Total lifecycle cost: ${total_investment_capital_needed:,.0f}")
            print(f"   Funding shortfall: ${shortfall:,.0f}")
        
        return max_stations > 0

    def _calculate_location_revenue_multipliers(self):
        """
        Calculate revenue multipliers based on location attributes.
        Calculate revenue multipliers for orthogonal value drivers.
        Ensures no double-counting with Bayesian probability weights.
        Higher multipliers for better locations (interchanges, accessibility, etc.)
        """
        multiplier = np.ones(len(self.candidates))  # Start with 1.0x base
        
        # Bayesian captures: Demand capture probability due to traffic patterns
        # Multipliers capture: Revenue per unit due to operational advantages
        
        # OPERATIONAL EFFICIENCY 
        # Interchange benefits (major revenue driver)
        if 'is_interchange' in self.candidates.columns:
            # Interchanges enable premium pricing due to convenience, not just higher demand
            operational_premium = 0.10 * self.candidates['is_interchange']  
            multiplier += operational_premium
            
            # INFRASTRUCTURE SYNERGIES
            # Interchanges often have better infrastructure (e.g., easier access, better facilities)
            # Interstate access reduces operational costs (fuel for delivery trucks)
            if 'has_interstate' in self.candidates.columns:
                cost_savings_as_margin = 0.08 * self.candidates['has_interstate']
                multiplier += cost_savings_as_margin
            
            
            if 'interchange_score' in self.candidates.columns:
                # Additional boost for high-quality interchanges
                multiplier += 0.10 * self.candidates['interchange_score'].clip(0.6, 1.0) - 0.06
        
        
        # NETWORK EFFECTS
        # Trip position factor (mid-trip locations are premium)
        if 'trip_position_factor' in self.candidates.columns:
            # Mid-trip positions can charge premium due to necessity, not just volume
            # Convert sin curve (0-1) to multiplier (0.9-1.3)
            # ToDo: confirm this model against trip position factor model. 
            # the necessity premium should be highest where P(need_fuel) is highest
            # Necessity premium model passes if that occurs farther into the trip (exp rather than sinusoidal/quadratic)
            necessity_premium = 0.2 * self.candidates['trip_position_factor']
            multiplier += necessity_premium - 0.1  # Baseline adjustment, maximum of 10% at 0.50 position on route, trip position factor of sin(pi*0.5)
        
        # Market presence considerations
        # ToDo: lower this and add rest areas to the mix
        if 'dist_to_gas_miles' in self.candidates.columns:
            # Optimal distance to gas stations (infrastructure benefits vs competition)
            optimal_gas_distance = 0.5  # miles
            left_mask = self.candidates['dist_to_gas_miles'] <= optimal_gas_distance
            # set maximum gas_factor at optimal distance
            peak_gas_factor = 0.5
            # specify power facator for growth/steepness of curve at distances < optimal_gas_distance
            left_k = 7.0
            gas_factor = np.zeros(len(self.candidates))
            gas_factor[left_mask] = peak_gas_factor * ((self.candidates['dist_to_gas_miles'][left_mask] / optimal_gas_distance) ** left_k)
            # specify bandwidth for decay at distances > optimal_gas_distance
            right_sigma = 0.45
            gas_factor[~left_mask] = peak_gas_factor * np.exp(-1.0 * ((self.candidates['dist_to_gas_miles'][~left_mask] - optimal_gas_distance) / right_sigma) ** 2.0 )
            # define a model that starts highest at approx 0.5 miles and decays up to 1 mile
            # alternative: can make this a piecewise power-left gaussian-rigth function with a peak at optimal_gas_distance
            # decaying to reach 0 at 1 mile
            # old model: gas_factor = np.exp(-0.5 * ((self.candidates['dist_to_gas_miles'] - optimal_gas_distance) / 1)**2)
            # max 5% premium
            multiplier += 0.10 * gas_factor
            
        # Availability of rest facilities correlated to longer refuel times (ie more time spent at station, more h2 dispensed)
        if 'dist_to_rest_miles' in self.candidates.columns:
            # Closer rest areas = more time spent at stations
            optimal_rest_distance = 0.0  # miles
            # define model that starts highest at approx 0 mileand decays up to 1 mile
            rest_factor = np.exp(-0.5 * ((self.candidates['dist_to_rest_miles'] - optimal_rest_distance) / 0.5)**2)
            multiplier += 0.20 * rest_factor # max 20% premium
            
        
        # Network connectivity bonus
        if 'nearby_routes_count' in self.candidates.columns:
            # More routes = more traffic sources
            route_bonus = 0.05 * (self.candidates['nearby_routes_count'] - 1).clip(0, 5) / 5
            multiplier += route_bonus
        
        multiplier = multiplier.clip(0.8, 2.0)  # Reasonable bounds: 80% to 150% of base revenue
        # save the location multipliers for debugging and validation
        
        #self.candidates['location_revenue_multiplier'] = multiplier # Reasonable bounds: 80% to 200% of base revenue
        #self.location_reveue_multiplier = multiplier
        
        return multiplier


    def debug_candidate_structure(self):
        """Debug method to inspect candidate DataFrame structure"""
        
        print("CANDIDATE DATAFRAME DEBUG INFO")
        print("=" * 50)
        print(f"Shape: {self.candidates.shape}")
        print(f"Columns: {list(self.candidates.columns)}")
        print(f"Index: {self.candidates.index[:5].tolist()}...")
        
        print("\nFIRST ROW SAMPLE:")
        if not self.candidates.empty:
            first_row = self.candidates.iloc[0]
            for col in first_row.index:
                value = first_row[col]
                print(f"  {col}: {value} ({type(value).__name__})")
        
        print("\nGEOMETRY ANALYSIS:")
        if 'geometry' in self.candidates.columns:
            geom_sample = self.candidates['geometry'].iloc[0]
            print(f"  Geometry type: {type(geom_sample)}")
            if hasattr(geom_sample, 'x'):
                print(f"  Sample coordinates: ({geom_sample.x}, {geom_sample.y})")
            print(f"  Geometry column exists: Passed")
        else:
            print(f"  Geometry column exists: Failed")
        
        # Check for coordinate columns
        coord_cols = ['longitude', 'latitude', 'lon', 'lat', 'x', 'y']
        found_coords = [col for col in coord_cols if col in self.candidates.columns]
        print(f"  Coordinate columns found: {found_coords}")
        
        print("=" * 50)
            

    def debug_workflow_state(self):
        """Debug the current state of the workflow"""
        
        print("WORKFLOW STATE DEBUG")
        print("=" * 40)
        
        # Check what attributes exist
        attrs_to_check = [
            'candidates', 'highway_network', 'truck_routes', 
            'port_data', 'rest_area_data', 'gas_station_data',
            'config'
        ]
        
        for attr in attrs_to_check:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is None:
                    status = " None"
                elif hasattr(value, 'shape'):
                    status = f" Shape: {value.shape}"
                elif hasattr(value, '__len__'):
                    status = f" Length: {len(value)}"
                else:
                    status = f" Exists ({type(value).__name__})"
            else:
                status = " Missing"
            
            print(f"  {attr}: {status}")
        
        print("=" * 40)
            


    """""
    Depracated method
    def _validate_double_counting(self):
        #Check for correlation between Bayesian weights and location multipliers#
        
        if 'bayesian_weights' not in self.candidates.columns or 'location_revenue_multiplier' not in self.candidates.columns:
            print("Cannot validate - missing Bayesian weights or multipliers")
            return
        
        import scipy.stats as stats
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(
            self.candidates['bayesian_weights'], 
            self.candidates['location_revenue_multiplier']
        )
        
        print(f"Correlation between Bayesian weights and revenue multipliers: {correlation:.3f}")
        print(f"P-value: {p_value:.3f}")
        
        if abs(correlation) > 0.5:
            print("WARNING: High correlation suggests potential double-counting")
            print("Consider removing overlapping factors from revenue multipliers")
        else:
            print("Low correlation suggests factors are largely orthogonal")
            
        return correlation
    """

    def _validate_double_counting(self):
        """
        Check for correlation between Bayesian stop probability and location revenue multipliers.
        
        Validates that we're not double-counting location advantages by checking if:
        - Bayesian probability (P(stop|need) - affects VOLUME)
        - Revenue multipliers (pricing power - affects REVENUE PER UNIT)
        are highly correlated.
        """
        
        # The key insight: We need to compare the FINAL probability (which drives volume)
        # against the revenue multiplier (which drives pricing)
        
        # Check if we have the calculated Bayesian probability
        prob_columns = ['utilization_prob', 'bayesian_probability', 'refuel_probability', 'utilization_probability', 'bayesian_prob']
        prob_col = None
        
        for col in prob_columns:
            if col in self.candidates.columns:
                prob_col = col
                break
        
        if prob_col is None:
            print("Cannot validate - Bayesian probability not calculated")
            print("  Run calculate_utilization_probability() first")
            return
        
        # Check for revenue multiplier column
        if 'location_revenue_multiplier' not in self.candidates.columns:
            print("Cannot validate - location_revenue_multiplier column not found")
            try:
                revenue_multipliers = self._calculate_location_revenue_multipliers()
                self.candidates['location_revenue_multiplier'] = revenue_multipliers
                print("  Revenue multipliers calculated and stored")
            except Exception as e:
                print(f"  Failed to calculate revenue multipliers: {e}")
                return
        
        import scipy.stats as stats
        
        # Remove NaN values for correlation calculation
        mask = (self.candidates[prob_col].notna() & 
                self.candidates['location_revenue_multiplier'].notna())
        
        if mask.sum() < 2:
            print("Insufficient data for correlation analysis")
            return
        
        # Calculate correlation between final probability and revenue multiplier
        correlation, p_value = stats.pearsonr(
            self.candidates.loc[mask, prob_col], 
            self.candidates.loc[mask, 'location_revenue_multiplier']
        )
        
        print(f"\nDouble-counting validation:")
        print(f"  Comparing: {prob_col} (volume driver) vs location_revenue_multiplier (price driver)")
        print(f"  Correlation: {correlation:.3f}")
        print(f"  P-value: {p_value:.3f}")
        
        # Interpretation
        if abs(correlation) > 0.7:
            print("     Warning: HIGH CORRELATION - Potential double-counting detected!")
            print("     Locations with high traffic capture also have high pricing power")
            print("     Consider: Are the same attributes driving both volume AND pricing?")
        elif abs(correlation) > 0.5:
            print("     Flag: MODERATE correlation - Some overlap in value drivers")
            print("     This may be acceptable if justified by business logic")
        else:
            print("     Pass: LOW correlation - Good orthogonality")
            print("     Volume and pricing drivers are reasonably independent")
        
        # Additional diagnostic: Show which attributes contribute most to each
        if hasattr(self, 'bayesian_weights') and correlation > 0.5:
            print("\n  Diagnostic - Top weighted attributes:")
            sorted_weights = sorted(self.bayesian_weights.items(), key=lambda x: x[1], reverse=True)
            for attr, weight in sorted_weights[:2]:
                print(f"    {attr}: {weight:.1%} of stop probability")
            print("  Consider if these same attributes also drive revenue multipliers")




    def _calculate_location_cost_multipliers(self):
        """
        Calculate cost multipliers based on location characteristics.
        Higher multipliers for more expensive locations to build/operate.
        """
        multiplier = np.ones(len(self.candidates))  # Start with 1.0x base cost
        
        # Interchange locations may have higher land costs but easier permitting
        if 'is_interchange' in self.candidates.columns:
            # Slight cost increase for prime real estate, but net positive due to easier development
            multiplier += 0.10 * self.candidates['is_interchange']  # 10% cost increase
        
        # Distance to infrastructure affects development costs
        if 'dist_to_gas_miles' in self.candidates.columns:
            # Very remote locations cost more to develop
            remote_penalty = 0.15 * (self.candidates['dist_to_gas_miles'] / 20).clip(0, 1)
            multiplier += remote_penalty
        
        # Port proximity might increase costs due to industrial land prices
        if 'dist_to_port_miles' in self.candidates.columns:
            port_cost_factor = 0.05 * np.exp(-self.candidates['dist_to_port_miles'] / 30)
            multiplier += port_cost_factor
        
        return multiplier.clip(0.8, 1.4)  # Reasonable bounds: 80% to 140% of base cost

    def _print_economic_summary(self):
        """Print summary of economic calculations"""
        print("\n" + "="*60)
        print("ECONOMIC PROXY SUMMARY")
        print("="*60)
        
        # Revenue metrics
        avg_daily_revenue = self.candidates['daily_revenue'].mean()
        avg_revenue_multiplier = self.candidates['location_revenue_multiplier'].mean()
        print(f"Average daily revenue: ${avg_daily_revenue:,.0f}")
        print(f"Average location revenue multiplier: {avg_revenue_multiplier:.2f}x")
        
        # Cost metrics
        avg_daily_opex = self.candidates['daily_opex'].mean()
        avg_cost_multiplier = self.candidates['location_cost_multiplier'].mean()
        print(f"Average daily OPEX: ${avg_daily_opex:,.0f}")
        print(f"Average location cost multiplier: {avg_cost_multiplier:.2f}x")
        
        # Profitability metrics
        avg_daily_profit = self.candidates['daily_profit'].mean()
        avg_npv = self.candidates['npv_proxy'].mean()
        profitable_count = (self.candidates['npv_proxy'] > 0).sum()
        
        print(f"Average daily profit: ${avg_daily_profit:,.0f}")
        print(f"Average NPV: ${avg_npv:,.0f}")
        print(f"Profitable locations: {profitable_count}/{len(self.candidates)} ({profitable_count/len(self.candidates)*100:.1f}%)")
        
        # Interchange impact
        if 'is_interchange' in self.candidates.columns:
            interchange_count = self.candidates['is_interchange'].sum()
            if interchange_count > 0:
                interchange_npv = self.candidates[self.candidates['is_interchange'] == 1]['npv_proxy'].mean()
                non_interchange_npv = self.candidates[self.candidates['is_interchange'] == 0]['npv_proxy'].mean()
                print(f"\nInterchange Analysis:")
                print(f"  Interchange locations: {interchange_count}")
                print(f"  Avg NPV at interchanges: ${interchange_npv:,.0f}")
                print(f"  Avg NPV at non-interchanges: ${non_interchange_npv:,.0f}")
                print(f"  Interchange premium: {(interchange_npv/non_interchange_npv - 1)*100:.1f}%")
        
        # Top performers
        top_5_npv = self.candidates.nlargest(5, 'npv_proxy')['npv_proxy']
        print(f"\nTop 5 NPV values: ${top_5_npv.min():,.0f} to ${top_5_npv.max():,.0f}")
        
        print("="*60)
    
    
    """"    
    def identify_location_clusters(self, n_clusters=100):
        
        #Depracated method replaced with analyze_regional_clusters
        #Identify top location clusters using DBSCAN clustering.
        
        
        warnings.warn(
            "identify_location_clusters is deprecated. "
            "Use run_iterative_station_selection() for individual station analysis.",
            DeprecationWarning
        )
        print("Identifying location clusters...")
        
        # Filter to profitable locations
        profitable = self.candidates[self.candidates['npv_proxy'] > 0].copy()
        
        if len(profitable) < n_clusters:
            print(f"  Warning: Only {len(profitable)} profitable locations found")
            self.clusters = profitable
            return
        
        # Standardize features for clustering
        features = ['expected_demand_kg_day', 'utilization_rate', 'npv_proxy']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(profitable[features])
        
        # Spatial clustering with DBSCAN
        coords = np.c_[profitable.geometry.x, profitable.geometry.y]
        
        # Add spatial coordinates to features (with lower weight)
        spatial_weight = 0.3
        scaled_coords = coords / (self.config['min_station_spacing_miles'] * 1609.34)
        combined_features = np.hstack([scaled_features, scaled_coords * spatial_weight])
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(combined_features)
        profitable['cluster'] = clustering.labels_
        
        # Calculate cluster statistics
        cluster_stats = []
        
        for cluster_id in range(max(clustering.labels_) + 1):
            cluster_points = profitable[profitable['cluster'] == cluster_id]
            if len(cluster_points) > 0:
                # Create cluster polygon (convex hull)
                cluster_geom = unary_union(cluster_points.geometry).convex_hull
                
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'geometry': cluster_geom,
                    'n_locations': len(cluster_points),
                    'total_demand_kg_day': cluster_points['expected_demand_kg_day'].sum(),
                    'avg_utilization': cluster_points['utilization_rate'].mean(),
                    'total_npv': cluster_points['npv_proxy'].sum(),
                    'avg_payback_years': cluster_points['payback_years'].mean(),
                    'centroid': cluster_geom.centroid
                })
        
        # Rank clusters by total NPV
        cluster_df = pd.DataFrame(cluster_stats)
        cluster_df = cluster_df.sort_values('total_npv', ascending=False)
        
        # Select top n_clusters
        self.clusters = gpd.GeoDataFrame(
            cluster_df.head(n_clusters), 
            geometry='geometry',
            crs="EPSG:3310"
        )
        
        # Calculate tipping values for each cluster
        self._calculate_tipping_values()
    
    """
    
    """
    def _calculate_tipping_values(self):
        
        #Depracated method
        #Calculate economic tipping points using iterative optimization.
        
        print("Calculating tipping values...")
        
        # For each cluster, calculate how much capacity would need to be added
        # before its NPV drops below the next cluster
        
        print("Calculating economic tipping values for station clusters...")
        
        self.clusters['tipping_capacity_kg_day'] = 0
        self.clusters['tipping_npv_threshold'] = 0
        self.clusters['demand_capture_radius_mi'] = self.config.get('service_radius_miles', 2.0)
        
        
        # for each cluster, calculate the tipping point
        #for i in range(len(self.clusters) - 1):
        for i in range(len(self.clusters)):
            current_cluster = self.clusters.iloc[i]
            #next_cluster = self.clusters.iloc[i + 1]
            cluster_stations = self.candidates[self.candidates.get('cluster', -1) == current_cluster.get('cluster_id', i)]
            
            if len(cluster_stations) == 0:
                print(f"  No stations in cluster {current_cluster['cluster_id']}")
                continue
            
            # Calculate the weighted average demand and NPV for cluster
            total_demand = cluster_stations['expected_demand_kg_day'].sum()
            #mean_npv = cluster_stations['npv_proxy'].mean()
            avg_npv_per_kg = cluster_stations['npv_proxy'].sum() / total_demand if total_demand > 0 else 0
            
            ##### Depracated placeholder method ######
            # Simplified: capacity that would reduce utilization to unprofitable
            #current_demand = current_cluster['total_demand_kg_day']
            
            # Tipping point is where added capacity reduces NPV below next cluster
            # This is a simplification - in reality would need iterative calculation
            #tipping_capacity = current_demand * 0.5  # 50% oversupply typically tips economics
            
            #self.clusters.loc[self.clusters.index[i], 'tipping_capacity_kg_day'] = tipping_capacity
            
            ##### END Depracated placeholder method ######
             
            # Find next best alternative (best cluster or best non-clustered station)
            if i < len(self.clusters) - 1:
                next_cluster = self.clusters.iloc[i + 1]    
                next_best_npv = self.clusters.iloc[i + 1]['total_npv']
                
            else:
                
                # No next cluster, use the best non-clustered station
                non_clustered = self.candidates[self.candidates.get('cluster', -1) == -1]
                next_best_npv = non_clustered['npv_proxy'].max() if len(non_clustered) > 0 else 0
            
            # Calculate tipping capacity usimg economic optimization
            # pass to iterative optimization method
            tipping_capacity = self._optimize_tipping_capacity(
                current_demand = total_demand,
                current_npv = current_cluster['total_npv'],
                next_best_npv = next_best_npv,
                avg_npv_per_kg = avg_npv_per_kg
            )
            
            self.clusters.loc[self.clusters.index[i], 'tipping_capacity_kg_day'] = tipping_capacity
            self.clusters.loc[self.clusters.index[i], 'tipping_npv_threshold'] = next_best_npv
    """        
            
            
    """        
    def _optimize_tipping_capacity(self, current_demand, current_npv, next_best_npv, avg_npv_per_kg):
        
        
        #Depracated method.
        #Returns simple estimate
        
        
        return current_demand * 2.0
    
    """
    
    
    
    def run_iterative_station_selection(self, max_stations=200, min_npv_threshold=0):
        """
        Iteratively select individual stations with demand cannibalization.
        Optimized version with backward compatibility.
        """
        import time
        start_time = time.time()
        
        print(f"\nRunning iterative station selection (max {max_stations} stations)...")
        print(f"Capacity optimization range: 500-12,000 kg/day")
        
        # Initialize capacity optimization cache
        if not hasattr(self, '_capacity_optimization_cache'):
            self._capacity_optimization_cache = {}
        
        # Store original state (exactly as before)
        original_candidates = self.candidates.copy()
        original_capacity = self.config['station_capacity_kg_per_day']
        original_existing = self.existing_stations.copy() if hasattr(self, 'existing_stations') else pd.DataFrame()
        
        # Work with profitable candidates
        candidates = self.candidates[self.candidates['npv_proxy'] > min_npv_threshold].copy()
        
        self._current_working_candidates = candidates
        
        if len(candidates) == 0:
            print("No profitable candidates found")
            return pd.DataFrame()
        
        # Pre-calculate distance matrix for cannibalization
        from scipy.spatial.distance import cdist
        candidate_coords = np.c_[candidates.geometry.x, candidates.geometry.y]
        distance_matrix = cdist(candidate_coords, candidate_coords)
        idx_to_position = {idx: i for i, idx in enumerate(candidates.index)}
        
        # Verify competition graph (as before)
        #if self._competition_graph is None:
            #print("Warning: Competition graph not initialized.")
            # Recheck graph initialization
        
        # Verify competition graph exists 
        if self._competition_graph is None:
            print("Warning: Competition graph not initialized. This should have been done in calculate_bayesian_probabilities.")
            print("Initializing now with existing infrastructure...")
            self._competition_graph = self.CompetitionGraph(self.config)
            
            # Add existing infrastructure to graph (if any)
            if len(original_existing) > 0:
                for idx, existing in original_existing.iterrows():
                    self._competition_graph.add_or_update_station(
                        f"existing_{idx}", existing, 'existing'
                    )
        else:
            print(f"Using existing competition graph with {len(self._competition_graph.nodes)} nodes and {len(self._competition_graph.edges)} edges")
        
        
        # Verify NPV exists
        if 'npv_proxy' not in self.candidates.columns:
            raise ValueError("Economic proxy not calculated. Run calculate_economic_proxy() first.")
   
        
        # Initialize results tracking
        selected_stations = []
        remaining_candidates = candidates.copy()
        cumulative_investment = 0
        cumulative_npv = 0
        
        # Add iteration tracking structures
        iteration_history = {}  # {iteration: {'station_id': idx, 'base_demand': X, 'current_demand': Y}}
        station_demand_matrix = {}  # {station_idx: {iteration: demand}} - for final matrix
        
        
        # Service radius
        service_radius_m = self.config.get('service_radius_miles', 2.0) * 1609.34
        
        # Iterate through station selection
        for iteration in range(min(max_stations, len(candidates))):
            if len(remaining_candidates) == 0:
                break
            
            print(f"\n  Iteration {iteration + 1}:")
            
            # Update candidate economics
            remaining_candidates = self._update_candidate_economics(remaining_candidates)
            
            # Filter for profitable stations with npv > 0
            profitable = remaining_candidates[remaining_candidates['current_npv'] > 0]
            if len(profitable) == 0:
                print("  No more profitable candidates")
                break
            
            # Skipping filtration for minimum distance requirement step 
            # (applied to valid subset of remaining candidates except in first iteration)
            
            # Select best station based on npv
            best_idx = profitable['current_npv'].idxmax()
            best_station = profitable.loc[best_idx]
            
            # Get actual next best stationwith flexible capacity optimization
            if len(profitable) > 1:
                next_best_idx = profitable['current_npv'].nlargest(2).index[1]
                next_best_station = profitable.loc[next_best_idx]
            else:
                next_best_station = None
            
            # Next best NPV
            #next_best_npv = profitable['current_npv'].nlargest(2).iloc[-1] if len(profitable) > 1 else 0
            
            # Optimized tipping metrics calculation
            tipping_metrics = self._calculate_station_tipping_metrics_fast(
                best_station, next_best_station, remaining_candidates
            )
            
            print(f"  Selected: Station {best_idx} at ({best_station.geometry.x:.0f}, {best_station.geometry.y:.0f})")
            print(f"  Expected demand: {best_station['expected_demand_kg_day']:.0f} kg/day")
            print(f"  Optimal capacity: {tipping_metrics['optimal_capacity_kg_day']} kg/day")
            print(f"  Optimal NPV: ${tipping_metrics['optimal_npv']:,.0f}")
            print(f" Breakeven demand: {tipping_metrics['breakeven_demand_kg_day']:.0f} kg/day")
            if 'tipping_capacity_kg_day' in tipping_metrics:
                print(f" Tipping capacity: {tipping_metrics['tipping_capacity_kg_day']:.0f} kg/day")
            
            # Update cumulative metrics
            cumulative_investment += tipping_metrics['optimal_capex']
            cumulative_npv += tipping_metrics['optimal_npv']
            
            # Calculate current demand for this station given current network state
            # This is a "what-if" calculation - what would its demand be with current competition
            
            current_demand = self._competition_graph.calculate_market_share(best_idx) * best_station.get(
                'competition_agnostic_adjusted_demand', best_station['initial_demand_post_existing'] * best_station['p_need_fuel'] * best_station['p_stop_given_need']
            )
            
            # Store iteration data
            iteration_history[iteration] = {
                'station_id': best_idx,
                'base_demand': best_station.get('competition_agnostic_adjusted_demand', best_station['initial_demand_post_existing'] * best_station['p_need_fuel'] * best_station['p_stop_given_need']),
                'current_demand': current_demand,
                'tipping_capacity_kg_day': tipping_metrics['tipping_capacity_kg_day'],
            }
            
            # Initialize station's demand history 
            if best_idx not in station_demand_matrix:
                station_demand_matrix[best_idx] = {i: 0.0 for i in range(max_stations)}
            station_demand_matrix[best_idx][iteration] = current_demand
            
            # Update previously selected stations' demands 
            for prev_iteration, prev_data in iteration_history.items():
                if prev_iteration < iteration:
                    station_id = prev_data['station_id']
                    if station_id in self._competition_graph.nodes:
                        market_share = self._competition_graph.calculate_market_share(station_id)
                        base = prev_data.get('competition_agnostic_adjusted_demand', 
                                            prev_data['base_demand'])
                        station_demand_matrix[station_id][iteration] = base * market_share
        
            
            
            
            # Store station record 
            station_record = {
                'iteration': iteration + 1,
                'station_id': len(selected_stations) + 1,
                'idx': best_idx,
                'geometry': best_station.geometry,
                'expected_demand_kg_day': best_station['expected_demand_kg_day'],
                'total_demand_kg_day': best_station['expected_demand_kg_day'],
                'total_demand': best_station['expected_demand_kg_day'],
                'breakeven_demand_kg_day': tipping_metrics['breakeven_demand_kg_day'],
                'tipping_capacity_kg_day': tipping_metrics['tipping_capacity_kg_day'],
                'npv': tipping_metrics['optimal_npv'],
                'utilization': tipping_metrics['optimal_utilization'],
                'capacity_kg_day': tipping_metrics['optimal_capacity_kg_day'],
                'demand_cushion_pct': tipping_metrics['demand_cushion_pct'],
                'cumulative_investment': cumulative_investment,
                'cumulative_npv': cumulative_npv,
                'total_investment_at_tipping': tipping_metrics['optimal_capex'],    
                'cumulative_investment': cumulative_investment,
                'cumulative_npv': cumulative_npv,
                
                # Include all rich attributes from economic proxy model
                
                'location_revenue_multiplier': best_station['location_revenue_multiplier'],
                'location_cost_multiplier': best_station['location_cost_multiplier'],
                'daily_revenue': best_station.get('daily_revenue', 0),
                'daily_opex': best_station.get('daily_opex', 0),
                'capex': best_station.get('capex', self.config['station_capex']),
                'daily_profit': best_station.get('daily_profit', 0),
                'payback_years': best_station.get('payback_years', 999),
                'total_cost': best_station.get('total_cost', self.config['station_capex']),
                'irr': best_station.get('irr', 0),
                'is_interchange': best_station.get('is_interchange', False),
                'interchange_score': best_station.get('interchange_score', 0),
                'dist_to_gas_miles': best_station.get('dist_to_gas_miles', 0),
                'dist_to_rest_miles': best_station.get('dist_to_rest_miles', 0),
                'nearby_routes_count': best_station.get('nearby_routes_count', 0),
                
                # Include all tipping metrics
                **tipping_metrics,
            }
            
            selected_stations.append(station_record)
            
            # OPTIMIZATION 3: Vectorized demand cannibalization
            remaining_candidates = self._apply_demand_cannibalization_fast(
                remaining_candidates, best_station, service_radius_m,
                distance_matrix, idx_to_position
            )
            
            # Remove selected station
            remaining_candidates = remaining_candidates.drop(best_idx)

            # Progress update
            if (iteration + 1) % 10 == 0:
                print(f"\n  Progress: {iteration + 1} stations, cumulative NPV: ${cumulative_npv:,.0f}")
        
        # Final network demand update
        #if self._competition_graph:
            #self._competition_graph.update_network_demands()
            
            # Update selected stations with final equilibrium demands
            #for record in selected_stations:
                #station_id = record['idx']
                #if station_id in self._competition_graph.nodes:
                    #node_data = self._competition_graph.nodes[station_id]
                    #record['final_equilibrium_demand'] = node_data['demand']
                    #record['final_market_share'] = node_data['market_share']
                    
        
        # Create iteration tracking DataFrame
        iteration_data = []
        for station_id, demand_history in station_demand_matrix.items():
            for iteration_num in range(len(iteration_history)):
                iteration_data.append({
                    'iteration': iteration_num,
                    'station_id': station_id,
                    'base_demand': iteration_history.get(iteration_num, {}).get('base_demand', 0) 
                                if iteration_history.get(iteration_num, {}).get('station_id') == station_id else 0,
                    'current_demand': demand_history.get(iteration_num, 0),
                    'was_selected': iteration_history.get(iteration_num, {}).get('station_id') == station_id
                })
        
        iteration_df = pd.DataFrame(iteration_data)

        # Create pivot table for easier analysis
         # Create pivot table for analysis 
        if len(iteration_data) > 0:
            demand_evolution_matrix = iteration_df.pivot_table(
                index='station_id',
                columns='iteration', 
                values='current_demand',
                fill_value=0
            )
        else:
            demand_evolution_matrix = pd.DataFrame()

        # Store both in class for access
        self.iteration_history = iteration_history
        self.demand_evolution_matrix = demand_evolution_matrix
        
        # Create results GeoDataFrame
        results_gdf = gpd.GeoDataFrame(selected_stations, crs=candidates.crs)
        results_gdf['selection_order'] = results_gdf.index  # Order of selection
        
        # Add iteration history as attribute
        results_gdf.attrs['iteration_history'] = iteration_history
        results_gdf.attrs['demand_evolution'] = demand_evolution_matrix

        
        # Store as instance attribute for later use
        self.iterative_selection_results = results_gdf
        
        # Restore original state
        self.candidates = original_candidates
        self.config['station_capacity_kg_per_day'] = original_capacity
        self.existing_stations = original_existing
        
        print(f"\n Iterative selection complete: {len(results_gdf)} stations selected")
        print(f"Total NPV: ${cumulative_npv:,.0f}")
        print(f"Total Investment: ${cumulative_investment:,.0f}")
        print(f"Average capacity: {results_gdf['capacity_kg_day'].mean():.0f} kg/day")
        
        # Store competition graph for analysis
        self.competition_graph = self._competition_graph
        
        # Clear capacity optimization cache
        self._capacity_optimization_cache = {}
        elapsed_time = time.time() - start_time
        
        
        return results_gdf
    

    
    def _update_candidate_economics(self, candidates_df):
        """
        Recalculate NPV for candidates based on current demand.
        """
        # save state
        original_candidates = self.candidates
        
        # update candidates
        self.candidates = candidates_df.copy()
        
        # Recalculate NPV for each candidate
        self.calculate_economic_proxy()
        
        # Copy NPV to columns for iteration
        self.candidates['current_npv'] = self.candidates['npv_proxy']
        result = self.candidates.copy()
        
        # Restore original state
        self.candidates = original_candidates
        
        return result
    
    
    def _find_optimal_capacity_cached(self, station):
        """
        Find optimal capacity for a station with caching to avoid redundant calculations.
        """
        # Create cache key
        station_id = station.name if hasattr(station, 'name') else id(station)
        cache_key = (
            station_id, 
            round(station['expected_demand_kg_day'], 1),
            round(station.get('location_revenue_multiplier', 1.0), 2),
            round(station.get('location_cost_multiplier', 1.0), 2)
        )
        
        # Check cache
        if hasattr(self, '_capacity_optimization_cache') and cache_key in self._capacity_optimization_cache:
            return self._capacity_optimization_cache[cache_key]
        
        # Calculate optimal capacity
        result = self._find_optimal_capacity(station)
        
        # Cache result
        if not hasattr(self, '_capacity_optimization_cache'):
            self._capacity_optimization_cache = {}
        
        self._capacity_optimization_cache[cache_key] = result
        
        # Limit cache size
        if len(self._capacity_optimization_cache) > 500:
            # Remove oldest entries
            keys_to_keep = list(self._capacity_optimization_cache.keys())[-250:]
            self._capacity_optimization_cache = {
                k: self._capacity_optimization_cache[k] for k in keys_to_keep
            }
        
        return result
    
    
    def _find_optimal_capacity(self, station):
        """
        Find optimal capacity for a station by testing different capacity options.
        """
        # Save original capacity
        original_capacity = self.config['station_capacity_kg_per_day']
        
        # Test fewer capacities initially
        # Coarse search: every 1000kg instead of 500kg
        # Define flexible range of capacities at 500kg/day intervals
        min_capacity = self.config.get('min_iteration_capacity_kg_per_day', 500.0)
        max_capacity = self.config.get('max_iteration_capacity_kg_per_day', 12000.0)
        capacity_step = self.config.get('capacity_step_size_kg_per_day', 500.0)
        coarse_capacities = np.arange(min_capacity, max_capacity + capacity_step, capacity_step)
        
        # Pre-create single station DataFrame
        station_gdf = gpd.GeoDataFrame(
            [station], 
            geometry=[station.geometry],
            crs=self.candidates.crs
        )
        
        # Test coarse capacities
        coarse_results = []
        for capacity in coarse_capacities:
            self.config['station_capacity_kg_per_day'] = capacity
            
            # Temporary assignment
            saved_candidates = self.candidates
            self.candidates = station_gdf
            self.calculate_economic_proxy()
            
            # Extract result immediately
            result = self.candidates.iloc[0]
            self.candidates = saved_candidates
            
            coarse_results.append({
                'capacity': capacity,
                'npv': result['npv_proxy'],
                'expected_demand_kg_day': station['expected_demand_kg_day'],
                'utilization': min(station['expected_demand_kg_day'] / capacity, 1.0),
                'capex': result.get('adjusted_capex', result.get('capex', (self.config['base_capex_multiplier'] * self.config['station_capex']))),
                'daily_revenue': result.get('daily_revenue', 0),
                'daily_opex': result.get('daily_opex', 0),
                'daily_profit': result.get('daily_profit',0),
                'payback_years': result.get('payback_years', 999),
                'location_revenue_multiplier': result['location_revenue_multiplier'],
                'location_cost_multiplier': result['location_cost_multiplier'],
                'irr': result['irr_proxy'],
                'total_cost': result['total_cost'],
                
            })
        
        # Find coarse optimal
        coarse_df = pd.DataFrame(coarse_results)
        coarse_optimal_idx = coarse_df['npv'].idxmax()
        coarse_optimal = coarse_df.iloc[coarse_optimal_idx]
        
        # Refined search only in promising range
        refined_min = max(500, coarse_optimal['capacity'] - 1500)
        refined_max = min(12000, coarse_optimal['capacity'] + 1500)
        refined_capacities = np.arange(refined_min, refined_max + 100, 100)
        
        # Test refined capacities
        refined_results = []
        for capacity in refined_capacities:
            if capacity in coarse_capacities:
                continue
                
            self.config['station_capacity_kg_per_day'] = capacity
            
            saved_candidates = self.candidates
            self.candidates = station_gdf
            self.calculate_economic_proxy()
            
            result = self.candidates.iloc[0]
            self.candidates = saved_candidates
            
            refined_results.append({
                'capacity': capacity,
                'npv': result['npv_proxy'],
                'utilization': min(station['expected_demand_kg_day'] / capacity, 1.0),
                'capex': result.get('adjusted_capex', result.get('capex', self.config['base_capex_multiplier'] * self.config['station_capex'])),
                # Extract all needed fields
                'daily_revenue': result.get('daily_revenue', 0),
                'daily_opex': result.get('daily_opex', 0),
                'daily_profit': result.get('daily_profit', 0),
                'payback_years': result.get('payback_years', 999),
                'location_revenue_multiplier': result['location_revenue_multiplier'],
                'location_cost_multiplier': result['location_cost_multiplier'],
                'irr': result.get('irr', 0),
                'total_cost': result.get('total_cost', 0),
            })
        
        # Combine and find optimal
        all_results = refined_results + [{k: v for k, v in r.items()} for r in coarse_results]
        all_results_df = pd.DataFrame(all_results)
        optimal_idx = all_results_df['npv'].idxmax()
        optimal_result = all_results_df.iloc[optimal_idx].to_dict()
        
        #print(f"  Refined optimal capacity: {optimal_result['capacity']} kg/day")
        
        # Restore original capacity
        self.config['station_capacity_kg_per_day'] = original_capacity
        
        # Return comprehensive result
        return {
            # Optimal capacity results
            'capacity': optimal_result['capacity'],
            'npv': optimal_result['npv'],
            'utilization': optimal_result['utilization'],
            'capex': optimal_result['capex'],
            'all_results_df': all_results_df,
            'idx': optimal_idx,
            'irr': optimal_result['irr'],
            'daily_revenue': optimal_result['daily_revenue'],
            'daily_opex': optimal_result['daily_opex'],
            'daily_profit': optimal_result['daily_profit'],
            'payback_years': optimal_result['payback_years'],
            'location_revenue_multiplier': optimal_result['location_revenue_multiplier'],
            'location_cost_multiplier': optimal_result['location_cost_multiplier'],
            'total_cost': optimal_result['total_cost']
            
        }
        


    def _find_capacity_where_npv_equals(self, station, target_npv, all_results_df):
        """
        Find the capacity at which this station's NPV equals the target NPV.
        This is the true tipping point where another location becomes preferable.
        """
        if target_npv <= 0:
            # No viable alternative - tipping point is where NPV = 0
            zero_crossing = all_results_df[all_results_df['npv'] <= 0]
            if len(zero_crossing) > 0:
                # Find the last positive NPV capacity
                positive_npv = all_results_df[all_results_df['npv'] > 0]
                if len(positive_npv) > 0:
                    return float(positive_npv['capacity'].max())
            return float(all_results_df['capacity'].max())
        
        # Sort by capacity
        sorted_results = all_results_df.sort_values('capacity')
        
        # Find where NPV crosses target_npv
        above_target = sorted_results[sorted_results['npv'] > target_npv]
        below_target = sorted_results[sorted_results['npv'] <= target_npv]
        
        if len(above_target) > 0 and len(below_target) > 0:
            # Find adjacent points for interpolation
            last_above = above_target.iloc[-1]
            first_below = below_target.iloc[0]
            
            # Linear interpolation
            capacity_diff = first_below['capacity'] - last_above['capacity']
            npv_diff = last_above['npv'] - first_below['npv']
            
            if npv_diff > 0:
                fraction = (last_above['npv'] - target_npv) / npv_diff
                tipping_capacity = last_above['capacity'] + fraction * capacity_diff
                return float(tipping_capacity)
        
        # If no crossover found
        if len(above_target) == 0:
            # All NPVs are below target
            return float(sorted_results['capacity'].min())
        else:
            # All NPVs are above target
            return float(sorted_results['capacity'].max())


    def _calculate_station_tipping_metrics_fast(self, station, next_best_station, all_candidates):
        """
        Optimized tipping metrics calculation with minimal state changes.
        Calculate comprehensive tipping metrics for a station.
        Find optimal capacity for each station and calculate realistic risk metrics.
        
        """
        # Store state
        original_capacity = self.config['station_capacity_kg_per_day']
        
        # # Get optimal capacity for current station
        print(f"  Finding optimal capacity for current station...")
        current_optimal = self._find_optimal_capacity_cached(station)
        
        # Get optimal capacity for next best station if it exists
        if next_best_station is not None:
            print(f"  Finding optimal capacity for next best station...")
            next_best_optimal = self._find_optimal_capacity_cached(next_best_station)
            true_next_best_npv = next_best_optimal['npv']
        else:
            true_next_best_npv = 0
            next_best_optimal = None
            
            # Calculate true tipping capacity
        true_tipping_capacity = self._find_capacity_where_npv_equals(
            station,
            target_npv=true_next_best_npv,
            all_results_df=current_optimal['all_results_df']
        )
        
        
        # Calculate breakeven demand using demand steps
        self.config['station_capacity_kg_per_day'] = current_optimal['capacity']
        breakeven_demand = self._calculate_breakeven_demand(
            station,
            current_optimal['capacity'],
            self.candidates  # Use full candidates set which has already been reset with saved_candidates
        )
        
        # Calculate demand cushion 
        # Assuming selected candidates have positive NPV, expected_demand_kg_day > breakeven_demand
        # Demand cushion measures how much demand can be lost before NPV turns negative - a form of risk
        demand_cushion_pct = ((station['expected_demand_kg_day'] - breakeven_demand) / 
                            station['expected_demand_kg_day']) * 100 if station['expected_demand_kg_day'] > 0 else 0
        
            
        
        # Calculate remaining metrics (unchanged)
        service_radius_m = self.config.get('service_radius_miles', 2.0) * 1609.34
        
        # Count competitors
        station_coord = np.array([[station.geometry.x, station.geometry.y]])
        all_coords = np.c_[all_candidates.geometry.x, all_candidates.geometry.y]
        
        from scipy.spatial import cKDTree
        tree = cKDTree(all_coords)
        nearby_indices = tree.query_ball_point(station_coord[0], service_radius_m)
        competitors_within_radius = len(nearby_indices) - 1
        
        # Market share from competition graph
        station_id = station.name if hasattr(station, 'name') else station.index
        demand_capture_probability = self._competition_graph.calculate_market_share(station_id)
        
        # Calculate minimum viable capacity (what was previously called tipping capacity)
        min_viable_npv = self.config.get('min_viable_npv', 0)
        viable_results = current_optimal['all_results_df'][current_optimal['all_results_df']['npv'] >= min_viable_npv]
        
        if len(viable_results) > 0:
            min_viable_capacity = viable_results['capacity'].min()
        else:
            min_viable_capacity = current_optimal['all_results_df']['capacity'].max()
        
        
        # Calculate capacity sensitivity 
        capacity_sensitivity = self._calculate_capacity_sensitivity(
            current_optimal['all_results_df'], 
            current_optimal['all_results_df'].iloc[current_optimal['idx']]
            )
    
        
        # Restore config
        self.config['station_capacity_kg_per_day'] = original_capacity
        
        # Return all metrics 
        return {

            # Optimal capacity results
            'optimal_capacity_kg_day': current_optimal['capacity'],
            'optimal_npv': current_optimal['npv'],
            'optimal_utilization': current_optimal['utilization'],
            'optimal_capex': current_optimal['capex'],
            'tipping_capacity_kg_day': true_tipping_capacity,
            
            # Risk metrics
            'breakeven_demand_kg_day': breakeven_demand,
            'demand_cushion_pct': demand_cushion_pct,
            'min_viable_utilization': breakeven_demand / current_optimal['capacity'] if current_optimal['capacity'] > 0 else 0,
            'min_viable_capacity_kg_day': min_viable_capacity,
            
            
            
            # Economic metrics
            'daily_revenue_optimal': current_optimal['daily_revenue'],
            'daily_opex_optimal': current_optimal['daily_opex'],
            'daily_profit_optimal': current_optimal['daily_profit'],
            'payback_years_optimal': current_optimal['payback_years'],
            'location_revenue_multiplier_optimal': current_optimal['location_revenue_multiplier'],
            'location_cost_multiplier_optimal': current_optimal['location_cost_multiplier'],
            'irr_optimal': current_optimal['irr'],
            'total_cost_optimal': current_optimal['total_cost'],
            
            # Capacity analysis
            'capacities_tested': len(current_optimal['all_results_df']),
            'capacity_range_tested': f"{current_optimal['all_results_df']['capacity'].min()}-{current_optimal['all_results_df']['capacity'].max()} kg/day",
            'default_capacity_npv': current_optimal['all_results_df'][current_optimal['all_results_df']['capacity'] == original_capacity]['npv'].iloc[0] 
                if original_capacity in current_optimal['all_results_df']['capacity'].values else None,
            'npv_improvement_vs_default': current_optimal['npv'] - 
                (current_optimal['all_results_df'][current_optimal['all_results_df']['capacity'] == original_capacity]['npv'].iloc[0]
                 if original_capacity in current_optimal['all_results_df']['capacity'].values else 0),
            
            
            # Competitive landscape metrics
            'competitors_within_radius': competitors_within_radius,
            'demand_capture_probability': demand_capture_probability,
            'competitive_advantage_npv': current_optimal['npv'] - true_next_best_npv,
            
            # Sensitivity analysis
            'capacity_sensitivity': capacity_sensitivity
            
        }

            

    def _calculate_breakeven_demand(self, station, capacity, original_candidates):
        """
        Calculate breakeven demand for a station based on capacity and economic model.
        Uses flexible demand steps to find the point where NPV = 0.
        Uses average refuel amount per truck per refuel as step size for realistic granularity.
        
        """
        
        # Use config refuel amount for step size
        demand_step = self.config.get('avg_refuel_amount_kg', 60.0)
        
        # Test demands from 0 to capacity in demand_step increments
        test_demands = np.arange(0, capacity + demand_step, demand_step)
        npv_results = []
        
        
        print(f" Finding breakeven demand for {station.name} with capacity {capacity} kg/day...")
        
        for test_demand in test_demands:
            
            # create testt candidate with adjusted demand
            test_station = station.copy()
            test_station['expected_demand_kg_day'] = test_demand
            
            temp_candidates_gdf = gpd.GeoDataFrame(
                [test_station], 
                geometry=[station.geometry],
                crs=original_candidates.crs
            )
            
            # Calculate NPV for this test candidate
            self.candidates = temp_candidates_gdf
            self.calculate_economic_proxy()
            test_npv = self.candidates.iloc[0]['npv_proxy']
            
            npv_results.append({
                'demand': test_demand,
                'npv': test_npv
            })
            
            # Early exit if NPV has crossed zero
            if len(npv_results) > 1 and npv_results[-2]['npv'] < 0 and npv_results[-1]['npv'] >= 0:
                break
            
            # Find where NPV crosses zero
            npv_df = pd.DataFrame(npv_results)
            
            # Find the demand where NPV changes from negative to positive
            positive_npv = npv_df[npv_df['npv'] >= 0]
            
            if len(positive_npv) == 0:
                # Station is never profitable
                return capacity # Return full capacity as breakeven demand in worst case 
            
            elif len(positive_npv) == len(npv_df):
                # Station is always profitable
                return min(positive_npv['demand'].min(), 0) # Should retun 0 in this case, meaning station is profitable even at 0 demand in best case
            
            else:
                # Normal case - find the first positive NPV
                
                breakeven_idx = positive_npv.index[0]
                
                if breakeven_idx > 0:
                    # Interpolate between the last negative and first positive NPV
                    neg_demand = npv_df.iloc[breakeven_idx - 1]['demand']
                    neg_npv = npv_df.iloc[breakeven_idx - 1]['npv']
                    pos_demand = npv_df.iloc[breakeven_idx]['demand']
                    pos_npv = npv_df.iloc[breakeven_idx]['npv']
                    
                    # Interpolate to find breakeven demand (use linear interpolation)
                    breakeven = neg_demand + (pos_demand - neg_demand) * (0 - neg_npv) / (pos_npv - neg_npv)
                    
                    return max(breakeven, 0)  # Ensure non-negative demand
                
                else:
                    # No negative NPV before breakeven
                    return positive_npv.iloc[0]['demand']  # Return first positive demand
                
    
    
    def _calculate_capacity_sensitivity(self, results_df, optimal):
        """
        Calculate sensitivity of NPV to capacity changes.
        
        """
        
        # Calculate NPV variance around optimal capacity
        # Use a small window around the optimal capacity
        capacity_window = results_df[
            (results_df['capacity'] >= optimal['capacity'] - 1000) &
            (results_df['capacity'] <= optimal['capacity'] + 1000)
        ]
        
        if len(capacity_window) > 1:
            npv_std = capacity_window['npv'].std()
            npv_mean = capacity_window['npv'].mean()
            cv = npv_std / abs(npv_mean) if npv_mean != 0 else 0 # capacity variance
            
            if cv < 0.1:
                sensitivity = "Low - robust to capacity choice"
                
            elif cv < 0.25:
                sensitivity = "Medium - moderate sensitivity to capacity choice"
                
            else:
                sensitivity = "High - sensitive to capacity choice; careful sizing needed. Consider refining capacity more precisely."
                
        else:
            sensitivity = "Unknown - not enough data to assess sensitivity"
                
        
        return sensitivity
    
    
    
            

    def _apply_demand_cannibalization_old(self, competition_candidates, new_station):
        """
        Apply demand reduction to nearby stations.
        Uses a gravity model and calls the unified calculate_competition_adjusted_demand method
        
        Parameters:
        - remaining_candidates: DataFrame of candidates still to be selected wherein those in the vicinity of the new station will have their demand adjusted
        - new_station: The station just selected (Series)
        - service_radius_m: Service radius in meters (kept for compatibility)
        
        Returns:
        - Updated remaining_candidates with adjusted demand

        """
        
        if len(competition_candidates) == 0:
            return competition_candidates
        
    
        updated_candidates = competition_candidates.copy()
        
        # Add all candidates to graph if not already there
        #for idx, candidate in updated_candidates.iterrows():
            #self._competition_graph.add_or_update_station(idx, candidate, 'candidate')
            
        new_station_df = pd.DataFrame([new_station], index=[new_station.name if hasattr(new_station, 'name') else id(new_station)])    
            
        # Add new station to graph if not there
        new_station_id = new_station.name if hasattr(new_station, 'name') else id(new_station)
        if new_station_id not in self._competition_graph.nodes:
            self._competition_graph.add_or_update_station(new_station_id, new_station, 'selected')
        else:
            # Update type from 'candidate' to 'selected'
            self._competition_graph.nodes[new_station_id]['type'] = 'selected'    
        
        # Create edges between all stations in the network
        #all_station_ids = list(self._competition_graph.nodes.keys())
        #for i, id1 in enumerate(all_station_ids):
            #for id2 in all_station_ids[i+1:]:
                #self._competition_graph.add_competition_edge(id1, id2)
        
        
        # Update each remaining candidate's demand using the graph
        for comp_idx in updated_candidates.index:
            competitor_candidate = updated_candidates.loc[comp_idx]
            
            # Calculate this candidates adjusted demand based on competition with new_station with graph-aware parameters
            adjusted_demand = self.calculate_competition_adjusted_demand(
                competitor_candidate, # the competitor candidate station whose demand we are adjusting
                new_station_df, # the selected station that is now competing 
                include_existing=True,
                include_candidates=True
            )
            
            updated_candidates.loc[comp_idx, 'expected_demand_kg_day'] = adjusted_demand
            
        return updated_candidates
    
    
    """""
    def run_iterative_station_selection(self, max_stations=200, min_npv_threshold=0):
        
        #Iteratively select individual stations with demand cannibalization.
        #Optimized version with backward compatibility.
        
        import time
        start_time = time.time()
        
        print(f"\nRunning iterative station selection (max {max_stations} stations)...")
        print(f"Capacity optimization range: 500-12,000 kg/day")
        
        # Store original state (exactly as before)
        original_candidates = self.candidates.copy()
        original_capacity = self.config['station_capacity_kg_per_day']
        original_existing = self.existing_stations.copy() if hasattr(self, 'existing_stations') else pd.DataFrame()
        
        # Work with profitable candidates
        candidates = self.candidates[self.candidates['npv_proxy'] > min_npv_threshold].copy()
        
        if len(candidates) == 0:
            print("No profitable candidates found")
            return pd.DataFrame()
        
        # OPTIMIZATION 1: Pre-calculate distance matrix for cannibalization
        from scipy.spatial.distance import cdist
        candidate_coords = np.c_[candidates.geometry.x, candidates.geometry.y]
        distance_matrix = cdist(candidate_coords, candidate_coords)
        idx_to_position = {idx: i for i, idx in enumerate(candidates.index)}
        
        # Verify competition graph (as before)
        if self._competition_graph is None:
            print("Warning: Competition graph not initialized.")
            # Recheck graph initialization
        
        # Results tracking
        selected_stations = []
        remaining_candidates = candidates.copy()
        cumulative_investment = 0
        cumulative_npv = 0
        
        # Add iteration tracking structures
        iteration_history = {}  # {iteration: {'station_id': idx, 'base_demand': X, 'current_demand': Y}}
        station_demand_matrix = {}  # {station_idx: {iteration: demand}} - for final matrix
        
        
        # Service radius
        service_radius_m = self.config.get('service_radius_miles', 2.0) * 1609.34
        
        # Main iteration loop
        for iteration in range(min(max_stations, len(candidates))):
            if len(remaining_candidates) == 0:
                break
            
            print(f"\n  Iteration {iteration + 1}:")
            
            # Update economics (unchanged)
            remaining_candidates = self._update_candidate_economics(remaining_candidates)
            
            # Filter profitable
            profitable = remaining_candidates[remaining_candidates['current_npv'] > 0]
            if len(profitable) == 0:
                print("  No more profitable candidates")
                break
            
            # Select best
            best_idx = profitable['current_npv'].idxmax()
            best_station = profitable.loc[best_idx]
            
            # Next best NPV
            next_best_npv = profitable['current_npv'].nlargest(2).iloc[-1] if len(profitable) > 1 else 0
            
            # OPTIMIZATION 2: Optimized tipping metrics calculation
            tipping_metrics = self._calculate_station_tipping_metrics_fast(
                best_station, next_best_npv, remaining_candidates
            )
            
            print(f"  Selected: Station {best_idx}")
            print(f"  Expected demand: {best_station['expected_demand_kg_day']:.0f} kg/day")
            print(f"  Optimal capacity: {tipping_metrics['optimal_capacity_kg_day']} kg/day")
            print(f"  Optimal NPV: ${tipping_metrics['optimal_npv']:,.0f}")
            
            # Update cumulative metrics
            cumulative_investment += tipping_metrics['optimal_capex']
            cumulative_npv += tipping_metrics['optimal_npv']
            
            # Store station record (unchanged)
            station_record = {
                'iteration': iteration + 1,
                'station_id': len(selected_stations) + 1,
                'idx': best_idx,
                'geometry': best_station.geometry,
                'expected_demand_kg_day': best_station['expected_demand_kg_day'],
                'npv': tipping_metrics['optimal_npv'],
                'utilization': tipping_metrics['optimal_utilization'],
                'capacity_kg_day': tipping_metrics['optimal_capacity_kg_day'],
                'demand_cushion_pct': tipping_metrics['demand_cushion_pct'],
                'breakeven_demand_kg_day': tipping_metrics['breakeven_demand_kg_day'],
                'total_investment_at_tipping': tipping_metrics['optimal_capex'],
                'cumulative_investment': cumulative_investment,
                'cumulative_npv': cumulative_npv,
                **tipping_metrics,
            }
            
            selected_stations.append(station_record)
            
            # OPTIMIZATION 3: Vectorized demand cannibalization
            remaining_candidates = self._apply_demand_cannibalization_fast(
                remaining_candidates, best_station, service_radius_m,
                distance_matrix, idx_to_position
            )
            
            # Remove selected station
            remaining_candidates = remaining_candidates.drop(best_idx)
            
            # Calculate current demand for this station given current network state
            # This is a "what-if" calculation - what would its demand be with current competition
            current_demand = self._competition_graph.calculate_market_share(best_idx) * best_station.get('initial_demand_post_existing', best_station['expected_demand_kg_day'])
            
            
            # Store iteration data
            iteration_history[iteration] = {
                'station_id': best_idx,
                'base_demand': best_station.get('initial_demand_post_existing', best_station['expected_demand_kg_day']),
                'current_demand': current_demand
            }
            
            # Initialize this station's demand history
            if best_idx not in station_demand_matrix:
                station_demand_matrix[best_idx] = {i: 0.0 for i in range(max_stations)}
            station_demand_matrix[best_idx][iteration] = current_demand
            
            # Update all previously selected stations' current demands
            for prev_iteration, prev_data in iteration_history.items():
                if prev_iteration < iteration:
                    station_id = prev_data['station_id']
                    if station_id in self._competition_graph.nodes:
                        market_share = self._competition_graph.calculate_market_share(station_id)
                        base = prev_data['base_demand']
                        station_demand_matrix[station_id][iteration] = base * market_share
            
            
            # Progress update
            if (iteration + 1) % 10 == 0:
                print(f"\n  Progress: {iteration + 1} stations, cumulative NPV: ${cumulative_npv:,.0f}")
        
        # Final network demand update
        if self._competition_graph:
            self._competition_graph.update_network_demands()
            
            # Update selected stations with final equilibrium demands
            for record in selected_stations:
                station_id = record['idx']
                if station_id in self._competition_graph.nodes:
                    node_data = self._competition_graph.nodes[station_id]
                    record['final_equilibrium_demand'] = node_data['demand']
                    record['final_market_share'] = node_data['market_share']
                    
        
        # Create iteration tracking DataFrame
        iteration_data = []
        for station_id, demand_history in station_demand_matrix.items():
            for iteration_num in range(len(iteration_history)):
                iteration_data.append({
                    'iteration': iteration_num,
                    'station_id': station_id,
                    'base_demand': iteration_history.get(iteration_num, {}).get('base_demand', 0) 
                                if iteration_history.get(iteration_num, {}).get('station_id') == station_id else 0,
                    'current_demand': demand_history.get(iteration_num, 0),
                    'was_selected': iteration_history.get(iteration_num, {}).get('station_id') == station_id
                })
        
        iteration_df = pd.DataFrame(iteration_data)

        # Create pivot table for easier analysis
        demand_evolution_matrix = iteration_df.pivot_table(
            index='station_id',
            columns='iteration', 
            values='current_demand',
            fill_value=0
        )

        # Store both in class for access
        self.iteration_history = iteration_history
        self.demand_evolution_matrix = demand_evolution_matrix
        
        # Create results GeoDataFrame
        results_gdf = gpd.GeoDataFrame(selected_stations, crs=candidates.crs)
        results_gdf['selection_order'] = results_gdf.index  # Order of selection
        
        # Add iteration history as attribute
        results_gdf.attrs['iteration_history'] = iteration_history
        results_gdf.attrs['demand_evolution'] = demand_evolution_matrix

        
        # Store as instance attribute for later use
        self.iterative_selection_results = results_gdf
        
        # Restore original state
        self.candidates = original_candidates
        self.config['station_capacity_kg_per_day'] = original_capacity
        self.existing_stations = original_existing
        
        print(f"\n Iterative selection complete: {len(results_gdf)} stations selected")
        print(f"Total NPV: ${cumulative_npv:,.0f}")
        print(f"Total Investment: ${cumulative_investment:,.0f}")
        print(f"Average capacity: {results_gdf['capacity_kg_day'].mean():.0f} kg/day")
        
        # Store competition graph for analysis
        self.competition_graph = self._competition_graph
        
        
        return results_gdf

    ########
    
    def _calculate_station_tipping_metrics_fast(self, station, next_best_npv, all_candidates):
        
        #Optimized tipping metrics calculation with minimal state changes.
        #Maintains exact same logic as original.
        
        # Store state
        original_capacity = self.config['station_capacity_kg_per_day']
        
        # OPTIMIZATION: Test fewer capacities initially
        # Coarse search: every 1000kg instead of 500kg
        coarse_capacities = np.arange(500, 12500, 1000)
        
        # Pre-create single station DataFrame
        station_gdf = gpd.GeoDataFrame(
            [station], 
            geometry=[station.geometry],
            crs=self.candidates.crs
        )
        
        # Test coarse capacities
        coarse_results = []
        for capacity in coarse_capacities:
            self.config['station_capacity_kg_per_day'] = capacity
            
            # Temporary assignment
            saved_candidates = self.candidates
            self.candidates = station_gdf
            self.calculate_economic_proxy()
            
            # Extract result immediately
            result = self.candidates.iloc[0]
            self.candidates = saved_candidates
            
            coarse_results.append({
                'capacity': capacity,
                'npv': result['npv_proxy'],
                'utilization': min(station['expected_demand_kg_day'] / capacity, 1.0),
                'capex': result.get('adjusted_capex', result.get('capex', self.config['station_capex'])),
            })
        
        # Find coarse optimal
        coarse_df = pd.DataFrame(coarse_results)
        coarse_optimal_idx = coarse_df['npv'].idxmax()
        coarse_optimal = coarse_df.iloc[coarse_optimal_idx]
        
        # OPTIMIZATION: Refined search only in promising range
        refined_min = max(500, coarse_optimal['capacity'] - 1500)
        refined_max = min(12000, coarse_optimal['capacity'] + 1500)
        refined_capacities = np.arange(refined_min, refined_max + 100, 100)
        
        # Test refined capacities
        refined_results = []
        for capacity in refined_capacities:
            if capacity in coarse_capacities:
                continue
                
            self.config['station_capacity_kg_per_day'] = capacity
            
            saved_candidates = self.candidates
            self.candidates = station_gdf
            self.calculate_economic_proxy()
            
            result = self.candidates.iloc[0]
            self.candidates = saved_candidates
            
            refined_results.append({
                'capacity': capacity,
                'npv': result['npv_proxy'],
                'utilization': min(station['expected_demand_kg_day'] / capacity, 1.0),
                'capex': result.get('adjusted_capex', result.get('capex', self.config['station_capex'])),
                # Extract all needed fields
                'daily_revenue': result.get('daily_revenue', 0),
                'daily_opex': result.get('daily_opex', 0),
                'daily_profit': result.get('daily_profit', 0),
                'payback_years': result.get('payback_years', 999),
                'location_revenue_multiplier': result['location_revenue_multiplier'],
                'location_cost_multiplier': result['location_cost_multiplier'],
                'irr': result.get('irr', 0),
                'total_cost': result.get('total_cost', 0),
            })
        
        # Combine and find optimal
        all_results = refined_results + [{k: v for k, v in r.items()} for r in coarse_results]
        all_results_df = pd.DataFrame(all_results)
        optimal_idx = all_results_df['npv'].idxmax()
        optimal_result = all_results_df.iloc[optimal_idx].to_dict()
        
        # Calculate remaining metrics (unchanged)
        service_radius_m = self.config.get('service_radius_miles', 2.0) * 1609.34
        
        # Count competitors
        station_coord = np.array([[station.geometry.x, station.geometry.y]])
        all_coords = np.c_[all_candidates.geometry.x, all_candidates.geometry.y]
        
        from scipy.spatial import cKDTree
        tree = cKDTree(all_coords)
        nearby_indices = tree.query_ball_point(station_coord[0], service_radius_m)
        competitors_within_radius = len(nearby_indices) - 1
        
        # Market share from competition graph
        station_id = station.name if hasattr(station, 'name') else station.index
        demand_capture_probability = self._competition_graph.calculate_market_share(station_id)
        
        # Restore config
        self.config['station_capacity_kg_per_day'] = original_capacity
        
        # Return all metrics (unchanged structure)
        return {
            'optimal_capacity_kg_day': optimal_result['capacity'],
            'optimal_npv': optimal_result['npv'],
            'optimal_utilization': optimal_result['utilization'],
            'optimal_capex': optimal_result['capex'],
            # ... all other metrics as before
            'competitors_within_radius': competitors_within_radius,
            'demand_capture_probability': demand_capture_probability,
            'competitive_advantage_npv': optimal_result['npv'] - next_best_npv,
            'capacities_tested': len(all_results),
            # Include all the detailed metrics
            'daily_revenue_optimal': optimal_result.get('daily_revenue', 0),
            'daily_opex_optimal': optimal_result.get('daily_opex', 0),
            'daily_profit_optimal': optimal_result.get('daily_profit', 0),
            'payback_years_optimal': optimal_result.get('payback_years', 999),
            'location_revenue_multiplier_optimal': optimal_result.get('location_revenue_multiplier', 1.0),
            'location_cost_multiplier_optimal': optimal_result.get('location_cost_multiplier', 1.0),
            'irr_optimal': optimal_result.get('irr', 0),
            'total_cost_optimal': optimal_result.get('total_cost', 0),
        }
        """

    def _apply_demand_cannibalization_fast(self, candidates_df, new_station, service_radius_m, 
                                        distance_matrix, idx_to_position):
        """
        Vectorized demand cannibalization using pre-computed distances.
        """
        if len(candidates_df) == 0:
            return candidates_df
        
        updated_candidates = candidates_df.copy()
        
        # Update competition graph
        new_station_id = new_station.name if hasattr(new_station, 'name') else id(new_station)
        if new_station_id in self._competition_graph.nodes:
            self._competition_graph.nodes[new_station_id]['type'] = 'selected'
        
        # Get position in distance matrix
        if new_station.name in idx_to_position:
            new_pos = idx_to_position[new_station.name]
            
            # Vectorized distance lookup and cannibalization
            for idx in updated_candidates.index:
                if idx in idx_to_position and idx != new_station.name:
                    pos = idx_to_position[idx]
                    distance = distance_matrix[new_pos, pos]
                    
                    if distance < service_radius_m:
                        # Same decay formula as original
                        decay_factor = np.exp(-2 * distance / service_radius_m)
                        market_share_loss = 0.3 * decay_factor
                        
                        # Apply reduction
                        current_demand = updated_candidates.loc[idx, 'expected_demand_kg_day']
                        updated_candidates.loc[idx, 'expected_demand_kg_day'] = current_demand * (1 - market_share_loss)
        
        return updated_candidates
    
               
        """"
        # Station coordinates
        station_coord = np.array([new_station.geometry.x, new_station.geometry.y])
        
        # Calculate distances to all candidates
        for idx, candidate in candidates_df.iterrows():
            candidate_coord = np.array([candidate.geometry.x, candidate.geometry.y])
            distance = np.linalg.norm(candidate_coord - station_coord)
            
            if distance < service_radius_m and idx != new_station.name:
                # Distance decay function (based on research)
                decay_factor = np.exp(-2 * distance / service_radius_m)
                
                # Market share loss (calibrated to real-world data)
                market_share_loss = 0.3 * decay_factor  # Max 30% loss at same location
                
                # Capacity competition factor
                capacity_ratio = (self.config['station_capacity_kg_per_day'] / 
                                new_station['expected_demand_kg_day'])
                capacity_factor = min(capacity_ratio, 2.0)  # Cap at 2x impact
                
                # Apply demand reduction
                demand_reduction = market_share_loss * capacity_factor
                new_demand = candidate['expected_demand_kg_day'] * (1 - demand_reduction)
                
                candidates_df.loc[idx, 'expected_demand_kg_day'] = max(new_demand, 0)
        
        return candidates_df
        """
         
            
    def _calculate_distance(self, pos1, pos2, coord_system='auto'):
        """
        **** not needed in current implementation *****
        
        Calculate distance between two positions in miles.
        Handles both (lat, lon) and (x, y) coordinate systems
        
        Parameters:
        -----------
        pos1, pos2 : tuple
            Position coordinates
        coord_system : str
            'latlon' for (longitude, latitude) in decimal degrees
            'meters' for projected coordinates in meters  
            'feet' for projected coordinates in feet
            'auto' to auto-detect (default)
        """
        import math
        
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Auto-detect coordinate system if not specified
        if coord_system == 'auto':
            if abs(x1) <= 180 and abs(x2) <= 180 and abs(y1) <= 90 and abs(y2) <= 90:
                coord_system = 'latlon'
            elif max(abs(x1), abs(x2), abs(y1), abs(y2)) > 10000:
                coord_system = 'meters'  # Large numbers suggest projected coordinates
            else:
                coord_system = 'latlon'  # Small numbers suggest decimal degrees
        
        if coord_system == 'latlon':
            # Haversine formula for lat/lon coordinates
            try:
                lon1, lat1 = x1, y1  # Clarify variable names
                lon2, lat2 = x2, y2
                
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                
                return c * 3959  # Earth's radius in miles
                
            except (ValueError, OverflowError):
                # Fall back to Euclidean if Haversine fails
                euclidean_degrees = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                return euclidean_degrees * 69.0  # Rough degrees to miles conversion
        
        elif coord_system == 'meters':
            # Euclidean distance in meters, convert to miles
            euclidean_meters = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return euclidean_meters * 0.000621371  # meters to miles
        
        elif coord_system == 'feet':
            # Euclidean distance in feet, convert to miles
            euclidean_feet = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            return euclidean_feet / 5280  # feet to miles
        
        else:
            raise ValueError(f"Unknown coordinate system: {coord_system}")

                
    def _prefilter_candidates_by_distance(self, candidates, min_station_spacing_miles=5):
        """
        
        **** not needed in current implementation *****
        
        Pre-filter candidates to remove spatial conflicts
        This eliminates the need for quadratic constraints
        
        """
        if not candidates:
            return []
        
        # Sort candidates by NPV (descending)
        sorted_candidates = sorted(
            candidates, 
            key=lambda x: x.get('expected_npv', 0), 
            reverse=True
        )
        
        filtered_candidates = []
        
        for candidate in sorted_candidates:
            # Check if this candidate conflicts with any already selected
            conflicts = False
            
            for selected in filtered_candidates:
                distance = self._calculate_distance(
                    candidate.get('position', (0, 0)),
                    selected.get('position', (0, 0))
                )
                
                if distance < min_station_spacing_miles:
                    conflicts = True
                    break
            
            if not conflicts:
                filtered_candidates.append(candidate)
        
        print(f"Pre-filtered {len(candidates)} candidates to {len(filtered_candidates)} non-conflicting options")
        
        return filtered_candidates

    
    def optimize_portfolio(self, budget=None, n_stations=5, use_gurobi=None, 
                           use_flexible_capacity=None, precomputed_results=None): 
        """
        Optimize station portfolio using integer linear programming with PuLP and the option to use Gurobi.
        
        Intelligently chooses between:
        1. Flexible capacity optimization for full system (optional)
        2. Traditional MILP solvers for a developer portfolio 
        Parameters:
        -----------
        budget : float
            Budget constraint in USD
        n_stations : int
            Maximum number of stations to select
        use_gurobi : None or bool
            Calls solver when provided for instances  where traditional MILP optimization is needed.
            If None, will use Gurobi if available, otherwise will use PuLP.
            Whether to use Gurobi (True) or PuLP (False)
        use_flexible_capacity : bool or None
            Whether to use flexible capacity optimization (True) or not (False).
            If None, autodetects based on results and will use default capacity from config.
        precomputed_results : GeoDataFrame or None
            Precomputed results to use from iterative station selection with flexible capacity.
            If provided, will skip optimization and return these results directly.
            
        Returns:
        --------
        
        dict: optimization_results 
            
        """
        
        # Auto-detect whether to use flexible capacity
        if use_flexible_capacity is None:
            # Use flexible if we have iterative results or precomputed results
            use_flexible_capacity = (
                precomputed_results is not None or 
                hasattr(self, 'iterative_selection_results')
            )
            
        print("Optimizing station portfolio...")
            
        if use_flexible_capacity:
            print("Auto-detected flexible capacity results available")
            
        # Route to appropriate portfolio optimization method
        if use_flexible_capacity:
            # Use flexible capacity optimization method
            return self.optimize_developer_portfolio(
                budget=budget, 
                n_stations=n_stations, 
                precomputed_results=precomputed_results,
                use_flexible_capacity=True
            )
        else:
            # Use traditional MILP optimization method
            print(f"Using traditional MILP optimization with {'Gurobi' if use_gurobi else 'PuLP'}")

        
            if use_gurobi is None and use_flexible_capacity is False:
                # Use Gurobi if available, otherwise use PuLP
            
                print("No solver specified, checking for Gurobi...")
                use_gurobi = True
                
            if use_gurobi:
                try:
                    return self._optimize_with_gurobi(
                        budget=budget, 
                        n_stations=n_stations)
                except Exception as e:
                    print(f"Gurobi optimization failed: {e}")
                    print("Falling back to PuLP solver.")
                    use_gurobi = False
            else:
                return self._optimize_with_pulp(
                    budget=budget, 
                    n_stations=n_stations
                )
    

    def _optimize_with_gurobi(self, budget=None, n_stations=5):    
        """Gurobi optimization using existing distance matrix and candidate structure"""
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            print("Gurobi not available. Falling back to PuLP fixed version.")
            return self._optimize_with_pulp(budget, n_stations)
        
        if budget is None:
            base_capex_multiplier = self.config.get('base_capex_multiplier', 1.4)  # Default 40% buffer
            budget = n_stations * self.config.get('station_capex', 12000000) * base_capex_multiplier  # Default budget (include 40% buffer)
            print(f"No budget specified, using default budget assumption of {100.0*base_capex_multiplier:,.0f}% of configured ${self.config.get('station_capex', 12000000):,} per station")
        
        if budget > 0:
            print(f"Optimizing portfolio with Gurobi: budget ${budget:,}, {n_stations} stations")

        
        if self.candidates.empty:
            raise ValueError("No candidates available. Run calculate_utilization_probability() first.")
    

        # Use top candidates from dataframe directly (no prefiltering needed)
        # Use top candidates by NPV proxy (but ensure minimum viable set)
        min_candidates = min(n_stations+20, len(self.candidates))  # does not fail if less than desired number of 10 candidates
    
        if self.candidates['npv_proxy'].max() > self.candidates['npv_proxy'].min():
            # Use top performers
            candidates = self.candidates.nlargest(min_candidates, 'npv_proxy').copy()
        else:
            # If all NPV values are the same, just use first N
            candidates = self.candidates.head(min_candidates).copy()
        
        print(f"Using top {len(candidates)} candidates for optimization")
        print(f"NPV range: ${candidates['npv_proxy'].min():,.0f} to ${candidates['npv_proxy'].max():,.0f}")
    
        
        candidates['idx'] = range(len(candidates))
        
        # Calculate pairwise distances
        coords = np.c_[candidates.geometry.x, candidates.geometry.y]
        from scipy.spatial.distance import cdist
        distances = cdist(coords, coords) / 1609.34  # Convert to miles
            
        # Create Gurobi model
        model = gp.Model("H2_Station_Portfolio_Gurobi")
        model.params.OutputFlag = 1
        model.params.MIPGap = 0.01  # 1% optimality gap
        model.params.TimeLimit = 900  # 15 minute time limit
        
        # Decision variables using candidate indices
        x = {}
        for i in candidates.index:
            x[i] = model.addVar(vtype=GRB.BINARY, name=f"station_{i}")
        
        # Objective: Maximize NPV using existing npv_proxy
        
        objective = gp.quicksum(
            candidates.loc[i, 'npv_proxy'] * x[i] for i in candidates.index
        )
        
        """"
        # Allowing violation of minimum spacing with a penalty 
        # Penalize stations that are too close together with a function of their NPV
        # Add network competition penalty
        for i in candidates.index:
            for j in candidates.index:
                if i < j and distances[candidates.loc[i, 'idx'], candidates.loc[j, 'idx']] < min_spacing:
                    # Penalty for violating minimum spacing
                    penalty = 0.5 * min(candidates.loc[i, 'npv_proxy'], candidates.loc[j, 'npv_proxy'])
                    objective -= penalty * x[i] * x[j]
                    constraint_count += 1
        """      
        
        model.setObjective(objective, GRB.MAXIMIZE)
        
        # check budget sufficiency if specified
        if budget is not None and budget > 0:
            if not self._validate_budget_approach(budget):
                raise ValueError("Budget insufficient for any station portfolio under the specified approach")

            # Budget constraint if specified
            
            budget_constraint = gp.quicksum(
                candidates.loc[i, 'capex'] * x[i] for i in candidates.index
            )
            
            model.addConstr(budget_constraint <= budget, name="budget")
        

        # Number of stations constraints
        num_stations = gp.quicksum(x[i] for i in candidates.index)
        model.addConstr(num_stations <= n_stations, name="max_stations")
        model.addConstr(num_stations >= max(1, n_stations - 2), name="min_stations")
        
        # Spatial constraints using existing distance matrix
        min_spacing = self.config['min_station_spacing_miles']
        constraint_count = 0

        
        # Enforce minimum spacing strictly with linear constraints
        for i in candidates.index:
            for j in candidates.index:
                if i < j and distances[candidates.loc[i, 'idx'], candidates.loc[j, 'idx']] < min_spacing:
                    # Linear constraint: cannot select both stations if too close
                    model.addConstr(
                        x[i] + x[j] <= 1,
                        name=f"min_spacing_{i}_{j}"
                    )
                    constraint_count += 1
                    
            
        print(f"Added {constraint_count} minimum spacing constraints")
        
        # Optimize the model
        
        model.optimize()

        # Extract results
        selected_candidates = []
        total_selected_cost = 0
        total_selected_npv = 0
        
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            selected_indices = []
            
            for i in candidates.index:
                if x[i].x > 0.5:  # Binary variable is 1
                    selected_indices.append(i)
                    candidate_data = candidates.loc[i].to_dict()
                    selected_candidates.append({
                        'candidate_index': i,
                        'candidate_data': candidate_data,
                        'cost': candidate_data['total_cost'],
                        'capacity_kg_day': candidate_data['capacity_kg_day'],
                        'npv': candidate_data['npv_proxy'],
                        'position': [candidate_data['longitude'], candidate_data['latitude']],  # Assuming longitude is the position
                        'demand': candidate_data['total_demand'],
                        'utilization': candidate_data['total_demand'] / candidate_data['capacity_kg_day'],
                    })
                    total_selected_cost += candidate_data['total_cost']
                    total_selected_npv += candidate_data['npv_proxy']
            
            # Create portfolio DataFrame matching PuLP implementation
            self.portfolio = candidates.loc[selected_indices].copy()
            
            # Store results
            self.optimization_results = {
                'selected_candidates': selected_candidates,
                'total_cost': total_selected_cost,
                'total_npv': total_selected_npv,
                'num_stations': len(selected_candidates),
                'model_status': 'Optimal' if model.status == GRB.OPTIMAL else 'Time_Limit',
                'solver_used': 'Gurobi',
                'solve_time': model.Runtime,
                'mip_gap': model.mipGap,
                'budget_used': total_selected_cost,
                'budget_utilization': total_selected_cost / budget if budget else 0,
                'objective_value': model.objVal
            }
            
            print(f"Gurobi optimization complete: {len(selected_candidates)} stations selected")
            print(f"Total cost: ${total_selected_cost:,.0f}")
            print(f"Total NPV: ${total_selected_npv:,.0f}")
            print(f"Solve time: {model.Runtime:.2f} seconds")
            print(f"MIP Gap: {model.mipGap:.2%}")
            
        else:
            print(f"Gurobi optimization failed with status: {model.status}")
            return self._optimize_with_pulp(budget, n_stations)
        
        return self.optimization_results
    
            
        
        
    def _optimize_with_pulp(self, budget=None, n_stations=5):
        """
        Optimize station portfolio using PuLP.
        Method avoids quadratic constraints, enforcing minimum distance spacing strictly with linear constraints.
        Optimization with capex budget approach
    
        Parameters:
        - budget: Available budget
        - n_stations: Target number of stations
        """
        import pulp
        
        
        if budget is None:
            base_capex_multiplier = self.config.get('base_capex_multiplier', 1.4)  # Default 40% buffer
            budget = n_stations * self.config.get('station_capex', 12000000) * base_capex_multiplier  # Default budget (include 40% buffer)
            print(f"No budget specified, using default budget assumption of {100.0*base_capex_multiplier:,.0f}% of configured ${self.config.get('station_capex', 12000000):,} per station")
        
            
        print(f"Optimizing portfolio with PuLP: {n_stations} stations, budget ${budget:,.0f}") 
        
        
        if self.candidates.empty:
            raise ValueError("No candidates available for optimization. Run demand estimation and candidate generation first.")
        
        
        if budget is not None and budget > 0:
            if not self._validate_budget_approach(budget):
                raise ValueError("Budget insufficient for any station portfolio under the specified approach")

        
        # Ensure we have required columns
        if 'npv_proxy' not in self.candidates.columns:
            print("Warning: npv_proxy not found, calculating economic metrics...")
            self.calculate_economic_proxy()
        
        # Check if we have valid NPV values
        #valid_npv = self.candidates['npv_proxy'].notna()
        valid_mask = self.candidates['npv_proxy'].notna() & np.isfinite(self.candidates['npv_proxy'])
        valid_candidates = self.candidates[valid_mask]
        
        if len(valid_candidates) == 0:
            print("ERROR: No valid candidates with finite NPV values")
            print("Creating fallback candidates based on demand...")
            # Create simple NPV proxy based on demand
            self.candidates['npv_proxy'] = (
                self.candidates.get('expected_demand_kg_day', 100) * 365 * 2 * 6.71 - 4000000
            )
            # Recalculate valid candidates after fixing NPV
            valid_mask = self.candidates['npv_proxy'].notna() & np.isfinite(self.candidates['npv_proxy'])
            valid_candidates = self.candidates[valid_mask].copy()
        
        #if not valid_npv.any():
            #print("Warning: No valid NPV values found. Using demand-based selection...")
            # Fallback to demand-based selection
            #self.candidates['npv_proxy'] = self.candidates.get('expected_demand_kg_day', 100) * 1000 
            # ToDo: Replace with actual demand-based proxy
        
        
        # Use top candidates from dataframe directly (no prefiltering needed)
        # Use top candidates by NPV proxy (but ensure minimum viable set)
        min_candidates = min(n_stations+20, len(valid_candidates))  # does not fail if less than desired number of 10 candidates
    
    
        # Filter out NaN NPV values first
        #valid_candidates = self.candidates[self.candidates['npv_proxy'].notna()]
     
        #if len(valid_candidates) == 0:
            #raise ValueError("No candidates with valid NPV values available for optimization.")
    
    
        if valid_candidates['npv_proxy'].max() > valid_candidates['npv_proxy'].min():
            # Use top performers
            candidates = valid_candidates.nlargest(min_candidates, 'npv_proxy').copy()
        else:
            # If all NPV values are the same, just use first N
            candidates = valid_candidates.head(min_candidates).copy()
        
        print(f"Using top {len(candidates)} candidates for optimization")
        print(f"NPV range: ${candidates['npv_proxy'].min():,.0f} to ${candidates['npv_proxy'].max():,.0f}")
    
        # Reset index to ensure continuous indexing for optimization
        candidates = candidates.reset_index(drop=True)
        candidates['idx'] = range(len(candidates))
        
        # Calculate pairwise distances
        coords = np.c_[candidates.geometry.x, candidates.geometry.y]
        from scipy.spatial.distance import cdist
        distances = cdist(coords, coords) / 1609.34  # Convert to miles
        
        # Create optimization problem
        model = LpProblem("H2_Station_Portfolio_PuLP", LpMaximize)
        
        # Decision variables
        x = LpVariable.dicts("station", candidates.index, cat='Binary')
        
        # Objective: maximize total NPV with network effects
        # Penalty for stations too close together
        objective = lpSum([candidates.loc[i, 'npv_proxy'] * x[i] for i in candidates.index])
        
        """""
        # Add network competition penalty
        
        for i in candidates.index:
            for j in candidates.index:
                if i < j and distances[candidates.loc[i, 'idx'], candidates.loc[j, 'idx']] < min_spacing:
                    # Penalty for violating minimum spacing
                    penalty = 0.5 * min(candidates.loc[i, 'npv_proxy'], candidates.loc[j, 'npv_proxy'])
                    objective -= penalty * x[i] * x[j]
        
        """
        
        
        model += objective
        
        # Constraints
        # Number of stations
        model += lpSum([x[i] for i in candidates.index]) <= n_stations
        model += lpSum(x[i] for i in candidates.index) >= max(1, n_stations - 2)
        
        # Budget constraint if specified
        if budget is not None and budget > 0:
            model += lpSum([candidates.loc[i, 'adjusted_capex'] * x[i] for i in candidates.index]) <= budget
            
        # Minimum spacing constraint (linearized)
        min_spacing = self.config['min_station_spacing_miles']
        
        for i in candidates.index:
            for j in candidates.index:
                if i < j and distances[candidates.loc[i, 'idx'], candidates.loc[j, 'idx']] < min_spacing:
                    # Linear constraint: cannot select both stations if too close
                    model += x[i] + x[j] <= 1,  f"spacing_constraint_{i}_{j}"
        
        # Solve
        model.solve(PULP_CBC_CMD(msg=0))
        
        # Extract solution
        selected_indices = [i for i in candidates.index if x[i].varValue == 1]
        self.portfolio = candidates.loc[selected_indices].copy()
        
        # Extract results using existing candidate structure
        selected_candidates = []
        total_selected_cost = 0
        total_selected_npv = 0
        
        for i in candidates.index:
            if x[i].value() == 1:
                candidate_data = candidates.loc[i].to_dict()
                selected_candidates.append({
                    'candidate_index': i,
                    'candidate_data': candidate_data,
                    'capacity_kg_day': candidate_data['capacity_kg_day'],
                    'cost': candidate_data['total_cost'],
                    'npv': candidate_data['npv_proxy'],
                    'position': [candidate_data['longitude'], candidate_data['latitude']],
                    'demand': candidate_data['expected_demand_kg_day'],
                    'utilization': candidate_data['expected_demand_kg_day'] / candidate_data['capacity_kg_day'],
                })
                total_selected_cost += candidate_data['total_cost']
                total_selected_npv += candidate_data['npv_proxy']
        
        # Store results in existing format
        self.optimization_results = {
            'selected_candidates': selected_candidates,
            'total_cost': total_selected_cost,
            'total_npv': total_selected_npv,
            'num_stations': len(selected_candidates),
            'model_status': pulp.LpStatus[model.status],
            'solver_used': 'PuLP_Fixed',
            'budget_used': total_selected_cost,
            'budget_utilization': total_selected_cost / budget if budget else 0
        }
        
        print(f"PuLP optimization complete: {len(selected_candidates)} stations selected")
        print(f"Total cost: ${total_selected_cost:,.0f}")
        print(f"Total NPV: ${total_selected_npv:,.0f}")
        if budget:
            print(f"Budget utilization: {total_selected_cost/budget:.1%}")
        else:
            print("Budget utilization: No budget constraint applied")
        print(f"  Estimated total portfolio investment: ${len(self.portfolio) * self.config['station_capex']:,.0f}")
        
        
        #print(f"  Selected {len(self.portfolio)} stations")
        #print(f"  Total NPV: ${self.portfolio['npv_proxy'].sum():,.0f}")
        
        return self.optimization_results
    
    

    def optimize_developer_portfolio(self, budget=None, n_stations=None, 
                                precomputed_results=None, use_flexible_capacity=True):
        
        """
        Optimize portfolio for a capital-constrained developer.
        
        Two-stage approach:
        1. Use iterative selection results (with optimal capacities)
        2. Select subset within budget constraints
        
        Parameters:
        -----------
        budget : float
            Capital budget constraint
        n_stations : int
            Maximum number of stations to select
        precomputed_results : GeoDataFrame
            Results from run_iterative_station_selection (optional)
        use_flexible_capacity : bool
            Whether to use flexible capacity results (True) or fixed capacity (False)
            
        Returns:
        --------
        dict : Optimization results matching existing format    

        """
        print(f"\nOptimizing developer portfolio...")
        print(f"Budget: ${budget:,.0f}" if budget else "No budget constraint")
        print(f"Max stations: {n_stations}" if n_stations else "No station limit")
        
        # Stage 1: Get system-optimal candidate stations if not provided
        if precomputed_results is not None:
            # Use provided pre-optimized results
            all_stations = precomputed_results.copy()
            print(f"Using {len(all_stations)} pre-optimized stations")
        elif use_flexible_capacity and hasattr(self, 'iterative_selection_results'):
            # Use results from flexible capacity optimization
            all_stations = self.iterative_selection_results.copy()
            print(f"Using {len(all_stations)} stations from iterative selection")
        else:
            # Fall back to profitable candidates with fixed capacity
            print("Using standard profitable candidates")
            all_stations = self.candidates[self.candidates['npv_proxy'] > 0].copy()
            
        # Ensure consistent column names
        if 'npv' not in all_stations.columns:
            all_stations['npv'] = all_stations['npv_proxy']
        if 'capacity_kg_day' not in all_stations.columns:
            all_stations['capacity_kg_day'] = self.config['station_capacity_kg_per_day']
        
        if 'optimal_capex' not in all_stations.columns:
        # Try to find capex in various possible column names
            capex_columns = ['capex', 'adjusted_capex', 'total_adjusted_capex']
            capex_found = False
            
            for col in capex_columns:
                if col in all_stations.columns:
                    all_stations['optimal_capex'] = all_stations[col]
                    capex_found = True
                    #print(f"  Using '{col}' column for capex values")
                    break
            
            if not capex_found:
                # Fall back to config value with multiplier
                base_capex_multiplier = self.config.get('base_capex_multiplier', 1.4)
                default_capex = self.config['station_capex'] * base_capex_multiplier
                all_stations['optimal_capex'] = default_capex
                print(f"  No capex column found, using default: ${default_capex:,.0f} per station")
        

        # Stage 2: Apply constraints using CAPEX
        # Sort by NPV/capex ratio for greedy selection
        all_stations['npv_capex_ratio'] = all_stations['npv'] / all_stations['optimal_capex']
        all_stations['utilization'] = all_stations['expected_demand_kg_day'] / all_stations['capacity_kg_day']
        stations_sorted = all_stations.sort_values('npv_capex_ratio', ascending=False)
        
        selected_indices = []
        selected_candidates = []
        total_investment_capex = 0
        total_portfolio_lifetime_cost = 0
        total_npv = 0
        
        
        # Iterate through sorted stations and select within budget
        for idx, station in stations_sorted.iterrows():
            station_cost = station['optimal_capex']
            
            
            if budget and total_investment_capex + station_cost  > budget:
                continue
            if n_stations and len(selected_indices) >= n_stations:
                break
            
            # Add station to portfolio    
            selected_indices.append(idx)
            
            # Build candidate information for optimization_results data structure
            candidate_info = {
                'candidate': station.to_dict(),
                'index': idx,
                'capital_cost': station_cost,
                'npv': station['npv'],
                'capacity': station['capacity_kg_day'],
                'utilization': station['utilization'],
                'position': [station['longitude'], station['latitude']],
                'demand': station['expected_demand_kg_day'],
            }
            
            # Add lifetime total NPV cost of station if available for reporting
            if 'total_cost' in station:
                candidate_info['total_lifetime_cost'] = station['total_cost']
                
            selected_candidates.append(candidate_info)
            # Update total costs and NPV
            total_investment_capex += station_cost
            total_npv += station['npv']
            total_portfolio_lifetime_cost += candidate_info['total_lifetime_cost']
            
        # Create final portfolio GeoDataFrame
        if selected_indices:
            self.portfolio = all_stations.loc[selected_indices].copy()
            
            
            # Apply competition adjustment if multiple developer mode is enabled
            if self.config.get('multi_developer_mode', False):
                self.portfolio = self._adjust_for_competition(portfolio, all_stations)
                
                # Recalculate total NPV after competition adjustment
                total_npv = self.portfolio['npv'].sum()
        else:
            print("No stations selected within budget and constraints for the developer.")
            self.portfolio = gpd.GeoDataFrame(columns=all_stations.columns)
            
        # Create optimization_results in exact format expected by downstream methods
        self.optimization_results = {
            'selected_candidates': selected_candidates,
            'total_investment_capex': total_investment_capex,  # This is total CAPEX (initial investment)
            'total_npv': total_npv,
            'total_cost': total_portfolio_lifetime_cost,
            'num_stations': len(selected_candidates),
            'model_status': 'Optimal',
            'solver_used': 'GreedyKnapsack',
            'budget_used': total_investment_capex,
            'budget_utilization': total_investment_capex / budget if budget and budget > 0 else 0,
            'selection_method': 'flexible_capacity' if use_flexible_capacity else 'fixed_capacity',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add clarifying fields for downstream analysis
        if 'total_cost' in all_stations.columns:
            # Calculate total lifetime cost if available
            total_lifetime_cost = sum(all_stations.loc[idx, 'total_cost'] for idx in selected_indices)
            self.optimization_results['total_lifetime_cost'] = total_lifetime_cost
            self.optimization_results['lifetime_cost_note'] = "total_cost includes capex + PV(opex)"
        
        print(f"\nOptimization complete:")
        print(f"  Selected stations: {len(selected_candidates)}")
        print(f"  Total CAPEX (initial investment): ${total_investment_capex:,.0f}")
        print(f"  Total NPV: ${total_npv:,.0f}")
        print(f"  ROI: {(total_npv / total_investment_capex * 100):.1f}%")
        if budget:
            print(f"  Budget utilization: {total_investment_capex/budget:.1%}")
        
        return self.optimization_results
                

        
    
    
    def _adjust_for_competition(self, portfolio, all_stations):
        """
        Adjust portfolio NPV for competitive effects when multiple developers exist.
        
        Parameters:
        -----------
        portfolio : GeoDataFrame
            Selected stations for this developer
        all_stations : GeoDataFrame
            All available stations (including those selected by others)
            
        Returns:
        --------
        portfolio : GeoDataFrame
            Portfolio with adjusted NPV values accounting for competition
        """
        # If no competition adjustment needed, return as-is
        if not self.config.get('competition_adjustment_factor'):
            return portfolio
        
        portfolio = portfolio.copy()
        competition_factor = self.config.get('competition_adjustment_factor', 0.8)
        service_radius_m = self.config.get('service_radius_miles', 2.0) * 1609.34
        
        # For each station in portfolio, check for nearby competitors
        for idx in portfolio.index:
            station_geom = portfolio.loc[idx, 'geometry']
            
            # Find other stations within service radius
            nearby_mask = all_stations.geometry.distance(station_geom) < service_radius_m
            n_competitors = nearby_mask.sum() - 1  # Exclude self
            
            if n_competitors > 0:
                # Reduce NPV based on competition
                # More competitors = lower NPV due to demand splitting
                adjustment = competition_factor ** n_competitors
                portfolio.loc[idx, 'npv'] *= adjustment
                portfolio.loc[idx, 'competition_adjusted'] = True
                portfolio.loc[idx, 'n_competitors'] = n_competitors
            else:
                portfolio.loc[idx, 'competition_adjusted'] = False
                portfolio.loc[idx, 'n_competitors'] = 0
        
        return portfolio
  
        
    def analyze_regional_clusters(self, selected_stations, n_regions=2):
        """
        Group selected stations into regions for management purposes.
        Prior station selection from iterative optimization is required.
        """
        # Cluster the already-selected stations
        coords = np.c_[selected_stations.geometry.x, selected_stations.geometry.y]
        clustering = KMeans(n_clusters=n_regions).fit(coords)
        
        selected_stations['region'] = clustering.labels_
        
        # Create spatial representation of clusters
        from shapely.ops import unary_union
        cluster_geometries = []
        
        for region in range(n_regions):
            region_stations = selected_stations[selected_stations['region'] == region]
            if len(region_stations) > 0:
                # Create convex hull for the region
                cluster_geom = unary_union(region_stations.geometry).convex_hull
                
                # Calculate statistics
                stats = {
                    'region': region,
                    'geometry': cluster_geom,
                    'n_stations': len(region_stations),
                    'total_capacity': region_stations['capacity_kg_day'].sum(),
                    'avg_capacity': region_stations['capacity_kg_day'].mean(),
                    'total_npv': region_stations['npv'].sum(),
                    'total_demand': region_stations['expected_demand_kg_day'].sum(),
                    'total_capex': region_stations['optimal_capex'].sum()
                }
                cluster_geometries.append(stats)
        
        
        # Return GeoDataFrame instead of regular DataFrame
        return gpd.GeoDataFrame(cluster_geometries, geometry='geometry', crs=selected_stations.crs)        
          
        
    def create_continuous_score_surface(self):
        """
        Create continuous marginal utility surface for entire geography.
        """
        print("Creating continuous score surface...")
        
        # Check if demand surface exists
        if not hasattr(self, 'demand_surface') or self.demand_surface is None:
            print("  Warning: No demand surface available. Creating score surface requires demand surface.")
            return
        
        # Get or reconstruct grid points
        if 'points' in self.demand_surface:
            grid_points = self.demand_surface['points']
        else:
            # Reconstruct grid points from x,y arrays
            if 'x' in self.demand_surface and 'y' in self.demand_surface:
                xx = self.demand_surface['x']
                yy = self.demand_surface['y']
                # Create grid points by flattening and stacking
                grid_points = np.column_stack([xx.ravel(), yy.ravel()])
            else:
                print("  Error: Cannot create grid points from demand surface")
                return
        
        
        # For each grid point, calculate marginal utility score
        scores = np.zeros(len(grid_points))
        
        # Create KDTree for fast neighbor queries
        if len(self.candidates) > 0:
            candidate_coords = np.c_[self.candidates.geometry.x, self.candidates.geometry.y]
            candidate_tree = cKDTree(candidate_coords)
            
            for i, pt in enumerate(grid_points):
                # Find nearest candidates
                distances, indices = candidate_tree.query(pt, k=min(5, len(self.candidates)))
                
                # Weight scores by inverse distance
                weights = 1 / (distances + 1000)  # Add small constant to avoid division by zero
                weights = weights / weights.sum()
                
                # Weighted average of nearby candidate scores
                nearby_scores = self.candidates.iloc[indices]['npv_proxy'].values
                scores[i] = np.sum(nearby_scores * weights)
        
        # Reshape to grid
        self.score_surface = {
            'x': self.demand_surface['x'],
            'y': self.demand_surface['y'],
            'scores': scores.reshape(self.demand_surface['x'].shape),
            'marginal_utility': scores
        }
    
    def validate_results(self):
        """
        Validate results against existing infrastructure.
        Analyzes spatial correlation, route alignment, and coverage patterns.
        """
        
        print("\n" + "="*60)
        print("Validating results...")
        print("="*60)
        
        if not hasattr(self, 'portfolio') or self.portfolio.empty:
            print(" No portfolio to validate. Run optimization first.")
            return
        
        if len(self.existing_stations) == 0:
            print("  No existing stations for validation")
            self.validation_results = {'status': 'no_existing_infrastructure'}
            return
        
        # Initialize validation results dictionary
        self.validation_results = {}
        
        # DISTANCE ANALYSIS - Portfolio to Existing Infrastructure
        print("\n1. DISTANCE ANALYSIS")
        print("-" * 40)
        
        # Calculate distances from each selected site to nearest existing station
        portfolio_coords = np.c_[self.portfolio.geometry.x, self.portfolio.geometry.y]
        existing_coords = np.c_[self.existing_stations.geometry.x, self.existing_stations.geometry.y]
        
        
        existing_tree = cKDTree(existing_coords)
        distances_m, nearest_indices = existing_tree.query(portfolio_coords)
        distances_miles = distances_m / 1609.34
        
        # Store detailed distance information
        self.portfolio['dist_to_nearest_existing_miles'] = distances_miles
        self.portfolio['nearest_existing_idx'] = nearest_indices
        
        # Calculate statistics
        avg_distance = distances_miles.mean()
        median_distance = np.median(distances_miles)
        std_distance = distances_miles.std()
        
        print(f"  Average distance to existing stations: {avg_distance:.1f} miles")
        print(f"  Median distance: {median_distance:.1f} miles")
        print(f"  Standard deviation: {std_distance:.1f} miles")
        print(f"  Range: {distances_miles.min():.1f} - {distances_miles.max():.1f} miles")
        
        # Identify complementary vs competitive placements
        competitive_threshold = self.config.get('min_station_spacing_miles', 5.0)
        competitive_sites = (distances_miles < competitive_threshold).sum()
        complementary_sites = len(self.portfolio) - competitive_sites
        
        print(f"\n  Site Classification:")
        print(f"    Competitive (<{competitive_threshold} mi): {competitive_sites} ({competitive_sites/len(self.portfolio)*100:.1f}%)")
        print(f"    Complementary (≥{competitive_threshold} mi): {complementary_sites} ({complementary_sites/len(self.portfolio)*100:.1f}%)")
        
        
        
        # COVERAGE ANALYSIS - Service Area Overlap
        print("\n2. COVERAGE ANALYSIS")
        print("-" * 40)
        
        service_radius = self.config['service_radius_miles'] * 1609.34  # Convert to meters
        
        # Check overlap with existing infrastructure
        existing_buffer = self.existing_stations.buffer(service_radius)
        existing_union = unary_union(existing_buffer)
        
        # Check which selected sites are within existing service areas
        portfolio_covered = []
        for idx, station in self.portfolio.iterrows():
            if station.geometry.within(existing_union):
                portfolio_covered.append(True)
            else:
                portfolio_covered.append(False)
        
        coverage_rate = sum(portfolio_covered) / len(portfolio_covered) if len(portfolio_covered) > 0 else 0
        
        print(f"  {coverage_rate*100:.1f}% of selected sites near existing infrastructure")
        
        # ROUTE-BASED ANALYSIS
        print("\n3. ROUTE-BASED ANALYSIS")
        print("-" * 40)
        
        
        # Identify which routes have existing stations
        routes_with_existing = set()
        routes_with_selected = set()
        
        # Buffer existing stations and find intersecting routes
        existing_buffer_small = self.existing_stations.buffer(1000)  # 1km buffer
        for idx, route in self.routes.iterrows():
            if route.geometry.intersects(existing_buffer_small.unary_union):
                routes_with_existing.add(idx)
        
        # Buffer selected stations and find intersecting routes
        portfolio_buffer_small = self.portfolio.buffer(1000)  # 1km buffer
        for idx, route in self.routes.iterrows():
            if route.geometry.intersects(portfolio_buffer_small.unary_union):
                routes_with_selected.add(idx)
        
        # Calculate route coverage metrics
        routes_both = routes_with_existing.intersection(routes_with_selected)
        routes_only_existing = routes_with_existing - routes_with_selected
        routes_only_selected = routes_with_selected - routes_with_existing
        routes_neither = set(self.routes.index) - routes_with_existing - routes_with_selected
        
        total_routes = len(self.routes)
        print(f"  Total routes analyzed: {total_routes}")
        print(f"  Routes with both existing & selected: {len(routes_both)} ({len(routes_both)/total_routes*100:.1f}%)")
        print(f"  Routes with only existing: {len(routes_only_existing)} ({len(routes_only_existing)/total_routes*100:.1f}%)")
        print(f"  Routes with only selected: {len(routes_only_selected)} ({len(routes_only_selected)/total_routes*100:.1f}%)")
        print(f"  Routes with neither: {len(routes_neither)} ({len(routes_neither)/total_routes*100:.1f}%)")
                
        
        # DEMAND CORRELATION ANALYSIS
        print("\n4. DEMAND CORRELATION ANALYSIS")
        print("-" * 40)
        
        # Check if existing stations correlate with high-demand areas
        if hasattr(self, 'candidates') and 'expected_demand_kg_day' in self.candidates.columns:
            # Find demand at existing station locations
            existing_demand = []
            candidate_coords = np.c_[self.candidates.geometry.x, self.candidates.geometry.y]
            candidate_tree = cKDTree(candidate_coords)
            
            for existing_coord in existing_coords:
                # Find nearest candidate to estimate demand
                dist, idx = candidate_tree.query(existing_coord)
                if dist < 5000:  # Within 5km
                    existing_demand.append(self.candidates.iloc[idx]['expected_demand_kg_day'])
                else:
                    existing_demand.append(0)
            
            avg_existing_demand = np.mean(existing_demand) if existing_demand else 0
            avg_portfolio_demand = self.portfolio['expected_demand_kg_day'].mean()
            avg_candidate_demand = self.candidates['expected_demand_kg_day'].mean()
            avg_portfolio_utilization = self.portfolio['utilization'].mean()
            
            
            print(f"  Average daily demand (kg H2):")
            print(f"    At existing stations: {avg_existing_demand:.0f}")
            print(f"    At selected stations: {avg_portfolio_demand:.0f}")
            print(f"    Overall candidate average: {avg_candidate_demand:.0f}")
            
            demand_improvement = (avg_portfolio_demand - avg_existing_demand) / avg_existing_demand * 100 if avg_existing_demand > 0 else float('inf')
            print(f"  Demand improvement: {demand_improvement:+.1f}%")
                    
        # SPATIAL CLUSTERING ANALYSIS
        print("\n5. SPATIAL CLUSTERING ANALYSIS")
        print("-" * 40)            
        
        # Analyze spatial patterns using convex hulls
        if len(self.existing_stations) > 3 and len(self.portfolio) > 3:
            existing_hull = self.existing_stations.unary_union.convex_hull
            portfolio_hull = self.portfolio.unary_union.convex_hull
            
            existing_area = existing_hull.area / 1e6  # Convert to km²
            portfolio_area = portfolio_hull.area / 1e6
            overlap_area = existing_hull.intersection(portfolio_hull).area / 1e6
            
            print(f"  Existing network coverage area: {existing_area:,.0f} km²")
            print(f"  Selected network coverage area: {portfolio_area:,.0f} km²")
            print(f"  Overlapping area: {overlap_area:,.0f} km²")
            print(f"  Network expansion: {(portfolio_area - overlap_area):,.0f} km² new area")

        
        # REGIONAL SPARSITY ANALYSIS
        print("\n6. REGIONAL SPARSITY ANALYSIS")
        print("-" * 40)
        
        # Divide California into regions and analyze distribution
        bounds = self.routes.total_bounds  # minx, miny, maxx, maxy
        
        # Create a 3x3 grid for regional analysis
        x_bins = np.linspace(bounds[0], bounds[2], 4)
        y_bins = np.linspace(bounds[1], bounds[3], 4)
        
        regions = ['SW', 'S', 'SE', 'W', 'Central', 'E', 'NW', 'N', 'NE']
        region_analysis = {}
        
        for i, region in enumerate(regions):
            row = i // 3
            col = i % 3
            
            # Create region bounds
            region_bounds = [x_bins[col], y_bins[row], x_bins[col+1], y_bins[row+1]]
            
            # Count stations in region
            existing_in_region = 0
            selected_in_region = 0
            
            for station in self.existing_stations.geometry:
                if (region_bounds[0] <= station.x <= region_bounds[2] and 
                    region_bounds[1] <= station.y <= region_bounds[3]):
                    existing_in_region += 1
            
            for station in self.portfolio.geometry:
                if (region_bounds[0] <= station.x <= region_bounds[2] and 
                    region_bounds[1] <= station.y <= region_bounds[3]):
                    selected_in_region += 1
            
            region_analysis[region] = {
                'existing': existing_in_region,
                'selected': selected_in_region
            }
        
        print("  Regional Distribution (3x3 grid):")
        print("  Region | Existing | Selected | Change")
        print("  -------|----------|----------|--------")
        for region, counts in region_analysis.items():
            change = counts['selected'] - counts['existing']
            print(f"  {region:6} | {counts['existing']:8} | {counts['selected']:8} | {change:+6}")
                     
        
        self.validation_results = {
            'coverage_rate': coverage_rate,
            'avg_distance_to_existing': avg_distance,
            'avg_existing_demand': avg_existing_demand,
            'avg_portfolio_demand': avg_portfolio_demand,
            'avg_candidate_demand': avg_candidate_demand,
            'avg_portfolio_utilization': avg_portfolio_utilization,
            'median_distance_to_existing': median_distance,
            'competitive_sites': competitive_sites,
            'complementary_sites': complementary_sites,
            'routes_coverage': {
                'both': len(routes_both),
                'only_existing': len(routes_only_existing),
                'only_selected': len(routes_only_selected),
                'neither': len(routes_neither)
            },
            'demand_improvement': demand_improvement if 'demand_improvement' in locals() else None,
            'regional_distribution': region_analysis,
            'distance_distribution': distances_miles.tolist()
        }
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)

            
   
    
    
    def analyze_results(self):
        """Analyze optimization results using the refined structure"""
        if not hasattr(self, 'optimization_results') or not self.optimization_results:
            print("No optimization results available. Run optimize_portfolio() first.")
            return None
        
        results = self.optimization_results
        selected = results['selected_candidates']
        
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS ANALYSIS")
        print("="*60)
        
        # Summary statistics
        print(f"Solver Used: {results['solver_used']}")
        print(f"Model Status: {results['model_status']}")
        print(f"Selected Stations: {results['num_stations']}")
        print(f"Total NPV Investment: ${results['total_cost']:,.0f}")
        print(f"Total Capital Investment: ${results['capex']:,.0f}")
        print(f"Budget Utilization: {results['budget_utilization']:.1%}")
        print(f"Total NPV: ${results['total_npv']:,.0f}")
        
        if 'solve_time' in results:
            print(f"Solve Time: {results['solve_time']:.2f} seconds")
        if 'mip_gap' in results:
            print(f"Optimality Gap: {results['mip_gap']:.2%}")
        
        # Detailed station analysis
        print(f"\nSelected Station Details:")
        print(f"{'Index':<8} {'Cost ($M)':<12} {'NPV ($M)':<12} {'Demand (kg/day)':<15} {'Location':<20}")
        print("-" * 80)
        
        for station in selected:
            cost_m = station['cost'] / 1_000_000
            npv_m = station['npv'] / 1_000_000
            demand = station['demand']
            location = f"({station['position'][0]:.3f}, {station['position'][1]:.3f})"
            
            print(f"{station['candidate_index']:<8} {cost_m:<12.2f} {npv_m:<12.2f} {demand:<15.0f} {location:<20}")
        
        # Geographic distribution analysis
        if len(selected) > 1:
            positions = [s['position'] for s in selected]
            lons = [p[0] for p in positions]
            lats = [p[1] for p in positions]
            
            print(f"\nGeographic Distribution:")
            print(f"Longitude range: {min(lons):.3f} to {max(lons):.3f}")
            print(f"Latitude range: {min(lats):.3f} to {max(lats):.3f}")
            
            # Calculate average inter-station distance using existing distance matrix
            distances_between_selected = []
            for i, station_i in enumerate(selected):
                for j, station_j in enumerate(selected):
                    if i < j:
                        idx_i = station_i['candidate_data']['idx']
                        idx_j = station_j['candidate_data']['idx']
                        if (idx_i < len(self.distances) and 
                            idx_j < len(self.distances[0])):
                            distances_between_selected.append(self.distances[idx_i, idx_j])
            
            if distances_between_selected:
                avg_distance = sum(distances_between_selected) / len(distances_between_selected)
                min_distance = min(distances_between_selected)
                print(f"Average inter-station distance: {avg_distance:.1f} miles")
                print(f"Minimum inter-station distance: {min_distance:.1f} miles")
        
        return results

    def export_results(self, filename=None):
        """Export optimization results to CSV and JSON"""
        if not hasattr(self, 'optimization_results') or not self.optimization_results:
            print("No results to export. Run optimize_portfolio() first.")
            return
        
        import pandas as pd
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            base_filename = f"h2_optimization_results_{timestamp}"
        else:
            base_filename = filename
        
        results = self.optimization_results
        
        # Export selected stations to CSV
        stations_data = []
        for station in results['selected_candidates']:
            
            station_row = {
                'candidate_index': station['index'],
                'longitude': station['position'][0],
                'latitude': station['position'][1],
                'capex_usd': station['capital_cost'],
                'npv_usd': station['npv'],
                'capacity': station['capacity'],
                'expected_demand': station['demand'],
                'total_lifetime_cost_million': station['total_lifetime_cost'] / 1_000_000,
                'npv_million': station['npv'] / 1_000_000
            }
            # Add all candidate data fields
            for key, value in station['candidate'].items():
                if key not in station_row:  # Avoid duplicates
                    station_row[f'candidate_{key}'] = value
            
            stations_data.append(station_row)
        
        stations_df = pd.DataFrame(stations_data)
        csv_filename = f"{base_filename}_stations.csv"
        stations_df.to_csv(csv_filename, index=False)
        print(f"Selected stations exported to: {csv_filename}")
        
        # Export summary results to JSON
        summary_results = {
            'optimization_summary': {
                'timestamp': timestamp,
                'solver_used': results['solver_used'],
                'model_status': results['model_status'],
                'num_stations_selected': results['num_stations'],
                'total_cost_usd': results['total_cost'],
                'total_npv_usd': results['total_npv'],
                'budget_utilization_pct': results['budget_utilization'] * 100
            },
            'performance_metrics': {
                k: v for k, v in results.items() 
                if k in ['solve_time', 'mip_gap', 'objective_value']
            },
            'selected_station_indices': [s['index'] for s in results['selected_candidates']]
        }
        
        json_filename = f"{base_filename}_summary.json"
        with open(json_filename, 'w') as f:
            json.dump(summary_results, f, indent=2)
        print(f"Summary results exported to: {json_filename}")
        
        return csv_filename, json_filename
        
    
    
    
    def generate_outputs(self, output_dir="h2_station_outputs"):
        """
        Generate all output files and visualizations.
        """
        print(f"Generating outputs to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save portfolio locations
        if hasattr(self, 'portfolio') and len(self.portfolio) > 0:
            portfolio_output = self.portfolio.copy()
            portfolio_output['rank'] = range(1, len(portfolio_output) + 1)
            portfolio_output.to_file(
                os.path.join(output_dir, "optimal_portfolio.geojson"), 
                driver="GeoJSON"
            )
        
        # 2. Save cluster analysis
        if hasattr(self, 'clusters') and len(self.clusters) > 0:
            self.clusters.to_file(
                os.path.join(output_dir, "location_clusters.geojson"),
                driver="GeoJSON"
            )
        
        # 3. Save scored candidates
        if hasattr(self, 'candidates') and len(self.candidates) > 0:
            num_top_candidates = min(100, len(self.candidates))
            # Top 100 candidates
            
            
            # check that column includes a column name that includes 'npv' or similar
            if 'npv' not in self.candidates.columns and 'optimal_npv' not in self.candidates.columns and 'npv_proxy' not in self.candidates.columns:
                print("No NPV data available for top potfolio stations.")
                return
            
            if 'npv' in self.candidates.columns:
                # get the subset of results for top 10 stations where npv is sorted in descending order (do not use nlargest)
                top_candidates = self.candidates.sort_values(by='npv', ascending=False).head(num_top_candidates)
            elif 'optimal_npv' in self.candidates.columns:
                top_candidates = self.candidates.sort_values(by='optimal_npv', ascending=False).head(num_top_candidates)
            elif 'npv_proxy' in self.candidates.columns:
                top_candidates = self.candidates.sort_values(by='npv_proxy', ascending=False).head(num_top_candidates)
            else:
                print("Warning: No NPV column found for top stations.")
                # Fallback to first num_candidates rows if no NPV data
                top_candidates = self.candidates.head(num_top_candidates)
            
            
            #top_candidates = self.candidates.nlargest(num_top_candidates, 'npv')
            
            top_candidates.to_file(
                os.path.join(output_dir, "top_100_candidates.geojson"),
                driver="GeoJSON"
            )
        
        # 4. Create visualizations
        self._create_visualizations(output_dir)
        
        # 5. Generate validation visualizations
        if hasattr(self, 'validation_results'):
            print("\nGenerating validation visualizations...")
            validation_dir = os.path.join(output_dir, 'validation')
            self.visualize_validation_results(save_plots=True, output_dir=validation_dir)
            print(f"  Validation visualizations saved to {validation_dir}/")

        
        # 6. Generate summary report
        self._generate_summary_report(output_dir)



    def _create_visualizations(self, output_dir):
        """Create visualization plots."""
        
        # 1. Demand surface heatmap
        plt.figure(figsize=(12, 10))
        plt.contourf(
            self.demand_surface['x'] / 1609.34,  # Convert to miles
            self.demand_surface['y'] / 1609.34,
            self.demand_surface['demand'],
            levels=20,
            cmap='YlOrRd'
        )
        plt.colorbar(label='H2 Demand (kg/day)')
        plt.xlabel('X (miles)')
        plt.ylabel('Y (miles)')
        plt.title('Hydrogen Demand Surface')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'demand_surface.png'), dpi=300)
        plt.close()
        
        # 2. Score surface heatmap
        if hasattr(self, 'score_surface'):
            plt.figure(figsize=(12, 10))
            plt.contourf(
                self.score_surface['x'] / 1609.34,
                self.score_surface['y'] / 1609.34,
                self.score_surface['scores'],
                levels=20,
                cmap='viridis'
            )
            plt.colorbar(label='Marginal Utility Score')
            plt.xlabel('X (miles)')
            plt.ylabel('Y (miles)')
            plt.title('Station Siting Score Surface')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'score_surface.png'), dpi=300)
            plt.close()
        
        # 3. Portfolio map
        if hasattr(self, 'portfolio') and len(self.portfolio) > 0:
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Plot routes with traffic volume
            if hasattr(self, 'routes'):
                self.routes.plot(
                    ax=ax,
                    column='truck_aadt',
                    cmap='Blues',
                    linewidth=1,
                    alpha=0.6
                )
            
            # Plot existing stations
            if len(self.existing_stations) > 0:
                self.existing_stations.plot(
                    ax=ax,
                    color='gray',
                    marker='s',
                    markersize=50,
                    alpha=0.5,
                    label='Existing Stations'
                )
            
            # Plot selected portfolio
            self.portfolio.plot(
                ax=ax,
                color='red',
                marker='*',
                markersize=200,
                edgecolor='black',
                linewidth=1,
                label='Selected Stations'
            )
            
            # Add labels
            for idx, row in self.portfolio.iterrows():
                ax.annotate(
                    f"{row.name + 1}",
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold'
                )
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_title('Optimal H2 Station Portfolio')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'portfolio_map.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Economic metrics distribution
        if hasattr(self, 'candidates') and len(self.candidates) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # NPV distribution
            axes[0, 0].hist(self.candidates['npv_proxy'] / 1e6, bins=50, edgecolor='black')
            axes[0, 0].set_xlabel('NPV Proxy ($M)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('NPV Distribution')
            axes[0, 0].axvline(0, color='red', linestyle='--', label='Break-even')
            axes[0, 0].legend()
            
            # Utilization distribution
            axes[0, 1].hist(self.candidates['utilization_rate'], bins=50, edgecolor='black')
            axes[0, 1].set_xlabel('Utilization Rate')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Utilization Rate Distribution')
            
            # Payback period distribution
            payback_clipped = self.candidates['payback_years'].clip(0, 20)
            axes[1, 0].hist(payback_clipped, bins=50, edgecolor='black')
            axes[1, 0].set_xlabel('Payback Period (years)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Payback Period Distribution')
            
            # Demand vs NPV scatter
            axes[1, 1].scatter(
                self.candidates['expected_demand_kg_day'],
                self.candidates['npv_proxy'] / 1e6,
                alpha=0.5,
                s=20
            )
            axes[1, 1].set_xlabel('Expected Demand (kg/day)')
            axes[1, 1].set_ylabel('NPV Proxy ($M)')
            axes[1, 1].set_title('Demand vs NPV Relationship')
            axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'economic_metrics.png'), dpi=300)
            plt.close()

    
    def visualize_competition_graph_old(self, save_path=None):
        """Visualize the competition network graph."""
        if not self._competition_graph:
            print("No competition graph to visualize")
            return
            
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node_id, node_data in self._competition_graph.nodes.items():
            G.add_node(node_id, 
                      demand=node_data['demand'],
                      type=node_data['type'],
                      market_share=node_data.get('market_share', 0))
        
        # Add edges
        for edge_key, edge_data in self._competition_graph.edges.items():
            G.add_edge(edge_key[0], edge_key[1], 
                      weight=1/edge_data['weight'],  # Inverse for visualization
                      distance=edge_data['distance'])
        
        # Create layout
        pos = {}
        for node_id, node_data in self._competition_graph.nodes.items():
            geom = node_data['geometry']
            pos[node_id] = (geom.x, geom.y)
        
        # Draw
        plt.figure(figsize=(12, 10))
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            node_type = self._competition_graph.nodes[node]['type']
            if node_type == 'existing':
                node_colors.append('gray')
            elif node_type == 'selected':
                node_colors.append('red')
            else:  # candidate
                node_colors.append('lightblue')
        
        # Size nodes by demand
        node_sizes = [self._competition_graph.nodes[node]['demand'] / 10 for node in G.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Add labels for selected stations
        labels = {node: str(node) for node in G.nodes() 
                 if self._competition_graph.nodes[node]['type'] == 'selected'}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title("Competition Network Graph")
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def visualize_competition_graph(self, save_path=None, filename='competition_graph.png'):
        """Visualize the competition graph network"""
        
        if self._competition_graph is None:
            print("No competition graph to visualize")
            return
            
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create networkx graph from competition graph
        G = nx.Graph()
        
        # Add nodes with their attributes
        for node_id, node_data in self._competition_graph.nodes.items():
            G.add_node(node_id, **node_data)
        
        # Add edges
        for edge_key, edge_data in self._competition_graph.edges.items():
            parts = edge_key.split('-')
            if len(parts) == 2:
                # Convert edge node IDs to appropriate types
                node1_str, node2_str = parts
                
                # Find matching nodes
                node1, node2 = None, None
                for n in G.nodes():
                    if str(n) == node1_str:
                        node1 = n
                    if str(n) == node2_str:
                        node2 = n
                        
                if node1 is not None and node2 is not None:
                    G.add_edge(node1, node2, **edge_data)
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Get positions
        pos = {}
        for node_id in G.nodes():
            if node_id in self._competition_graph.nodes:
                node_data = self._competition_graph.nodes[node_id]
                if 'x' in node_data and 'y' in node_data:
                    pos[node_id] = (node_data['x'], node_data['y'])
        
        # Color nodes by type
        node_colors = []
        for node_id in G.nodes():
            if node_id in self._competition_graph.nodes:
                node_type = self._competition_graph.nodes[node_id].get('type', 'candidate')
                if node_type == 'existing':
                    node_colors.append('red')
                elif node_type == 'selected':
                    node_colors.append('green')
                else:
                    node_colors.append('lightblue')
            else:
                node_colors.append('gray')  # Unknown nodes
        
        # Draw the graph
        if pos:
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50, alpha=0.6)
            
            # Draw edges
            edges = G.edges()
            if edges:
                nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        plt.title("Competition Network Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        if save_path:
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        
        
        print(f"Competition graph visualization saved to {filepath}")
    
    
    
    def get_competition_summary(self):
        """Get summary statistics from competition graph."""
        if not self._competition_graph:
            return None
            
        graph = self._competition_graph
        
        summary = {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'nodes_by_type': {},
            'average_market_share': {},
            'total_demand_by_type': {}
        }
        
        # Aggregate by type
        for node_id, node_data in graph.nodes.items():
            node_type = node_data['type']
            
            if node_type not in summary['nodes_by_type']:
                summary['nodes_by_type'][node_type] = 0
                summary['average_market_share'][node_type] = []
                summary['total_demand_by_type'][node_type] = 0
            
            summary['nodes_by_type'][node_type] += 1
            summary['average_market_share'][node_type].append(node_data.get('market_share', 0))
            summary['total_demand_by_type'][node_type] += node_data['demand']
        
        # Calculate averages
        for node_type in summary['average_market_share']:
            shares = summary['average_market_share'][node_type]
            summary['average_market_share'][node_type] = np.mean(shares) if shares else 0
        
        return summary



    def visualize_validation_results(self, save_plots=True, output_dir='validation_plots'):
        """
        Create comprehensive visualizations for validation results.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Circle
        import contextily as ctx
        
        if not hasattr(self, 'validation_results'):
            print("No validation results to visualize. Run validate_results() first.")
            return
        
        # Create output directory if saving
        if save_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(24, 16))
        
        # Use GridSpec for better control over subplot sizes
        gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.3)  # Increased spacing
        
        
        # 1. MAIN MAP - Existing vs Selected Stations with Routes
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_infrastructure_comparison(ax1)
        
        # 2. DISTANCE DISTRIBUTION
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_distance_distribution(ax2)
        
        # 3. ROUTE COVERAGE ANALYSIS
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_route_coverage(ax3)
        
        # 4. DEMAND HEAT MAP WITH STATIONS
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_demand_correlation(ax4)
        
        # 5. REGIONAL DISTRIBUTION
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_regional_distribution(ax5)
        
        # 6. COVERAGE EXPANSION MAP
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_coverage_expansion(ax6)
        
        # 7. COMPETITIVE VS COMPLEMENTARY PIE CHART
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_site_classification(ax7)
        
        # 8. DEMAND COMPARISON
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_site_classification(ax8)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/validation_comprehensive.png", dpi=300, bbox_inches='tight')
            print(f"  Main validation plot saved to {output_dir}/validation_comprehensive.png")
        plt.close()
        
        # Create additional detailed maps
        try:
            self._create_detailed_maps(save_plots, output_dir)
            
        except Exception as e:
            print(f"Warning: Detailed comparison map failed: {e}")
            
        

        #self._create_detailed_comparison_map(save_plots, output_dir)



    def _plot_infrastructure_comparison(self, ax):
        
        
        """Plot existing vs selected stations with route network."""
        # Plot routes with low alpha
        self.routes.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
        
        # Plot existing stations
        if len(self.existing_stations) > 0:
            self.existing_stations.plot(ax=ax, color='blue', markersize=100, 
                                    marker='o', label='Existing H2 Stations', 
                                    edgecolor='darkblue', linewidth=2, alpha=0.7)
        
        # Plot selected stations
        self.portfolio.plot(ax=ax, color='red', markersize=150, 
                        marker='*', label='Selected Stations', 
                        edgecolor='darkred', linewidth=2)
        
        # Add distance lines for nearest pairs
        for idx, row in self.portfolio.iterrows():
            if 'nearest_existing_idx' in row:
                nearest_existing = self.existing_stations.iloc[int(row['nearest_existing_idx'])]
                ax.plot([row.geometry.x, nearest_existing.geometry.x],
                    [row.geometry.y, nearest_existing.geometry.y],
                    'k--', alpha=0.3, linewidth=1)
        
        ax.set_title('Infrastructure Comparison: Existing vs Selected Stations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add basemap if available
        try:
            ctx.add_basemap(ax, crs=self.routes.crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass

    def _plot_distance_distribution(self, ax):
        """Plot distribution of distances to existing infrastructure."""
        distances = self.portfolio['dist_to_nearest_existing_miles'].values
        
        # Handle potential infinity or NaN values
        distances = distances[np.isfinite(distances)]
    
        if len(distances) == 0:
            ax.text(0.5, 0.5, 'No valid distance data available', 
                    transform=ax.transAxes, ha='center', va='center')
            return
        
        # Create histogram with KDE
        #sns.histplot(data=distances, bins=20, kde=True, ax=ax, color='skyblue', edgecolor='black')
        # Seaboprn breaks. try matplotlib
        
        # Create histogram using matplotlib instead of seaborn
        n, bins, patches = ax.hist(distances, bins=20, color='skyblue', 
                                edgecolor='black', alpha=0.7, density=True)
        
        # Add KDE using scipy instead of seaborn
        try:
            from scipy import stats
            if len(distances) > 1:
                kde = stats.gaussian_kde(distances)
                x_range = np.linspace(distances.min(), distances.max(), 200)
                kde_values = kde(x_range)
                ax2 = ax.twinx()
                ax2.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
                ax2.set_ylabel('Density')
                ax2.set_ylim(bottom=0)
                # Hide y-axis ticks for cleaner look
                ax2.set_yticks([])
        except Exception as e:
            print(f"Warning: Could not add KDE overlay: {e}")
        


        
        # Add vertical lines for key statistics
        ax.axvline(distances.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {distances.mean():.1f} mi')
        ax.axvline(np.median(distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.1f} mi')
        
        # Add competitive threshold
        threshold = self.config.get('min_station_spacing_miles', 5.0)
        ax.axvline(threshold, color='orange', linestyle=':', linewidth=2, label=f'Competitive Threshold: {threshold} mi')
        
        ax.set_xlabel('Distance to Nearest Existing Station (miles)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Distances to Existing Infrastructure', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_route_coverage(self, ax):
        """Plot route coverage analysis."""
        coverage_data = self.validation_results['routes_coverage']
        
        # Create stacked bar chart
        categories = ['Existing\nOnly', 'Both', 'Selected\nOnly', 'Neither']
        values = [coverage_data['only_existing'], coverage_data['both'], 
                coverage_data['only_selected'], coverage_data['neither']]
        colors = ['blue', 'purple', 'red', 'gray']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{value}\n({value/sum(values)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Routes')
        ax.set_title('Route Coverage Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_demand_correlation(self, ax):
        """Plot demand heat map with station locations."""
        if hasattr(self, 'demand_surface'):
            # Plot demand surface
            im = ax.contourf(self.demand_surface['x'], self.demand_surface['y'], 
                            self.demand_surface['demand'], levels=20, cmap='YlOrRd', alpha=0.7)
            
            # Add contour lines
            ax.contour(self.demand_surface['x'], self.demand_surface['y'], 
                    self.demand_surface['demand'], levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            # Plot stations on top
            if len(self.existing_stations) > 0:
                ax.scatter(self.existing_stations.geometry.x, self.existing_stations.geometry.y,
                        c='blue', s=100, marker='o', edgecolor='darkblue', linewidth=2,
                        label='Existing Stations', zorder=5)
            
            ax.scatter(self.portfolio.geometry.x, self.portfolio.geometry.y,
                    c='red', s=150, marker='*', edgecolor='darkred', linewidth=2,
                    label='Selected Stations', zorder=6)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='H2 Demand (kg/day)')
            
            ax.set_title('Demand Intensity vs Station Locations', fontsize=14, fontweight='bold')
            ax.legend()
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            ax.text(0.5, 0.5, 'Demand surface not available', 
                    transform=ax.transAxes, ha='center', va='center')



    def _plot_regional_distribution(self, ax):
        """Plot regional distribution comparison."""
        regional_data = self.validation_results['regional_distribution']
        
        regions = list(regional_data.keys())
        existing_counts = [regional_data[r]['existing'] for r in regions]
        selected_counts = [regional_data[r]['selected'] for r in regions]
        
        x = np.arange(len(regions))
        width = 0.35
        
        # Create grouped bar chart
        ax.bar(x - width/2, existing_counts, width, label='Existing', color='blue', edgecolor='black')
        ax.bar(x + width/2, selected_counts, width, label='Selected', color='red', edgecolor='black')
        
        ax.set_xlabel('Region')
        ax.set_ylabel('Number of Stations')
        ax.set_title('Regional Distribution of Stations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_coverage_expansion(self, ax):
        """Plot service area coverage expansion."""
        service_radius_m = self.config['service_radius_miles'] * 1609.34
        
        # Plot base map
        self.routes.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.2)
        
        # Plot existing coverage areas
        if len(self.existing_stations) > 0:
            existing_buffer = self.existing_stations.buffer(service_radius_m)
            existing_union = unary_union(existing_buffer)
            #gpd.GeoSeries([existing_union]).plot(ax=ax, alpha=0.3, color='blue', 
                #edgecolor='darkblue', linewidth=1)
    
            
            if hasattr(existing_union, 'geoms'):
                for geom in existing_union.geoms:
                    x, y = geom.exterior.xy
                    ax.fill(x, y, alpha=0.3, color='blue', edgecolor='darkblue', linewidth=1)
            else:
                x, y = existing_union.exterior.xy
                ax.fill(x, y, alpha=0.3, color='blue', edgecolor='darkblue', linewidth=1)
        
        # Plot new coverage areas (non-overlapping parts)
        portfolio_buffer = self.portfolio.buffer(service_radius_m)
        portfolio_union = unary_union(portfolio_buffer)
        
        if len(self.existing_stations) > 0:
            new_coverage = portfolio_union.difference(existing_union)
            
            #if new_coverage.area > 0:
                #gpd.GeoSeries([new_coverage]).plot(ax=ax, alpha=0.4, color='red', 
                                                #edgecolor='darkred', linewidth=1)
                    
            
            if hasattr(new_coverage, 'geoms'):
                for geom in new_coverage.geoms:
                    if geom.area > 0:
                        x, y = geom.exterior.xy
                        ax.fill(x, y, alpha=0.4, color='red', edgecolor='darkred', linewidth=1)
                        
            elif new_coverage.area > 0:
                x, y = new_coverage.exterior.xy
                ax.fill(x, y, alpha=0.4, color='red', edgecolor='darkred', linewidth=1)
        
        # Plot stations
        if len(self.existing_stations) > 0:
            self.existing_stations.plot(ax=ax, color='blue', markersize=50, marker='o', zorder=5)
        self.portfolio.plot(ax=ax, color='red', markersize=75, marker='*', zorder=6)
        
        ax.set_title('Service Area Coverage Expansion', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.3, edgecolor='darkblue', label='Existing Coverage'),
            Patch(facecolor='red', alpha=0.4, edgecolor='darkred', label='New Coverage'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Existing Stations'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label='Selected Stations')
        ]
        ax.legend(handles=legend_elements, loc='best')

    def _plot_site_classification(self, ax):
        """Plot pie chart of competitive vs complementary sites."""
        val = self.validation_results
        sizes = [val['competitive_sites'], val['complementary_sites']]
        labels = ['Competitive', 'Complementary']
        colors = ['orange', 'green']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                        startangle=90, textprops={'fontsize': 12})
        
        ax.set_title('Site Classification')


    def _plot_demand_comparison(self, ax):
        """Plot demand comparison between existing and selected locations."""
        if self.validation_results.get('demand_improvement') is not None:
            # Extract data from candidates
            avg_existing = 0  # Will be calculated from validation results
            avg_selected = self.portfolio['total_demand'].mean()
            avg_all = self.candidates['total_demand'].mean()
            
            categories = ['Existing\nStations', 'Selected\nStations', 'All\nCandidates']
            values = [avg_existing, avg_selected, avg_all]
            colors = ['blue', 'red', 'gray']
            
            bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                        f'{value:.0f}',
                        ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Average Daily Demand (kg H2)')
            ax.set_title('Demand Comparison')
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Demand comparison not available', 
                    transform=ax.transAxes, ha='center', va='center')


    def _create_detailed_comparison_map(self, save_plots, output_dir):
        """Create detailed comparison map with competitive/complementary classification."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Plot routes
        self.routes.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
        
        # Plot existing stations
        if len(self.existing_stations) > 0:
            self.existing_stations.plot(ax=ax, color='blue', markersize=100, 
                                    marker='o', alpha=0.5, edgecolor='darkblue',
                                    label='Existing Stations')
        
        # Plot selected stations colored by distance
        threshold = self.config.get('min_station_spacing_miles', 25)
        competitive_mask = self.portfolio['dist_to_nearest_existing_miles'] < threshold
        
        # Competitive sites (close to existing)
        if competitive_mask.any():
            self.portfolio[competitive_mask].plot(ax=ax, color='orange', markersize=150,
                                                marker='*', label=f'Competitive (<{threshold} mi)',
                                                edgecolor='darkorange', linewidth=2)
        
        # Complementary sites (far from existing)
        if (~competitive_mask).any():
            self.portfolio[~competitive_mask].plot(ax=ax, color='green', markersize=150,
                                                marker='*', label=f'Complementary (≥{threshold} mi)',
                                                edgecolor='darkgreen', linewidth=2)
        
        # Add distance lines for nearest pairs
        for idx, row in self.portfolio.iterrows():
            if 'nearest_existing_idx' in row:
                nearest_existing = self.existing_stations.iloc[int(row['nearest_existing_idx'])]
                ax.plot([row.geometry.x, nearest_existing.geometry.x],
                    [row.geometry.y, nearest_existing.geometry.y],
                    'k--', alpha=0.3, linewidth=1)
        
        ax.set_title('Competitive vs Complementary Site Selection', fontsize=16, fontweight='bold')
        ax.legend()
        ax.set_aspect('equal')
        
        if save_plots:
            plt.savefig(f"{output_dir}/competitive_vs_complementary.png", dpi=300, bbox_inches='tight')
        plt.close()




    def _create_detailed_maps(self, save_plots, output_dir):
        """Create additional detailed maps."""
        # 1. Competitive vs Complementary Sites Map
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot routes
        self.routes.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.3)
        
        # Plot existing stations
        if len(self.existing_stations) > 0:
            self.existing_stations.plot(ax=ax, color='blue', markersize=100, 
                                    marker='o', alpha=0.5, edgecolor='darkblue')
        
        # Plot selected stations colored by distance
        threshold = self.config.get('min_station_spacing_miles', 25)
        competitive_mask = self.portfolio['dist_to_nearest_existing_miles'] < threshold
        
        # Competitive sites (close to existing)
        self.portfolio[competitive_mask].plot(ax=ax, color='orange', markersize=150,
                                            marker='*', label=f'Competitive (<{threshold} mi)',
                                            edgecolor='darkorange', linewidth=2)
        
        # Complementary sites (far from existing)
        self.portfolio[~competitive_mask].plot(ax=ax, color='green', markersize=150,
                                            marker='*', label=f'Complementary (≥{threshold} mi)',
                                            edgecolor='darkgreen', linewidth=2)
        
        ax.set_title('Competitive vs Complementary Site Selection', fontsize=16, fontweight='bold')
        ax.legend()
        
        if save_plots:
            plt.savefig(f"{output_dir}/competitive_vs_complementary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Route Alignment Analysis Map
        self._create_route_alignment_map(save_plots, output_dir)
        

    def _create_route_alignment_map(self, save_plots, output_dir):
        """Create detailed route alignment analysis map."""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Color routes by coverage status
        coverage_data = self.validation_results['routes_coverage']
        
        # Get route indices for each category
        routes_with_existing = set()
        routes_with_selected = set()
        
        # Identify routes near stations
        buffer_dist = 1000  # 1km
        
        if len(self.existing_stations) > 0:
            existing_buffer = self.existing_stations.buffer(buffer_dist).unary_union
            for idx, route in self.routes.iterrows():
                if route.geometry.intersects(existing_buffer):
                    routes_with_existing.add(idx)
        
        portfolio_buffer = self.portfolio.buffer(buffer_dist).unary_union
        for idx, route in self.routes.iterrows():
            if route.geometry.intersects(portfolio_buffer):
                routes_with_selected.add(idx)
        
        # Plot routes by category
        for idx, route in self.routes.iterrows():
            if idx in routes_with_existing and idx in routes_with_selected:
                color, width, label = 'purple', 2, 'Both'
            elif idx in routes_with_existing:
                color, width, label = 'blue', 1.5, 'Existing Only'
            elif idx in routes_with_selected:
                color, width, label = 'red', 1.5, 'Selected Only'
            else:
                color, width, label = 'lightgray', 0.5, 'Neither'
            
            route_gdf = gpd.GeoDataFrame([route], crs=self.routes.crs)
            route_gdf.plot(ax=ax, color=color, linewidth=width, alpha=0.7)
        
        # Plot stations
        if len(self.existing_stations) > 0:
            self.existing_stations.plot(ax=ax, color='blue', markersize=100,
                                    marker='o', edgecolor='darkblue', linewidth=2, zorder=5)
        self.portfolio.plot(ax=ax, color='red', markersize=150,
                        marker='*', edgecolor='darkred', linewidth=2, zorder=6)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='purple', linewidth=3, label='Routes with Both'),
            Line2D([0], [0], color='blue', linewidth=2, label='Routes with Existing Only'),
            Line2D([0], [0], color='red', linewidth=2, label='Routes with Selected Only'),
            Line2D([0], [0], color='lightgray', linewidth=1, label='Routes with Neither'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                markersize=10, label='Existing Stations', linestyle=''),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                markersize=12, label='Selected Stations', linestyle='')
        ]
        
        ax.legend(handles=legend_elements, loc='best', framealpha=0.9)
        ax.set_title('Route Coverage Alignment Analysis', fontsize=16, fontweight='bold')
        
        if save_plots:
            plt.savefig(f"{output_dir}/route_alignment_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_enhanced_visualizations(self, output_dir='enhanced_outputs'):
        """
        Updated visualizations to show flexible capacity and tipping point analyses results.
        
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'iterative_selection_results') or len(self.iterative_selection_results) == 0:
            print("No iterative selection results to visualize. Run 'run_iterative_station_selection' first.")
            return
        
        results = self.iterative_selection_results
        
        print("Creating enhanced visualizations...")
        
        #  New visualization showing capacity distribution
        self._create_capacity_distribution_chart(results, output_dir)
        
        # Updated existing visualizations
        self._create_demand_capacity_map(results, output_dir)
        self._create_npv_evolution_chart(results, output_dir)
        self._create_competitive_landscape(results, output_dir)
        self._create_economic_dashboard(results, output_dir)
        
        print(f"Visualizations saved to {output_dir}/")


    def _create_capacity_distribution_chart(self, results, output_dir):
        """
         New method to visualize capacity distribution across stations.
        """
        
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Capacity distribution histogram
        ax1.hist(results['capacity_kg_day'], bins=24, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(results['capacity_kg_day'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {results["capacity_kg_day"].mean():.0f} kg/day')
        ax1.set_xlabel('Station Capacity (kg/day)')
        ax1.set_ylabel('Number of Stations')
        ax1.set_title('Distribution of Optimal Station Capacities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Capacity vs Demand scatter
        scatter = ax2.scatter(results['expected_demand_kg_day'], 
                            results['capacity_kg_day'],
                            c=results['optimal_utilization'], 
                            cmap='RdYlGn',
                            s=100, alpha=0.7, edgecolors='black')
        
        # Add perfect sizing line
        max_val = max(results['expected_demand_kg_day'].max(), results['capacity_kg_day'].max())
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 line')
        
        # Add typical sizing ratios
        ax2.plot([0, max_val], [0, max_val * 1.2], 'g--', alpha=0.3, label='1.2x sizing')
        ax2.plot([0, max_val], [0, max_val * 1.5], 'b--', alpha=0.3, label='1.5x sizing')
        
        plt.colorbar(scatter, ax=ax2, label='Utilization')
        ax2.set_xlabel('Expected Demand (kg/day)')
        ax2.set_ylabel('Optimal Capacity (kg/day)')
        ax2.set_title('Capacity Sizing vs Demand')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. NPV improvement from optimization
        if 'npv_improvement_vs_default' in results.columns:
            improvements = results['npv_improvement_vs_default'] / 1e6
            positive = improvements[improvements > 0]
            negative = improvements[improvements < 0]
            
            ax3.hist([positive, negative], bins=20, label=['Improvement', 'Reduction'],
                    color=['green', 'red'], alpha=0.7, edgecolor='black')
            ax3.axvline(0, color='black', linestyle='-', linewidth=2)
            ax3.set_xlabel('NPV Change vs Default Capacity ($M)')
            ax3.set_ylabel('Number of Stations')
            ax3.set_title('NPV Impact of Capacity Optimization')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Demand cushion distribution
        ax4.hist(results['demand_cushion_pct'], bins=20, edgecolor='black', alpha=0.7)
        
        # Add risk zones
        ax4.axvspan(0, 20, alpha=0.2, color='red', label='High Risk (<20%)')
        ax4.axvspan(20, 40, alpha=0.2, color='yellow', label='Moderate Risk')
        ax4.axvspan(40, 100, alpha=0.2, color='green', label='Low Risk (>40%)')
        
        ax4.set_xlabel('Demand Cushion (%)')
        ax4.set_ylabel('Number of Stations')
        ax4.set_title('Risk Profile: Demand Cushion Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Capacity Optimization Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/capacity_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _create_npv_evolution_chart(self, results, output_dir):
        """
        CHANGED: Updated to show flexible capacity results.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # NPV per station with optimal capacity
        stations = results.head(50)  # Top 50 for clarity
        x = range(len(stations))
        
        # CHANGED: Color bars by capacity tier
        capacity_colors = []
        for cap in stations['capacity_kg_day']:
            if cap <= 1000:
                capacity_colors.append('#3498db')  # Blue - small
            elif cap <= 2000:
                capacity_colors.append('#2ecc71')  # Green - medium
            elif cap <= 3000:
                capacity_colors.append('#f39c12')  # Orange - large
            else:
                capacity_colors.append('#e74c3c')  # Red - XL
        
        bars = ax1.bar(x, stations['npv'] / 1e6, color=capacity_colors, 
                    edgecolor='black', linewidth=0.5)
        
        # CHANGED: Show capacity as line overlay
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, stations['capacity_kg_day'], 
                    'k-o', markersize=4, linewidth=2,
                    label='Optimal Capacity', alpha=0.8)
        
        ax1.set_xlabel('Station Rank')
        ax1.set_ylabel('NPV ($ millions)')
        ax1_twin.set_ylabel('Optimal Capacity (kg/day)', color='black')
        ax1_twin.tick_params(axis='y', labelcolor='black')
        ax1.set_title('Station NPV with Optimal Capacity Selection')
        ax1.grid(True, alpha=0.3)
        
        # Add capacity tier legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='≤1000 kg/day'),
            Patch(facecolor='#2ecc71', label='1001-2000 kg/day'),
            Patch(facecolor='#f39c12', label='2001-3000 kg/day'),
            Patch(facecolor='#e74c3c', label='>3000 kg/day')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', title='Capacity Tier')
        
        # Cumulative NPV and ROI
        ax2.plot(results['cumulative_investment'] / 1e6,
                results['cumulative_npv'] / 1e6,
                'g-', linewidth=3, label='Cumulative NPV')
        
        # CHANGED: Show investment efficiency
        investment_efficiency = results['cumulative_npv'] / results['cumulative_investment']
        ax2_twin = ax2.twinx()
        ax2_twin.plot(results['cumulative_investment'] / 1e6, 
                    investment_efficiency,
                    'r--', linewidth=2, label='NPV/Investment Ratio', alpha=0.8)
        
        ax2.fill_between(results['cumulative_investment'] / 1e6,
                        0, results['cumulative_npv'] / 1e6,
                        alpha=0.3, color='green')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Cumulative Investment ($ millions)')
        ax2.set_ylabel('Cumulative NPV ($ millions)')
        ax2_twin.set_ylabel('NPV/Investment Ratio', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2.set_title('Investment Performance with Optimized Capacities')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/npv_evolution_flexible_capacity.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _create_demand_capacity_map(self, results, output_dir):
        """
        Updated to show actual selected capacities.
        """
        
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot base network if available
        if hasattr(self, 'routes') and self.routes is not None:
            self.routes.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5)
            
            
        # Size markers by actual capacity, not just utilization
        # Normalize capacity for marker sizing
        min_cap = results['capacity_kg_day'].min()
        max_cap = results['capacity_kg_day'].max()
        
        if max_cap == min_cap:
            results['marker_size'] = 200  # uniform size if all capacities equal
        else:
            results['marker_size'] = (
                50 + 450 * (results['capacity_kg_day'] - min_cap) / (max_cap - min_cap)
        )


        # Create scatter plot colored by utilization
        scatter = ax.scatter(
            results.geometry.x,
            results.geometry.y,
            s=results['marker_size'],
            c=results['optimal_utilization'],
            cmap='RdYlGn',
            vmin=0.3, vmax=1.0,  # Focus color range on typical utilization
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )

        
        # Add labels for top stations
        for idx, row in results.head(20).iterrows():
            ax.annotate(
                f"{row['station_id']}\n{row['capacity_kg_day']:.0f}kg",
                (row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
            
        # Colorbar for utilization
        cbar = plt.colorbar(scatter, ax=ax, label='Utilization Rate', shrink=0.8)
        
        
        def area_to_radius(area_pts2):
            # Convert scatter area (points²) → marker radius (points).
            return math.sqrt(area_pts2 / math.pi)
        
        # Determine capacity bins from data
        # Optional tolerance (round down/up to nearest 500 kg/day)
        bin_step = 500
        lo = bin_step * math.floor(min_cap / bin_step)
        hi = bin_step * math.ceil(max_cap / bin_step)

        # Make 6–8 nicely spaced ticks across [lo, hi]; fall back to step=bin_step
        n_bins = 7
        if hi - lo < n_bins * bin_step:
            bins = np.arange(lo, hi + bin_step, bin_step)
        else:
            bins = np.linspace(lo, hi, n_bins, dtype=int)
        
        # Build proxy handles for legend
        
        handles, labels = [], []
        for cap in bins:
            if max_cap == min_cap:
                marker_area = 200
            else:
                marker_area = 50 + 450 * (cap - min_cap) / (max_cap - min_cap)

            handles.append(
                mlines.Line2D(
                    [], [], marker='o', linestyle='none',
                    markerfacecolor='gray', markeredgecolor='black', alpha=0.7,
                    markersize=area_to_radius(marker_area)
                )
            )
            labels.append(f'{cap:,} kg/day')

        ax.legend(
            handles, labels,
            loc='upper right',
            title='Station Capacity',
            framealpha=0.9,
            ncol=2
        )
        
        
        ax.set_title('H2 Station Network - Optimized Capacities and Utilization', fontsize=16)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
        os.path.join(output_dir, "station_network_optimized_capacity.png"),
        dpi=300, bbox_inches='tight'
        )
        
        plt.close()



    def _create_competitive_landscape(self, results, output_dir):
            """Analyze competitive dynamics."""
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Competitors within radius distribution
            ax1.hist(results['competitors_within_radius'], bins=20, 
                    edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_xlabel('Number of Competitors within Service Radius')
            ax1.set_ylabel('Number of Stations')
            ax1.set_title('Competition Density Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 2. Demand capture probability
            scatter = ax2.scatter(results['competitors_within_radius'],
                                results['demand_capture_probability'],
                                c=results['npv'] / 1e6, cmap='RdYlGn',
                                s=100, alpha=0.7, edgecolors='black')
            ax2.set_xlabel('Competitors within Radius')
            ax2.set_ylabel('Demand Capture Probability')
            ax2.set_title('Market Share vs Competition')
            plt.colorbar(scatter, ax=ax2, label='NPV ($M)')
            ax2.grid(True, alpha=0.3)
            
            # 3. Competitive advantage
            ax3.scatter(results['iteration'], 
                    results['competitive_advantage_npv'] / 1e6,
                    c=results['utilization'], cmap='plasma',
                    s=80, alpha=0.7)
            ax3.set_xlabel('Selection Order')
            ax3.set_ylabel('Competitive Advantage NPV ($M)')
            ax3.set_title('Competitive Advantage Over Time')
            ax3.grid(True, alpha=0.3)
            
            # 4. Tipping capacity vs actual demand
            ax4.scatter(results['expected_demand_kg_day'],
                    results['tipping_capacity_kg_day'],
                    c=results['iteration'], cmap='viridis',
                    s=100, alpha=0.7, edgecolors='black')
            
            # Add reference lines
            max_val = max(results['tipping_capacity_kg_day'].max(),
                        results['expected_demand_kg_day'].max())
            ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='1:1 line')
            ax4.plot([0, max_val], [0, max_val * 1.5], 'g--', alpha=0.5, label='1.5x line')
            
            ax4.set_xlabel('Expected Demand (kg/day)')
            ax4.set_ylabel('Tipping Capacity (kg/day)')
            ax4.set_title('Economic Resilience: Tipping Capacity vs Demand')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/competitive_landscape.png', dpi=300, bbox_inches='tight')
            plt.close()
        
    def _create_economic_dashboard(self, results, output_dir):
        """Create comprehensive economic performance dashboard."""
        fig = plt.figure(figsize=(20, 12))
        
        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Utilization heatmap (top left, 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # Create utilization matrix by location bins
        x_bins = pd.cut(results.geometry.x, bins=10)
        y_bins = pd.cut(results.geometry.y, bins=10)
        utilization_matrix = results.groupby([x_bins, y_bins])['utilization'].mean().unstack()
        
        sns.heatmap(utilization_matrix, cmap='RdYlGn', center=0.7,
                cbar_kws={'label': 'Average Utilization'},
                ax=ax1, square=True)
        ax1.set_title('Geographic Utilization Heatmap', fontsize=14)
        ax1.set_xlabel('Longitude Bins')
        ax1.set_ylabel('Latitude Bins')
        
        # 2. Key metrics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        metrics_text = f"""
        KEY METRICS
        
        Total Stations: {len(results)}
        Total Investment: ${results['cumulative_investment'].iloc[-1]/1e9:.2f}B
        Total NPV: ${results['cumulative_npv'].iloc[-1]/1e9:.2f}B
        Average ROI: {(results['cumulative_npv'].iloc[-1]/results['cumulative_investment'].iloc[-1]*100):.1f}%
        Avg Utilization: {results['utilization'].mean():.1%}
        Avg Payback: {(results['cumulative_investment'].iloc[-1]/results['npv'].sum()*results['station_id'].max()/365):.1f} years
        """
        
        ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # 3. Demand served progression (middle right)
        ax3 = fig.add_subplot(gs[1, 2])
        
        cumulative_demand = results['expected_demand_kg_day'].cumsum()
        cumulative_capacity = results['station_id'] * self.config['station_capacity_kg_per_day']
        
        ax3.fill_between(results['station_id'], 0, cumulative_demand / 1000,
                        alpha=0.5, color='blue', label='Demand Served')
        ax3.plot(results['station_id'], cumulative_capacity / 1000,
                'r--', linewidth=2, label='Total Capacity')
        
        ax3.set_xlabel('Number of Stations')
        ax3.set_ylabel('H2 Volume (tons/day)')
        ax3.set_title('Demand Coverage Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. NPV distribution by utilization (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        
        utilization_bins = pd.cut(results['utilization'], bins=[0, 0.5, 0.7, 0.85, 1.0],
                                labels=['Low (<50%)', 'Medium (50-70%)', 
                                        'High (70-85%)', 'Very High (>85%)'])
        
        results_binned = results.groupby(utilization_bins)['npv'].mean() / 1e6
        bars = ax4.bar(range(len(results_binned)), results_binned.values,
                    color=['red', 'orange', 'lightgreen', 'darkgreen'])
        ax4.set_xticks(range(len(results_binned)))
        ax4.set_xticklabels(results_binned.index, rotation=45, ha='right')
        ax4.set_ylabel('Average NPV ($ millions)')
        ax4.set_title('NPV by Utilization Category')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Station selection pace (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Group by 10-station increments
        grouped = results.groupby(results['station_id'] // 10 * 10)
        avg_npv_by_group = grouped['npv'].mean() / 1e6
        
        ax5.plot(avg_npv_by_group.index, avg_npv_by_group.values,
                'b-o', linewidth=2, markersize=6)
        ax5.fill_between(avg_npv_by_group.index, 0, avg_npv_by_group.values,
                        alpha=0.3, color='blue')
        ax5.set_xlabel('Station Group (by 10s)')
        ax5.set_ylabel('Average NPV per Station ($M)')
        ax5.set_title('NPV Quality Over Selection Process')
        ax5.grid(True, alpha=0.3)
        
        # 6. Risk metrics (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        
        # Calculate risk score based on tipping capacity margin
        results['tipping_margin'] = (results['tipping_capacity_kg_day'] - 
                                    results['expected_demand_kg_day']) / results['expected_demand_kg_day']
        
        risk_categories = pd.cut(results['tipping_margin'], 
                            bins=[0, 0.2, 0.5, 1.0, np.inf],
                            labels=['High Risk', 'Medium Risk', 'Low Risk', 'Very Low Risk'])
        
        risk_counts = risk_categories.value_counts()
        colors = ['darkred', 'orange', 'lightgreen', 'darkgreen']
        ax6.pie(risk_counts.values, labels=risk_counts.index, colors=colors,
            autopct='%1.1f%%', startangle=90)
        ax6.set_title('Portfolio Risk Distribution')
        
        plt.suptitle('H2 Station Network Economic Performance Dashboard', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/economic_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_iterative_selection_progress(self, save_path=None):
        """
        Visualize how demand changes during iterative selection.
        """
        
        # Check if data exists
        if not hasattr(self, 'iteration_history') or not hasattr(self, 'demand_evolution_matrix'):
            print("No iteration data found. Run iterative selection first.")
            return

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Create a 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])  # Bottom spans both columns
        
        
        # 1. Plot Individual Station Demands at Selection Time
        iterations = sorted(self.iteration_history.keys())
        selection_demands = []
        base_demands = []
        station_labels = []
        
        for iter_num in iterations:
            data = self.iteration_history[iter_num]
            selection_demands.append(data['current_demand'])
            base_demands.append(data['base_demand'])
            station_labels.append(f"S{data['station_id']}")
        
        x = np.arange(len(iterations))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, base_demands, width, 
                        label='Base Demand', alpha=0.8, color='lightblue')
        bars2 = ax1.bar(x + width/2, selection_demands, width, 
                        label='Demand at Selection', alpha=0.8, color='darkblue')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('Selection Order')
        ax1.set_ylabel('Demand (kg/day)')
        ax1.set_title('Station Demand at Time of Selection')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Iter {i+1}' for i in iterations], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot Cumulative Network Demand Evolution
        cumulative_original = np.cumsum(base_demands)
        cumulative_adjusted = []
        
        # Calculate cumulative demand at each iteration
        for iter_num in iterations:
            total_demand = 0
            for station_id, history in self.demand_evolution_matrix.items():
                if iter_num in history and history[iter_num] > 0:
                    total_demand += history[iter_num]
            cumulative_adjusted.append(total_demand)
        
        iteration_numbers = [i+1 for i in iterations]
        
        ax2.plot(iteration_numbers, cumulative_original, 'o-', linewidth=2, 
                markersize=8, label='Cumulative Base Demand', color='lightblue')
        ax2.plot(iteration_numbers, cumulative_adjusted, 's-', linewidth=2, 
                markersize=8, label='Cumulative Adjusted Demand', color='darkblue')
        
        # Fill area between
        ax2.fill_between(iteration_numbers, cumulative_original, cumulative_adjusted, 
                        alpha=0.3, color='red', label='Demand Loss')
        
        # Add annotations
        if len(iterations) > 0:
            final_loss = cumulative_original[-1] - cumulative_adjusted[-1]
            final_loss_pct = (final_loss / cumulative_original[-1]) * 100 if cumulative_original[-1] > 0 else 0
            
            ax2.text(len(iterations) * 0.7, cumulative_adjusted[-1] + final_loss/2,
                    f'Total Loss: {final_loss:.0f} kg/day\n({final_loss_pct:.1f}%)',
                    fontsize=10, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        ax2.set_xlabel('Number of Stations Selected')
        ax2.set_ylabel('Cumulative Demand (kg/day)')
        ax2.set_title('Cumulative Network Demand Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot Demand Evolution Heatmap
        # Convert demand evolution matrix to DataFrame for plotting
        demand_matrix_df = self.demand_evolution_matrix.copy()
        
        # Only show iterations that were actually used
        demand_matrix_df = demand_matrix_df.iloc[:, :len(iterations)]
        
        # Create heatmap
        sns.heatmap(demand_matrix_df, 
                    ax=ax3, 
                    cmap='YlOrRd', 
                    cbar_kws={'label': 'Demand (kg/day)'},
                    fmt='.0f',
                    linewidths=0.5,
                    annot=False)  # Set to True if you want values in cells
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Station ID')
        ax3.set_title('Station Demand Evolution Across Iterations')
        ax3.set_xticklabels([f'{i+1}' for i in range(len(iterations))])
        
        # Overall title
        plt.suptitle('Iterative Station Selection Progress with Demand Evolution', 
                    fontsize=16, fontweight='bold')
        
        # Add summary statistics
        summary_text = f"Total Stations: {len(self.iteration_history)}\n"
        if len(iterations) > 0:
            avg_base = np.mean(base_demands)
            avg_selected = np.mean(selection_demands)
            avg_retention = (avg_selected / avg_base * 100) if avg_base > 0 else 0
            summary_text += f"Avg Base Demand: {avg_base:.0f} kg/day\n"
            summary_text += f"Avg Selected Demand: {avg_selected:.0f} kg/day\n"
            summary_text += f"Avg Demand Retention: {avg_retention:.1f}%"
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig


    def plot_demand_evolution_detail(self, station_ids=None, save_path=None):
        """
        Detailed plot showing demand evolution for specific stations.
        
        Parameters:
        - station_ids: List of station IDs to plot (default: first 5 selected)
        - save_path: Path to save the figure
        """
        if not hasattr(self, 'demand_evolution_matrix'):
            print("No demand evolution data found.")
            return
        
        # Default to first 5 selected stations if not specified
        if station_ids is None:
            station_ids = list(self.iteration_history[i]['station_id'] 
                            for i in sorted(self.iteration_history.keys())[:5])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each station's demand evolution
        for station_id in station_ids:
            if station_id in self.demand_evolution_matrix.index:
                demands = self.demand_evolution_matrix.loc[station_id].values
                iterations = range(len(demands))
                
                # Find when this station was selected
                selection_iter = None
                for iter_num, data in self.iteration_history.items():
                    if data['station_id'] == station_id:
                        selection_iter = iter_num
                        break
                
                # Plot with marker at selection point
                line = ax.plot(iterations, demands, 'o-', label=f'Station {station_id}', 
                            markersize=6, linewidth=2)
                
                if selection_iter is not None:
                    ax.scatter(selection_iter, demands[selection_iter], 
                            s=200, marker='*', color=line[0].get_color(),
                            edgecolor='black', linewidth=2, zorder=5)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Demand (kg/day)')
        ax.set_title('Individual Station Demand Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines for each selection
        for iter_num in self.iteration_history.keys():
            ax.axvline(x=iter_num, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig

    def save_iteration_results(self, output_dir='iteration_results'):
        """Save all iteration tracking data to files."""
        import os
        import json
        from datetime import datetime
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                return super(NumpyEncoder, self).default(obj)
    
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save iteration history as CSV
        iteration_df = pd.DataFrame.from_dict(self.iteration_history, orient='index')
        iteration_df.index.name = 'iteration'
        iteration_df.to_csv(os.path.join(output_dir, 'iteration_history.csv'))
        
        # 2. Save demand evolution matrix
        self.demand_evolution_matrix.to_csv(os.path.join(output_dir, 'demand_evolution_matrix.csv'))
        
        # 3. Create a summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_iterations': len(self.iteration_history),
            'total_stations_selected': len(self.iteration_history),
            'stations_selected': list(self.iteration_history[i]['station_id'] 
                                    for i in sorted(self.iteration_history.keys())),
            'final_total_demand': self.demand_evolution_matrix.iloc[:, -1].sum(),
            'average_demand_retention': (self.demand_evolution_matrix.iloc[:, -1].sum() / 
                                        sum(d['base_demand'] for d in self.iteration_history.values())) * 100
        }
        
            # When saving JSON, use the custom encoder
        with open(f"{output_dir}/iteration_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        #with open(os.path.join(output_dir, 'selection_summary.json'), 'w') as f:
        #    json.dump(summary, f, indent=2)
        
        # 4. Create a detailed iteration report (human-readable)
        with open(os.path.join(output_dir, 'iteration_report.txt'), 'w') as f:
            f.write("Iterative Station Selection Report\n")
            f.write("=" * 50 + "\n\n")
            
            for iter_num in sorted(self.iteration_history.keys()):
                data = self.iteration_history[iter_num]
                f.write(f"Iteration {iter_num + 1}:\n")
                f.write(f"  Station ID: {data['station_id']}\n")
                f.write(f"  Base Demand: {data['base_demand']:.1f} kg/day\n")
                f.write(f"  Demand at Selection: {data['current_demand']:.1f} kg/day\n")
                f.write(f"  Demand Retention: {(data['current_demand']/data['base_demand']*100):.1f}%\n")
                f.write("-" * 30 + "\n")
        
        print(f"Iteration results saved to {output_dir}/")
        print(f"  - iteration_history.csv: Selection details by iteration")
        print(f"  - demand_evolution_matrix.csv: Complete demand evolution")
        print(f"  - selection_summary.json: Summary statistics")
        print(f"  - iteration_report.txt: Human-readable report")
        
        return output_dir
            
            
            

    def generate_enhanced_report(self, output_dir='enhanced_outputs'):
        """
        Enhanced report for iterative station selection to include flexible capacity results.
        """
        import os
        from datetime import datetime
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'iterative_selection_results'):
            print("No results to report. Run analysis first.")
            return
        
        results = self.iterative_selection_results
        
        # Create a clean DataFrame for operations like nlargest()
        # Drop any columns that might contain complex objects
        scalar_columns = []
        for col in results.columns:
            # Check if column contains only scalar values
            try:
                # Try to convert to numeric - if it fails, check if it's a simple string/bool
                pd.to_numeric(results[col], errors='coerce')
                scalar_columns.append(col)
            except:
                # Check if it's a string or boolean column
                if results[col].dtype in ['object', 'bool', 'string']:
                    # Check if values are simple (not lists, dicts, etc.)
                    sample_value = results[col].dropna().iloc[0] if len(results[col].dropna()) > 0 else None
                    if sample_value is None or isinstance(sample_value, (str, bool, int, float)):
                        scalar_columns.append(col)
        
        # Create a clean DataFrame with only scalar columns
        results_clean = results[scalar_columns].copy()

        
        report_lines = [
            "=" * 80,
            "ENHANCED H2 STATION SITING ANALYSIS REPORT WITH ITERATUVE STATION SELECTION AND FLEXIBLE CAPACITIES",
            "Flexible Capacity Optimization (500-12,000 kg/day)",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "METHODOLOGY:",
            "-" * 40,
            "- Flexible capacity selection from 500-12,000 kg/day",
            "- Mesh refinement for precise optimization",
            "- Breakeven analysis using 60kg demand steps",
            "- Iterative selection with demand cannibalization",
            "- All economics calculated via consistent model",
            "",
            "EXECUTIVE SUMMARY:",
            "-" * 40,
            f"Total stations selected: {len(results)}",
            f"Total capital investment: ${results['cumulative_investment'].iloc[-1]:,.0f}",
            f"Total NPV generated: ${results['cumulative_npv'].iloc[-1]:,.0f}",
            f"Portfolio ROI: {(results['cumulative_npv'].iloc[-1]/results['cumulative_investment'].iloc[-1]*100):.1f}%",
            f"Average station capacity: {results['capacity_kg_day'].mean():.0f} kg/day",
            f"Capacity range deployed: {results['capacity_kg_day'].min():.0f} - {results['capacity_kg_day'].max():.0f} kg/day",
            f"Average utilization: {results['optimal_utilization'].mean():.1%}",
            f"Total H2 capacity: {results['capacity_kg_day'].sum():,} kg/day",
            f"Total H2 demand served: {results['expected_demand_kg_day'].sum():,.0f} kg/day",
            "",
            "CAPACITY DISTRIBUTION:",
            "-" * 40
        ]
        
        # Capacity tier analysis
        capacity_tiers = [
            ('Small (≤1000)', results[results['capacity_kg_day'] <= 1000]),
            ('Medium (1001-2000)', results[(results['capacity_kg_day'] > 1000) & (results['capacity_kg_day'] <= 2000)]),
            ('Large (2001-3000)', results[(results['capacity_kg_day'] > 2000) & (results['capacity_kg_day'] <= 3000)]),
            ('XL (>3000)', results[results['capacity_kg_day'] > 3000])
        ]
        
        for tier_name, tier_data in capacity_tiers:
            if len(tier_data) > 0:
                report_lines.extend([
                    f"{tier_name}: {len(tier_data)} stations ({len(tier_data)/len(results)*100:.1f}%)",
                    f"  Avg NPV: ${tier_data['npv'].mean():,.0f}",
                    f"  Avg Utilization: {tier_data['optimal_utilization'].mean():.1%}"
                ])
        
        report_lines.extend([
            "",
            "TOP 10 STATIONS BY NPV:",
            "-" * 40
        ])
        
        # Top 10 stations with flexible capacity details using cleaned sclar results DataFrame
        # check that column includes a column name that includes 'npv' or similar
        if 'npv' not in results_clean.columns and 'optimal_npv' not in results_clean.columns and 'npv_proxy' not in results_clean.columns:
            print("No NPV data available for top stations.")
            return
        
        if 'npv' in results_clean.columns:
            # get the subset of results for top 10 stations where npv is sorted in descending order (do not use nlargest)
            top_10 = results_clean.sort_values(by='npv', ascending=False).head(10)
        elif 'optimal_npv' in results_clean.columns:
            top_10 = results_clean.sort_values(by='optimal_npv', ascending=False).head(10)
        elif 'npv_proxy' in results_clean.columns:
            top_10 = results_clean.sort_values(by='npv_proxy', ascending=False).head(10)
        else:
            print("Warning: No NPV column found for top stations.")
            # Fallback to first 10 rows if no NPV data
            top_10 = results.head(10)
            
        
        for idx, station in top_10.iterrows():
            report_lines.extend([
                f"\nStation #{station['station_id']} (Selection {station['iteration']})",
                f"  Location: ({station.geometry.x:.0f}, {station.geometry.y:.0f})",
                f"  Optimal Capacity: {station['capacity_kg_day']:.0f} kg/day",
                f"  Expected Demand: {station['expected_demand_kg_day']:.0f} kg/day",
                f"  NPV: ${station['npv']:,.0f}",
                f"  CAPEX: ${station['optimal_capex']:,.0f}",
                f"  Utilization: {station['optimal_utilization']:.1%}",
                f"  Breakeven: {station['breakeven_demand_kg_day']:.0f} kg/day",
                f"  Demand Cushion: {station['demand_cushion_pct']:.0f}%",
                f"  Payback: {station.get('payback_years_optimal', 0):.1f} years"
            ])
        
        # Value of optimization
        if 'npv_improvement_vs_default' in results.columns:
            total_improvement = results['npv_improvement_vs_default'].sum()
            avg_improvement = results['npv_improvement_vs_default'].mean()
            
            report_lines.extend([
                "",
                "VALUE OF CAPACITY OPTIMIZATION:",
                "-" * 40,
                f"Total NPV improvement vs default capacity: ${total_improvement:,.0f}",
                f"Average NPV improvement per station: ${avg_improvement:,.0f}",
                f"Stations with improved NPV: {len(results[results['npv_improvement_vs_default'] > 0])}",
                f"Stations with reduced NPV: {len(results[results['npv_improvement_vs_default'] < 0])}"
            ])
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        # Write report
        report_path = os.path.join(output_dir, 'flexible_capacity_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        
        # For GeoJSON, we need the geometry column
        if 'geometry' in results.columns:
            # Create a GeoDataFrame with essential columns only
            geo_columns = ['geometry'] + [col for col in scalar_columns if col != 'geometry']
            results_geo = results[geo_columns].copy()
            results_geo.to_file(
                os.path.join(output_dir, 'selected_stations.geojson'),
                driver='GeoJSON'
            )
        
        # Save detailed results
        results_df = results.drop('geometry', axis=1)
        results_df.to_csv(os.path.join(output_dir, 'station_selection_flexible_capacity.csv'), index=False)
        
        print(f"Enhanced report generated: {report_path}")

    
    def _generate_summary_report(self, output_dir):
        """Generate text summary report."""
        
        report_lines = [
            "=" * 80,
            "HYDROGEN REFUELING STATION SITING ANALYSIS",
            "=" * 80,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "CONFIGURATION PARAMETERS:",
            "-" * 40
        ]
        
        # Add config parameters
        for key, value in self.config.items():
            report_lines.append(f"  {key}: {value}")
        
        report_lines.extend([
            "",
            "NETWORK ANALYSIS:",
            "-" * 40,
            f"  Total route segments: {len(self.routes) if hasattr(self, 'routes') else 0}",
            f"  Total route miles: {self.routes['length_miles'].sum():.0f}" if hasattr(self, 'routes') else "  N/A",
            f"  Average truck AADT: {self.routes['truck_aadt'].mean():.0f}" if hasattr(self, 'routes') else "  N/A",
            f"  Total H2 demand: {self.routes['h2_demand_kg_day'].sum():.0f} kg/day" if hasattr(self, 'routes') else "  N/A",
            "",
            "CANDIDATE LOCATIONS:",
            "-" * 40,
            f"  Total candidates evaluated: {len(self.candidates) if hasattr(self, 'candidates') else 0}",
            f"  Profitable locations: {len(self.candidates[self.candidates['npv_proxy'] > 0]) if hasattr(self, 'candidates') else 0}",
            f"  Average utilization rate: {self.candidates['utilization_rate'].mean():.2f}" if hasattr(self, 'candidates') else "  N/A",
            "",
            "TOP 10 LOCATION CLUSTERS:",
            "-" * 40
        ])
        
        if hasattr(self, 'clusters') and len(self.clusters) > 0:
            for idx, cluster in self.clusters.head(10).iterrows():
                report_lines.extend([
                    f"  Cluster {cluster['cluster_id']}:",
                    f"    Total NPV: ${cluster['total_npv']:,.0f}",
                    f"    Total demand: {cluster['total_demand_kg_day']:.0f} kg/day",
                    f"    Avg utilization: {cluster['avg_utilization']:.2%}",
                    f"    Tipping capacity: {cluster['tipping_capacity_kg_day']:.0f} kg/day",
                    ""
                ])
        
        report_lines.extend([
            "OPTIMAL PORTFOLIO:",
            "-" * 40
        ])
        
        if hasattr(self, 'portfolio') and len(self.portfolio) > 0:
            report_lines.extend([
                f"  Selected stations: {len(self.portfolio)}",
                f"  Total investment: ${self.portfolio['capex'].sum():,.0f}",
                f"  Total NPV: ${self.portfolio['npv_proxy'].sum():,.0f}",
                f"  Average payback: {self.portfolio['payback_years'].mean():.1f} years",
                f"  Average utilization: {self.portfolio['utilization_rate'].mean():.2%}",
                "",
                "  Top 5 stations by NPV:"
            ])
            
            for i, (idx, station) in enumerate(self.portfolio.nlargest(5, 'npv_proxy').iterrows(), 1):
                report_lines.append(
                    f"    {i}. NPV: ${station['npv_proxy']:,.0f}, "
                    f"Demand: {station['expected_demand_kg_day']:.0f} kg/day, "
                    f"Utilization: {station['utilization_rate']:.2%}"
                )
        
        if hasattr(self, 'validation_results'):
            report_lines.extend([
                "",
                "VALIDATION RESULTS:",
                "-" * 40,
                f"  Coverage of existing areas: {self.validation_results['coverage_rate']:.1%}",
                f"  Avg distance to existing: {self.validation_results['avg_distance_to_existing']:.1f} miles" 
                if self.validation_results['avg_distance_to_existing'] else "  N/A"
            ])
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        # Write report
        with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))
    
    def run_full_analysis(self, route_file, loi_files, existing_stations=None, 
                         budget=None, n_stations=10, output_dir="h2_station_outputs"):
        """
        Run complete analysis pipeline.
        """
        print("Starting H2 station siting analysis...")
        print("=" * 60)
        
        # Load data
        self.load_data(route_file, loi_files, existing_stations)
        
        # Run analysis steps
        self.estimate_demand_surface()
        self.calculate_utilization_probability()
        self.calculate_economic_proxy()
        #self.identify_location_clusters(n_clusters=20)
        self.optimize_portfolio(budget=budget, n_stations=n_stations)
        self.optimize_developer_portfolio(budget=budget, n_stations=n_stations)
        results=self.run_iterative_station_selection()
        self.analyze_regional_clusters(results)
        self.create_continuous_score_surface()
        self.validate_results()
        
        # Generate outputs
        self.generate_outputs(output_dir)
        
        print("=" * 60)
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}")
        
        return self.portfolio if hasattr(self, 'portfolio') else None


# Example usage
if __name__ == "__main__":
    # Initialize model with custom configuration
    config = {
        'h2_consumption_kg_per_mile': 0.1,
        'station_capacity_kg_per_day': 1500,
        'station_capex': 4000000,
        'h2_price_per_kg': 12.5,
        'service_radius_miles': 2.0,
        'min_station_spacing_miles': 5
    }
    
    model = H2StationSitingModel(config)
    
    # Define input files
    route_file = "freight_analysis/data/combined_network.geojson"
    
    loi_files = [
        "downloaded_data/Airports.geojson",
        "downloaded_data/Designated Truck Parking Simplified.geojson",
        "downloaded_data/Major Traffic Generators.geojson",
        "downloaded_data/Seaports.geojson",
        "downloaded_data/CA_Energy_Commission_-_Gas_Stations.geojson",
        "downloaded_data/CA_rest_areas.csv"
    ]
    
    existing_stations = "downloaded_data/CA_Hy_extract_USA_MHD_ZE_Infrastructure.geojson"
    
    # Run analysis
    portfolio = model.run_full_analysis(
        route_file=route_file,
        loi_files=loi_files,
        existing_stations=existing_stations,
        budget=50000000,  # $50M budget
        n_stations=10,
        output_dir="h2_station_analysis_results"
    )
    
    print("\nExample of how to access results:")
    print("  model.portfolio - Selected station locations")
    print("  model.clusters - Top location clusters") 
    print("  model.candidates - All evaluated candidates with scores")
    print("  model.score_surface - Continuous utility surface")
    
    
