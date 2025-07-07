#!/usr/bin/env python3
"""
Comprehensive Configuration for H2 Station Siting Model
Designed for transportation network analysis with complete parameter coverage
"""

import numpy as np
from typing import Dict, Any

def get_comprehensive_config() -> Dict[str, Any]:
    """
    Returns a comprehensive configuration for H2 station siting model that addresses
    all computational and transportation network modeling requirements.
    
    Based on California transportation infrastructure and hydrogen fuel economics,
    with parameters validated against industry standards and research.
    """
    
    config = {
        # ==================== MODEL METADATA ====================
        'model_name': 'H2StationSitingModel',
        'model_version': '2.0.0',
        'config_version': '1.0.0',
        'created_date': '2025-01-07',
        'description': 'Comprehensive transportation network H2 infrastructure siting model',
        
        # ==================== COMPUTATIONAL PARAMETERS ====================
        'random_seed': 42,
        'parallel_processing': True,
        'max_workers': 4,
        'memory_limit_gb': 16,
        'temp_dir': None,  # Use system default
        'cleanup_temp_files': True,
        
        # ==================== VALIDATION AND DIAGNOSTICS ====================
        'save_diagnostic_plots': True,
        'create_enhanced_plots': True,
        'validate_demand_surface': True,
        'validate_results': True,
        'verbose_logging': True,
        'debug_mode': False,
        
        # ==================== SPATIAL ANALYSIS PARAMETERS ====================
        # Core spatial parameters
        'target_crs': 'EPSG:3310',  # California Albers equal-area projection
        'service_radius_miles': 2.0,  # Station service radius
        'min_station_spacing_miles': 1.0,  # Minimum distance between stations
        'max_loi_route_distance_miles': 2.0,  # Max distance for LOI-route matching
        
        # Spatial indexing and performance
        'spatial_index_type': 'rtree',  # rtree, quadtree, or grid
        'spatial_grid_resolution_m': 1000,  # Grid cell size for spatial operations
        'interchange_detection_radius_m': 1000.0,  # Highway interchange detection
        'network_buffer_m': 500,  # Buffer around network for candidate generation
        
        # Demand surface parameters
        'demand_kernel_bandwidth_miles': 2.0,  # Gaussian kernel bandwidth
        'demand_grid_resolution_m': 500,  # Demand surface grid resolution
        'demand_smoothing_factor': 0.5,  # Spatial smoothing parameter
        'demand_decay_exponent': 1.5,  # Distance decay for demand
        
        # ==================== NETWORK TOPOLOGY PARAMETERS ====================
        'candidate_interval_miles': 2.0,  # Spacing between candidate locations
        'min_candidate_truck_aadt': 200,  # Minimum AADT for candidates
        'min_truck_aadt_highway': 1000,  # Minimum AADT for highway candidates
        'max_candidates_per_segment': 5,  # Limit candidates per route segment
        'candidate_snap_tolerance_m': 100,  # Snap tolerance for network alignment
        
        # ==================== VEHICLE AND FUEL PARAMETERS ====================
        # Hydrogen consumption and range
        'h2_consumption_kg_per_mile': 0.1,  # Based on Class 8 truck efficiency
        'tank_capacity_kg': 80.0,  # Typical H2 tank capacity
        'typical_range_miles': 450.0,  # Range on full tank
        'refuel_threshold': 0.25,  # Refuel when tank drops to 25%
        'avg_refuel_amount_kg': 60.0,  # Average refueling amount
        
        # Refueling behavior parameters
        'refuel_probability_base': 0.15,  # Base probability of refueling
        'range_anxiety_factor': 1.5,  # Multiplier for range anxiety effects
        'detour_tolerance_miles': 5.0,  # Max detour for refueling
        'time_penalty_per_mile': 2.0,  # Minutes penalty per mile detour
        
        # ==================== ECONOMIC PARAMETERS ====================
        # Pricing and revenue
        'h2_price_per_kg': 28.50,  # Retail hydrogen price ($/kg)
        'price_elasticity': -0.3,  # Demand price elasticity
        'wholesale_h2_cost_per_kg': 8.50,  # Wholesale hydrogen cost
        'delivery_cost_per_kg': 2.00,  # Delivery cost per kg
        'operating_margin': 0.20,  # Target operating margin
        
        # Capital expenditures
        'station_capex': 12000000,  # Base station CAPEX ($)
        'capex_scaling_exponent': 0.7,  # Economy of scale factor
        'base_capex_multiplier': 1.4,  # Regional cost adjustment
        'land_cost_per_sqft': 25.0,  # Land cost ($/sq ft)
        'site_preparation_cost': 500000,  # Site preparation cost
        'permitting_cost': 150000,  # Permitting and regulatory costs
        
        # Operating expenditures
        'base_opex_daily': 5000,  # Fixed daily operating cost
        'variable_opex_per_kg': 6.0,  # Variable cost per kg sold
        'maintenance_cost_annual': 300000,  # Annual maintenance cost
        'insurance_cost_annual': 120000,  # Annual insurance cost
        'property_tax_rate': 0.012,  # Property tax rate
        'utilities_cost_monthly': 15000,  # Monthly utilities cost
        'labor_cost_annual': 180000,  # Annual labor cost
        
        # Financial parameters
        'discount_rate': 0.10,  # Discount rate for NPV calculations
        'inflation_rate': 0.025,  # Annual inflation rate
        'station_lifetime_years': 15.0,  # Station economic lifetime
        'depreciation_years': 10,  # Depreciation schedule
        'tax_rate': 0.28,  # Corporate tax rate
        'min_viable_npv': 0.0,  # Minimum NPV for viability
        
        # ==================== STATION CAPACITY PARAMETERS ====================
        'station_capacity_kg_per_day': 2000,  # Base station capacity
        'min_iteration_capacity_kg_per_day': 500,  # Minimum capacity
        'max_iteration_capacity_kg_per_day': 12000,  # Maximum capacity
        'capacity_step_size_kg_per_day': 500,  # Capacity optimization steps
        'capacity_utilization_target': 0.70,  # Target utilization rate
        'peak_to_average_ratio': 1.8,  # Peak vs average demand ratio
        'storage_to_daily_ratio': 1.5,  # Storage capacity relative to daily throughput
        
        # ==================== DEMAND MODELING PARAMETERS ====================
        # Base utilization and demand
        'base_utilization_rate': 0.05,  # Base probability of station use
        'min_daily_visits': 5.0,  # Minimum viable daily visits
        'demand_blending_factor': 0.7,  # Blend factor for demand estimates
        'seasonal_demand_variation': 0.15,  # Seasonal variation coefficient
        'weekly_demand_pattern': [0.9, 1.0, 1.1, 1.1, 1.2, 1.0, 0.8],  # Daily patterns
        
        # Market penetration and adoption
        'market_penetration_rate': 0.02,  # Current H2 truck penetration
        'adoption_growth_rate': 0.25,  # Annual adoption growth rate
        'fleet_conversion_factor': 0.8,  # Fleet vs individual adoption
        'early_adopter_premium': 1.3,  # Premium willingness to pay
        
        # ==================== BEHAVIORAL PARAMETERS ====================
        # Location attractiveness weights
        'rest_area_attraction_weight': 0.25,
        'port_proximity_weight': 0.20,
        'highway_access_weight': 0.20,
        'existing_station_weight': 0.10,
        'gas_station_proximity_weight': 0.15,
        'amenity_weight': 0.10,
        'urban_proximity_weight': 0.12,
        'freight_corridor_weight': 0.18,
        
        # Driver behavior parameters
        'convenience_factor': 1.2,  # Preference for convenient locations
        'brand_loyalty_factor': 0.9,  # Brand loyalty effect
        'payment_preference_factor': 1.1,  # Payment method preferences
        'service_quality_factor': 1.15,  # Service quality importance
        
        # ==================== COMPETITION PARAMETERS ====================
        # Competition modeling
        'use_gravity_competition_model': True,
        'distance_decay_exponent': 2.0,  # Distance decay for competition
        'existing_station_competition_weight': 1.0,
        'potential_station_competition_weight': 0.7,
        'competition_distance_offset': 100,  # Meters offset to avoid division by zero
        'market_share_calculation_method': 'gravity',  # gravity, huff, or logit
        
        # Market dynamics
        'capacity_saturation_factor': 0.7,
        'competition_decay_rate': 0.3,
        'competition_adjustment_factor': 0.8,
        'network_externality_factor': 1.1,  # Network effects benefit
        'first_mover_advantage': 1.2,  # First mover advantage multiplier
        
        # ==================== EXISTING INFRASTRUCTURE ====================
        'existing_station_utilization': 0.7,  # Assumed utilization of existing stations
        'existing_capacity_utilization': 0.65,  # Capacity utilization of existing
        'expansion_potential_factor': 1.5,  # Potential for capacity expansion
        
        # ==================== OPTIMIZATION PARAMETERS ====================
        # Portfolio optimization
        'multiple_developer_mode': True,
        'optimization_method': 'mixed_integer',  # mixed_integer, genetic, simulated_annealing
        'max_optimization_iterations': 1000,
        'convergence_tolerance': 1e-6,
        'solution_pool_size': 10,
        
        # Iterative selection
        'max_iterative_stations': 500,  # Maximum stations in iterative selection
        'iteration_convergence_threshold': 0.01,  # Convergence threshold
        'demand_update_frequency': 5,  # Update demand every N iterations
        
        # ==================== PERFORMANCE PARAMETERS ====================
        # Computational performance
        'spatial_query_cache_size': 10000,
        'demand_calculation_batch_size': 1000,
        'optimization_timeout_seconds': 3600,
        'memory_cleanup_frequency': 100,  # Cleanup every N iterations
        
        # Quality thresholds
        'min_model_r_squared': 0.7,  # Minimum model fit quality
        'max_cv_rmse': 0.3,  # Maximum cross-validation RMSE
        'spatial_autocorrelation_threshold': 0.1,  # Max spatial autocorrelation in residuals
        
        # ==================== VALIDATION PARAMETERS ====================
        # Cross-validation
        'cv_folds': 5,  # Number of cross-validation folds
        'validation_holdout_fraction': 0.2,  # Fraction for validation
        'bootstrap_samples': 100,  # Bootstrap samples for uncertainty
        
        # Sensitivity analysis
        'sensitivity_parameter_range': 0.2,  # ±20% for sensitivity analysis
        'monte_carlo_samples': 1000,  # Monte Carlo simulation samples
        'confidence_interval': 0.95,  # Confidence interval level
        
        # ==================== OUTPUT AND REPORTING ====================
        # Output control
        'save_intermediate_results': True,
        'export_geojson': True,
        'export_csv': True,
        'create_summary_report': True,
        'generate_visualizations': True,
        
        # Reporting parameters
        'map_dpi': 300,  # Map resolution
        'plot_style': 'seaborn-v0_8',  # Matplotlib style
        'color_palette': 'viridis',  # Color palette for plots
        'figure_size': (12, 8),  # Default figure size
        
        # ==================== ERROR HANDLING ====================
        # Robustness parameters
        'max_retries': 3,  # Maximum retries for failed operations
        'timeout_seconds': 300,  # Timeout for individual operations
        'fallback_methods_enabled': True,  # Enable fallback methods
        'graceful_degradation': True,  # Continue with partial results
        
        # Data quality
        'missing_data_threshold': 0.1,  # Maximum fraction of missing data
        'outlier_detection_method': 'iqr',  # iqr, zscore, or isolation_forest
        'outlier_threshold': 3.0,  # Threshold for outlier detection
        'data_consistency_checks': True,  # Enable consistency checks
    }
    
    return config

def validate_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate configuration parameters for consistency and feasibility.
    
    Returns:
        Dictionary of validation errors (empty if all valid)
    """
    errors = {}
    
    # Economic parameter validation
    if config['discount_rate'] <= 0 or config['discount_rate'] >= 1:
        errors['discount_rate'] = "Must be between 0 and 1"
    
    if config['h2_price_per_kg'] <= config['wholesale_h2_cost_per_kg']:
        errors['pricing'] = "Retail price must be greater than wholesale cost"
    
    # Spatial parameter validation
    if config['service_radius_miles'] <= 0:
        errors['service_radius_miles'] = "Must be positive"
    
    if config['min_station_spacing_miles'] > config['service_radius_miles']:
        errors['spacing'] = "Minimum spacing cannot exceed service radius"
    
    # Capacity parameter validation
    if config['min_iteration_capacity_kg_per_day'] >= config['max_iteration_capacity_kg_per_day']:
        errors['capacity_range'] = "Minimum capacity must be less than maximum"
    
    # Utilization parameter validation
    if not (0 <= config['capacity_utilization_target'] <= 1):
        errors['capacity_utilization_target'] = "Must be between 0 and 1"
    
    return errors

def get_scenario_configs() -> Dict[str, Dict[str, Any]]:
    """
    Return predefined scenario configurations for different analysis contexts.
    """
    base_config = get_comprehensive_config()
    
    scenarios = {
        'conservative': {
            **base_config,
            'h2_price_per_kg': 32.0,  # Higher price
            'market_penetration_rate': 0.01,  # Lower penetration
            'adoption_growth_rate': 0.15,  # Slower adoption
            'capacity_utilization_target': 0.60,  # Lower utilization
        },
        
        'aggressive': {
            **base_config,
            'h2_price_per_kg': 25.0,  # Lower price
            'market_penetration_rate': 0.05,  # Higher penetration
            'adoption_growth_rate': 0.35,  # Faster adoption
            'capacity_utilization_target': 0.80,  # Higher utilization
        },
        
        'high_density': {
            **base_config,
            'service_radius_miles': 1.5,  # Smaller service radius
            'min_station_spacing_miles': 0.5,  # Closer spacing
            'candidate_interval_miles': 1.0,  # More candidates
            'min_candidate_truck_aadt': 500,  # Higher AADT threshold
        },
        
        'rural_focus': {
            **base_config,
            'service_radius_miles': 5.0,  # Larger service radius
            'min_station_spacing_miles': 3.0,  # Wider spacing
            'min_candidate_truck_aadt': 100,  # Lower AADT threshold
            'capacity_utilization_target': 0.50,  # Lower expected utilization
        }
    }
    
    return scenarios

def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of key configuration parameters."""
    print("Configuration Summary")
    print("=" * 50)
    print(f"Model: {config['model_name']} v{config['model_version']}")
    print(f"Service Radius: {config['service_radius_miles']} miles")
    print(f"H2 Price: ${config['h2_price_per_kg']:.2f}/kg")
    print(f"Station CAPEX: ${config['station_capex']:,.0f}")
    print(f"Discount Rate: {config['discount_rate']:.1%}")
    print(f"Target Utilization: {config['capacity_utilization_target']:.1%}")
    print(f"Competition Model: {config['use_gravity_competition_model']}")
    print(f"Multiple Developers: {config['multiple_developer_mode']}")

if __name__ == '__main__':
    # Test configuration
    config = get_comprehensive_config()
    errors = validate_config(config)
    
    if errors:
        print("Configuration Validation Errors:")
        for param, error in errors.items():
            print(f"  {param}: {error}")
    else:
        print("✓ Configuration validation passed")
        print_config_summary(config)
        
    # Test scenarios
    scenarios = get_scenario_configs()
    print(f"\nAvailable scenarios: {list(scenarios.keys())}")