#!/usr/bin/env python3
"""
H2 Station Siting Analysis - Quick Setup Script for pre-processed LOI-route matching data
Fixed version with organized output directory structure
"""

import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import the main model
from h2_station_model import H2StationSitingModel


def get_next_version_dir(base_dir="h2_station_quick_results"):
    """
    Get the next version number for output directory.
    Returns a directory name like 'h2_station_quick_results_v1', 'h2_station_quick_results_v2', etc.
    """
    version = 1
    while os.path.exists(f"{base_dir}_v{version}"):
        version += 1
    return f"{base_dir}_v{version}"


def save_iteration_results(model, output_dir):
    """Save all iteration tracking data to files with proper type conversion."""
    import os
    import json
    from datetime import datetime
    import pandas as pd
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return super(NumpyEncoder, self).default(obj)
    
    # 1. Save iteration history as CSV
    if hasattr(model, 'iteration_history') and model.iteration_history:
        iteration_df = pd.DataFrame.from_dict(model.iteration_history, orient='index')
        iteration_df.index.name = 'iteration'
        iteration_df.to_csv(os.path.join(output_dir, 'iteration_history.csv'))
    
    # 2. Save demand evolution matrix if it exists
    if hasattr(model, 'demand_evolution_matrix') and not model.demand_evolution_matrix.empty:
        model.demand_evolution_matrix.to_csv(os.path.join(output_dir, 'demand_evolution_matrix.csv'))
    
    # 3. Create a summary report
    if hasattr(model, 'iteration_history') and model.iteration_history:
        # Calculate safe values with NaN handling
        stations_selected = []
        total_base_demand = 0
        total_final_demand = 0
        
        for i in sorted(model.iteration_history.keys()):
            hist_entry = model.iteration_history[i]
            if 'station_id' in hist_entry:
                stations_selected.append(int(hist_entry['station_id']))
            if 'base_demand' in hist_entry:
                total_base_demand += hist_entry['base_demand']
        
        if hasattr(model, 'demand_evolution_matrix') and not model.demand_evolution_matrix.empty:
            total_final_demand = float(model.demand_evolution_matrix.iloc[:, -1].sum())
            avg_demand_retention = float((total_final_demand / total_base_demand * 100)) if total_base_demand > 0 else 0
        else:
            total_final_demand = 0
            avg_demand_retention = 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_iterations': len(model.iteration_history),
            'total_stations_selected': len(stations_selected),
            'stations_selected': stations_selected,
            'final_total_demand': total_final_demand,
            'average_demand_retention': avg_demand_retention
        }
        
        with open(os.path.join(output_dir, 'selection_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    # 4. Create a detailed iteration report (human-readable)
    if hasattr(model, 'iteration_history') and model.iteration_history:
        with open(os.path.join(output_dir, 'iteration_report.txt'), 'w') as f:
            f.write("Iterative Station Selection Report\n")
            f.write("=" * 50 + "\n\n")
            
            for iter_num in sorted(model.iteration_history.keys()):
                data = model.iteration_history[iter_num]
                f.write(f"Iteration {iter_num + 1}:\n")
                f.write(f"  Station ID: {data.get('station_id', 'N/A')}\n")
                
                # Handle different possible data structures
                if 'base_demand' in data and 'current_demand' in data:
                    base_demand = data['base_demand']
                    current_demand = data['current_demand']
                    f.write(f"  Base Demand: {base_demand:.1f} kg/day\n")
                    f.write(f"  Demand at Selection: {current_demand:.1f} kg/day\n")
                    if base_demand > 0:
                        f.write(f"  Demand Retention: {(current_demand/base_demand*100):.1f}%\n")
                elif 'npv' in data:
                    f.write(f"  NPV: ${data['npv']:,.0f}\n")
                if 'capacity' in data:
                    f.write(f"  Capacity: {data['capacity']:.0f} kg/day\n")
                if 'utilization' in data:
                    f.write(f"  Utilization: {data['utilization']:.1%}\n")
                f.write("-" * 30 + "\n")
    
    print(f"Iteration results saved to {output_dir}/")
    return output_dir


def validate_competition_graph_implementation(model):
    """Validate that the competition graph prevents double counting."""
    print("\n=== COMPETITION GRAPH VALIDATION ===\n")
    
    # 1. Check graph initialization
    print("1. Checking graph initialization after calculate_bayesian_probabilities...")
    if model._competition_graph is None:
        print("   ERROR: Competition graph not initialized!")
        return False
    else:
        print(f"   Passed: Graph initialized with {len(model._competition_graph.nodes)} nodes")
        print(f"   Passed: Graph has {len(model._competition_graph.edges)} edges")
    
    # 2. Check existing infrastructure in graph
    print("\n2. Checking existing infrastructure nodes...")
    existing_count = sum(1 for n, d in model._competition_graph.nodes.items() 
                        if d['type'] == 'existing')
    print(f"   Passed: Found {existing_count} existing station nodes")
    
    # 3. Check candidate nodes
    print("\n3. Checking candidate nodes...")
    candidate_count = sum(1 for n, d in model._competition_graph.nodes.items() 
                         if d['type'] == 'candidate')
    print(f"   Passed: Found {candidate_count} candidate nodes")
    
    # 4. Test market share calculation
    print("\n4. Testing market share calculation...")
    sample_candidates = model.candidates.head(1)
    for idx, candidate in sample_candidates.iterrows():
        market_share = model._competition_graph.calculate_market_share(idx)
        print(f"   Sample candidate {idx} market share: {market_share:.3f}")
    
    print("\n=== COMPETITION GRAPH VALIDATION COMPLETE ===")
    return True


def test_no_competition_double_counting(model):
    """Test that competition effects aren't double-counted."""
    print("\n=== TESTING FOR DOUBLE COUNTING IN COMPETITION  ===")
    
    # Select a sample candidate to trace through the calculations
    sample_idx = model.candidates.index[0]
    sample_candidate = model.candidates.loc[sample_idx]
    
    print(f"Testing candidate {sample_idx}:")
    print(f"  Original demand: {sample_candidate.get('original_demand', 'N/A')} kg/day")
    print(f"  After existing competition: {sample_candidate.get('initial_demand_post_existing', 'N/A')} kg/day")
    print(f"  Current demand: {sample_candidate.get('expected_demand_kg_day', 'N/A')} kg/day")
    
    # Check if this candidate has edges in the competition graph
    print(f"  Checking graph edges for this candidate...")
    edge_count = 0
    sample_idx_str = str(sample_idx)  # Convert to string for edge comparison
    
    for edge_key in model._competition_graph.edges:
        if sample_idx_str in edge_key.split('-'):
            edge_count += 1
    
    print(f"  Found {edge_count} competition edges for this candidate")
    
    # Verify market share calculation
    market_share = model._competition_graph.calculate_market_share(sample_idx)
    print(f"  Market share from graph: {market_share:.3f}")
    
    print("\n=== DOUBLE COUNTING TEST COMPLETE ===")


def run_quick_analysis(route_file, merged_loi_file, gas_stations_csv=None, existing_stations=None, budget=None, n_stations=5, visualize_validation=True):
    """
    Run a quick 5-minute analysis with your data.
    All outputs will be organized in a single versioned directory.
    """
    
    # Create main output directory with version number
    main_output_dir = get_next_version_dir()
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"\nAll outputs will be saved to: {main_output_dir}/")
    
    # Create subdirectories for different output types
    subdirs = {
        'data': os.path.join(main_output_dir, 'data'),
        'validation': os.path.join(main_output_dir, 'validation'),
        'visualizations': os.path.join(main_output_dir, 'visualizations'),
        'iteration_results': os.path.join(main_output_dir, 'iteration_results'),
        'reports': os.path.join(main_output_dir, 'reports'),
        'competition_analysis': os.path.join(main_output_dir, 'competition_analysis')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    # Get comprehensive configuration and customize for quick run
    try:
        from comprehensive_config import get_comprehensive_config
        quick_config = get_comprehensive_config()
        print("✓ Using comprehensive configuration with all validated parameters")
    except ImportError:
        print("⚠ Comprehensive config not available, using fallback configuration")
        # Fallback to ensure all required parameters are present
        quick_config = {
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
    
    # Customize configuration for quick run (override specific settings for speed)
    quick_config.update({
        'save_diagnostic_plots': True,
        'create_enhanced_plots': True,
        'validate_demand_surface': True,
        'n_stations_quick_run': 50,  # Default for quick runs
        'quick_run_mode': True,
    })
    
    # Initialize model
    print("\nInitializing H2 Station Siting Model...")
    model = H2StationSitingModel(quick_config)
    
    # Load all data files
    print("Loading network and location data...")
    model.load_data(route_file, merged_loi_file, gas_stations_csv, existing_stations)
    
    print("\nStarting analysis pipeline...")
    
    try:
        # Create demand surface
        print("\nEstimating demand surface...")
        model.estimate_demand_surface()
        
        if hasattr(model, 'demand_surface') and 'validation' in model.demand_surface:
            model.create_gas_station_validation_plots()
            
        # Generate candidates
        model.generate_candidates(strategy='hybrid')  # or 'route_based', 'loi_based'
        
        # Generate and evaluate candidates
        print("\nGenerating and evaluating candidates...")
        model.calculate_utilization_probability()
        
        # Calculate economics
        print("\nCalculating economic metrics...")
        model.calculate_economic_proxy()
        
       
        # Run competition graph validation
        validate_competition_graph_implementation(model)
        test_no_competition_double_counting(model)
        
        # Run portfolio optimization first (needed for validation)
        print("\nRunning portfolio optimization...")
        model.optimize_portfolio(budget=budget, n_stations=n_stations)
        
        # Run developer portfolio optimization
        print("\nRunning developer portfolio optimization...")
        model.optimize_developer_portfolio(budget=budget, n_stations=n_stations)
        
        # Iteratively determine optimal station locations with tipping points
        print("\nIteratively determining optimal station locations with tipping point capacity and true optimal competition npv....")
        results = model.run_iterative_station_selection(max_stations=500)
        
        # Analyze regional clusters
        print("\nAnalyzing regional clusters...")
        model.analyze_regional_clusters(results)
        
        # Create continuous score surface
        print("\nCreating continuous score surface...")
        model.create_continuous_score_surface()
        
        # Save iteration results
        print("\nSaving iteration tracking results...")
        save_iteration_results(model, subdirs['iteration_results'])
        
        # Run validation
        print("\nValidating results...")
        model.validate_results()
        
        # Visualizations
        if visualize_validation:
            print("\nCreating validation visualizations...")
            model.visualize_validation_results(
                save_plots=True, 
                output_dir=subdirs['validation']
            )
            print(f"   Visualizations saved to {subdirs['validation']}/")
        
        # Create enhanced visualizations with error handling for NaN values
        print("\nCreating enhanced visualizations for iterative station selection...")
        try:
            model.create_enhanced_visualizations(output_dir=subdirs['visualizations'])
        except ValueError as e:
            if "cannot convert float NaN to integer" in str(e):
                print(f"Warning: Skipping some visualizations due to NaN values in data")
                print("This typically occurs when there are no low/medium/high risk stations")
            else:
                raise
        
        # Visualize competition network
        try:
            print("\nVisualizing competition network...")
            competition_graph_path = os.path.join(subdirs['competition_analysis'], 'competition_network.png')
            model.visualize_competition_graph(competition_graph_path)
        except Exception as e:
            print(f"Warning: Could not create competition graph visualization: {e}")
        
        # Plot iterative selection progress
        print("Plotting iterative selection progress...")
        progress_plot_path = os.path.join(subdirs['visualizations'], 'selection_progress.png')
        model.plot_iterative_selection_progress(save_path=progress_plot_path)
        
        # Save model configuration
        config_file = os.path.join(main_output_dir, "model_config.json")
        with open(config_file, 'w') as f:
            json.dump(model.config, f, indent=4)
        print(f"   Model configuration saved to {config_file}")
        
        # Generate enhanced report
        print("\nGenerating enhanced report for iterative selection...")
        model.generate_enhanced_report(output_dir=subdirs['reports'])
        
        # Generate main outputs
        print("\nGenerating final outputs...")
        model.generate_outputs(subdirs['data'])
        
        # Export results
        print("\nExporting results...")
        model.export_results(subdirs['data'])
        
        # Create summary report in main directory
        summary_path = os.path.join(main_output_dir, 'analysis_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"H2 Station Quick Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {main_output_dir}\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  - Number of stations: {n_stations}\n")
            f.write(f"  - Budget constraint: ${budget:,.0f}\n" if budget else "  - Budget constraint: None\n")
            f.write(f"  - Service radius: {quick_config['service_radius_miles']} miles\n")
            f.write(f"\nResults:\n")
            if hasattr(model, 'iterative_selection_results') and len(model.iterative_selection_results) > 0:
                f.write(f"  - Stations selected: {len(model.iterative_selection_results)}\n")
                f.write(f"  - Total NPV: ${model.iterative_selection_results['npv'].sum():,.0f}\n")
                f.write(f"  - Total Investment: ${model.iterative_selection_results['capacity_kg_day'].sum() * quick_config['station_capex'] / 2000:,.0f}\n")
            f.write(f"\nOutput Structure:\n")
            f.write(f"  - /data: Raw data and results\n")
            f.write(f"  - /validation: Validation analysis and plots\n")
            f.write(f"  - /visualizations: All visualization outputs\n")
            f.write(f"  - /iteration_results: Detailed iteration tracking\n")
            f.write(f"  - /reports: Comprehensive analysis reports\n")
            f.write(f"  - /competition_analysis: Competition network analysis\n")
        
        print("\n" + "=" * 60)
        print("QUICK ANALYSIS COMPLETE!")
        print(f"All results saved to: {main_output_dir}/")
        print("=" * 60)
        
        
        return model
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\nAnalysis failed. Please check error messages above.")
        return None


if __name__ == "__main__":
    # File paths - adjust these to your actual file locations
    ROUTE_FILE = "combined_network.geojson"
    MERGED_LOI_FILE = "all_merged_loi_routes.geojson"
    GAS_STATIONS_CSV = "gas_stations_from_CA_Energy_Commission_Gas_Stations.csv"
    EXISTING_STATIONS_FILE = "existing_ze_infrastructure.geojson"
    
    # Check for required files
    if not os.path.exists(ROUTE_FILE):
        print(f"ERROR: Route file not found: {ROUTE_FILE}")
        print("Please ensure combined_network.geojson is in the current directory")
        exit(1)
    
    if not os.path.exists(MERGED_LOI_FILE):
        print(f"ERROR: Merged LOI file not found: {MERGED_LOI_FILE}")
        print("Please ensure all_merged_loi_routes.geojson is in the current directory")
        exit(1)
    
    # Run the quick analysis
    print("Starting H2 Station Siting Quick Analysis")
    print("This should complete in approximately 5 minutes...")
    print()
    
    budget = None
    model = run_quick_analysis(
        route_file=ROUTE_FILE,
        merged_loi_file=MERGED_LOI_FILE,
        gas_stations_csv=GAS_STATIONS_CSV if os.path.exists(GAS_STATIONS_CSV) else None,
        existing_stations=EXISTING_STATIONS_FILE if os.path.exists(EXISTING_STATIONS_FILE) else None,
        budget=budget,
        n_stations=50,
        visualize_validation=True,
    )
    
    if model:
        print("\nAnalysis completed successfully!")
        print("\nNext steps for full analysis:")
        print("1. Review results in the output directory")
        print("2. Adjust configuration parameters based on initial results")
        print("3. Run full analysis with higher resolution and more candidates")
        print("4. Consider geocoding gas station addresses for better coverage")
    else:
        print("\nAnalysis failed. Please check error messages above.")