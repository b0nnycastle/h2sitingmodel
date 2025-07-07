#!/usr/bin/env python3
"""
Easy-to-use H2 Station Siting Model Runner
Updated to work with comprehensive configuration system
"""

import os
import sys
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run H2 Station Siting Model')
    
    # Basic parameters
    parser.add_argument('--scenario', choices=['default', 'conservative', 'aggressive', 'high_density', 'rural_focus'], 
                       default='default', help='Configuration scenario to use')
    parser.add_argument('--n_stations', type=int, default=50, 
                       help='Number of stations to optimize for')
    parser.add_argument('--budget', type=float, default=None,
                       help='Budget constraint in dollars (e.g., 50000000 for $50M)')
    
    # File paths
    parser.add_argument('--routes', default='combined_network.geojson',
                       help='Path to route network file')
    parser.add_argument('--lois', default='all_merged_loi_routes.geojson', 
                       help='Path to merged LOI file')
    parser.add_argument('--gas_stations', default='gas_stations_from_CA_Energy_Commission_Gas_Stations.csv',
                       help='Path to gas stations CSV file')
    parser.add_argument('--existing_stations', default='existing_ze_infrastructure.geojson',
                       help='Path to existing stations file')
    
    # Analysis options
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip validation visualizations for faster run')
    parser.add_argument('--output_dir', default=None,
                       help='Custom output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("H2 STATION SITING MODEL - COMPREHENSIVE ANALYSIS")
    print("="*60)
    print(f"Scenario: {args.scenario}")
    print(f"Target stations: {args.n_stations}")
    print(f"Budget: ${args.budget:,.0f}" if args.budget else "Budget: No constraint")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if required files exist
    required_files = [args.routes, args.lois]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ ERROR: Required files not found:")
        for f in missing_files:
            print(f"   - {f}")
        print()
        print("Please ensure the following files are in the current directory:")
        print("   - combined_network.geojson (route network)")
        print("   - all_merged_loi_routes.geojson (LOI data)")
        print()
        print("Or specify custom paths using --routes and --lois arguments")
        return 1
    
    # Import after file check
    try:
        from comprehensive_config import get_comprehensive_config, get_scenario_configs
        from h2_station_quick_setup import run_quick_analysis
        
        print("✓ Loaded comprehensive configuration system")
        
        # Get configuration
        if args.scenario == 'default':
            config = get_comprehensive_config()
        else:
            scenarios = get_scenario_configs()
            if args.scenario in scenarios:
                config = scenarios[args.scenario]
                print(f"✓ Using {args.scenario} scenario configuration")
            else:
                print(f"❌ ERROR: Scenario '{args.scenario}' not found")
                return 1
                
        # Override specific parameters from command line
        if args.budget:
            config['station_capex'] = args.budget / args.n_stations  # Rough per-station budget
            
        print(f"✓ Configuration loaded: {len(config)} parameters")
        
        # Run analysis
        print("\nStarting analysis...")
        
        model = run_quick_analysis(
            route_file=args.routes,
            merged_loi_file=args.lois,
            gas_stations_csv=args.gas_stations if os.path.exists(args.gas_stations) else None,
            existing_stations=args.existing_stations if os.path.exists(args.existing_stations) else None,
            budget=args.budget,
            n_stations=args.n_stations,
            visualize_validation=not args.skip_validation
        )
        
        if model:
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print()
            print("Results have been saved to the output directory.")
            print("Check the analysis_summary.txt file for key findings.")
            print()
            return 0
        else:
            print("\n❌ ANALYSIS FAILED")
            print("Check error messages above for details.")
            return 1
            
    except ImportError as e:
        print(f"❌ ERROR: Missing dependencies: {e}")
        print()
        print("Please ensure all required modules are installed:")
        print("   pip install geopandas pandas numpy shapely matplotlib")
        return 1
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())