#!/usr/bin/env python3
"""
Configuration Customization Tool for H2 Station Siting Model
Allows easy modification of key parameters without editing the full config
"""

import json
import os
from comprehensive_config import get_comprehensive_config, get_scenario_configs

def create_custom_config():
    """Interactive tool to create custom configuration"""
    print("="*60)
    print("H2 STATION SITING MODEL - CONFIGURATION CUSTOMIZER")
    print("="*60)
    
    # Start with base configuration
    config = get_comprehensive_config()
    print(f"Loaded base configuration with {len(config)} parameters")
    
    print("\nKey parameters you can customize:")
    print("-" * 40)
    
    # Economic parameters
    print("\n1. ECONOMIC PARAMETERS:")
    current_h2_price = config['h2_price_per_kg']
    new_h2_price = input(f"   H2 price per kg [current: ${current_h2_price:.2f}]: ")
    if new_h2_price.strip():
        config['h2_price_per_kg'] = float(new_h2_price)
        
    current_capex = config['station_capex']
    new_capex = input(f"   Station CAPEX [current: ${current_capex:,.0f}]: ")
    if new_capex.strip():
        config['station_capex'] = float(new_capex)
        
    current_discount = config['discount_rate']
    new_discount = input(f"   Discount rate [current: {current_discount:.1%}]: ")
    if new_discount.strip():
        config['discount_rate'] = float(new_discount)
    
    # Spatial parameters
    print("\n2. SPATIAL PARAMETERS:")
    current_radius = config['service_radius_miles']
    new_radius = input(f"   Service radius (miles) [current: {current_radius}]: ")
    if new_radius.strip():
        config['service_radius_miles'] = float(new_radius)
        
    current_spacing = config['min_station_spacing_miles']
    new_spacing = input(f"   Min station spacing (miles) [current: {current_spacing}]: ")
    if new_spacing.strip():
        config['min_station_spacing_miles'] = float(new_spacing)
    
    # Capacity parameters
    print("\n3. CAPACITY PARAMETERS:")
    current_capacity = config['station_capacity_kg_per_day']
    new_capacity = input(f"   Station capacity (kg/day) [current: {current_capacity:,.0f}]: ")
    if new_capacity.strip():
        config['station_capacity_kg_per_day'] = float(new_capacity)
        
    current_utilization = config['capacity_utilization_target']
    new_utilization = input(f"   Target utilization [current: {current_utilization:.1%}]: ")
    if new_utilization.strip():
        config['capacity_utilization_target'] = float(new_utilization)
    
    # Market parameters
    print("\n4. MARKET PARAMETERS:")
    current_penetration = config['market_penetration_rate']
    new_penetration = input(f"   Market penetration rate [current: {current_penetration:.1%}]: ")
    if new_penetration.strip():
        config['market_penetration_rate'] = float(new_penetration)
        
    current_growth = config['adoption_growth_rate']
    new_growth = input(f"   Adoption growth rate [current: {current_growth:.1%}]: ")
    if new_growth.strip():
        config['adoption_growth_rate'] = float(new_growth)
    
    # Save custom configuration
    print("\n" + "="*60)
    config_name = input("Enter name for custom configuration [default: custom]: ").strip()
    if not config_name:
        config_name = "custom"
        
    filename = f"config_{config_name}.json"
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"✓ Custom configuration saved to: {filename}")
    print(f"✓ Configuration contains {len(config)} parameters")
    
    return filename

def load_custom_config(filename):
    """Load a custom configuration file"""
    if not os.path.exists(filename):
        print(f"❌ Configuration file not found: {filename}")
        return None
        
    with open(filename, 'r') as f:
        config = json.load(f)
        
    print(f"✓ Loaded custom configuration from: {filename}")
    print(f"✓ Configuration contains {len(config)} parameters")
    
    return config

def show_scenarios():
    """Show available predefined scenarios"""
    scenarios = get_scenario_configs()
    
    print("AVAILABLE PREDEFINED SCENARIOS:")
    print("="*50)
    
    for name, config in scenarios.items():
        print(f"\n{name.upper()}:")
        print(f"   H2 Price: ${config['h2_price_per_kg']:.2f}/kg")
        print(f"   Market Penetration: {config['market_penetration_rate']:.1%}")
        print(f"   Service Radius: {config['service_radius_miles']} miles")
        print(f"   Target Utilization: {config['capacity_utilization_target']:.1%}")
        
        if name == 'conservative':
            print("   → Higher prices, lower penetration, conservative assumptions")
        elif name == 'aggressive':
            print("   → Lower prices, higher penetration, optimistic assumptions")
        elif name == 'high_density':
            print("   → Smaller service areas, closer spacing, urban focus")
        elif name == 'rural_focus':
            print("   → Larger service areas, wider spacing, rural optimization")

def compare_configs(config1_name, config2_name):
    """Compare two configurations"""
    scenarios = get_scenario_configs()
    
    # Load configurations
    if config1_name in scenarios:
        config1 = scenarios[config1_name]
    elif os.path.exists(config1_name):
        config1 = load_custom_config(config1_name)
    else:
        print(f"❌ Configuration not found: {config1_name}")
        return
        
    if config2_name in scenarios:
        config2 = scenarios[config2_name]
    elif os.path.exists(config2_name):
        config2 = load_custom_config(config2_name)
    else:
        print(f"❌ Configuration not found: {config2_name}")
        return
    
    print(f"CONFIGURATION COMPARISON: {config1_name} vs {config2_name}")
    print("="*60)
    
    # Key parameters to compare
    key_params = [
        ('h2_price_per_kg', 'H2 Price ($/kg)'),
        ('station_capex', 'Station CAPEX ($)'),
        ('discount_rate', 'Discount Rate'),
        ('service_radius_miles', 'Service Radius (miles)'),
        ('station_capacity_kg_per_day', 'Station Capacity (kg/day)'),
        ('market_penetration_rate', 'Market Penetration'),
        ('capacity_utilization_target', 'Target Utilization'),
    ]
    
    for param, label in key_params:
        val1 = config1.get(param, 'N/A')
        val2 = config2.get(param, 'N/A')
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if param in ['discount_rate', 'market_penetration_rate', 'capacity_utilization_target']:
                print(f"{label:<25} {val1:.1%}  vs  {val2:.1%}")
            elif param in ['station_capex']:
                print(f"{label:<25} ${val1:,.0f}  vs  ${val2:,.0f}")
            elif param in ['h2_price_per_kg']:
                print(f"{label:<25} ${val1:.2f}  vs  ${val2:.2f}")
            else:
                print(f"{label:<25} {val1}  vs  {val2}")
        else:
            print(f"{label:<25} {val1}  vs  {val2}")

def main():
    """Main configuration customization interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='H2 Model Configuration Customizer')
    parser.add_argument('--create', action='store_true', 
                       help='Create a new custom configuration interactively')
    parser.add_argument('--show-scenarios', action='store_true',
                       help='Show available predefined scenarios')
    parser.add_argument('--compare', nargs=2, metavar=('CONFIG1', 'CONFIG2'),
                       help='Compare two configurations')
    parser.add_argument('--load', metavar='FILENAME',
                       help='Load and validate a custom configuration file')
    
    args = parser.parse_args()
    
    if args.create:
        create_custom_config()
    elif args.show_scenarios:
        show_scenarios()
    elif args.compare:
        compare_configs(args.compare[0], args.compare[1])
    elif args.load:
        config = load_custom_config(args.load)
        if config:
            from comprehensive_config import validate_config
            errors = validate_config(config)
            if errors:
                print(f"❌ Configuration validation errors:")
                for param, error in errors.items():
                    print(f"   {param}: {error}")
            else:
                print("✅ Configuration validation passed")
    else:
        # Interactive mode
        print("H2 Station Siting Model - Configuration Customizer")
        print("Options:")
        print("  1. Create custom configuration")
        print("  2. Show predefined scenarios") 
        print("  3. Compare configurations")
        print("  4. Validate configuration file")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            create_custom_config()
        elif choice == '2':
            show_scenarios()
        elif choice == '3':
            config1 = input("First configuration (scenario name or file): ").strip()
            config2 = input("Second configuration (scenario name or file): ").strip()
            compare_configs(config1, config2)
        elif choice == '4':
            filename = input("Configuration file to validate: ").strip()
            config = load_custom_config(filename)
            if config:
                from comprehensive_config import validate_config
                errors = validate_config(config)
                if errors:
                    print(f"❌ Validation errors:")
                    for param, error in errors.items():
                        print(f"   {param}: {error}")
                else:
                    print("✅ Configuration validation passed")

if __name__ == "__main__":
    main()