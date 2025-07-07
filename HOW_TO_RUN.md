# How to Run the H2 Station Siting Model (Updated)

## Overview
The H2 Station Siting Model has been updated with comprehensive configuration management and improved error handling. Here are the different ways to run the model after the recent updates.

## Quick Start (Recommended)

### Option 1: Using the Easy Runner Script
```bash
# Basic run with default settings
python run_h2_model.py

# Run with specific scenario
python run_h2_model.py --scenario conservative --n_stations 25

# Run with budget constraint
python run_h2_model.py --scenario aggressive --n_stations 50 --budget 100000000

# Run with custom files
python run_h2_model.py --routes my_routes.geojson --lois my_lois.geojson --n_stations 30
```

### Option 2: Using the Original Quick Setup (Fixed)
```bash
# This now works with the comprehensive configuration system
python h2_station_quick_setup.py
```

## Available Scenarios

The model now supports multiple predefined scenarios:

- **`default`**: Balanced assumptions for general planning
- **`conservative`**: Higher prices, lower adoption, cautious projections
- **`aggressive`**: Lower prices, higher adoption, optimistic projections  
- **`high_density`**: Urban focus with smaller service areas
- **`rural_focus`**: Rural optimization with larger service areas

## Configuration Management

### View Available Scenarios
```bash
python customize_config.py --show-scenarios
```

### Create Custom Configuration
```bash
# Interactive configuration creator
python customize_config.py --create

# This will guide you through customizing key parameters:
# - H2 price per kg
# - Station CAPEX
# - Service radius
# - Market penetration rate
# - etc.
```

### Compare Configurations
```bash
# Compare two scenarios
python customize_config.py --compare conservative aggressive

# Compare scenario with custom config
python customize_config.py --compare default my_custom_config.json
```

## Required Files

Ensure these files are in your working directory:
- ✅ `combined_network.geojson` (route network)
- ✅ `all_merged_loi_routes.geojson` (locations of interest)
- ⚠️ `gas_stations_from_CA_Energy_Commission_Gas_Stations.csv` (optional)
- ⚠️ `existing_ze_infrastructure.geojson` (optional)

## Command Line Options

### run_h2_model.py Options
```bash
python run_h2_model.py --help

Options:
  --scenario {default,conservative,aggressive,high_density,rural_focus}
  --n_stations N        Number of stations to optimize for
  --budget BUDGET       Budget constraint in dollars
  --routes FILE         Path to route network file
  --lois FILE          Path to merged LOI file
  --gas_stations FILE  Path to gas stations CSV
  --existing_stations FILE  Path to existing stations
  --skip_validation    Skip validation visualizations
  --output_dir DIR     Custom output directory
```

## Example Use Cases

### 1. Conservative Planning Analysis
```bash
python run_h2_model.py \
  --scenario conservative \
  --n_stations 20 \
  --budget 50000000
```

### 2. Aggressive Market Development
```bash
python run_h2_model.py \
  --scenario aggressive \
  --n_stations 75 \
  --budget 200000000
```

### 3. Urban High-Density Network
```bash
python run_h2_model.py \
  --scenario high_density \
  --n_stations 100 \
  --budget 150000000
```

### 4. Rural Coverage Analysis
```bash
python run_h2_model.py \
  --scenario rural_focus \
  --n_stations 30 \
  --budget 75000000
```

### 5. Custom Configuration Run
```bash
# First create custom config
python customize_config.py --create

# Then run with custom settings
python run_h2_model.py --n_stations 40
```

## Output Structure

Results are saved to timestamped directories (e.g., `h2_station_quick_results_v1/`):

```
h2_station_quick_results_v1/
├── analysis_summary.txt          # Key findings summary
├── model_config.json            # Configuration used
├── data/                         # Raw results and data
│   ├── selected_stations.geojson
│   ├── all_candidates.geojson
│   ├── optimal_portfolio.geojson
│   └── summary_report.txt
├── visualizations/               # Maps and charts
│   ├── portfolio_map.png
│   ├── demand_surface.png
│   └── economic_dashboard.png
├── validation/                   # Validation analysis
│   ├── validation_comprehensive.png
│   └── route_alignment_analysis.png
├── reports/                      # Detailed reports
│   └── selected_stations.geojson
└── iteration_results/            # Detailed iteration tracking
    ├── iteration_history.csv
    └── selection_summary.json
```

## Troubleshooting

### Common Issues and Solutions

#### 1. KeyError: 'min_iteration_capacity_kg_per_day'
**Problem**: Old configuration missing required parameters
**Solution**: Use the updated scripts which include comprehensive configuration

```bash
# Use the fixed runner instead of old scripts
python run_h2_model.py
```

#### 2. Missing Data Files
**Problem**: Required geospatial files not found
**Solution**: Ensure data files are in the correct location

```bash
# Check for required files
ls -la combined_network.geojson
ls -la all_merged_loi_routes.geojson

# If missing, specify custom paths
python run_h2_model.py --routes path/to/routes.geojson --lois path/to/lois.geojson
```

#### 3. Memory Issues with Large Datasets
**Problem**: Out of memory errors
**Solution**: Use smaller analysis scope or increase system memory

```bash
# Reduce number of stations for memory efficiency
python run_h2_model.py --n_stations 25

# Skip validation visualizations to save memory
python run_h2_model.py --skip_validation
```

#### 4. Slow Performance
**Problem**: Analysis takes too long
**Solution**: Use conservative scenario or reduce scope

```bash
# Use conservative scenario (fewer candidates)
python run_h2_model.py --scenario conservative --n_stations 20
```

## Advanced Usage

### Running with Custom Data
```bash
# Prepare your data files
# - routes.geojson: Highway/route network with truck AADT
# - lois.geojson: Locations of interest (ports, rest areas, etc.)

python run_h2_model.py \
  --routes your_routes.geojson \
  --lois your_lois.geojson \
  --n_stations 40
```

### Batch Analysis
```bash
# Run multiple scenarios for comparison
for scenario in conservative default aggressive; do
  python run_h2_model.py --scenario $scenario --n_stations 30
done
```

### Integration with Other Tools
```python
# Python script integration
from comprehensive_config import get_comprehensive_config
from h2_station_model import H2StationSitingModel

# Load configuration
config = get_comprehensive_config()

# Customize parameters
config['h2_price_per_kg'] = 25.0
config['service_radius_miles'] = 3.0

# Run model
model = H2StationSitingModel(config)
# ... continue with analysis
```

## Configuration Parameters

### Key Economic Parameters
- `h2_price_per_kg`: Retail hydrogen price ($/kg)
- `station_capex`: Station capital expenditure ($)
- `discount_rate`: Financial discount rate
- `operating_margin`: Target operating margin

### Key Spatial Parameters
- `service_radius_miles`: Station service radius
- `min_station_spacing_miles`: Minimum distance between stations
- `demand_kernel_bandwidth_miles`: Demand estimation bandwidth

### Key Market Parameters
- `market_penetration_rate`: Current H2 truck penetration
- `adoption_growth_rate`: Annual adoption growth rate
- `capacity_utilization_target`: Target station utilization

## Getting Help

### View Configuration Options
```bash
python customize_config.py --show-scenarios
```

### Compare Scenarios
```bash
python customize_config.py --compare conservative aggressive
```

### Validate Custom Configuration
```bash
python customize_config.py --load my_config.json
```

### Command Line Help
```bash
python run_h2_model.py --help
python customize_config.py --help
```

## Updates from Previous Version

### What Changed
1. **Comprehensive Configuration**: 147 validated parameters instead of partial configuration
2. **Error Handling**: Improved validation and error recovery
3. **Multiple Scenarios**: Predefined scenarios for different planning contexts
4. **Better Documentation**: Clear usage instructions and troubleshooting
5. **Simplified Interface**: Easy-to-use runner scripts

### Migration from Old Version
```bash
# Old way (now fixed but comprehensive)
python h2_station_quick_setup.py

# New recommended way
python run_h2_model.py

# Both now work with comprehensive configuration system
```

The model is now production-ready with robust error handling, comprehensive parameter validation, and multiple deployment scenarios for California hydrogen infrastructure planning.