# H2 Station Siting Model

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/yourusername/h2-station-siting)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

A comprehensive optimization model for hydrogen refueling station placement along freight corridors, designed to support the transition to zero-emission freight transportation.

## Overview

The H2 Station Siting Model uses advanced spatial analysis, demand estimation, and optimization techniques to identify optimal locations for hydrogen refueling infrastructure. The model considers truck traffic patterns, existing infrastructure, competition dynamics, and economic viability to recommend station placements that maximize network coverage and financial returns.

### Key Features

- **Demand Surface Estimation**: Kernel density estimation of hydrogen demand based on truck traffic data
- **Multi-Strategy Candidate Generation**: Route-based, LOI-based, and hybrid approaches
- **Competition Modeling**: Sophisticated market share calculation using Huff gravity models
- **Iterative Selection**: Advanced algorithm with demand cannibalization and tipping point analysis
- **Flexible Capacity Optimization**: Dynamic station sizing based on local demand conditions
- **Comprehensive Validation**: Built-in validation against existing infrastructure patterns

## Installation

### Prerequisites

- Python 3.9 or higher
- GDAL/OGR libraries for geospatial operations
- Optional: Gurobi solver for enhanced optimization performance

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/h2-station-siting.git
cd h2-station-siting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run quick analysis
python h2_station_quick_setup.py

Usage
Basic Usage

from h2_station_model import H2StationSitingModel

# Initialize model with default configuration
model = H2StationSitingModel()

# Load data
model.load_data(
    route_file='data/combined_network.geojson',
    loi_files=['data/gas_stations.csv', 'data/rest_areas.csv'],
    existing_stations='data/existing_h2_stations.csv'
)

# Generate candidates and run analysis
model.generate_candidates(strategy='hybrid')
model.calculate_utilization_probability()
model.run_iterative_station_selection(max_stations=50)


Advanced Configuration

Custom configuration
custom_config = {
    'h2_price_per_kg': 28.50,
    'station_capacity_kg_per_day': 2000,
    'service_radius_miles': 2.0,
    'min_station_spacing_miles': 1.0
}

model = H2StationSitingModel(config=custom_config)

Data Requirements
Input Data Formats

Route Network (GeoJSON/Shapefile)

Required fields: geometry, TOT_TRK_AADT, length_miles
Optional fields: route_id, DESIG, h2_demand_daily_kg


Locations of Interest (CSV/GeoJSON)

Required fields: latitude, longitude (or geometry)
Optional fields: type, name, capacity


Existing Stations (CSV)

Required fields: latitude, longitude, station_capacity
Optional fields: name, operator, commission_date



See data/README.md for detailed data specifications.
Model Components
1. Demand Estimation

Kernel density estimation with configurable bandwidth
Route-based demand allocation
Consideration of truck traffic volumes and trip lengths

2. Candidate Generation

Route-based: Regular intervals along high-traffic routes
LOI-based: Existing infrastructure locations (gas stations, rest areas)
Hybrid: Combination of both approaches

3. Competition Analysis

Huff gravity model for market share calculation
Network effects consideration
Dynamic demand adjustment

4. Optimization

Iterative greedy selection with look-ahead
Mixed-integer linear programming (MILP) option
Multi-objective optimization capabilities

Output Files
The model generates comprehensive outputs in the specified directory:

output/
├── data/
│   ├── selected_stations.geojson      # Final station locations
│   ├── optimal_portfolio.csv          # Station details with metrics
│   └── iteration_history.csv          # Selection process tracking
├── visualizations/
│   ├── demand_surface.png             # Hydrogen demand heatmap
│   ├── network_coverage.png           # Service area coverage
│   └── competition_analysis.png       # Market dynamics
├── reports/
│   ├── analysis_summary.pdf           # Executive summary
│   ├── technical_report.txt           # Detailed technical results
│   └── validation_report.html         # Model validation results
└── config/
    └── model_config.json              # Configuration used for run



Contributing
We welcome contributions! Please see CONTRIBUTING.md for guidelines.
Citation
If you use this model in your research, please cite:

@software{
  h2_station_siting_model,
  title = {H2 Station Siting Model: Optimization for Hydrogen Infrastructure},
  version = {1.0.0},
  year = {2025},
  url = {https://github.com/b0nnyCastle/h2-station-siting}
}


License
This project is licensed under the MIT License - see LICENSE file for details.
Acknowledgments

[Placeholder: Funding Agency]
[Placeholder: Data Provider Organizations]
[Prof. Sally Benson]

Contact
For questions or support, please open an issue or contact [fayoola@alumni.stanford.edu].