#!/usr/bin/env python3
"""
Performance Benchmarking Tests for H2 Station Siting Model
Focused on computational efficiency and scalability for transportation networks
"""

import unittest
import time
import psutil
import gc
import sys
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from memory_profiler import profile
import cProfile
import pstats
from io import StringIO

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2_station_model import H2StationSitingModel
from comprehensive_config import get_comprehensive_config
from test_fixtures import TestDataGenerator

warnings.filterwarnings('ignore')


class PerformanceBenchmark:
    """Base class for performance benchmarking"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results = {}
        
    def measure_memory(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def measure_performance(self, func, *args, **kwargs):
        """Measure execution time and memory usage of a function"""
        gc.collect()  # Clean up before measurement
        
        start_memory = self.measure_memory()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = self.measure_memory()
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_start': start_memory,
            'memory_end': end_memory,
            'memory_delta': end_memory - start_memory
        }


class TestSpatialAlgorithmPerformance(unittest.TestCase):
    """Test performance of spatial algorithms critical for transportation networks"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.benchmark = PerformanceBenchmark()
        self.config = get_comprehensive_config()
        self.generator = TestDataGenerator(seed=42)
        
    def test_spatial_indexing_performance(self):
        """Benchmark spatial indexing performance across different data sizes"""
        print("\n--- Benchmarking Spatial Indexing Performance ---")
        
        dataset_sizes = [100, 500, 1000, 2500, 5000]
        results = {}
        
        for size in dataset_sizes:
            print(f"Testing spatial index with {size} geometries...")
            
            # Generate test data
            routes = self.generator.generate_california_routes(n_routes=size)
            
            # Benchmark spatial index creation
            def create_spatial_index():
                try:
                    from rtree import index as rtree_index
                    idx = rtree_index.Index()
                    for i, geom in enumerate(routes.geometry):
                        if geom is not None:
                            idx.insert(i, geom.bounds)
                    return idx
                except ImportError:
                    # Fallback to manual spatial operations
                    return None
                    
            perf = self.benchmark.measure_performance(create_spatial_index)
            
            # Benchmark spatial queries
            def spatial_query_test():
                if perf['result'] is None:
                    return 0
                    
                # Test 100 random point queries
                query_count = 0
                for _ in range(100):
                    x = np.random.uniform(-122, -117)
                    y = np.random.uniform(33, 38)
                    # Convert to projected coordinates (approximate)
                    px = (x + 120) * 100000
                    py = (y - 33) * 100000
                    
                    bbox = (px-1000, py-1000, px+1000, py+1000)
                    hits = list(perf['result'].intersection(bbox))
                    query_count += len(hits)
                    
                return query_count
                
            query_perf = self.benchmark.measure_performance(spatial_query_test)
            
            results[size] = {
                'index_time': perf['execution_time'],
                'index_memory': perf['memory_delta'],
                'query_time': query_perf['execution_time'],
                'query_count': query_perf['result']
            }
            
            print(f"  Index creation: {perf['execution_time']:.3f}s, "
                  f"Memory: {perf['memory_delta']:.1f}MB")
            print(f"  Query performance: {query_perf['execution_time']:.3f}s for 100 queries")
            
        # Analyze scalability
        print("\nSpatial Indexing Scalability Analysis:")
        for i, size in enumerate(dataset_sizes[1:], 1):
            prev_size = dataset_sizes[i-1]
            time_ratio = results[size]['index_time'] / results[prev_size]['index_time']
            size_ratio = size / prev_size
            
            print(f"  {prev_size} -> {size}: Time scaling = {time_ratio:.2f}x "
                  f"(ideal: {size_ratio:.2f}x)")
                  
        self.results_spatial_indexing = results
        return results
        
    def test_distance_calculation_performance(self):
        """Benchmark distance calculation methods"""
        print("\n--- Benchmarking Distance Calculation Performance ---")
        
        # Generate test geometries
        points = [Point(np.random.uniform(-2000000, -1500000), 
                       np.random.uniform(500000, 1000000)) for _ in range(1000)]
        lines = [LineString([(p.x, p.y), (p.x + np.random.uniform(-5000, 5000), 
                           p.y + np.random.uniform(-5000, 5000))]) for p in points[:500]]
        
        point_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:3310")
        line_gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:3310")
        
        # Test different distance calculation approaches
        methods = {
            'point_to_point': lambda: [p1.distance(p2) for p1 in points[:100] for p2 in points[100:200]],
            'point_to_line': lambda: [point.distance(line) for point in points[:100] for line in lines[:100]],
            'vectorized_point_line': lambda: point_gdf.head(100).geometry.apply(
                lambda p: line_gdf.head(100).geometry.distance(p).min()),
            'bulk_distance_matrix': lambda: point_gdf.head(50).geometry.apply(
                lambda p: point_gdf.geometry.distance(p))
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            print(f"Testing {method_name}...")
            perf = self.benchmark.measure_performance(method_func)
            
            results[method_name] = {
                'time': perf['execution_time'],
                'memory': perf['memory_delta']
            }
            
            print(f"  {method_name}: {perf['execution_time']:.3f}s, "
                  f"Memory: {perf['memory_delta']:.1f}MB")
                  
        # Find most efficient method
        fastest_method = min(results.keys(), key=lambda k: results[k]['time'])
        print(f"\nMost efficient method: {fastest_method}")
        
        self.results_distance_calculation = results
        return results
        
    def test_demand_surface_computation_scaling(self):
        """Test demand surface computation performance scaling"""
        print("\n--- Benchmarking Demand Surface Computation ---")
        
        grid_sizes = [10, 25, 50, 100]  # Grid resolution multipliers
        results = {}
        
        for grid_mult in grid_sizes:
            print(f"Testing demand surface with {grid_mult}x{grid_mult} grid...")
            
            # Create mock demand points
            n_demand_points = grid_mult ** 2
            demand_points = []
            
            for i in range(n_demand_points):
                x = np.random.uniform(-2000000, -1500000)
                y = np.random.uniform(500000, 1000000)
                demand = np.random.uniform(100, 2000)
                demand_points.append({
                    'geometry': Point(x, y),
                    'demand_kg_day': demand
                })
                
            demand_gdf = gpd.GeoDataFrame(demand_points, crs="EPSG:3310")
            
            # Benchmark demand surface creation
            def create_demand_surface():
                # Simplified demand surface calculation
                bounds = demand_gdf.total_bounds
                x_range = np.linspace(bounds[0], bounds[2], grid_mult)
                y_range = np.linspace(bounds[1], bounds[3], grid_mult)
                
                surface = np.zeros((len(y_range), len(x_range)))
                
                for i, x in enumerate(x_range):
                    for j, y in enumerate(y_range):
                        grid_point = Point(x, y)
                        # Simple inverse distance weighting
                        weights = 1 / (demand_gdf.geometry.distance(grid_point) + 100)
                        weighted_demand = (demand_gdf['demand_kg_day'] * weights).sum()
                        surface[j, i] = weighted_demand
                        
                return surface
                
            perf = self.benchmark.measure_performance(create_demand_surface)
            
            surface_size = perf['result'].size
            results[grid_mult] = {
                'time': perf['execution_time'],
                'memory': perf['memory_delta'],
                'surface_size': surface_size,
                'points_processed': n_demand_points
            }
            
            print(f"  Grid {grid_mult}x{grid_mult}: {perf['execution_time']:.3f}s, "
                  f"Memory: {perf['memory_delta']:.1f}MB")
                  
        self.results_demand_surface = results
        return results
        
    def test_competition_graph_performance(self):
        """Test competition graph algorithms performance"""
        print("\n--- Benchmarking Competition Graph Performance ---")
        
        candidate_counts = [50, 100, 200, 500]
        results = {}
        
        for n_candidates in candidate_counts:
            print(f"Testing competition graph with {n_candidates} candidates...")
            
            # Create mock candidates
            candidates_data = []
            for i in range(n_candidates):
                candidates_data.append({
                    'geometry': Point(np.random.uniform(-2000000, -1500000),
                                    np.random.uniform(500000, 1000000)),
                    'expected_demand_kg_day': np.random.uniform(100, 2000),
                    'npv_proxy': np.random.uniform(-100000, 500000)
                })
                
            candidates_gdf = gpd.GeoDataFrame(candidates_data, crs="EPSG:3310")
            
            # Test competition graph creation
            def create_competition_graph():
                model = H2StationSitingModel(self.config)
                model.candidates = candidates_gdf
                
                # Initialize competition graph
                if hasattr(model, 'CompetitionGraph'):
                    comp_graph = model.CompetitionGraph(model.config)
                    
                    # Add all candidates
                    for idx, candidate in candidates_gdf.iterrows():
                        comp_graph.add_or_update_station(idx, candidate, 'candidate')
                        
                    # Add competition edges (limit to avoid O(n²) explosion)
                    max_edges = min(1000, n_candidates * 10)
                    edge_count = 0
                    
                    for i in range(min(n_candidates, 50)):  # Limit outer loop
                        for j in range(i+1, min(n_candidates, i+20)):  # Limit inner loop
                            if edge_count < max_edges:
                                comp_graph.add_competition_edge(i, j)
                                edge_count += 1
                            else:
                                break
                        if edge_count >= max_edges:
                            break
                            
                    return comp_graph, edge_count
                else:
                    return None, 0
                    
            perf = self.benchmark.measure_performance(create_competition_graph)
            
            if perf['result'][0] is not None:
                graph, edge_count = perf['result']
                
                # Test market share calculation
                def test_market_share():
                    if graph and hasattr(graph, 'calculate_market_share'):
                        # Test market share for first 10 candidates
                        shares = []
                        for i in range(min(10, n_candidates)):
                            try:
                                share = graph.calculate_market_share(i)
                                shares.append(share)
                            except:
                                shares.append(0)
                        return shares
                    return []
                    
                share_perf = self.benchmark.measure_performance(test_market_share)
                
                results[n_candidates] = {
                    'graph_creation_time': perf['execution_time'],
                    'graph_memory': perf['memory_delta'],
                    'edge_count': edge_count,
                    'market_share_time': share_perf['execution_time'],
                    'market_shares': len(share_perf['result'])
                }
                
                print(f"  {n_candidates} candidates: Graph creation {perf['execution_time']:.3f}s, "
                      f"Edges: {edge_count}, Market share calc: {share_perf['execution_time']:.3f}s")
            else:
                print(f"  {n_candidates} candidates: Competition graph not available")
                results[n_candidates] = {'error': 'CompetitionGraph not available'}
                
        self.results_competition_graph = results
        return results


class TestOptimizationPerformance(unittest.TestCase):
    """Test performance of optimization algorithms"""
    
    def setUp(self):
        """Set up optimization performance tests"""
        self.benchmark = PerformanceBenchmark()
        self.config = get_comprehensive_config()
        
    def test_portfolio_optimization_scaling(self):
        """Test portfolio optimization performance with different problem sizes"""
        print("\n--- Benchmarking Portfolio Optimization ---")
        
        problem_sizes = [10, 25, 50, 100]
        results = {}
        
        for size in problem_sizes:
            print(f"Testing portfolio optimization with {size} candidates...")
            
            # Create mock optimization problem
            model = H2StationSitingModel(self.config)
            
            # Generate mock candidates with economic data
            candidates_data = []
            for i in range(size):
                candidates_data.append({
                    'geometry': Point(i*1000, i*1000),
                    'npv_proxy': np.random.uniform(-50000, 300000),
                    'expected_demand_kg_day': np.random.uniform(100, 1500),
                    'station_capex': np.random.uniform(8000000, 15000000)
                })
                
            model.candidates = gpd.GeoDataFrame(candidates_data, crs="EPSG:3310")
            
            # Test optimization
            def run_optimization():
                try:
                    # Mock portfolio optimization
                    n_select = min(size // 3, 15)  # Select up to 1/3 of candidates
                    
                    # Simple greedy selection based on NPV
                    sorted_candidates = model.candidates.sort_values('npv_proxy', ascending=False)
                    selected_indices = sorted_candidates.head(n_select).index.tolist()
                    
                    return selected_indices
                except Exception as e:
                    return []
                    
            perf = self.benchmark.measure_performance(run_optimization)
            
            selected_count = len(perf['result'])
            results[size] = {
                'optimization_time': perf['execution_time'],
                'memory_usage': perf['memory_delta'],
                'candidates_evaluated': size,
                'candidates_selected': selected_count
            }
            
            print(f"  {size} candidates: {perf['execution_time']:.3f}s, "
                  f"Selected: {selected_count}, Memory: {perf['memory_delta']:.1f}MB")
                  
        self.results_optimization = results
        return results
        
    def test_iterative_selection_performance(self):
        """Test iterative station selection algorithm performance"""
        print("\n--- Benchmarking Iterative Selection ---")
        
        # Test with different iteration counts
        iteration_counts = [5, 10, 20, 50]
        results = {}
        
        for max_iter in iteration_counts:
            print(f"Testing iterative selection with {max_iter} iterations...")
            
            model = H2StationSitingModel(self.config)
            
            # Create larger candidate set for iterative selection
            n_candidates = 100
            candidates_data = []
            for i in range(n_candidates):
                candidates_data.append({
                    'geometry': Point(np.random.uniform(-2000000, -1500000),
                                    np.random.uniform(500000, 1000000)),
                    'npv_proxy': np.random.uniform(-100000, 400000),
                    'expected_demand_kg_day': np.random.uniform(100, 2000),
                    'capacity_kg_day': np.random.uniform(1000, 3000)
                })
                
            model.candidates = gpd.GeoDataFrame(candidates_data, crs="EPSG:3310")
            
            # Mock iterative selection
            def run_iterative_selection():
                selected_stations = []
                remaining_candidates = model.candidates.copy()
                
                for iteration in range(max_iter):
                    if len(remaining_candidates) == 0:
                        break
                        
                    # Select best candidate (simplified)
                    best_idx = remaining_candidates['npv_proxy'].idxmax()
                    selected_stations.append(best_idx)
                    
                    # Remove selected candidate and nearby candidates (simplified competition)
                    selected_geom = remaining_candidates.loc[best_idx, 'geometry']
                    distances = remaining_candidates.geometry.distance(selected_geom)
                    nearby_mask = distances < 5000  # 5km exclusion radius
                    
                    remaining_candidates = remaining_candidates[~nearby_mask]
                    
                    # Simulate demand update (simplified)
                    if len(remaining_candidates) > 0:
                        demand_decay = 0.95  # 5% demand decay per iteration
                        remaining_candidates.loc[:, 'expected_demand_kg_day'] *= demand_decay
                        remaining_candidates.loc[:, 'npv_proxy'] *= demand_decay
                        
                return selected_stations
                
            perf = self.benchmark.measure_performance(run_iterative_selection)
            
            selected_count = len(perf['result'])
            results[max_iter] = {
                'selection_time': perf['execution_time'],
                'memory_usage': perf['memory_delta'],
                'iterations_completed': min(max_iter, selected_count),
                'stations_selected': selected_count,
                'time_per_iteration': perf['execution_time'] / max(max_iter, 1)
            }
            
            print(f"  {max_iter} iterations: {perf['execution_time']:.3f}s, "
                  f"Selected: {selected_count}, Time/iter: {results[max_iter]['time_per_iteration']:.3f}s")
                  
        self.results_iterative_selection = results
        return results


class TestMemoryScaling(unittest.TestCase):
    """Test memory usage and scaling characteristics"""
    
    def setUp(self):
        """Set up memory scaling tests"""
        self.benchmark = PerformanceBenchmark()
        self.config = get_comprehensive_config()
        
    def test_memory_scaling_with_data_size(self):
        """Test how memory usage scales with input data size"""
        print("\n--- Testing Memory Scaling with Data Size ---")
        
        data_sizes = [100, 500, 1000, 2500]
        results = {}
        
        for size in data_sizes:
            print(f"Testing memory usage with {size} routes and {size*2} candidates...")
            
            gc.collect()  # Clean memory before test
            initial_memory = self.benchmark.measure_memory()
            
            # Create model and data
            model = H2StationSitingModel(self.config)
            
            # Generate routes
            routes_data = []
            for i in range(size):
                start_x = np.random.uniform(-2000000, -1500000)
                start_y = np.random.uniform(500000, 1000000)
                end_x = start_x + np.random.uniform(-10000, 10000)
                end_y = start_y + np.random.uniform(-10000, 10000)
                
                routes_data.append({
                    'geometry': LineString([(start_x, start_y), (end_x, end_y)]),
                    'tot_truck_aadt': np.random.randint(200, 5000),
                    'route_id': f'route_{i}'
                })
                
            model.routes = gpd.GeoDataFrame(routes_data, crs="EPSG:3310")
            
            after_routes_memory = self.benchmark.measure_memory()
            
            # Generate candidates
            candidates_data = []
            for i in range(size * 2):
                candidates_data.append({
                    'geometry': Point(np.random.uniform(-2000000, -1500000),
                                    np.random.uniform(500000, 1000000)),
                    'expected_demand_kg_day': np.random.uniform(100, 2000),
                    'truck_aadt': np.random.randint(200, 3000),
                    'npv_proxy': np.random.uniform(-100000, 500000)
                })
                
            model.candidates = gpd.GeoDataFrame(candidates_data, crs="EPSG:3310")
            
            after_candidates_memory = self.benchmark.measure_memory()
            
            # Add spatial index
            try:
                from rtree import index as rtree_index
                spatial_index = rtree_index.Index()
                for i, geom in enumerate(model.routes.geometry):
                    if geom is not None:
                        spatial_index.insert(i, geom.bounds)
            except ImportError:
                spatial_index = None
                
            after_index_memory = self.benchmark.measure_memory()
            
            results[size] = {
                'initial_memory': initial_memory,
                'routes_memory': after_routes_memory - initial_memory,
                'candidates_memory': after_candidates_memory - after_routes_memory,
                'index_memory': after_index_memory - after_candidates_memory,
                'total_memory': after_index_memory - initial_memory,
                'memory_per_route': (after_routes_memory - initial_memory) / size,
                'memory_per_candidate': (after_candidates_memory - after_routes_memory) / (size * 2)
            }
            
            print(f"  Routes: {results[size]['routes_memory']:.1f}MB "
                  f"({results[size]['memory_per_route']:.3f}MB per route)")
            print(f"  Candidates: {results[size]['candidates_memory']:.1f}MB "
                  f"({results[size]['memory_per_candidate']:.3f}MB per candidate)")
            print(f"  Spatial index: {results[size]['index_memory']:.1f}MB")
            print(f"  Total: {results[size]['total_memory']:.1f}MB")
            
            # Cleanup for next iteration
            del model
            gc.collect()
            
        self.results_memory_scaling = results
        return results
        
    def test_memory_leak_detection(self):
        """Test for memory leaks in repetitive operations"""
        print("\n--- Testing Memory Leak Detection ---")
        
        initial_memory = self.benchmark.measure_memory()
        memory_samples = [initial_memory]
        
        # Run repetitive operations
        for iteration in range(10):
            print(f"Iteration {iteration + 1}/10...")
            
            # Create and destroy model multiple times
            for _ in range(5):
                model = H2StationSitingModel(self.config)
                
                # Add some data
                routes_data = [{
                    'geometry': LineString([(i*1000, i*1000), ((i+1)*1000, (i+1)*1000)]),
                    'tot_truck_aadt': 1000 + i*100
                } for i in range(50)]
                
                model.routes = gpd.GeoDataFrame(routes_data, crs="EPSG:3310")
                
                # Simulate some processing
                model.routes['length'] = model.routes.geometry.length
                
                # Delete model
                del model
                
            # Force garbage collection
            gc.collect()
            
            # Sample memory
            current_memory = self.benchmark.measure_memory()
            memory_samples.append(current_memory)
            
            print(f"  Memory: {current_memory:.1f}MB "
                  f"(delta: {current_memory - initial_memory:+.1f}MB)")
                  
        # Analyze memory trend
        memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
        final_memory = memory_samples[-1]
        
        print(f"\nMemory leak analysis:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Net change: {final_memory - initial_memory:+.1f}MB")
        print(f"  Memory trend: {memory_trend:+.3f}MB per iteration")
        
        # Flag potential memory leaks
        if memory_trend > 1.0:  # More than 1MB per iteration trend
            print("  ⚠ Potential memory leak detected!")
        elif abs(final_memory - initial_memory) > 50:  # More than 50MB net increase
            print("  ⚠ Significant memory increase detected!")
        else:
            print("  ✓ No significant memory leaks detected")
            
        return {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_trend': memory_trend,
            'memory_samples': memory_samples
        }


def run_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    print("="*80)
    print("PERFORMANCE BENCHMARKS FOR H2 STATION SITING MODEL")
    print("="*80)
    
    # Check available profiling tools
    profiling_available = True
    try:
        import memory_profiler
    except ImportError:
        print("Warning: memory_profiler not available. Install with: pip install memory-profiler")
        profiling_available = False
        
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestSpatialAlgorithmPerformance))
    suite.addTest(loader.loadTestsFromTestCase(TestOptimizationPerformance))
    suite.addTest(loader.loadTestsFromTestCase(TestMemoryScaling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    
    result = runner.run(suite)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total benchmark time: {total_time:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    # Performance recommendations
    print("\nPERFORMANCE RECOMMENDATIONS:")
    print("-" * 40)
    
    if total_time < 60:
        print("✓ Benchmarks completed quickly - good performance")
    elif total_time < 180:
        print("⚠ Moderate benchmark time - consider optimization")
    else:
        print("✗ Slow benchmark time - optimization recommended")
        
    print("• Use spatial indexing for large datasets")
    print("• Implement caching for repeated calculations")
    print("• Consider parallel processing for independent operations")
    print("• Monitor memory usage with large candidate sets")
    print("• Profile critical paths for optimization opportunities")
    
    if not profiling_available:
        print("\nFor detailed memory profiling, install memory-profiler:")
        print("  pip install memory-profiler")
        print("  pip install psutil")
        
    return result.testsRun - len(result.failures) - len(result.errors), result.testsRun


if __name__ == '__main__':
    passed, total = run_performance_benchmarks()
    
    print(f"\nBenchmark completion: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL PERFORMANCE BENCHMARKS COMPLETED!")
        sys.exit(0)
    else:
        print("⚠️ Some performance issues detected")
        sys.exit(1)