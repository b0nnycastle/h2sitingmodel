# -------------------------  CONFIG  ---------------------------------
import os, glob, warnings, json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, Polygon, mapping
from shapely.ops import nearest_points
from rtree import index

# File locations -----------------------------------------------------
ROUTE_FILE = "freight_analysis/data/combined_network.geojson"

# any mix of *.shp / *.geojson / *.csv (csv needs lat/long columns)
LOI_SOURCES = [
    "downloaded_data/Airports.geojson",
    "downloaded_data/Designated Truck Parking Simplified.geojson",
    "downloaded_data/Major Traffic Generators.geojson",
    "downloaded_data/Seaports.geojson",
    "downloaded_data/CA_Hy_extract_USA_MHD_ZE_Infrastructure.geojson"
    "downloaded_data_strategic_freight/*.geojson",
    "CA_MDHD_refueling_locations_hy_with_geom.csv", 
    "downloaded_data/CA_Energy_Commission_-_Gas_Stations.geojson",
    "downloaded_data/CA_rest_areas.csv"]


# Create new directory structure
BASE_OUTPUT_DIR = "loi_route_matching_output"
INPUT_LOI_DIR = os.path.join(BASE_OUTPUT_DIR, "locations_of_interest")
RESULTS_DIR = os.path.join(BASE_OUTPUT_DIR, "assignment_results")

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_LOI_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# search radius (miles → metres)
RADIUS_MILES = 1.0
RADIUS_M = RADIUS_MILES * 1609.34

# metric CRS for CA (equal-area Albers)
TARGET_EPSG = 3310   # NAD83 / California Albers
# --------------------------------------------------------------------

# ---------------------------  HELPERS -------------------------------

def get_geometry_bounds(geom):
    """
    Get the bounding box for any geometry type.
    Returns the minx, miny, maxx, maxy bounds.
    """
    return geom.bounds


def load_loi_file(path: str) -> gpd.GeoDataFrame:
    """Load any of our LOI formats into a GeoDataFrame (WGS-84)."""
    ext = os.path.splitext(path)[1].lower()
    file_basename = os.path.basename(path)
    
    try:
        if ext == ".csv":
            # expect lat / lon columns (case-insensitive)
            df = pd.read_csv(path)
            
            # Find latitude column flexibly
            lat_cols = [c for c in df.columns if "lat" in c.lower()]
            if not lat_cols:
                raise ValueError(f"No latitude column found in {path}")
            lat_col = lat_cols[0]
            
            # Find longitude column flexibly
            lon_cols = [c for c in df.columns if "lon" in c.lower()]
            if not lon_cols:
                lon_cols = [c for c in df.columns if "long" in c.lower()]
            if not lon_cols:
                raise ValueError(f"No longitude column found in {path}")
            lon_col = lon_cols[0]
            
            # Check for existing geometry column
            if 'geometry' in df.columns and 'geometry_wkt' in df.columns:
                # If there's a WKT geometry column
                try:
                    from shapely import wkt
                    geometries = [wkt.loads(g) if isinstance(g, str) else Point() for g in df['geometry_wkt']]
                    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                except Exception as e:
                    print(f"Error creating from WKT: {e}")
                    # If WKT conversion fails, use lat/lon instead
                    gdf = gpd.GeoDataFrame(
                        df,
                        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
                        crs="EPSG:4326",
                    )
            elif 'geometry' in df.columns:
                # Handle possible GeoJSON string in 'geometry' column
                try:
                    import json
                    from shapely.geometry import shape
                    geometries = []
                    for g in df['geometry']:
                        if isinstance(g, str):
                            try:
                                geom_dict = json.loads(g)
                                geometries.append(shape(geom_dict))
                            except:
                                # If JSON parsing fails, use lat/lon
                                geometries.append(Point(df.loc[df['geometry'] == g, lon_col].iloc[0], 
                                                      df.loc[df['geometry'] == g, lat_col].iloc[0]))
                        else:
                            # If not a string, use lat/lon
                            idx = df['geometry'] == g
                            geometries.append(Point(df.loc[idx, lon_col].iloc[0], 
                                                  df.loc[idx, lat_col].iloc[0]))
                    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
                except Exception as e:
                    print(f"Error creating from geometry column: {e}")
                    # Fall back to lat/lon
                    gdf = gpd.GeoDataFrame(
                        df,
                        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
                        crs="EPSG:4326",
                    )
            else:
                # Create geometry from lat/lon
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
                    crs="EPSG:4326",
                )
        else:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            else:
                gdf = gdf.to_crs("EPSG:4326")
        
        # add a globally unique LOI id and source filename
        gdf["loi_uid"] = (
            os.path.basename(path).split(".")[0] + "_" + gdf.index.astype(str)
        )
        gdf["source_file"] = file_basename
        
        # Add a column to indicate geometry type
        gdf["geom_type"] = gdf.geometry.geom_type
        
        # For non-Point geometries, create representative points
        gdf["rep_point"] = gdf.geometry.apply(
            lambda g: g if g.geom_type == 'Point' else g.representative_point()
        )
        
        # Copy original file to locations_of_interest directory
        output_path = os.path.join(INPUT_LOI_DIR, file_basename)
        if not os.path.exists(output_path):
            if ext == ".shp":
                # For shapefiles, need to copy all associated files
                base_path = path[:-4]  # Remove .shp
                for related_file in glob.glob(f"{base_path}.*"):
                    related_ext = os.path.splitext(related_file)[1]
                    output_related = os.path.join(INPUT_LOI_DIR, f"{file_basename[:-4]}{related_ext}")
                    import shutil
                    shutil.copy2(related_file, output_related)
            else:
                import shutil
                shutil.copy2(path, output_path)
        
        return gdf
    
    except Exception as e:
        warnings.warn(f"Error loading {path}: {e}")
        return None


def detect_flow_columns(cols):
    """Return (total_flow_col, ab_col, ba_col)."""
    total_col = None
    ab_col = ba_col = None
    for c in cols:
        if isinstance(c, str):
            if "tot" in c.lower() and "trk" in c.lower() and "aadt" in c.lower():
                total_col = c
            if "_ab" in c.lower() and "aadt" in c.lower():
                ab_col = c
            if "_ba" in c.lower() and "aadt" in c.lower():
                ba_col = c
    return total_col, ab_col, ba_col


def build_rtree(gdf):
    """Spatial index on *projected* GeoSeries."""
    idx = index.Index()
    for i, geom in enumerate(gdf.geometry):
        if geom is not None:
            idx.insert(i, geom.bounds)
    return idx


def process_loi_file(loi_gdf, routes, rtree_idx, tot_col, ab_col, ba_col):
    """Process a single LOI file and return results."""
    records = []
    
    # Track LOI hits on routes
    route_loi_mapping = {idx: [] for idx in routes.index}
    
    for i, loi in loi_gdf.iterrows():
        # Use representative point for spatial queries
        geom = loi.geometry
        
        # Get representative point for non-Point geometries
        if not hasattr(geom, 'x'):
            # For LineString, Polygon, or other geometry types
            rep_point = loi.rep_point
            minx, miny, maxx, maxy = rep_point.bounds
        else:
            # For Point geometries
            minx = geom.x - RADIUS_M
            miny = geom.y - RADIUS_M
            maxx = geom.x + RADIUS_M
            maxy = geom.y + RADIUS_M
        
        # quick bbox query using bounds
        cand_idx = list(
            rtree_idx.intersection((minx, miny, maxx, maxy))
        )
        
        if not cand_idx:
            # write a no-match record
            # Convert to dict safely, handling any Series with geometry objects
            no_rec = {k: v for k, v in loi.drop(["geometry", "rep_point"], errors='ignore').items() 
                     if not isinstance(v, (gpd.GeoSeries, pd.Series))}
            no_rec["no_route_found"] = 1
            records.append(no_rec)
            continue

        # compute true distances
        segs = routes.iloc[cand_idx].copy()
        segs["dist_m"] = segs.geometry.distance(geom)  # Use original geometry for true distance

        segs = segs[segs["dist_m"] <= RADIUS_M].sort_values("dist_m")
        if segs.empty:
            # Convert to dict safely, handling any Series with geometry objects
            no_rec = {k: v for k, v in loi.drop(["geometry", "rep_point"], errors='ignore').items() 
                     if not isinstance(v, (gpd.GeoSeries, pd.Series))}
            no_rec["no_route_found"] = 1
            records.append(no_rec)
            continue

        # total local flow (direction-aware)
        if tot_col:
            segs["local_flow"] = segs[tot_col]
        else:
            # Handle potential missing columns
            if ab_col is not None and ba_col is not None and ab_col in segs.columns and ba_col in segs.columns:
                segs["local_flow"] = segs[[ab_col, ba_col]].sum(axis=1)
            elif ab_col is not None and ab_col in segs.columns:
                segs["local_flow"] = segs[ab_col]
            elif ba_col is not None and ba_col in segs.columns:
                segs["local_flow"] = segs[ba_col]
            else:
                segs["local_flow"] = 0

        total_local_flow = segs["local_flow"].sum()

        n_cand = len(segs)

        # collect per-candidate rows
        for rank, (seg_idx, seg_row) in enumerate(segs.iterrows(), start=1):
            # Convert to dict safely, handling any Series with geometry objects
            out_dict = {}
            for k, v in loi.drop(["geometry", "rep_point"], errors='ignore').items():
                if isinstance(v, (gpd.GeoSeries, pd.Series)) or hasattr(v, 'is_empty'):
                    continue
                out_dict[k] = v
            
            # mark this route as linked
            route_loi_mapping[seg_idx].append(out_dict["loi_uid"])

            out_dict.update(
                {
                    "cand_id": rank,
                    "n_candidates": n_cand,
                    "dist_m": float(seg_row["dist_m"]),
                    "multi_flag": int(n_cand > 1),
                    "route_seg_id": seg_idx,
                    "route_length_m": float(seg_row.get("length_m", seg_row.geometry.length)),
                }
            )
            
            # copy flow
            if tot_col and tot_col in seg_row:
                out_dict["flow_total"] = seg_row[tot_col]
                out_dict["flow_share"] = (
                    seg_row[tot_col] / total_local_flow if total_local_flow > 0 else np.nan
                )
            else:
                if ab_col and ab_col in seg_row:
                    out_dict["flow_ab"] = seg_row[ab_col]
                if ba_col and ba_col in seg_row:
                    out_dict["flow_ba"] = seg_row[ba_col]
                out_dict["flow_share"] = (
                    seg_row["local_flow"] / total_local_flow
                    if total_local_flow > 0
                    else np.nan
                )

            records.append(out_dict)
    
    # Create dataframe from records
    if records:
        results_df = pd.DataFrame(records)
        
        # Get geometry from original LOI file for the matched records
        if "loi_uid" in results_df.columns:
            # Create a mapping from loi_uid to original geometry
            uid_to_geom = {uid: geom for uid, geom in zip(loi_gdf["loi_uid"], loi_gdf.geometry)}
            
            # Get geometry for each record
            geometries = [uid_to_geom.get(uid) for uid in results_df["loi_uid"]]
            
            # Create GeoDataFrame
            results_gdf = gpd.GeoDataFrame(
                results_df, 
                geometry=geometries,
                crs=TARGET_EPSG
            )
        else:
            # If no loi_uid column, create an empty GeoDataFrame
            results_gdf = gpd.GeoDataFrame(
                results_df,
                geometry=[],
                crs=TARGET_EPSG
            )
    else:
        # Create empty GeoDataFrame with minimal structure
        results_gdf = gpd.GeoDataFrame(geometry=[], crs=TARGET_EPSG)
    
    return results_gdf, route_loi_mapping


# ---------------------  1. LOAD DATA  -------------------------------

def main():
    """Main execution function"""
    print("Loading route layer …")
    routes = gpd.read_file(ROUTE_FILE)
    if routes.crs is None:
        warnings.warn("Route layer CRS missing – assuming EPSG:4326")
        routes.set_crs("EPSG:4326", inplace=True)
    routes = routes.to_crs(TARGET_EPSG)

    # detect flow cols
    tot_col, ab_col, ba_col = detect_flow_columns(routes.columns)
    if not any([tot_col, ab_col]):
        raise ValueError("No truck-flow column detected in route layer!")

    print(f"Flow columns ⇒ total={tot_col}, AB={ab_col}, BA={ba_col}")

    # Initialize route LOI mapping 
    route_loi_ids = {idx: [] for idx in routes.index}

    # Build the spatial index for routes
    rtree_idx = build_rtree(routes)

    # process each LOI file separately
    all_results = []

    for pat in LOI_SOURCES:
        for f in glob.glob(pat):
            print(f"Processing {f}...")
            loi_gdf = load_loi_file(f)
            
            if loi_gdf is None or len(loi_gdf) == 0:
                print(f"  Skipping {f} - no valid data found.")
                continue
                
            # Project to target CRS
            loi_gdf = loi_gdf.to_crs(TARGET_EPSG)
            
            # Process this LOI file
            try:
                result_gdf, file_route_mapping = process_loi_file(
                    loi_gdf, routes, rtree_idx, tot_col, ab_col, ba_col
                )
                
                # Update master route mapping
                for route_idx, loi_list in file_route_mapping.items():
                    route_loi_ids[route_idx].extend(loi_list)
                
                # Save individual result
                if len(result_gdf) > 0:
                    file_basename = os.path.basename(f)
                    output_name = f"{os.path.splitext(file_basename)[0]}_matched.geojson"
                    output_path = os.path.join(RESULTS_DIR, output_name)
                    result_gdf.to_file(output_path, driver="GeoJSON")
                    print(f"  {file_basename} results saved to {output_path}")
                    
                    # Add to all results for combined file
                    all_results.append(result_gdf)
                else:
                    print(f"  No matching results found for {f}")
            except Exception as e:
                print(f"  Error processing {f}: {e}")

    # Create combined results file if any results exist
    if all_results:
        try:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_gdf = gpd.GeoDataFrame(combined_results, geometry='geometry', crs=TARGET_EPSG)
            all_results_path = os.path.join(BASE_OUTPUT_DIR, "all_merged_loi_routes.geojson")
            combined_gdf.to_file(all_results_path, driver="GeoJSON")
            print(f" All combined LOI-route results written → {all_results_path}")
        except Exception as e:
            print(f" Error creating combined results file: {e}")

    # Update routes with LOI hits and save
    routes["loi_ids"] = [json.dumps(route_loi_ids.get(idx, [])) for idx in routes.index]
    routes_out = os.path.join(BASE_OUTPUT_DIR, "routes_with_loi_hits.gpkg")
    routes.to_file(routes_out, layer="routes", driver="GPKG")
    print(f" Routes with LOI hits written → {routes_out}")

    print("Done.")


if __name__ == "__main__":
    main()