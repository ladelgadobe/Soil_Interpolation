import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import Rbf
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import rasterio
from rasterio.transform import from_origin

# ----------------------------
# Utilities
# ----------------------------
# All comments are in English as requested

def tps_interpolation(x, y, z, xi, yi, epsilon):
    """Thin Plate Spline interpolation using scipy's Rbf with epsilon."""
    rbf = Rbf(x, y, z, function='thin_plate', epsilon=epsilon)
    return rbf(xi, yi)

def mean_absolute_error(ypred, yobs):
    """Mean Absolute Error."""
    return np.mean(np.abs(ypred - yobs))

def std_error(ypred, yobs):
    """Standard deviation of errors (residual dispersion)."""
    return np.std(ypred - yobs)

def concordance_correlation_coefficient(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient (LCCC)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    std_true, std_pred = np.std(y_true), np.std(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = std_true**2 + std_pred**2 + (mean_true - mean_pred)**2
    if denom == 0:
        return np.nan
    return (2 * cov) / denom

def calculate_isi(mae, sae, max_abs_ae, min_sae, max_sae):
    """ISI = normalized MAE + normalized SAE (lower is better)."""
    # Protect against zero division
    if max_abs_ae == 0:
        normalized_mae = 0.0
    else:
        normalized_mae = mae / max_abs_ae

    if (max_sae - min_sae) == 0:
        normalized_sae = 0.0
    else:
        normalized_sae = (sae - min_sae) / (max_sae - min_sae)

    return normalized_mae + normalized_sae

def optimize_epsilon(x, y, z, epsilons, k=5, random_state=42):
    """
    Optimize epsilon via K-Fold CV minimizing ISI.
    ISI uses global normalization across the tested epsilons.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # First pass: collect per-epsilon CV MAE/SAE
    epsilon_stats = []
    all_mae_vals = []
    all_sae_vals = []

    for epsilon in epsilons:
        fold_mae = []
        fold_sae = []
        for train_index, test_index in kf.split(x):
            x_tr, x_te = x[train_index], x[test_index]
            y_tr, y_te = y[train_index], y[test_index]
            z_tr, z_te = z[train_index], z[test_index]

            z_pred = tps_interpolation(x_tr, y_tr, z_tr, x_te, y_te, epsilon)
            fold_mae.append(mean_absolute_error(z_pred, z_te))
            fold_sae.append(std_error(z_pred, z_te))

        avg_mae = float(np.mean(fold_mae))
        avg_sae = float(np.mean(fold_sae))
        epsilon_stats.append({"epsilon": epsilon, "mae": avg_mae, "sae": avg_sae})
        all_mae_vals.append(avg_mae)
        all_sae_vals.append(avg_sae)

    # Global stats for normalization
    max_abs_ae = max(all_mae_vals) if len(all_mae_vals) else 1.0
    min_sae = min(all_sae_vals) if len(all_sae_vals) else 0.0
    max_sae = max(all_sae_vals) if len(all_sae_vals) else 1.0

    # Second pass: compute ISI and select best epsilon
    best_isi = float('inf')
    best_epsilon = None
    for rec in epsilon_stats:
        isi = calculate_isi(rec["mae"], rec["sae"], max_abs_ae, min_sae, max_sae)
        rec["isi"] = isi
        if isi < best_isi:
            best_isi = isi
            best_epsilon = rec["epsilon"]

    return best_epsilon, best_isi

def write_geotiff(output_path, polygon, pixel_size, xi_within, yi_within, zi):
    """Create a full grid GeoTIFF and fill predicted values at within-polygon cells."""
    minx, miny, maxx, maxy = polygon.total_bounds
    x_grid = np.arange(minx, maxx, pixel_size)
    y_grid = np.arange(miny, maxy, pixel_size)
    height, width = len(y_grid), len(x_grid)

    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    interpolated_grid = np.full((height, width), np.nan, dtype="float32")

    # Map point coords to array indices and fill
    for x_val, y_val, z_val in zip(xi_within, yi_within, zi):
        col = int((x_val - minx) / pixel_size)
        row = int((maxy - y_val) / pixel_size)
        if 0 <= row < height and 0 <= col < width:
            interpolated_grid[row, col] = z_val

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=polygon.crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(interpolated_grid, 1)

# ----------------------------
# Main interactive flow
# ----------------------------
def main():
    # --- Inputs ---
    csv_path = input("Enter the path to the CSV file: ").strip()
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read CSV with semicolon separator; try both decimal '.' and ',' if needed
    try:
        data = pd.read_csv(csv_path, sep=';')
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV with ';' separator: {e}")

    # Standard columns 'x','y' are required
    required = {'x', 'y'}
    if not required.issubset(set(map(str.lower, data.columns))):
        # Try to locate case-insensitively
        cols_lower = {c.lower(): c for c in data.columns}
        try:
            data.rename(columns={cols_lower['x']: 'x', cols_lower['y']: 'y'}, inplace=True)
        except Exception:
            raise ValueError("The CSV must contain 'x' and 'y' columns (case-insensitive).")

    # Coerce x,y to numeric
    data['x'] = pd.to_numeric(data['x'], errors='coerce')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')

    # Offer variable selection: comma-separated names or ALL
    print("\nAvailable columns:")
    print(", ".join([c for c in data.columns if c not in ['x','y']]))
    vars_input = input("\nEnter variable names separated by commas, or type ALL for all numeric columns (excluding x,y): ").strip()

    if vars_input.upper() == "ALL":
        candidate_cols = [c for c in data.columns if c not in ['x', 'y']]
        # Keep only numeric after coercion
        for c in candidate_cols:
            data[c] = pd.to_numeric(data[c].astype(str).str.replace(',', '.'), errors='coerce')
        variables = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(data[c])]
    else:
        variables = [v.strip() for v in vars_input.split(',') if v.strip()]
        # Coerce selected columns to numeric (handles decimal commas)
        for v in variables:
            if v not in data.columns:
                raise ValueError(f"Column '{v}' not found in CSV.")
            data[v] = pd.to_numeric(data[v].astype(str).str.replace(',', '.'), errors='coerce')

    if len(variables) == 0:
        raise ValueError("No variables selected. Aborting.")

    # Epsilon configuration
    try:
        eps_start = float(input("Epsilon start (default 0.0001): ") or "0.0001")
        eps_end   = float(input("Epsilon end (default 0.5): ") or "0.5")
        eps_step  = float(input("Epsilon step (default 0.001): ") or "0.001")
    except Exception:
        eps_start, eps_end, eps_step = 0.0001, 0.5, 0.001
    epsilons = np.arange(eps_start, eps_end + eps_step, eps_step)

    # CV config
    try:
        kfold_k = int(input("K for KFold optimization (default 5): ") or "5")
    except Exception:
        kfold_k = 5

    # Shapefile and pixel size
    shp_path = input("Enter the path to the shapefile for interpolation boundary: ").strip()
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    polygon = gpd.read_file(shp_path)
    if polygon.empty:
        raise ValueError("Shapefile has no geometries.")
    if polygon.crs is None:
        raise ValueError("Shapefile has no CRS. Please set a valid CRS.")

    try:
        pixel_size = float(input("Enter the pixel size for interpolation (e.g., 5): ").strip())
    except Exception:
        raise ValueError("Invalid pixel size.")

    # Output directory
    out_dir = input("Enter the output directory for rasters and metrics CSV: ").strip()
    os.makedirs(out_dir, exist_ok=True)

    # Prepare grid points within polygon
    minx, miny, maxx, maxy = polygon.total_bounds
    x_grid = np.arange(minx, maxx, pixel_size)
    y_grid = np.arange(miny, maxy, pixel_size)
    xi_mesh, yi_mesh = np.meshgrid(x_grid, y_grid)

    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xi_mesh.flatten(), yi_mesh.flatten()),
        crs=polygon.crs
    )
    union_poly = polygon.geometry.unary_union
    grid_within = grid_points[grid_points.geometry.within(union_poly)]
    xi_within = grid_within.geometry.x.values
    yi_within = grid_within.geometry.y.values

    # Metrics accumulator
    results = []

    # Main loop over variables
    for var in variables:
        print(f"\nProcessing variable: {var}")
        dfv = data[['x', 'y', var]].dropna().copy()
        if dfv.empty:
            print(f"Variable '{var}' has no valid data (after coercion). Skipping.")
            continue

        x = dfv['x'].values
        y = dfv['y'].values
        z = dfv[var].values

        # Optimize epsilon via K-Fold + ISI
        best_epsilon, best_isi = optimize_epsilon(x, y, z, epsilons, k=kfold_k)
        print(f"Optimized epsilon for {var}: {best_epsilon:.6f} | ISI: {best_isi:.6f}")

        # LOOCV
        loo = LeaveOneOut()
        y_true, y_pred = [], []
        for train_index, test_index in loo.split(x):
            x_tr, y_tr, z_tr = x[train_index], y[train_index], z[train_index]
            x_te, y_te, z_te = x[test_index], y[test_index], z[test_index]
            pred = tps_interpolation(x_tr, y_tr, z_tr, x_te, y_te, best_epsilon)
            y_true.append(z_te[0])
            y_pred.append(pred[0])

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        lccc = float(concordance_correlation_coefficient(y_true, y_pred))
        mae  = float(mean_absolute_error(np.array(y_pred), np.array(y_true)))
        sae  = float(std_error(np.array(y_pred), np.array(y_true)))

        # Save metrics to collector
        results.append({
            "variable": var,
            "n_points": len(z),
            "best_epsilon": best_epsilon,
            "ISI_opt": best_isi,
            "RMSE_LOOCV": rmse,
            "R2_LOOCV": r2,
            "LCCC_LOOCV": lccc,
            "MAE_LOOCV": mae,
            "SAE_LOOCV": sae
        })

        # Interpolate on grid within polygon
        zi = tps_interpolation(x, y, z, xi_within, yi_within, best_epsilon)

        # Export raster
        out_raster = os.path.join(out_dir, f"TPS_{var}.tif")
        write_geotiff(out_raster, polygon, pixel_size, xi_within, yi_within, zi)
        print(f"Raster saved: {out_raster}")

    # Export metrics CSV
    if results:
        metrics_df = pd.DataFrame(results)
        metrics_csv = os.path.join(out_dir, "TPS_metrics_summary.csv")
        metrics_df.to_csv(metrics_csv, sep=';', index=False)
        print(f"\nMetrics summary saved: {metrics_csv}")
    else:
        print("\nNo variables were processed. Please check your inputs/data.")

if __name__ == "__main__":
    main()
