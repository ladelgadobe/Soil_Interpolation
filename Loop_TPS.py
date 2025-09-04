import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import Rbf
from sklearn.model_selection import KFold
from rasterio.transform import from_origin
import rasterio

# =========================
# TPS interpolation (robust)
# =========================
# Ensures numeric dtype to avoid SciPy "object arrays are not supported".
def tps_interpolation(x, y, z, xi, yi, epsilon):
    # Force numeric float arrays
    x  = np.asarray(x,  dtype=float)
    y  = np.asarray(y,  dtype=float)
    z  = np.asarray(z,  dtype=float)
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)
    # Thin-plate radial basis function
    rbf = Rbf(x, y, z, function='thin_plate', epsilon=float(epsilon))
    zi = rbf(xi, yi)
    return zi

# =========================
# Metrics utilities
# =========================
def mean_absolute_error(ypred, yobs):
    return float(np.mean(np.abs(np.asarray(ypred) - np.asarray(yobs))))

def std_error(ypred, yobs):
    return float(np.std(np.asarray(ypred) - np.asarray(yobs)))

def calculate_isi(mae, sae, max_abs_ae, min_sae, max_sae):
    # Normalize MAE and SAE into [0,1] ranges and combine
    normalized_mae = mae / max_abs_ae if max_abs_ae > 0 else 0.0
    denom = (max_sae - min_sae)
    normalized_sae = (sae - min_sae) / denom if denom > 0 else 0.0
    return float(normalized_mae + normalized_sae)

def concordance_cc(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient (CCC), scalar output.
    Returns NaN if not computable (e.g., <2 valid pairs)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size < 2:
        return np.nan
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(ddof=1), yp.var(ddof=1)
    # Both constant
    if vt == 0 and vp == 0:
        return 1.0 if np.allclose(yt, yp) else np.nan
    cov = np.cov(yt, yp, ddof=1)[0, 1]
    denom = vt + vp + (mt - mp) ** 2
    if denom == 0:
        return np.nan
    return float((2.0 * cov) / denom)

# =========================
# Hyperparameter optimization (epsilon)
# =========================
def optimize_epsilon(x, y, z, epsilons, k=5):
    """Randomized K-fold CV over epsilon for TPS with ISI objective."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    n = z.size
    k_eff = int(min(max(k, 2), n))  # ensure 2 <= k <= n
    if k_eff < 2:
        raise ValueError("Not enough samples to run CV (need >= 2).")

    kf = KFold(n_splits=k_eff, shuffle=True, random_state=42)
    per_eps = []  # list of (epsilon, avg_mae, avg_sae)

    for epsilon in epsilons:
        mae_scores, sae_scores = [], []
        for tr, te in kf.split(x):
            x_tr, y_tr, z_tr = x[tr], y[tr], z[tr]
            x_te, y_te, z_te = x[te], y[te], z[te]
            z_pred = tps_interpolation(x_tr, y_tr, z_tr, x_te, y_te, epsilon)
            mae_scores.append(mean_absolute_error(z_pred, z_te))
            sae_scores.append(std_error(z_pred, z_te))
        per_eps.append((epsilon, float(np.mean(mae_scores)), float(np.mean(sae_scores))))

    # Global extrema for ISI normalization
    max_abs_ae = max(m for _, m, _ in per_eps)
    min_sae    = min(s for _, _, s in per_eps)
    max_sae    = max(s for _, _, s in per_eps)

    # Select epsilon with minimum ISI
    best_epsilon, best_isi = None, float('inf')
    for epsilon, mae, sae in per_eps:
        isi = calculate_isi(mae, sae, max_abs_ae, min_sae, max_sae)
        if isi < best_isi:
            best_isi, best_epsilon = isi, epsilon

    return best_epsilon, best_isi

# =========================
# LOOCV for TPS (fixed epsilon)
# =========================
def loocv_tps(x, y, z, epsilon):
    """Leave-One-Out CV for TPS with fixed epsilon.
    Returns (rmse, ccc, preds)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = z.size
    preds = np.full(n, np.nan, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        # Predict only the left-out point i
        z_hat = tps_interpolation(x[mask], y[mask], z[mask], x[~mask], y[~mask], epsilon)
        preds[i] = float(z_hat[0]) if np.ndim(z_hat) else float(z_hat)

    valid = np.isfinite(preds) & np.isfinite(z)
    if valid.sum() < 2:
        return np.nan, np.nan, preds
    rmse = float(np.sqrt(np.mean((preds[valid] - z[valid]) ** 2)))
    ccc  = concordance_cc(z[valid], preds[valid])
    return rmse, ccc, preds

# =========================
# Main script
# =========================
file_path = input("Enter the path to the CSV file: ")
shapefile_path = input("Enter the path to the shapefile: ")
output_folder = input("Enter the folder to save the output rasters: ")
os.makedirs(output_folder, exist_ok=True)

pixel_size = float(input("Enter the pixel size for interpolation (e.g., 10 for 10x10 meters): "))
parameter_file = os.path.join(output_folder, "optimized_parameters.csv")
metrics_file   = os.path.join(output_folder, "tps_loocv_metrics.csv")  # new

# Read data
data = pd.read_csv(file_path, delimiter=";")
polygon = gpd.read_file(shapefile_path)

# Optional: coerce all columns to numeric where possible (non-numeric -> NaN)
for c in data.columns:
    data[c] = pd.to_numeric(data[c], errors='ignore')  # keep strings for non-numeric

# Coordinates (force numeric)
data['x'] = pd.to_numeric(data['x'], errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
x_all = data['x'].values
y_all = data['y'].values

# Define grid over polygon bounds
minx, miny, maxx, maxy = polygon.total_bounds
x_grid = np.arange(minx, maxx, pixel_size)
y_grid = np.arange(miny, maxy, pixel_size)
xi, yi = np.meshgrid(x_grid, y_grid)

# Build grid points GeoDataFrame and filter inside polygon
grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xi.flatten(), yi.flatten()), crs=polygon.crs)
# Union geometry (compatibility across GeoPandas versions)
try:
    union_geom = polygon.geometry.union_all()
except Exception:
    union_geom = polygon.geometry.unary_union

grid_within = grid_points[grid_points.geometry.within(union_geom)]
xi_within = grid_within.geometry.x.values
yi_within = grid_within.geometry.y.values

# Epsilon search space
epsilons = np.arange(0.0001, 0.5001, 0.0001)

parameters = []   # to save best epsilon per variable
metrics    = []   # to save LOOCV RMSE/LCCC per variable

# Process each variable (excluding coordinates)
for column in data.columns:
    if column in ['x', 'y']:
        continue

    print(f"Processing variable: {column}")

    # Build clean numeric vectors for z and matching x,y
    z_series = pd.to_numeric(data[column], errors='coerce').dropna()
    valid_idx = z_series.index
    z = z_series.values.astype(float)

    x_valid = pd.to_numeric(data.loc[valid_idx, 'x'], errors='coerce').values.astype(float)
    y_valid = pd.to_numeric(data.loc[valid_idx, 'y'], errors='coerce').values.astype(float)

    # Guard against too few points
    if z.size < 2:
        print(f"  Skipped '{column}': not enough valid points (< 2).")
        continue

    # Optimize epsilon (KFold with k up to 5, but <= n)
    k_cv = min(5, z.size)
    try:
        best_epsilon, best_isi = optimize_epsilon(x_valid, y_valid, z, epsilons, k=k_cv)
    except Exception as e:
        print(f"  Optimization failed for '{column}': {e}")
        continue

    parameters.append({'Variable': column, 'Best Epsilon': best_epsilon, 'Best ISI': best_isi})
    print(f"  Best epsilon: {best_epsilon:.4f} | ISI: {best_isi:.4f}")

    # Interpolate to grid within polygon
    zi = tps_interpolation(x_valid, y_valid, z, xi_within, yi_within, best_epsilon)

    # Save raster
    output_raster = os.path.join(output_folder, f"{column}_TPS.tif")
    transform = from_origin(minx, maxy, pixel_size, pixel_size)
    grid_shape = (len(y_grid), len(x_grid))

    with rasterio.open(
        output_raster,
        'w',
        driver='GTiff',
        height=grid_shape[0],
        width=grid_shape[1],
        count=1,
        dtype='float32',
        crs=polygon.crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        interpolated_grid = np.full(grid_shape, np.nan, dtype='float32')
        for x_val, y_val, z_val in zip(xi_within, yi_within, zi):
            row = int((maxy - y_val) / pixel_size)
            col = int((x_val - minx) / pixel_size)
            if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
                interpolated_grid[row, col] = z_val
        dst.write(interpolated_grid, 1)

    print(f"  Raster saved: {output_raster}")

    # LOOCV evaluation (RMSE & Lin's CCC)
    loocv_rmse, loocv_ccc, _ = loocv_tps(x_valid, y_valid, z, best_epsilon)
    print(f"  LOOCV TPS → RMSE: {loocv_rmse:.4f} | LCCC: {loocv_ccc:.4f}")

    metrics.append({
        'Variable': column,
        'LOOCV_RMSE': loocv_rmse,
        'LOOCV_LCCC': loocv_ccc
    })

# Save optimized parameters and LOOCV metrics
if parameters:
    pd.DataFrame(parameters).to_csv(parameter_file, index=False)
    print(f"Optimized parameters saved to: {parameter_file}")
else:
    print("No parameters to save (no variables processed).")

if metrics:
    pd.DataFrame(metrics).to_csv(metrics_file, index=False)
    print(f"LOOCV metrics saved to: {metrics_file}")
else:
    print("No LOOCV metrics to save (no variables processed).")
