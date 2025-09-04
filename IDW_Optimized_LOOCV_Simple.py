import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from sklearn.model_selection import KFold
from scipy.spatial import distance_matrix


# ----------------- IDW core functions -----------------
def idw_interpolation(x, y, z, xi, yi, p, n):
    dist = distance_matrix(np.c_[xi, yi], np.c_[x, y])
    dist[dist == 0] = 1e-10  # Avoid division by zero
    sorted_indices = np.argsort(dist, axis=1)[:, :n]  # Select n neighbors
    dist = np.take_along_axis(dist, sorted_indices, axis=1)
    weights = 1 / dist ** p
    weights /= weights.sum(axis=1)[:, np.newaxis]
    z_neighbors = np.take_along_axis(z[None, :], sorted_indices, axis=1)
    zi = np.sum(weights * z_neighbors, axis=1)
    return zi


def mean_absolute_error(ypred, yobs):
    return np.mean(np.abs(ypred - yobs))


def std_error(ypred, yobs):
    return np.std(ypred - yobs)


def calculate_isi(mae, sae, max_abs_ae, min_sae, max_sae):
    normalized_mae = mae / max_abs_ae
    normalized_sae = (sae - min_sae) / (max_sae - min_sae)
    return normalized_mae + normalized_sae


def optimize_idw(x, y, z, k=5):
    p_values = np.arange(0.5, 6.5, 0.5)
    n_values = np.arange(4, 17, 1)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    all_mae = []
    all_sae = []

    for p in p_values:
        for n in n_values:
            mae_scores = []
            sae_scores = []
            for train_index, test_index in kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                z_train, z_test = z[train_index], z[test_index]

                z_pred = idw_interpolation(x_train, y_train, z_train, x_test, y_test, p, n)
                mae_scores.append(mean_absolute_error(z_pred, z_test))
                sae_scores.append(std_error(z_pred, z_test))

            avg_mae = np.mean(mae_scores)
            avg_sae = np.mean(sae_scores)
            all_mae.append(avg_mae)
            all_sae.append(avg_sae)
            results.append((p, n, avg_mae, avg_sae))

    max_abs_ae = max(all_mae)
    min_sae = min(all_sae)
    max_sae = max(all_sae)

    best_p, best_n, best_isi = None, None, float('inf')
    final_results = []

    for (p, n, mae, sae) in results:
        isi = calculate_isi(mae, sae, max_abs_ae, min_sae, max_sae)
        final_results.append((p, n, mae, sae, isi))
        if isi < best_isi:
            best_p, best_n, best_isi = p, n, isi

    return best_p, best_n, best_isi, final_results


def concordance_cc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if yt.size < 2:
        return np.nan
    mt, mp = yt.mean(), yp.mean()
    vt, vp = yt.var(ddof=1), yp.var(ddof=1)
    if vt == 0 and vp == 0:
        return 1.0 if np.allclose(yt, yp) else np.nan
    cov = np.cov(yt, yp, ddof=1)[0, 1]
    denom = vt + vp + (mt - mp) ** 2
    if denom == 0:
        return np.nan
    return float((2.0 * cov) / denom)


def loocv_idw(x, y, z, p, n):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n_pts = z.size
    preds = np.full(n_pts, np.nan, dtype=float)

    for i in range(n_pts):
        mask = np.ones(n_pts, dtype=bool)
        mask[i] = False
        n_eff = int(min(n, mask.sum()))
        z_hat = idw_interpolation(x[mask], y[mask], z[mask], x[~mask], y[~mask], p, n_eff)
        preds[i] = float(z_hat[0]) if np.ndim(z_hat) else float(z_hat)

    valid = np.isfinite(preds) & np.isfinite(z)
    if valid.sum() < 2:
        return np.nan, np.nan, preds
    rmse = float(np.sqrt(np.mean((preds[valid] - z[valid]) ** 2)))
    ccc = concordance_cc(z[valid], preds[valid])
    return rmse, ccc, preds


# ----------------- Main script -----------------
file_path = input("Enter the path to the CSV file: ")
data = pd.read_csv(file_path, delimiter=";")

shapefile_path = input("Enter the path to the shapefile: ")
polygon = gpd.read_file(shapefile_path)
pixel_size = float(input("Enter the pixel size for interpolation: "))
output_folder = input("Enter the output folder for rasters: ")
os.makedirs(output_folder, exist_ok=True)

# Ask user which variable to interpolate
print("\nAvailable columns in CSV:")
print(list(data.columns))
variable = input("\nEnter the name of the variable to interpolate: ").strip()

# Prepare data
data_var = data.dropna(subset=[variable])
x = data_var['x'].values
y = data_var['y'].values
z = data_var[variable].values

# Grid
minx, miny, maxx, maxy = polygon.total_bounds
x_grid = np.arange(minx, maxx, pixel_size)
y_grid = np.arange(miny, maxy, pixel_size)
xi, yi = np.meshgrid(x_grid, y_grid)

grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xi.flatten(), yi.flatten()), crs=polygon.crs)
grid_within = grid_points[grid_points.geometry.within(polygon.geometry.unary_union)]
xi_within = grid_within.geometry.x.values
yi_within = grid_within.geometry.y.values

# Optimize IDW
best_p, best_n, best_isi, results = optimize_idw(x, y, z)

# LOOCV
loocv_rmse, loocv_ccc, _ = loocv_idw(x, y, z, best_p, best_n)
print(f"\nLOOCV → RMSE: {loocv_rmse:.4f} | LCCC: {loocv_ccc:.4f}")

# Interpolate
zi = idw_interpolation(x, y, z, xi_within, yi_within, best_p, best_n)

# Save raster
output_file = os.path.join(output_folder, f"{variable}_IDW.tif")
transform = from_origin(minx, maxy, pixel_size, pixel_size)
grid_shape = (len(y_grid), len(x_grid))

with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=grid_shape[0],
        width=grid_shape[1],
        count=1,
        dtype='float32',
        crs=polygon.crs,
        transform=transform,
) as dst:
    interpolated_grid = np.full(grid_shape, np.nan)
    for i, (x_val, y_val, z_val) in enumerate(zip(xi_within, yi_within, zi)):
        row = int((maxy - y_val) / pixel_size)
        col = int((x_val - minx) / pixel_size)
        interpolated_grid[row, col] = z_val
    dst.write(interpolated_grid, 1)

print(f"\nRaster saved: {output_file}")

# Save results
results_df = pd.DataFrame([(variable, best_p, best_n, best_isi)],
                          columns=['Variable', 'Best_p', 'Best_n', 'Best_ISI'])
results_df.to_csv(os.path.join(output_folder, "IDW_optimization_results.csv"), index=False)

metrics_df = pd.DataFrame([(variable, loocv_rmse, loocv_ccc)],
                          columns=['Variable', 'LOOCV_RMSE', 'LOOCV_LCCC'])
metrics_df.to_csv(os.path.join(output_folder, "IDW_LOOCV_metrics.csv"), index=False)
