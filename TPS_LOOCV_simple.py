import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import Rbf
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
import rasterio


# Function for TPS Interpolation with epsilon parameter
def tps_interpolation(x, y, z, xi, yi, epsilon):
    rbf = Rbf(x, y, z, function='thin_plate', epsilon=epsilon)
    zi = rbf(xi, yi)
    return zi


# Functions for evaluation metrics
def mean_absolute_error(ypred, yobs):
    return np.mean(np.abs(ypred - yobs))


def std_error(ypred, yobs):
    return np.std(ypred - yobs)


def calculate_isi(mae, sae, max_abs_ae, min_sae, max_sae):
    normalized_mae = mae / max_abs_ae
    normalized_sae = (sae - min_sae) / (max_sae - min_sae)
    return normalized_mae + normalized_sae


def optimize_epsilon(x, y, z, epsilons, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    all_mae, all_sae = [], []

    for epsilon in epsilons:
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            z_train, z_test = z[train_index], z[test_index]

            z_pred = tps_interpolation(x_train, y_train, z_train, x_test, y_test, epsilon)
            all_mae.append(mean_absolute_error(z_pred, z_test))
            all_sae.append(std_error(z_pred, z_test))

    max_abs_ae = max(all_mae)
    min_sae = min(all_sae)
    max_sae = max(all_sae)

    best_isi, best_epsilon = float('inf'), None
    for epsilon in epsilons:
        avg_mae = np.mean(all_mae)
        avg_sae = np.mean(all_sae)
        isi = calculate_isi(avg_mae, avg_sae, max_abs_ae, min_sae, max_sae)
        if isi < best_isi:
            best_isi, best_epsilon = isi, epsilon

    return best_epsilon, best_isi


def concordance_correlation_coefficient(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    std_true, std_pred = np.std(y_true), np.std(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * cov) / (std_true ** 2 + std_pred ** 2 + (mean_true - mean_pred) ** 2)


# Load data
file_path = input("Enter the path to the CSV file: ")
data = pd.read_csv(file_path)
variable = input("Enter the name of the column containing the variable of interest: ")
data = data.dropna(subset=[variable])

x, y, z = data['x'].values, data['y'].values, data[variable].values

epsilons = np.arange(0.0001, 0.5001, 0.0001)
best_epsilon, best_isi = optimize_epsilon(x, y, z, epsilons)
print(f"\nOptimized epsilon: {best_epsilon:.5f} with ISI: {best_isi:.5f}")

# Load shapefile and define grid
shapefile_path = input("Enter the path to the shapefile for interpolation boundary: ")
polygon = gpd.read_file(shapefile_path)
pixel_size = float(input("Enter the pixel size for interpolation: "))

minx, miny, maxx, maxy = polygon.total_bounds
x_grid, y_grid = np.arange(minx, maxx, pixel_size), np.arange(miny, maxy, pixel_size)
xi, yi = np.meshgrid(x_grid, y_grid)

grid_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xi.flatten(), yi.flatten()), crs=polygon.crs)
grid_within = grid_points[grid_points.geometry.within(polygon.geometry.unary_union)]
xi_within, yi_within = grid_within.geometry.x.values, grid_within.geometry.y.values

# LOOCV Validation
y_true, y_pred = [], []
loo = LeaveOneOut()
for train_index, test_index in loo.split(x):
    x_train, y_train, z_train = x[train_index], y[train_index], z[train_index]
    x_test, y_test, z_test = x[test_index], y[test_index], z[test_index]
    z_pred = tps_interpolation(x_train, y_train, z_train, x_test, y_test, best_epsilon)
    y_true.append(z_test[0])
    y_pred.append(z_pred[0])

rmse, r2, lccc = np.sqrt(mean_squared_error(y_true, y_pred)), r2_score(y_true,
                                                                       y_pred), concordance_correlation_coefficient(
    y_true, y_pred)
print("\nLOOCV Evaluation:")
print(f"RMSE: {rmse:.5f}")
print(f"R²: {r2:.5f}")
print(f"LCCC: {lccc:.5f}")

# Interpolation
zi = tps_interpolation(x, y, z, xi_within, yi_within, best_epsilon)

# Export raster
output_file = input("Enter the path to save the output raster (GeoTIFF format): ")
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
        row, col = int((maxy - y_val) / pixel_size), int((x_val - minx) / pixel_size)
        interpolated_grid[row, col] = z_val
    dst.write(interpolated_grid, 1)

print(f"Interpolation complete. Raster saved to {output_file}")
