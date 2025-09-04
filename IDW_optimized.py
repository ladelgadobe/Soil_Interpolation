import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from sklearn.model_selection import KFold
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


def optimize_idw(x, y, z, k=5):
    # Verificar que hay suficientes datos
    if len(x) < k:
        raise ValueError(f"Not enough data points ({len(x)}) for {k}-fold cross-validation.")

    p_values = np.arange(0.5, 6.5, 0.5)
    n_values = np.arange(4, 17, 1)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    # Step 1: Precompute all ISI components
    all_mae = []
    all_sae = []

    for p in p_values:
        for n in n_values:
            mae_scores = []
            sae_scores = []
            for train_index, test_index in kf.split(x):
                # Verificar que los índices no excedan los límites
                if max(train_index) >= len(z) or max(test_index) >= len(z):
                    raise IndexError(f"Index out of bounds: train_index or test_index exceeds data size.")

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

    # Step 2: Calculate global min/max values
    max_abs_ae = max(all_mae)
    min_sae = min(all_sae)
    max_sae = max(all_sae)

    # Step 3: Recalculate ISI and find the best parameters
    best_p, best_n, best_isi = None, None, float('inf')
    final_results = []

    for (p, n, mae, sae) in results:
        isi = calculate_isi(mae, sae, max_abs_ae, min_sae, max_sae)
        final_results.append((p, n, mae, sae, isi))
        if isi < best_isi:
            best_p, best_n, best_isi = p, n, isi

    return best_p, best_n, best_isi, final_results
