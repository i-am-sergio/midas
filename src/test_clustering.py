import pandas as pd
import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# --- CONFIGURACIÓN ---
FUSION_MATRIX_FILE = 'jpetstore_results/jpetstore_fase3_fusion_matrix.csv'
K_CLUSTERS = 6  # Tu K fijo para la prueba
# ---------------------

print(f"--- Probando Clustering Espectral (K={K_CLUSTERS}) ---")

# 1. Cargar la Matriz Fusionada
try:
    df = pd.read_csv(FUSION_MATRIX_FILE, index_col=0)
    classes = df.index.tolist()
    affinity_matrix = df.values
    print(f"Matriz cargada: {affinity_matrix.shape}")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo {FUSION_MATRIX_FILE}")
    exit()

# 2. Ejecutar Spectral Clustering
sc = SpectralClustering(
    n_clusters=K_CLUSTERS,
    affinity='precomputed', # Importante: la matriz ya es de afinidad/similitud
    random_state=42,
    assign_labels='kmeans'
)

labels = sc.fit_predict(affinity_matrix)

# 3. Calcular Métricas
# Para Silhouette usamos Distancia (1 - Similitud)
distance_matrix = 1 - affinity_matrix
np.fill_diagonal(distance_matrix, 0) # Asegurar diagonal 0 para métrica

sil_score = silhouette_score(distance_matrix, labels, metric='precomputed')

# Para Calinski-Harabasz usamos la matriz de Similitud como características
ch_score = calinski_harabasz_score(affinity_matrix, labels)

print(f"\n--- Métricas de Calidad ---")
print(f"Silhouette Score: {sil_score:.4f} (Mayor es mejor, [-1, 1])")
print(f"Calinski-Harabasz: {ch_score:.4f} (Mayor es mejor)")

# 4. Organizar y Mostrar Resultados
clusters_result = {}
for idx, label in enumerate(labels):
    key = f"cluster_{label}"
    if key not in clusters_result:
        clusters_result[key] = []
    clusters_result[key].append(classes[idx])

# Ordenar para mejor visualización
sorted_clusters = dict(sorted(clusters_result.items()))

print(f"\n--- Resultados del Clustering (JSON) ---")
print(json.dumps(sorted_clusters, indent=2))