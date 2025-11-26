#!/usr/bin/env python3

"""
Aplica Clustering Espectral para un rango de K (de 2 a N-1).

Calcula:
  1. Coeficiente de Silhouette (basado en distancia).
  2. Índice Calinski-Harabasz (basado en varianza).

Guarda los puntajes en un CSV y cada resultado de clustering en JSON.
"""

import sys
import os
import pandas as pd
import json
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

class SpectralClusterer:
    """
    Encapsula la lógica para cargar la matriz y ejecutar
    el clustering espectral evaluando múltiples métricas.
    """

    def __init__(self, matrix_csv_path):
        print(f"Cargando matriz desde: {matrix_csv_path}")
        
        df = pd.read_csv(matrix_csv_path, index_col=0)
        
        self.class_names = df.index.tolist()
        self.n = len(self.class_names) # Número de clases
        
        # Matriz de Similitud (Features para Calinski-Harabasz y Clustering)
        self.similarity_matrix = df.values
        
        # Matriz de Distancia (Para Silhouette)
        # Distancia = 1 - Similitud
        self.distance_matrix = 1 - self.similarity_matrix
        
        # La diagonal debe ser 0 para silhouette_score precomputado
        np.fill_diagonal(self.distance_matrix, 0)
        
        self.labels = None
        self.current_k = None
        
        print(f"Matriz {self.n}x{self.n} cargada. Probando K de 2 a {self.n - 1}.")

    def run_for_k(self, k):
        """
        Ejecuta el clustering y calcula métricas para un K específico.
        Devuelve un diccionario con los puntajes.
        """
        self.current_k = k
        print(f"--- Ejecutando para K = {k} ---")

        # 1. Ejecutar Clustering
        model = SpectralClustering(
            n_clusters=self.current_k,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )
        self.labels = model.fit_predict(self.similarity_matrix)
        
        # 2. Calcular Silhouette (Usa Matriz de Distancia)
        # Rango: [-1, 1]. Mayor es mejor.
        sil_score = silhouette_score(
            self.distance_matrix, 
            self.labels, 
            metric='precomputed'
        )

        # 3. Calcular Calinski-Harabasz (Usa Matriz de Similitud como Features)
        # Rango: [0, inf). Mayor es mejor (densidad vs separación).
        # Nota: Pasamos la matriz de similitud tratando cada fila como un vector de características.
        ch_score = calinski_harabasz_score(
            self.similarity_matrix, 
            self.labels
        )
        
        print(f"   > Silhouette: {sil_score:.4f}")
        print(f"   > Calinski-Harabasz: {ch_score:.4f}")
        
        return {
            'k': k,
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score
        }

    def save_results_json(self, output_json_path):
        """Guarda los resultados del clustering actual en JSON."""
        clusters = {f"cluster_{i}": [] for i in range(self.current_k)}
        
        for class_name, cluster_label in zip(self.class_names, self.labels):
            clusters[f"cluster_{cluster_label}"].append(class_name)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(clusters, f, indent=4)


def main():
    if len(sys.argv) != 3:
        print("Uso: python optimize_k_spectral.py <INPUT_MATRIX_CSV> <OUTPUT_DIR>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    # Definir archivo de salida para los puntajes
    scores_csv = os.path.join(output_dir, "metrics_scores.csv")
    
    # Crear directorio de resultados JSON
    json_dir = os.path.join(output_dir, "clustering_results")
    os.makedirs(json_dir, exist_ok=True)

    print(f"Directorio de resultados: '{output_dir}/'")

    # Inicializar
    clusterer = SpectralClusterer(input_csv)
    
    all_scores = []
    
    # Iterar K desde 2 hasta N-1
    for k in range(2, clusterer.n):
        
        # Ejecutar métricas
        metrics = clusterer.run_for_k(k)
        all_scores.append(metrics)
        
        # Guardar JSON de la partición
        json_path = os.path.join(json_dir, f"k_{k}.json")
        clusterer.save_results_json(json_path)

    # --- Guardar CSV de Métricas ---
    print(f"\nGuardando resumen de métricas en: {scores_csv}")
    
    scores_df = pd.DataFrame(all_scores)
    
    # Ordenamos por Silhouette descendente por defecto, 
    # pero guardamos ambas métricas para análisis manual.
    scores_df = scores_df.sort_values(by='silhouette_score', ascending=False)
    
    scores_df.to_csv(scores_csv, index=False)
    
    print("\n[Completado] Optimización finalizada.")
    print("Top 3 configuraciones según Silhouette:")
    print(scores_df.head(3))
    print("\nTop 3 configuraciones según Calinski-Harabasz:")
    print(scores_df.sort_values(by='calinski_harabasz_score', ascending=False).head(3))


if __name__ == '__main__':
    main()