#!/usr/bin/env python3

"""
Aplica Clustering Espectral para un rango de K (de 2 a N-1).

Calcula el Coeficiente de Silhouette para cada K y guarda los puntajes
en un CSV. Guarda cada resultado de clustering en un JSON individual.

Uso:
    python optimize_k_spectral.py <INPUT_MATRIX_CSV>

Argumentos:
    <INPUT_MATRIX_CSV>: 
        Ruta a la matriz NxN normalizada (ej. jpetstore_structural_matrix.csv)

Salidas:
    - Un archivo 'silhouette_scores.csv' con los puntajes K.
    - Un directorio 'clustering_results/' con un JSON por cada K (ej. k_2.json).
"""

import sys
import os
import pandas as pd
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import numpy as np

class SpectralClusterer:
    """
    Encapsula la lógica para cargar la matriz y ejecutar
    el clustering espectral para múltiples valores de K.
    """

    def __init__(self, matrix_csv_path):
        print(f"Cargando matriz desde: {matrix_csv_path}")
        
        df = pd.read_csv(matrix_csv_path, index_col=0)
        
        self.class_names = df.index.tolist()
        self.n = len(self.class_names) # Número de clases
        
        self.similarity_matrix = df.values
        
        # Distancia = 1 - Similitud
        self.distance_matrix = 1 - self.similarity_matrix
        # La diagonal debe ser 0 para silhouette_score
        np.fill_diagonal(self.distance_matrix, 0)
        
        self.labels = None
        self.current_k = None
        
        print(f"Matriz {self.n}x{self.n} cargada. Probando K de 2 a {self.n - 1}.")

    def run_for_k(self, k):
        """
        Ejecuta el clustering y calcula silhouette para un K específico.
        Devuelve el puntaje de silhouette.
        """
        self.current_k = k
        print(f"--- Ejecutando para K = {k} ---")

        # 1. Ejecutar Clustering
        model = SpectralClustering(
            n_clusters=self.current_k,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42 # Para reproducibilidad
        )
        self.labels = model.fit_predict(self.similarity_matrix)
        print(f"Clustering completado para K={k}.")

        # 2. Calcular Silhouette
        score = silhouette_score(
            self.distance_matrix, 
            self.labels, 
            metric='precomputed'
        )
        print(f"Coeficiente de Silhouette: {score:.4f}")
        
        return score

    def save_results_json(self, output_json_path):
        """
        Guarda los resultados del clustering actual (self.labels) 
        en un archivo JSON.
        """
        
        clusters = {f"cluster_{i}": [] for i in range(self.current_k)}
        
        for class_name, cluster_label in zip(self.class_names, self.labels):
            clusters[f"cluster_{cluster_label}"].append(class_name)

        print(f"Guardando resultados en: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(clusters, f, indent=4)


def main():
    if len(sys.argv) != 2:
        print("Uso: python optimize_k_spectral.py <INPUT_MATRIX_CSV>")
        print("Ejemplo: python optimize_k_spectral.py jpetstore_structural_matrix.csv")
        sys.exit(1)

    input_csv = sys.argv[1]

    # Definir rutas de salida
    output_dir = "clustering_results_spectral"
    # scores_csv = "silhouette_scores_spectral.csv"
    scores_csv = os.path.join(output_dir, "silhouette_scores_spectral.csv")

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio de resultados JSON: '{output_dir}/'")

    # Inicializar el clusterer una vez
    clusterer = SpectralClusterer(input_csv)
    
    all_scores = []
    
    # Iterar K desde 2 hasta N-1
    # (El clustering no tiene sentido para K=1 o K=N)
    for k in range(2, clusterer.n):
        
        # Ejecutar y obtener puntaje
        score = clusterer.run_for_k(k)
        
        # Guardar puntaje
        all_scores.append({'k': k, 'silhouette_score': score})
        
        # Definir nombre de archivo JSON y guardar
        json_path = os.path.join(output_dir, f"k_{k}.json")
        clusterer.save_results_json(json_path)

    # --- Fin del bucle ---

    # Guardar todos los puntajes en un solo archivo CSV
    print("\nProceso de iteración completado.")
    print(f"Guardando todos los puntajes en: {scores_csv}")
    
    scores_df = pd.DataFrame(all_scores)
    # Ordenar por puntaje (descendente) para ver el mejor K primero
    scores_df = scores_df.sort_values(by='silhouette_score', ascending=False)
    
    scores_df.to_csv(scores_csv, index=False)
    
    print("\n[Completado] Optimización de K finalizada.")
    print(f"Mejor K encontrado (basado en Silhouette):")
    print(scores_df.iloc[0])


if __name__ == '__main__':
    main()