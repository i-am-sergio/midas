#!/usr/bin/env python3

"""
Calcula las métricas de calidad de la descomposición (SM, ICP, IFN, NED)
basado en un resultado de clustering y el grafo de relaciones crudo.

Uso:
    python calculate_metrics.py <CLUSTER_JSON> <RAW_RELATIONS_CSV>

Argumentos:
    <CLUSTER_JSON>: 
        Ruta al archivo JSON que define los clústeres
        (ej. clustering_results/k_5.json)
    <RAW_RELATIONS_CSV>: 
        Ruta al CSV con las relaciones filtradas *antes* de la normalización
        (ej. jPetStore_fase2_structural_view_filtered.csv)
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import ast
from itertools import combinations

class MetricsCalculator:
    """
    Calcula SM, ICP, IFN, y NED para una descomposición de clústeres dada.
    """

    def __init__(self, cluster_json_path, raw_relations_csv_path):
        
        # --- 1. Cargar Particiones (Clústeres) ---
        print(f"Cargando clústeres desde: {cluster_json_path}")
        with open(cluster_json_path, 'r') as f:
            # clusters = {'cluster_0': ['ClassA', ...], 'cluster_1': [...]}
            self.clusters = json.load(f)
        
        self.K = len(self.clusters)
        
        # Mapa inverso: {'ClassA': 'cluster_0', 'ClassB': 'cluster_1', ...}
        self.class_to_cluster = {}
        for cluster_name, classes in self.clusters.items():
            for class_name in classes:
                self.class_to_cluster[class_name] = cluster_name
        
        # Lista ordenada de todas las clases en el núcleo
        self.all_classes = sorted(list(self.class_to_cluster.keys()))
        self.N = len(self.all_classes)
        
        # Mapa de 'ClassName' -> índice (0..N-1)
        self.class_to_index = {name: i for i, name in enumerate(self.all_classes)}
        
        # --- 2. Construir las Matrices de Adyacencia ---
        print(f"Construyendo matrices desde: {raw_relations_csv_path}")
        self._build_matrices(raw_relations_csv_path)

        print(f"Calculadora lista: {self.N} clases, {self.K} clústeres.")

    def _build_matrices(self, raw_relations_csv_path):
        """
        Construye la matriz Asimétrica (A) y Simétrica (S) 
        *no normalizadas* desde el CSV de relaciones.
        """
        self.A = np.zeros((self.N, self.N)) # Matriz Asimétrica (para ICP, IFN)
        
        df = pd.read_csv(raw_relations_csv_path)
        
        for _, row in df.iterrows():
            source_class = row['class']
            relations = ast.literal_eval(row['relations'])
            
            # Asegurarse de que la clase está en nuestro núcleo
            if source_class not in self.class_to_index:
                continue
                
            i = self.class_to_index[source_class]
            
            for target_class, score in relations.items():
                if target_class in self.class_to_index:
                    j = self.class_to_index[target_class]
                    self.A[i, j] = score # A[i, j] = llamada de i -> j
        
        # S = Matriz Simétrica (para SM)
        # Representa un grafo no dirigido donde el peso es A + A.T
        self.S = self.A + self.A.T

    def calculate_sm(self):
        """Calcula la Modularidad Estructural (SM)"""
        
        cohesion_sum = 0.0
        coupling_sum = 0.0
        
        # Obtener todos los pares únicos de clústeres (i, j)
        cluster_indices = list(range(self.K))
        
        # 1. Calcular Cohesión (Término 1)
        for i in cluster_indices:
            cluster_name = f"cluster_{i}"
            classes_in_cluster = self.clusters[cluster_name]
            m_i = len(classes_in_cluster)
            
            if m_i == 0:
                continue

            # Obtener los índices de matriz para esta clase
            idx_i = [self.class_to_index[c] for c in classes_in_cluster]
            
            # Extraer la submatriz interna del clúster
            # Usamos la matriz Simétrica S
            internal_submatrix = self.S[np.ix_(idx_i, idx_i)]
            
            # mu_i = suma de todos los pesos internos. 
            # / 2 porque la matriz es simétrica (contamos cada borde dos veces)
            mu_i = np.sum(internal_submatrix) / 2
            
            cohesion_sum += mu_i / (m_i ** 2)

        avg_cohesion = cohesion_sum / self.K

        # 2. Calcular Acoplamiento (Término 2)
        num_pairs = self.K * (self.K - 1) / 2
        if num_pairs == 0:
            return avg_cohesion # Si K=1, el acoplamiento es 0

        for i, j in combinations(cluster_indices, 2):
            cluster_i_name = f"cluster_{i}"
            cluster_j_name = f"cluster_{j}"
            
            classes_i = self.clusters[cluster_i_name]
            classes_j = self.clusters[cluster_j_name]
            
            m_i = len(classes_i)
            m_j = len(classes_j)
            
            if m_i == 0 or m_j == 0:
                continue

            idx_i = [self.class_to_index[c] for c in classes_i]
            idx_j = [self.class_to_index[c] for c in classes_j]

            # Extraer la submatriz entre clústeres i y j
            external_submatrix = self.S[np.ix_(idx_i, idx_j)]
            
            # sigma_i_j = suma de todos los pesos entre i y j
            sigma_i_j = np.sum(external_submatrix)
            
            coupling_sum += sigma_i_j / (2 * m_i * m_j)

        avg_coupling = coupling_sum / num_pairs
        
        sm = avg_cohesion - avg_coupling
        return sm

    def calculate_icp(self):
        """Calcula el Porcentaje de Llamadas Internas (ICP)"""
        
        total_calls = np.sum(self.A) # Denominador
        if total_calls == 0:
            return 0.0 # Evitar división por cero

        external_calls = 0.0 # Numerador
        
        cluster_indices = list(range(self.K))

        for i in cluster_indices:
            idx_i = [self.class_to_index[c] for c in self.clusters[f"cluster_{i}"]]
            
            for j in cluster_indices:
                if i == j:
                    continue # Solo contamos llamadas *externas*

                idx_j = [self.class_to_index[c] for c in self.clusters[f"cluster_{j}"]]
                
                # c_i_j = llamadas desde el clúster i HASTA el clúster j
                # Usamos la matriz Asimétrica A
                calls_i_to_j = np.sum(self.A[np.ix_(idx_i, idx_j)])
                external_calls += calls_i_to_j
                
        icp = external_calls / total_calls
        return icp

    def calculate_ifn(self):
        """Calcula el Número de Interfaces (IFN)"""
        
        total_interfaces = 0
        
        for i in range(self.K):
            cluster_name = f"cluster_{i}"
            classes_in_cluster = self.clusters[cluster_name]
            
            idx_i = [self.class_to_index[c] for c in classes_in_cluster]
            
            # Índices de todas las clases que NO están en este clúster
            idx_external = [j for j in range(self.N) if j not in idx_i]
            
            if not idx_external:
                continue # No hay clases externas (ej. K=1)

            # Matriz de llamadas DESDE el exterior HACIA el interior
            # Filas = Externas, Columnas = Internas
            A_ext_to_int = self.A[np.ix_(idx_external, idx_i)]
            
            # Sumar las llamadas entrantes para cada clase interna (por columna)
            incoming_calls_per_class = np.sum(A_ext_to_int, axis=0)
            
            # ifn_i = número de clases con CUALQUIER llamada entrante (> 0)
            ifn_i = np.count_nonzero(incoming_calls_per_class)
            total_interfaces += ifn_i
            
        ifn = total_interfaces / self.K
        return ifn

    def calculate_ned(self):
        """Calcula la Distribución No Extrema (NED)"""
        
        acceptable_size_count = 0
        
        for i in range(self.K):
            m_i = len(self.clusters[f"cluster_{i}"])
            
            # Rango de tamaño aceptable [5, 20]
            if 5 <= m_i <= 20:
                acceptable_size_count += 1
                
        ned = 1.0 - (acceptable_size_count / self.K)
        return ned

    def calculate_all(self):
        """Ejecuta todos los cálculos y devuelve un diccionario."""
        metrics = {
            "K": self.K,
            "N": self.N,
            "SM (Higher is Better)": self.calculate_sm(),
            "ICP (Lower is Better)": self.calculate_icp(),
            "IFN (Lower is Better)": self.calculate_ifn(),
            "NED (Lower is Better)": self.calculate_ned()
        }
        return metrics

def main():
    if len(sys.argv) != 3:
        print("Uso: python calculate_metrics.py <CLUSTER_JSON> <RAW_RELATIONS_CSV>")
        print("Ejemplo: python calculate_metrics.py clustering_results/k_5.json jPetStore_fase2_structural_view_filtered.csv")
        sys.exit(1)

    cluster_json_path = sys.argv[1]
    raw_relations_csv_path = sys.argv[2]
    
    if not os.path.exists(cluster_json_path):
        print(f"Error: Archivo de clúster no encontrado: {cluster_json_path}")
        sys.exit(1)
        
    if not os.path.exists(raw_relations_csv_path):
        print(f"Error: Archivo de relaciones no encontrado: {raw_relations_csv_path}")
        sys.exit(1)

    # Inicializar y calcular
    calculator = MetricsCalculator(cluster_json_path, raw_relations_csv_path)
    metrics = calculator.calculate_all()

    # Imprimir resultados
    print("\n--- Resultados de Calidad de Descomposición ---")
    print(f"  Archivo de Clúster: {cluster_json_path}")
    print(f"  Clases (N): {metrics['N']}")
    print(f"  Microservicios (K): {metrics['K']}")
    print("-----------------------------------------------")
    print(f"  Modularidad Estructural (SM): {metrics['SM (Higher is Better)']:.4f}")
    print(f"  Porcentaje Llamadas Internas (ICP): {metrics['ICP (Lower is Better)']:.4f}")
    print(f"  Número de Interfaces (IFN): {metrics['IFN (Lower is Better)']:.4f}")
    print(f"  Distribución No Extrema (NED): {metrics['NED (Lower is Better)']:.4f}")
    print("-----------------------------------------------")

if __name__ == '__main__':
    main()