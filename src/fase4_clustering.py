# fase4_clustering.py
"""
FASE 4: Clustering Espectral y Optimización de K.

Toma la Matriz Fusionada (afinidad precomputada), ejecuta Spectral Clustering
para un rango de K (2 a N-1) y calcula métricas de calidad (Silhouette, CH).

Uso:
    python fase4_clustering.py <FUSION_MATRIX_CSV> <OUTPUT_DIR>
"""

import sys
import os
import pandas as pd
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt
import networkx as nx
from adjustText import adjust_text
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Silenciar warnings de grafos desconectados (comunes y esperados en microservicios limpios)
warnings.filterwarnings("ignore", category=UserWarning)

class SpectralClusterer:
    """
    Encapsula la lógica para cargar la matriz y ejecutar
    el clustering espectral evaluando múltiples métricas.
    """

    def __init__(self, matrix_csv_path):
        print(f"[Clustering] Cargando matriz fusionada desde: {matrix_csv_path}")
        
        try:
            df = pd.read_csv(matrix_csv_path, index_col=0)
        except Exception as e:
            print(f"Error cargando el archivo: {e}")
            sys.exit(1)
        
        self.class_names = df.index.tolist()
        self.n = len(self.class_names) # Número de clases
        
        # Matriz de Similitud (Features para Calinski-Harabasz y Clustering)
        # Aseguramos que sea numpy array
        self.similarity_matrix = df.values
        
        # Matriz de Distancia (Para Silhouette)
        # Distancia = 1 - Similitud
        # Clip para evitar valores negativos por errores de punto flotante
        self.distance_matrix = np.clip(1 - self.similarity_matrix, 0, 1)
        
        # La diagonal debe ser 0 para silhouette_score precomputado
        np.fill_diagonal(self.distance_matrix, 0)
        
        self.labels = None
        self.current_k = None
        
        print(f"[Clustering] Matriz {self.n}x{self.n} cargada.")
        print(f"[Clustering] Rango de exploración K: [2 ... {self.n - 1}]")

    def run_for_k(self, k):
        """
        Ejecuta el clustering y calcula métricas para un K específico.
        Devuelve un diccionario con los puntajes.
        """
        self.current_k = k
        
        # 1. Ejecutar Clustering
        # Usamos 'discretize' a veces es más estable que 'kmeans' para grafos complejos,
        # pero mantenemos 'kmeans' como en tu referencia.
        model = SpectralClustering(
            n_clusters=self.current_k,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=50 # Aumentamos intentos para estabilidad
        )
        self.labels = model.fit_predict(self.similarity_matrix)
        
        # 2. Calcular Silhouette (Usa Matriz de Distancia)
        # Rango: [-1, 1]. Mayor es mejor.
        try:
            sil_score = silhouette_score(
                self.distance_matrix, 
                self.labels, 
                metric='precomputed'
            )
        except Exception:
            sil_score = -1.0 # Fallback si falla (ej. solo 1 cluster encontrado)

        # 3. Calcular Calinski-Harabasz (Usa Matriz de Similitud como Features)
        # Rango: [0, inf). Mayor es mejor.
        try:
            ch_score = calinski_harabasz_score(
                self.similarity_matrix, 
                self.labels
            )
        except Exception:
            ch_score = 0.0

        # Log visual simple
        print(f"   K={k:02d} | Silhouette: {sil_score:.4f} | CH: {ch_score:.4f}")
        
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
    
    def generate_cluster_graph(self, k, output_path):
        """
        Genera y guarda un gráfico de red con etiquetas optimizadas para no solaparse.
        """
        try:
            from adjustText import adjust_text
        except ImportError:
            print("Error: Necesitas instalar la librería 'adjustText'. Ejecuta: pip install adjustText")
            return

        print(f"[Visualización] Generando gráfico para K={k}...")
        
        if self.current_k != k:
            self.run_for_k(k)
        
        G = nx.Graph()
        for i, name in enumerate(self.class_names):
            G.add_node(name, group=self.labels[i])
            
        rows, cols = np.where(self.similarity_matrix > 0.05)
        for r, c in zip(rows, cols):
            if r < c:
                weight = self.similarity_matrix[r, c]
                G.add_edge(self.class_names[r], self.class_names[c], weight=weight)

        plt.figure(figsize=(16, 14)) 
        
        # Aumentamos 'k' (distancia óptima entre nodos) para separarlos más
        pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)
        
        cmap = plt.get_cmap('tab10')
        node_colors = [cmap(self.labels[i]) for i in range(self.n)]
        
        # 1. Dibujar Aristas
        edges = G.edges(data=True)
        weights = [d['weight'] * 3 for u, v, d in edges] 
        nx.draw_networkx_edges(G, pos, alpha=1.0, width=weights, edge_color='gray')
        
        # 2. Dibujar Nodos
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_size=700, 
            node_color=node_colors, 
            alpha=0.9, 
            edgecolors='black'
        )
        
        # 3. Dibujar Etiquetas 
        texts = []
        for node, (x, y) in pos.items():
            # Creamos el objeto texto manualmente
            t = plt.text(
                x, y, 
                node, 
                fontsize=12, 
                fontweight='bold', 
                fontfamily='sans-serif'
            )
            texts.append(t)
        
        adjust_text(
            texts, 
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), # Dibuja una línea si se aleja mucho
            force_text=0.5,
            expand_points=(1.3, 1.3) # Expande el área alrededor de nodos y textos
        )

        plt.title(f'Grafo de Dependencias (K={k})', fontsize=16)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[Visualización] Gráfico guardado en: {output_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python fase4_clustering.py <INPUT_MATRIX_CSV> <OUTPUT_DIR>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_csv):
        print(f"Error: No existe el archivo de entrada {input_csv}")
        sys.exit(1)

    # Definir rutas
    scores_csv = os.path.join(output_dir, "metrics_scores.csv")
    json_dir = os.path.join(output_dir, "clustering_results")
    
    # Crear directorios
    os.makedirs(json_dir, exist_ok=True)

    print(f"[Clustering] Directorio de resultados: '{output_dir}/'")

    # Inicializar
    clusterer = SpectralClusterer(input_csv)
    
    all_scores = []
    
    # Iterar K desde 2 hasta N-1
    # Si N es muy pequeño (ej. < 3), ajustamos el rango
    max_k = clusterer.n
    if max_k > 20: max_k = 20 # Opcional: Limitar si es gigante para ahorrar tiempo
    else: max_k = clusterer.n

    for k in range(2, max_k):
        # Ejecutar métricas
        metrics = clusterer.run_for_k(k)
        all_scores.append(metrics)
        
        # Guardar JSON de la partición
        json_path = os.path.join(json_dir, f"k_{k}.json")
        clusterer.save_results_json(json_path)

    # --- Guardar CSV de Métricas ---
    print(f"\n[Clustering] Guardando resumen de métricas en: {scores_csv}")
    
    scores_df = pd.DataFrame(all_scores)
    
    # Ordenamos por Silhouette descendente
    if not scores_df.empty:
        scores_df = scores_df.sort_values(by='silhouette_score', ascending=False)
        scores_df.to_csv(scores_csv, index=False)
        
        print("\n--- Top 5 Configuraciones (Silhouette) ---")
        print(scores_df.head(5).to_string(index=False))
        
        print("\n--- Top 5 Configuraciones (Calinski-Harabasz) ---")
        print(scores_df.sort_values(by='calinski_harabasz_score', ascending=False).head(5).to_string(index=False))
    else:
        print("No se generaron puntajes (quizás N < 3).")

    # TARGET_K = 6
    # graph_output = os.path.join(output_dir, f"visualization_graph_k{TARGET_K}.png")
    # clusterer.generate_cluster_graph(k=TARGET_K, output_path=graph_output)

if __name__ == '__main__':
    main()