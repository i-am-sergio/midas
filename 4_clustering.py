import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Configuración
FUSION_MATRIX_PATH = 'fusion_matrix.csv'
K_CLUSTERS = 5
RANDOM_STATE = 42

# Output files
OUTPUT_CLUSTERS_CSV = 'clustering_results.csv'
OUTPUT_CLUSTER_STATS_CSV = 'cluster_statistics.csv'
OUTPUT_CLUSTER_PLOT = 'kmeans_clustering_results.png'
OUTPUT_SILHOUETTE_PLOT = 'silhouette_analysis.png'

class MicroservicesClusterer:
    def __init__(self, k_clusters=K_CLUSTERS):
        self.k_clusters = k_clusters
        self.fusion_matrix = None
        self.node_names = None
        self.kmeans = None
        self.labels = None
        self.cluster_results = None
        
    def load_fusion_matrix(self):
        """Carga la matriz fusionada"""
        print("Cargando matriz fusionada...")
        df = pd.read_csv(FUSION_MATRIX_PATH, index_col=0)
        self.fusion_matrix = df.values
        self.node_names = df.index.tolist()
        print(f"Matriz cargada: {self.fusion_matrix.shape}")
        print(f"Nodos: {len(self.node_names)}")
        print(f"Rango de similitud: {self.fusion_matrix.min():.3f} - {self.fusion_matrix.max():.3f}")
        
    def validate_matrix(self):
        """Valida que la matriz sea adecuada para clustering"""
        print("\nValidando matriz para clustering...")
        
        # Verificar que no hay NaN values
        if np.isnan(self.fusion_matrix).any():
            print("  Advertencia: La matriz contiene valores NaN, rellenando con 0...")
            self.fusion_matrix = np.nan_to_num(self.fusion_matrix)
        
        # Verificar simetría
        is_symmetric = np.allclose(self.fusion_matrix, self.fusion_matrix.T)
        print(f"  Matriz simétrica: {is_symmetric}")
        
        # Verificar rango de valores
        print(f"  Rango valores: [{self.fusion_matrix.min():.3f}, {self.fusion_matrix.max():.3f}]")
        
        return True
    
    def perform_kmeans(self):
        """Ejecuta algoritmo K-means"""
        print(f"\nEjecutando K-means con K={self.k_clusters}...")
        
        # Usar la matriz de similitud como features (cada fila es un vector de similitudes)
        X = self.fusion_matrix
        
        # Configurar e inicializar K-means
        self.kmeans = KMeans(
            n_clusters=self.k_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
            max_iter=300,
            tol=1e-4
        )
        
        # Ajustar el modelo
        self.labels = self.kmeans.fit_predict(X)
        
        print("K-means completado exitosamente")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Centroids shape: {self.kmeans.cluster_centers_.shape}")
        
        return self.labels
    
    def calculate_metrics(self):
        """Calcula métricas de evaluación del clustering"""
        print("\nCalculando métricas de evaluación...")
        
        X = self.fusion_matrix
        metrics = {}
        
        # Silhouette Score
        try:
            silhouette_avg = silhouette_score(X, self.labels)
            metrics['silhouette_score'] = silhouette_avg
            print(f"  Silhouette Score: {silhouette_avg:.3f}")
        except Exception as e:
            print(f"  Error calculando Silhouette: {e}")
            metrics['silhouette_score'] = np.nan
        
        # Calinski-Harabasz Index
        try:
            ch_score = calinski_harabasz_score(X, self.labels)
            metrics['calinski_harabasz'] = ch_score
            print(f"  Calinski-Harabasz Index: {ch_score:.3f}")
        except Exception as e:
            print(f"  Error calculando Calinski-Harabasz: {e}")
            metrics['calinski_harabasz'] = np.nan
        
        # Davies-Bouldin Index
        try:
            db_score = davies_bouldin_score(X, self.labels)
            metrics['davies_bouldin'] = db_score
            print(f"  Davies-Bouldin Index: {db_score:.3f}")
        except Exception as e:
            print(f"  Error calculando Davies-Bouldin: {e}")
            metrics['davies_bouldin'] = np.nan
        
        # Inercia (within-cluster sum of squares)
        metrics['inertia'] = self.kmeans.inertia_
        print(f"  Inertia: {self.kmeans.inertia_:.3f}")
        
        return metrics
    
    def analyze_clusters(self):
        """Analiza la distribución y características de los clusters"""
        print("\nAnalizando clusters...")
        
        cluster_counts = np.bincount(self.labels)
        cluster_stats = []
        
        for cluster_id in range(self.k_clusters):
            cluster_indices = np.where(self.labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            # Obtener nombres de nodos en el cluster
            cluster_nodes = [self.node_names[i] for i in cluster_indices]
            
            # Calcular similitud intra-cluster promedio
            cluster_matrix = self.fusion_matrix[cluster_indices][:, cluster_indices]
            intra_similarity = cluster_matrix.mean() if cluster_size > 0 else 0
            
            stats = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'intra_similarity': intra_similarity,
                'nodes': cluster_nodes,
                'node_names': ' | '.join([n.split('.')[-1] for n in cluster_nodes[:3]]) + 
                             ('...' if cluster_size > 3 else '')
            }
            cluster_stats.append(stats)
            
            print(f"  Cluster {cluster_id}: {cluster_size} elementos, similitud intra: {intra_similarity:.3f}")
            print(f"    Ejemplos: {stats['node_names']}")
        
        return cluster_stats
    
    def create_cluster_dataframe(self):
        """Crea DataFrame con resultados del clustering"""
        cluster_data = []
        
        for i, (node_name, label) in enumerate(zip(self.node_names, self.labels)):
            # Extraer nombre simple de la clase
            simple_name = node_name.split('.')[-1] if '.' in node_name else node_name
            
            cluster_data.append({
                'node_id': i,
                'full_name': node_name,
                'simple_name': simple_name,
                'cluster_id': label,
                'cluster_size': np.sum(self.labels == label)
            })
        
        self.cluster_results = pd.DataFrame(cluster_data)
        return self.cluster_results
    
    def visualize_clusters(self):
        """Visualiza los resultados del clustering"""
        print("\nGenerando visualizaciones...")
        
        # Reducción de dimensionalidad para visualización
        X = self.fusion_matrix
        
        # Usar PCA primero para reducir a 50 dimensiones
        pca = PCA(n_components=min(50, X.shape[0]), random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X)
        
        # Luego t-SNE para 2D
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, X.shape[0]-1))
        X_2d = tsne.fit_transform(X_pca)
        
        # Crear figura con subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot de clusters
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=self.labels, cmap='tab10', 
                            s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
        
        ax1.set_title(f'K-means Clustering (K={self.k_clusters})\nVisualización t-SNE', fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.grid(True, alpha=0.3)
        
        # Leyenda de clusters
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=plt.cm.tab10(i/self.k_clusters), 
                                    markersize=10, label=f'Cluster {i}') 
                         for i in range(self.k_clusters)]
        ax1.legend(handles=legend_elements, loc='best')
        
        # Heatmap de la matriz fusionada con clusters
        # Ordenar por clusters para mejor visualización
        sorted_indices = np.argsort(self.labels)
        sorted_matrix = self.fusion_matrix[sorted_indices][:, sorted_indices]
        
        im = ax2.imshow(sorted_matrix, cmap='viridis', aspect='auto')
        ax2.set_title('Matriz Fusionada Ordenada por Clusters', fontweight='bold')
        ax2.set_xlabel('Nodos (ordenados por cluster)')
        ax2.set_ylabel('Nodos (ordenados por cluster)')
        
        # Añadir líneas divisorias entre clusters
        cluster_sizes = np.bincount(self.labels)
        cumulative_sizes = np.cumsum(cluster_sizes)
        for size in cumulative_sizes[:-1]:
            ax2.axhline(y=size, color='red', linestyle='--', linewidth=2)
            ax2.axvline(x=size, color='red', linestyle='--', linewidth=2)
        
        plt.colorbar(im, ax=ax2, label='Similitud')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_CLUSTER_PLOT, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualización de clusters guardada en: {OUTPUT_CLUSTER_PLOT}")
        
        # Visualización de silhouette (opcional para matrices grandes)
        if len(self.labels) <= 1000:  # Solo para conjuntos razonables
            self.plot_silhouette_analysis()
    
    def plot_silhouette_analysis(self):
        """Análisis de silhouette para cada cluster"""
        from sklearn.metrics import silhouette_samples
        
        silhouette_vals = silhouette_samples(self.fusion_matrix, self.labels)
        cluster_labels = np.unique(self.labels)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        y_lower = 10
        for i, cluster_id in enumerate(cluster_labels):
            # Agrupar valores de silhouette por cluster
            cluster_silhouette_vals = silhouette_vals[self.labels == cluster_id]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.tab10(i / len(cluster_labels))
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Etiquetar el cluster
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
            
            y_lower = y_upper + 10
        
        ax.set_title("Análisis de Silhouette por Cluster", fontweight='bold')
        ax.set_xlabel("Coeficiente de Silhouette")
        ax.set_ylabel("Cluster")
        
        # Línea vertical para el promedio
        silhouette_avg = np.mean(silhouette_vals)
        ax.axvline(x=silhouette_avg, color="red", linestyle="--",
                  label=f"Promedio: {silhouette_avg:.3f}")
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_SILHOUETTE_PLOT, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Análisis de silhouette guardado en: {OUTPUT_SILHOUETTE_PLOT}")
    
    def save_results(self, metrics, cluster_stats):
        """Guarda todos los resultados"""
        print("\nGuardando resultados...")
        
        # Guardar asignación de clusters
        self.cluster_results.to_csv(OUTPUT_CLUSTERS_CSV, index=False)
        print(f"Resultados de clustering guardados en: {OUTPUT_CLUSTERS_CSV}")
        
        # Guardar estadísticas de clusters
        stats_df = pd.DataFrame(cluster_stats)
        stats_df.to_csv(OUTPUT_CLUSTER_STATS_CSV, index=False)
        print(f"Estadísticas de clusters guardadas en: {OUTPUT_CLUSTER_STATS_CSV}")
        
        # Mostrar resumen final
        print("\n" + "="*60)
        print("RESUMEN FINAL DE CLUSTERING")
        print("="*60)
        print(f"Total de nodos: {len(self.node_names)}")
        print(f"Número de clusters: {self.k_clusters}")
        print(f"Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.3f}")
        print(f"Inertia: {metrics.get('inertia', 'N/A'):.3f}")
        
        print("\nDistribución de clusters:")
        for stats in cluster_stats:
            print(f"  Cluster {stats['cluster_id']}: {stats['size']} elementos "
                  f"(similitud intra: {stats['intra_similarity']:.3f})")
    
    def run_clustering(self):
        """Ejecuta el pipeline completo de clustering"""
        print("="*60)
        print("INICIANDO CLUSTERING K-MEANS PARA MICROSERVICIOS")
        print("="*60)
        
        # 1. Cargar matriz fusionada
        self.load_fusion_matrix()
        
        # 2. Validar matriz
        self.validate_matrix()
        
        # 3. Ejecutar K-means
        self.perform_kmeans()
        
        # 4. Calcular métricas
        metrics = self.calculate_metrics()
        
        # 5. Analizar clusters
        cluster_stats = self.analyze_clusters()
        
        # 6. Crear DataFrame de resultados
        self.create_cluster_dataframe()
        
        # 7. Visualizar resultados
        self.visualize_clusters()
        
        # 8. Guardar resultados
        self.save_results(metrics, cluster_stats)
        
        return self.cluster_results, metrics

def main():
    """Función principal"""
    clusterer = MicroservicesClusterer(k_clusters=K_CLUSTERS)
    results, metrics = clusterer.run_clustering()
    
    print("\n" + "="*60)
    print("CLUSTERING COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return results, metrics

if __name__ == "__main__":
    main()