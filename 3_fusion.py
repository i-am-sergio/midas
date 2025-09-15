import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configuración
MATRIX_PATHS = {
    'structural': 'structural_similarity_matrix.csv',
    'functional': 'functional_similarity_matrix.csv', 
    'semantic': 'semantic_similarity_matrix.csv'
}

METADATA_PATHS = {
    'functional': 'functional_similarity_matrix_metadata.csv',
    'semantic': 'semantic_similarity_matrix_metadata.csv'
}

OUTPUT_FUSION_CSV = 'fusion_matrix.csv'
OUTPUT_WEIGHTS_CSV = 'fusion_weights.csv'
OUTPUT_FUSION_PLOT = 'fusion_process.png'

class MultiViewFusion:
    def __init__(self):
        self.matrices = {}
        self.metadata = {}
        self.fusion_matrix = None
        self.weights = None
        self.mapping = None
        
    def load_matrices(self):
        """Carga todas las matrices de similitud"""
        print("Cargando matrices de similitud...")
        
        for view, path in MATRIX_PATHS.items():
            try:
                df = pd.read_csv(path, index_col=0)
                self.matrices[view] = df.values
                print(f"  {view}: {df.shape} - {df.index[:3].tolist()}...")
            except Exception as e:
                print(f"Error cargando {view}: {e}")
                self.matrices[view] = None
        
        # Cargar metadata para mapeo
        print("\nCargando metadata para mapeo...")
        for view, path in METADATA_PATHS.items():
            try:
                df = pd.read_csv(path)
                self.metadata[view] = df
                print(f"  {view}: {len(df)} elementos")
            except Exception as e:
                print(f"Error cargando metadata {view}: {e}")
                self.metadata[view] = None
    
    def create_unified_mapping(self):
        """Crea un mapeo unificado entre todas las vistas"""
        print("\nCreando mapeo unificado...")
        
        # La vista estructural es nuestra base (clases Java)
        structural_nodes = pd.read_csv(MATRIX_PATHS['structural'], index_col=0).index.tolist()
        unified_nodes = structural_nodes.copy()
        
        # Mapeo para vistas adicionales
        self.mapping = {
            'structural': {node: node for node in unified_nodes},
            'functional': {},
            'semantic': {}
        }
        
        # Mapeo funcional: endpoints → clases (heurística basada en nombres)
        if 'functional' in self.metadata and self.metadata['functional'] is not None:
            print("  Mapeando endpoints funcionales a clases...")
            func_df = self.metadata['functional']
            for _, row in func_df.iterrows():
                endpoint_id = row['endpoint_id']
                path = row['path']
                
                # Heurística: buscar clases que coincidan con el path
                matching_classes = []
                for class_node in unified_nodes:
                    class_name = class_node.split('.')[-1].lower()
                    if any(keyword in path.lower() for keyword in [class_name, class_name[:-1]] if class_name):
                        matching_classes.append(class_node)
                
                if matching_classes:
                    self.mapping['functional'][endpoint_id] = matching_classes[0]
                    print(f"    {endpoint_id} → {matching_classes[0]}")
                else:
                    # Asignación por defecto si no hay match
                    self.mapping['functional'][endpoint_id] = unified_nodes[0]
        
        # Mapeo semántico: casos de uso → clases (asignación basada en contenido)
        if 'semantic' in self.metadata and self.metadata['semantic'] is not None:
            print("  Mapeando casos de uso a clases...")
            semantic_df = self.metadata['semantic']
            for _, row in semantic_df.iterrows():
                uc_id = row['use_case_id']
                uc_text = row['use_case_text'] if 'use_case_text' in row else ''
                
                # Heurística simple basada en palabras clave
                matching_classes = []
                for class_node in unified_nodes:
                    class_name = class_node.split('.')[-1].lower()
                    if class_name and class_name in uc_text.lower():
                        matching_classes.append(class_node)
                
                if matching_classes:
                    self.mapping['semantic'][uc_id] = matching_classes[0]
                else:
                    # Asignar al primer nodo si no hay match
                    self.mapping['semantic'][uc_id] = unified_nodes[0]
        
        print(f"  Mapeo completado: {len(unified_nodes)} nodos unificados")
        return unified_nodes
    
    def expand_matrix(self, matrix, view, unified_nodes):
        """Expande una matriz a la dimensión unificada"""
        if matrix is None:
            return None
            
        original_nodes = list(self.mapping[view].keys())
        n_unified = len(unified_nodes)
        expanded_matrix = np.zeros((n_unified, n_unified))
        count_matrix = np.zeros((n_unified, n_unified))
        
        # Mapear similitudes originales a la matriz expandida
        for i, node_i in enumerate(original_nodes):
            for j, node_j in enumerate(original_nodes):
                if i < matrix.shape[0] and j < matrix.shape[1]:
                    unified_i = unified_nodes.index(self.mapping[view][node_i])
                    unified_j = unified_nodes.index(self.mapping[view][node_j])
                    expanded_matrix[unified_i, unified_j] += matrix[i, j]
                    count_matrix[unified_i, unified_j] += 1
        
        # Promediar donde hay múltiples asignaciones
        np.divide(expanded_matrix, count_matrix, out=expanded_matrix, where=count_matrix > 0)
        
        return expanded_matrix
    
    def calculate_view_distance(self, view_matrix, fusion_matrix, weight):
        """Calcula distancia entre vista actual y fusión"""
        if view_matrix is None:
            return np.inf
            
        # Usar distancia Frobenius ponderada
        difference = view_matrix - fusion_matrix
        distance = np.linalg.norm(weight * difference, 'fro')
        return distance
    
    def fusion_objective(self, weights, view_matrices):
        """Función objetivo para optimización de pesos"""
        weighted_sum = sum(w * M for w, M in zip(weights, view_matrices) if M is not None)
        total_weight = sum(weights)
        
        if total_weight > 0:
            fusion_matrix = weighted_sum / total_weight
        else:
            fusion_matrix = np.zeros_like(view_matrices[0])
        
        # Calcular distancia total
        total_distance = 0
        valid_views = 0
        for i, view_matrix in enumerate(view_matrices):
            if view_matrix is not None:
                total_distance += self.calculate_view_distance(view_matrix, fusion_matrix, weights[i])
                valid_views += 1
        
        return total_distance / valid_views if valid_views > 0 else 0
    
    def optimize_weights(self, view_matrices, max_iterations=100):
        """Optimiza los pesos de forma automática"""
        print("\nOptimizando pesos automáticamente...")
        
        n_views = len(view_matrices)
        initial_weights = np.ones(n_views) / n_views  # Inicializar equitativamente
        
        # Definir constraints (pesos entre 0 y 1, suman 1)
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}  # w >= 0
        ]
        
        bounds = [(0, 1) for _ in range(n_views)]
        
        # Optimizar
        result = minimize(
            self.fusion_objective,
            initial_weights,
            args=(view_matrices,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        if result.success:
            optimized_weights = result.x
            print(f"  Optimización exitosa después {result.nit} iteraciones")
            print(f"  Pesos finales: {optimized_weights}")
            return optimized_weights
        else:
            print(f"  Optimización fallida: {result.message}")
            return initial_weights
    
    def perform_fusion(self):
        """Realiza la fusión multivista"""
        print("=" * 60)
        print("INICIANDO FUSIÓN MULTIVISTA AUTO-PONDERADA")
        print("=" * 60)
        
        # 1. Cargar matrices
        self.load_matrices()
        
        # 2. Crear mapeo unificado
        unified_nodes = self.create_unified_mapping()
        
        # 3. Expandir matrices a dimensión unificada
        print("\nExpandiendo matrices a dimensión unificada...")
        expanded_matrices = []
        view_names = []
        
        for view in ['structural', 'functional', 'semantic']:
            if view in self.matrices and self.matrices[view] is not None:
                expanded = self.expand_matrix(self.matrices[view], view, unified_nodes)
                expanded_matrices.append(expanded)
                view_names.append(view)
                print(f"  {view}: {self.matrices[view].shape} → {expanded.shape}")
            else:
                expanded_matrices.append(None)
                print(f"  {view}: No disponible")
        
        # 4. Optimizar pesos automáticamente
        optimized_weights = self.optimize_weights(expanded_matrices)
        
        # 5. Aplicar fusión ponderada
        print("\nAplicando fusión ponderada...")
        weighted_sum = np.zeros_like(expanded_matrices[0])
        total_weight = 0
        
        for i, (weight, matrix) in enumerate(zip(optimized_weights, expanded_matrices)):
            if matrix is not None:
                weighted_sum += weight * matrix
                total_weight += weight
                print(f"  {view_names[i]}: peso {weight:.3f}")
        
        if total_weight > 0:
            self.fusion_matrix = weighted_sum / total_weight
            self.weights = dict(zip(view_names, optimized_weights))
        
        # 6. Guardar resultados
        self.save_results(unified_nodes)
        
        # 7. Visualizar proceso
        self.visualize_fusion(expanded_matrices, view_names, unified_nodes)
        
        return self.fusion_matrix, self.weights
    
    def save_results(self, unified_nodes):
        """Guarda la matriz fusionada y los pesos"""
        # Guardar matriz fusionada
        fusion_df = pd.DataFrame(
            self.fusion_matrix,
            index=unified_nodes,
            columns=unified_nodes
        )
        fusion_df.to_csv(OUTPUT_FUSION_CSV)
        print(f"\nMatriz fusionada guardada en: {OUTPUT_FUSION_CSV}")
        
        # Guardar pesos
        weights_df = pd.DataFrame.from_dict(self.weights, orient='index', columns=['weight'])
        weights_df.to_csv(OUTPUT_WEIGHTS_CSV)
        print(f"Pesos guardados en: {OUTPUT_WEIGHTS_CSV}")
        
        # Mostrar estadísticas
        print(f"\n=== ESTADÍSTICAS DE FUSIÓN ===")
        print(f"Nodos unificados: {len(unified_nodes)}")
        print(f"Dimensión matriz fusionada: {self.fusion_matrix.shape}")
        print(f"Rango de similitud: {self.fusion_matrix.min():.3f} - {self.fusion_matrix.max():.3f}")
        print(f"Similitud promedio: {self.fusion_matrix.mean():.3f}")
        
        print("\nPesos finales de cada vista:")
        for view, weight in self.weights.items():
            print(f"  {view}: {weight:.3f} ({weight*100:.1f}%)")
    
    def visualize_fusion(self, view_matrices, view_names, unified_nodes):
        """Visualiza el proceso de fusión"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plotear vistas individuales
        for i, (matrix, name) in enumerate(zip(view_matrices, view_names)):
            if matrix is not None:
                sns.heatmap(
                    matrix,
                    ax=axes[i],
                    cmap='viridis',
                    center=0.5,
                    square=True,
                    xticklabels=False,
                    yticklabels=False,
                    cbar_kws={'label': 'Similitud'}
                )
                axes[i].set_title(f'Vista: {name}\nPeso: {self.weights[name]:.3f}', fontweight='bold')
        
        # Plotear fusión final
        sns.heatmap(
            self.fusion_matrix,
            ax=axes[3],
            cmap='plasma',
            center=0.5,
            square=True,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Similitud Fusionada'}
        )
        axes[3].set_title('Fusión Multivista Final', fontweight='bold')
        
        plt.suptitle('Proceso de Fusión Auto-Ponderada de Vistas', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_FUSION_PLOT, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualización del proceso guardada en: {OUTPUT_FUSION_PLOT}")

def main():
    """Función principal"""
    print("Iniciando fusión multivista auto-ponderada...")
    
    fusion_engine = MultiViewFusion()
    fusion_matrix, weights = fusion_engine.perform_fusion()
    
    print("\n" + "=" * 60)
    print("FUSIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    
    return fusion_matrix, weights

if __name__ == "__main__":
    main()