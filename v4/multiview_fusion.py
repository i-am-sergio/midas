#!/usr/bin/env python3

"""
Fase 3: Fusión Multivista Auto-Ponderada (Self-Weighted Multi-view Fusion).

Integra las vistas Estructural, Semántica y Funcional en una matriz unificada.
Utiliza un algoritmo iterativo que actualiza los pesos basándose en la 
consistencia de cada vista con la estructura de clústeres del consenso (Laplaciano).

Uso:
    python multiview_fusion.py <STR_CSV> <SEM_CSV> <FUN_CSV> <OUTPUT_BASE>
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

class MultiViewFusion:
    def __init__(self, str_path, sem_path, fun_path):
        print("[Fusion] Cargando vistas...")
        
        # Cargar matrices
        self.df_str = pd.read_csv(str_path, index_col=0)
        self.df_sem = pd.read_csv(sem_path, index_col=0)
        self.df_fun = pd.read_csv(fun_path, index_col=0)
        
        # 1. Alineación de Vistas (Intersección de clases)
        # Es crucial que las matrices tengan las mismas dimensiones y orden
        common_index = self.df_str.index.intersection(self.df_sem.index).intersection(self.df_fun.index)
        
        self.classes = sorted(list(common_index))
        self.n = len(self.classes)
        
        print(f"[Fusion] Clases comunes alineadas: {self.n}")
        
        # Convertir a matrices numpy alineadas
        self.A_str = self.df_str.loc[self.classes, self.classes].values
        self.A_sem = self.df_sem.loc[self.classes, self.classes].values
        self.A_fun = self.df_fun.loc[self.classes, self.classes].values
        
        # Lista de matrices para iterar fácil
        self.matrices = [self.A_str, self.A_sem, self.A_fun]
        self.view_names = ['Structural', 'Semantic', 'Functional']

    def _compute_normalized_laplacian(self, W):
        """Calcula el Laplaciano Normalizado: L = I - D^-1/2 * W * D^-1/2"""
        # Asegurar simetría
        W = (W + W.T) / 2
        # np.fill_diagonal(W, 0) # Opcional, depende de la definición estricta
        
        # Grados
        degrees = np.sum(W, axis=1)
        
        # Evitar división por cero
        degrees[degrees == 0] = 1e-10
        
        d_inv_sqrt = np.power(degrees, -0.5)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Identidad
        I = np.eye(self.n)
        
        # Laplaciano Normalizado
        L = I - np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)
        return L

    def _get_consensus_eigenvectors(self, U, k=5):
        """
        Obtiene los k autovectores más pequeños del Laplaciano de la matriz unificada U.
        Estos vectores (H) representan la estructura de clúster 'Consenso'.
        """
        L_u = self._compute_normalized_laplacian(U)
        
        # eigh es para matrices hermitianas/simétricas (más rápido y estable)
        # subset_by_index=[0, k-1] obtiene los k menores
        eigenvalues, eigenvectors = eigh(L_u, subset_by_index=[0, k-1])
        
        return eigenvectors

    def optimize_weights(self, max_iter=100, tol=1e-6, k_clusters=5):
        """
        Ejecuta el bucle de optimización auto-ponderada.
        """
        print(f"[Fusion] Iniciando optimización (Max Iter: {max_iter}, Tol: {tol}, K: {k_clusters})...")
        
        # Inicialización uniforme
        n_views = 3
        weights = np.ones(n_views) / n_views
        
        history = []

        for iteration in range(max_iter):
            # 1. Calcular Matriz Unificada Actual
            U = weights[0] * self.A_str + weights[1] * self.A_sem + weights[2] * self.A_fun
            
            # 2. Obtener Estructura de Consenso (H) de U
            # (Resolvemos el problema espectral sobre la combinación actual)
            H = self._get_consensus_eigenvectors(U, k=k_clusters)
            
            # 3. Evaluar la calidad de cada vista INDIVIDUAL contra el Consenso (H)
            # Métrica: Traza(H^T * L_v * H). Menor es mejor (más alineado).
            view_scores = []
            for i, matrix in enumerate(self.matrices):
                L_v = self._compute_normalized_laplacian(matrix)
                # Tr(H^T L H) = Suma de elementos diagonales de (H^T L H)
                # Matemáticamente eficiente: sum(sum(H * (L @ H))) por columnas
                score = np.trace(np.dot(H.T, np.dot(L_v, H)))
                view_scores.append(score)
            
            view_scores = np.array(view_scores)
            
            # 4. Actualizar Pesos
            # Usamos una función Softmax inversa: menor score (mejor calidad) -> mayor peso
            # Usamos un gamma para controlar la nitidez de la distribución (heurística)
            gamma = 2.0 
            exp_scores = np.exp(-gamma * view_scores)
            new_weights = exp_scores / np.sum(exp_scores)
            
            # Calcular métrica global (Traza del Laplaciano de U) para monitoreo
            # score_laplacian_U = np.trace(np.dot(H.T, np.dot(self._compute_normalized_laplacian(U), H)))
            
            # Guardar historial
            log_entry = {
                'iteration': iteration,
                'w_str': weights[0],
                'w_sem': weights[1],
                'w_fun': weights[2],
                'score_str': view_scores[0],
                'score_sem': view_scores[1],
                'score_fun': view_scores[2],
                'convergence_diff': np.linalg.norm(new_weights - weights)
            }
            history.append(log_entry)
            
            # 5. Verificar Convergencia
            if np.linalg.norm(new_weights - weights) < tol:
                print(f"[Fusion] Convergencia alcanzada en iteración {iteration}.")
                weights = new_weights
                break
            
            weights = new_weights

        self.final_weights = weights
        self.history_df = pd.DataFrame(history)
        
        # Matriz final
        self.U_final = weights[0] * self.A_str + weights[1] * self.A_sem + weights[2] * self.A_fun
        
        print(f"[Fusion] Pesos Finales -> STR: {weights[0]:.4f}, SEM: {weights[1]:.4f}, FUN: {weights[2]:.4f}")
        return self.U_final

    def save_results(self, output_base):
        # 1. Guardar Matriz Fusionada
        matrix_path = f"{output_base}_fused_matrix.csv"
        df_u = pd.DataFrame(self.U_final, index=self.classes, columns=self.classes)
        df_u.to_csv(matrix_path)
        print(f"[Fusion] Matriz fusionada guardada en: {matrix_path}")
        
        # 2. Guardar Historial de Convergencia
        history_path = f"{output_base}_fusion_convergence.csv"
        self.history_df.to_csv(history_path, index=False)
        print(f"[Fusion] Historial de convergencia guardado en: {history_path}")
        # 3. Guardar Imagen de la Matriz Fusionada
        output_png = f"{output_base}_fused_matrix.png"
        self.save_matrix_png(output_png)
    
    def save_matrix_png(self, output_png):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(self.U_final, cmap='viridis')
        plt.colorbar()
        plt.title('Matriz Fusionada Multivista')
        plt.savefig(output_png)
        print(f"[Fusion] Imagen de la matriz guardada en: {output_png}")

def main():
    if len(sys.argv) != 5:
        print("Uso: python multiview_fusion.py <STR_CSV> <SEM_CSV> <FUN_CSV> <OUTPUT_BASE>")
        sys.exit(1)

    str_path = sys.argv[1]
    sem_path = sys.argv[2]
    fun_path = sys.argv[3]
    output_base = sys.argv[4] # Ej: results/jpetstore

    # Validar entradas
    if not os.path.exists(str_path):
        print(f"Error: No existe {str_path}")
        sys.exit(1)

    fusion_engine = MultiViewFusion(str_path, sem_path, fun_path)
    
    # Ejecutar optimización (K=5 hardcodeado por consistencia con análisis previo)
    fusion_engine.optimize_weights(max_iter=50, tol=1e-6, k_clusters=5)
    
    fusion_engine.save_results(output_base)

if __name__ == '__main__':
    main()