# fase3_multiview_fusion.py

"""
FASE 3: Fusión Multivista Auto-Ponderada.

Integra las matrices A_str, S_sem, A_fun en una Matriz Unificada (U).
Utiliza optimización iterativa para aprender los pesos (w_str, w_sem, w_fun)
que maximizan la coherencia del clustering espectral.

Uso:
    python fase3_multiview_fusion.py <STR_CSV> <SEM_CSV> <FUN_CSV> <OUTPUT_MATRIX_CSV>
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.linalg import eigh

class MultiViewFusion:
    def __init__(self, str_path, sem_path, fun_path):
        print("[Fusion] Cargando matrices de vistas...")
        
        # 1. Cargar DataFrames
        try:
            self.df_str = pd.read_csv(str_path, index_col=0)
            self.df_sem = pd.read_csv(sem_path, index_col=0)
            self.df_fun = pd.read_csv(fun_path, index_col=0)
        except Exception as e:
            print(f"Error cargando archivos: {e}")
            sys.exit(1)
        
        # 2. Validación de Alineación
        # Las matrices deben tener exactamente las mismas clases en el mismo orden
        if not (self.df_str.index.equals(self.df_sem.index) and self.df_str.index.equals(self.df_fun.index)):
            print("Error Crítico: Las matrices no están alineadas (diferentes clases u orden).")
            print(f"   Str: {self.df_str.shape}, Sem: {self.df_sem.shape}, Fun: {self.df_fun.shape}")
            sys.exit(1)
            
        self.classes = self.df_str.index.tolist()
        self.n = len(self.classes)
        print(f"[Fusion] Matrices alineadas correctamente ({self.n} clases).")
        
        # 3. Convertir a Numpy Arrays
        self.A_str = self.df_str.values
        self.A_sem = self.df_sem.values
        self.A_fun = self.df_fun.values
        
        self.matrices = [self.A_str, self.A_sem, self.A_fun]
        self.view_names = ['STR', 'SEM', 'FUN']

    def _compute_normalized_laplacian(self, W):
        """
        Calcula el Laplaciano Normalizado: L = I - D^-1/2 * W * D^-1/2
        Es robusto para grafos con diferentes escalas de densidad.
        """
        # Asegurar simetría (por si acaso) y eliminar diagonal negativa
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0) 
        
        # Grados
        degrees = np.sum(W, axis=1)
        
        # Manejo seguro de división por cero (nodos desconectados)
        d_inv_sqrt = np.zeros_like(degrees)
        mask = degrees > 0
        d_inv_sqrt[mask] = np.power(degrees[mask], -0.5)
        
        D_inv_sqrt = np.diag(d_inv_sqrt)
        I = np.eye(self.n)
        
        # L = I - D^-0.5 * W * D^-0.5
        L = I - np.dot(np.dot(D_inv_sqrt, W), D_inv_sqrt)
        return L

    def _get_consensus_eigenvectors(self, U, k):
        """
        Obtiene los k autovectores (H) correspondientes a los autovalores más pequeños
        del Laplaciano de U. H representa la estructura de clúster latente.
        """
        L_u = self._compute_normalized_laplacian(U)
        
        # eigh es eficiente para matrices simétricas
        # subset_by_index=[0, k-1] obtiene los k menores autovalores
        vals, vecs = eigh(L_u, subset_by_index=[0, k-1])
        return vecs

    def run_optimization(self, k_clusters=5, max_iter=20, tol=1e-4):
        """
        Bucle principal de optimización de pesos.
        """
        print(f"[Fusion] Iniciando optimización iterativa (K={k_clusters})...")
        
        # Inicialización uniforme de pesos
        num_views = 3
        weights = np.ones(num_views) / num_views
        
        history = [] # Para guardar log de convergencia

        # Pre-calcular Laplacianos de vistas individuales para ahorrar cómputo
        # L_v no cambia, solo U cambia.
        L_views = [self._compute_normalized_laplacian(A) for A in self.matrices]

        for it in range(max_iter):
            # PASO 1: Calcular Matriz Unificada (U) con pesos actuales
            U = weights[0]*self.matrices[0] + weights[1]*self.matrices[1] + weights[2]*self.matrices[2]
            
            # PASO 2: Obtener Consenso (H) de U
            H = self._get_consensus_eigenvectors(U, k_clusters)
            
            # PASO 3: Calcular "Desacuerdo" (Loss) de cada vista respecto a H
            # Loss = Traza(H.T * L_v * H). Menor traza = Mejor vista.
            view_losses = []
            for i in range(num_views):
                # Traza eficiente: sum(sum(H * (L @ H)))
                loss = np.trace(np.dot(H.T, np.dot(L_views[i], H)))
                view_losses.append(loss)
            
            view_losses = np.array(view_losses)
            
            # PASO 4: Actualizar Pesos
            # Usamos una función exponencial inversa para penalizar losses altos.
            # Exponente gamma > 1 agudiza la selección (hace al algoritmo más "exigente")
            gamma = 1.5 # 2.0
            
            # Evitar overflow numérico restando el mínimo
            exp_vals = np.exp(-gamma * (view_losses - np.min(view_losses)))
            new_weights = exp_vals / np.sum(exp_vals)
            
            # Logging
            diff = np.linalg.norm(new_weights - weights)
            print(f"   Iter {it+1}: W_STR={new_weights[0]:.3f}, W_SEM={new_weights[1]:.3f}, W_FUN={new_weights[2]:.3f} (Diff: {diff:.6f})")
            
            history.append(list(new_weights) + list(view_losses))

            # Convergencia
            if diff < tol:
                print(f"✅ Convergencia alcanzada en la iteración {it+1}.")
                weights = new_weights
                break
            
            weights = new_weights

        # Construir U final
        self.U_final = weights[0]*self.matrices[0] + weights[1]*self.matrices[1] + weights[2]*self.matrices[2]
        self.final_weights = weights
        self.history = history

    def save_results(self, output_path):
        # 1. Guardar Matriz Fusionada
        df_u = pd.DataFrame(self.U_final, index=self.classes, columns=self.classes)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_u.to_csv(output_path)
        print(f"[Fusion] Matriz Unificada guardada en: {output_path}")

        # Guarda Heatmap de la Matriz Unificada
        png_path = output_path.replace(".csv", "_heatmap.png")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(self.U_final, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Fuerza de Acoplamiento')
        plt.title('Matriz Unificada (U) - Fusión Multivista')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[Fusion] Heatmap de la Matriz Unificada guardado en: {png_path}")
        
        # 2. Guardar Log de Pesos (cambiando extensión)
        log_path = output_path.replace(".csv", "_convergence_log.csv")
        cols = ['w_str', 'w_sem', 'w_fun', 'loss_str', 'loss_sem', 'loss_fun']
        df_log = pd.DataFrame(self.history, columns=cols)
        df_log.to_csv(log_path, index_label="iteration")
        print(f"[Fusion] Log de convergencia guardado en: {log_path}")

def main():
    if len(sys.argv) != 6:
        print("Uso: python fase3_multiview_fusion.py <STR> <SEM> <FUN> <OUTPUT> <K_TARGET>")
        sys.exit(1)

    str_csv = sys.argv[1]
    sem_csv = sys.argv[2]
    fun_csv = sys.argv[3]
    out_csv = sys.argv[4]

    k_target = int(sys.argv[5])

    if not os.path.exists(str_csv):
        print(f"Error: No existe {str_csv}")
        sys.exit(1)

    fusion = MultiViewFusion(str_csv, sem_csv, fun_csv)
    
    fusion.run_optimization(k_clusters=k_target)

    fusion.save_results(out_csv)

if __name__ == '__main__':
    main()