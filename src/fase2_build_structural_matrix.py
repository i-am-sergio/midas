# fase2_build_structural_matrix.py

"""
FASE 2.1: Construcción de la Matriz Estructural (A_str).

Transforma el CSV de relaciones crudas (pesos 15, 8, 5...) en una
matriz de adyacencia simétrica y normalizada [0, 1].

Uso:
    python fase2_build_structural_matrix.py <STRUCTURAL_RAW_CSV> <CORE_CLASSES_CSV> <OUTPUT_BASE_NAME>
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

class StructuralMatrixBuilder:
    def __init__(self, raw_relations_path, core_csv_path):
        print(f"[Structural] Cargando relaciones desde: {raw_relations_path}")
        
        self.df_raw = pd.read_csv(raw_relations_path)
        self.df_core = pd.read_csv(core_csv_path)
        
        # 1. Definir el NÚCLEO (Orden estricto de filas/columnas)
        # Usamos la lista 'class' del archivo core como la verdad absoluta
        self.class_names = self.df_core['class'].tolist()
        self.n = len(self.class_names)
        
        # Mapa rápido de nombre -> índice
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"[Structural] Matriz inicializada: {self.n}x{self.n} clases.")

    def build_adjacency_matrix(self):
        """Construye la matriz NxN basada en los pesos de relaciones."""
        adj_matrix = np.zeros((self.n, self.n))
        
        # Iterar sobre el archivo crudo de relaciones
        for _, row in self.df_raw.iterrows():
            source_class = row['class']
            
            # Si la clase fuente no está en el núcleo, la ignoramos
            if source_class not in self.class_to_index:
                continue
                
            source_idx = self.class_to_index[source_class]
            
            # Parsear el string de relaciones "{'Target': 15, ...}"
            try:
                relations = ast.literal_eval(row['relations'])
            except Exception as e:
                print(f"⚠️ Error parseando relaciones de {source_class}: {e}")
                relations = {}
            
            for target_class, weight in relations.items():
                # Si la clase destino es parte del núcleo, añadimos el peso
                if target_class in self.class_to_index:
                    target_idx = self.class_to_index[target_class]
                    adj_matrix[source_idx][target_idx] = weight

        return adj_matrix

    def process_matrix(self, adj_matrix):
        """Simetrización y Normalización."""
        print("[Structural] Procesando matriz (Simetrización + Normalización)...")
        
        # 1. Simetrización
        # Si A llama a B, existe un acoplamiento bidireccional.
        # A_sym = A + A.T (Sumamos pesos para reforzar la conexión fuerte)
        # Tu ejemplo usaba 0.5 * (...), que es un promedio. La suma es equivalente en escala relativa.
        # Usaremos la suma para mantener la "fuerza" del enlace.
        symmetric_matrix = adj_matrix + adj_matrix.T
        
        # 2. Normalización Min-Max [0, 1]
        # Esto es CRUCIAL para que la vista estructural sea comparable con 
        # la semántica (Coseno 0-1) en la fase de fusión.
        max_val = np.max(symmetric_matrix)
        
        if max_val > 0:
            normalized_matrix = symmetric_matrix / max_val
        else:
            normalized_matrix = symmetric_matrix

        # 3. FILTRADO DE CONEXIONES DÉBILES (Noise Reduction)
        # Esto ayuda a romper el efecto "God Class" y aislar utilidades.
        prev_edges = np.count_nonzero(normalized_matrix)
        WEAK_LINK_THRESHOLD = 0.1
        normalized_matrix[normalized_matrix < WEAK_LINK_THRESHOLD] = 0
        
        curr_edges = np.count_nonzero(normalized_matrix)
        removed = prev_edges - curr_edges
        
        if removed > 0:
            print(f"   Se eliminaron {removed} conexiones débiles (Threshold < {WEAK_LINK_THRESHOLD}).")
            print(f"   Conexiones restantes: {curr_edges}")

        # Asegurar diagonal en 0 (una clase no se 'depende' de sí misma para clustering)
        # Nota: En vista semántica la diagonal es 1. En estructural suele ser 0.
        # Esto no afecta mucho al clustering espectral (L = D - W), pero es más limpio.
        np.fill_diagonal(normalized_matrix, 0.0)
        
        return normalized_matrix

    def save_results(self, matrix, output_base):
        csv_path = f"{output_base}_matrix.csv"
        png_path = f"{output_base}_matrix.png"
        
        # Guardar CSV
        df_matrix = pd.DataFrame(matrix, index=self.class_names, columns=self.class_names)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_matrix.to_csv(csv_path)
        print(f"[Structural] Matriz guardada: {csv_path}")
        
        # Guardar Heatmap
        plt.figure(figsize=(10, 8))
        # Usamos 'viridis' para consistencia
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Fuerza de Acoplamiento (Normalizada)')
        plt.title(f'Matriz Estructural (A_str) - {self.n}x{self.n}')
        
        # Etiquetas si no son demasiadas
        if self.n <= 40:
            plt.xticks(ticks=np.arange(self.n), labels=self.class_names, rotation=90, fontsize=8)
            plt.yticks(ticks=np.arange(self.n), labels=self.class_names, fontsize=8)
            
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[Structural] Heatmap guardado: {png_path}")

def main():
    # Argumentos alineados con midas.sh
    if len(sys.argv) != 4:
        print("Uso: python fase2_build_structural_matrix.py <STRUCTURAL_RAW> <CORE_CSV> <OUTPUT_BASE>")
        sys.exit(1)

    raw_csv = sys.argv[1]
    core_csv = sys.argv[2]
    output_base = sys.argv[3]

    if not os.path.exists(raw_csv):
        print(f"Error: No existe el archivo de relaciones crudas: {raw_csv}")
        sys.exit(1)

    builder = StructuralMatrixBuilder(raw_csv, core_csv)
    
    # Construir
    adj_matrix = builder.build_adjacency_matrix()
    
    # Procesar
    final_matrix = builder.process_matrix(adj_matrix)
    
    # Guardar
    builder.save_results(final_matrix, output_base)

if __name__ == '__main__':
    main()