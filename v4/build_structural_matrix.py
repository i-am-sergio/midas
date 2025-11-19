#!/usr/bin/env python3

"""
Fase 2 (Parte B): Construcción de la Matriz Estructural.

Toma el CSV filtrado (núcleo funcional) y construye la
matriz de adyacencia A^(str) simétrica y normalizada.

Uso:
    python build_structural_matrix.py <INPUT_FILTERED_CSV> <OUTPUT_BASE_NAME>

Argumentos:
    <INPUT_FILTERED_CSV>: 
        Ruta al CSV filtrado (ej. jPetStore_fase2_structural_view_filtered.csv)
    <OUTPUT_BASE_NAME>: 
        Nombre base para los archivos de salida (ej. jpetstore_structural)

Genera:
    1. <OUTPUT_BASE_NAME>_matrix.csv: Matriz NxN normalizada.
    2. <OUTPUT_BASE_NAME>_matrix.png: Heatmap de la matriz.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

class MatrixBuilder:
    """
    Construye la matriz de adyacencia simétrica y normalizada
    a partir del grafo de relaciones filtrado.
    """

    def __init__(self, filtered_csv_path):
        self.df = pd.read_csv(filtered_csv_path)
        
        # Ordenamos las clases para un orden de matriz consistente
        self.class_names = sorted(list(self.df['class']))
        self.n = len(self.class_names)
        
        # Mapa de 'NombreClase' -> índice_matriz
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        
        # Inicializa la matriz N x N con ceros
        self.matrix = np.zeros((self.n, self.n))
        self.normalized_matrix = None
        
        print(f"[MatrixBuilder] Iniciando construcción de matriz {self.n}x{self.n}.")

    def build_matrix(self):
        """Puebla la matriz (A) desde el CSV (no simétrica aún)."""
        
        for _, row in self.df.iterrows():
            source_class = row['class']
            relations = ast.literal_eval(row['relations'])
            
            i = self.class_to_index[source_class]
            
            for target_class, score in relations.items():
                # El preprocesamiento anterior ya debería haber
                # eliminado los targets que no están en el núcleo.
                if target_class in self.class_to_index:
                    j = self.class_to_index[target_class]
                    self.matrix[i, j] = score
        
        print("[MatrixBuilder] Matriz de adyacencia inicial construida.")

    def symmetrize(self):
        """
        Hace la matriz simétrica sumando su transpuesta (A = A + A.T).
        Esto asegura que la relación (i, j) = (j, i).
        """
        print("[MatrixBuilder] Simetrizando matriz (A + A.T)...")
        self.matrix = self.matrix + self.matrix.T
        
        # Nos aseguramos de que la diagonal sea 0
        np.fill_diagonal(self.matrix, 0)

    def normalize(self):
        """Normaliza la matriz en el rango [0, 1] (Min-Max)."""
        print("[MatrixBuilder] Normalizando matriz [0, 1]...")
        
        max_val = np.max(self.matrix)
        
        if max_val > 0:
            self.normalized_matrix = self.matrix / max_val
        else:
            # Evitar división por cero si la matriz es todo ceros
            self.normalized_matrix = self.matrix

    def save_matrix_csv(self, output_matrix_csv):
        """Guarda la matriz normalizada en un CSV con cabeceras."""
        
        # Usamos pandas para guardar con nombres de filas/columnas
        matrix_df = pd.DataFrame(
            self.normalized_matrix,
            index=self.class_names,
            columns=self.class_names
        )
        
        matrix_df.to_csv(output_matrix_csv)
        print(f"[MatrixBuilder] Matriz normalizada guardada en: {output_matrix_csv}")
        
    def save_matrix_png(self, output_png):
        """Guarda una visualización (heatmap) de la matriz en PNG."""
        
        # Ajustar tamaño de la figura basado en N
        fig_size = max(8, self.n * 0.4) 
        
        plt.figure(figsize=(fig_size, fig_size))
        
        # Usamos 'viridis' para un buen contraste
        plt.imshow(self.normalized_matrix, cmap='viridis', interpolation='nearest')
        
        plt.title(f'Matriz Estructural (A_str) - {self.n}x{self.n}')
        plt.colorbar(label='Puntuación Normalizada [0, 1]')
        
        # Añadir etiquetas de clases a los ejes X e Y
        plt.xticks(ticks=np.arange(self.n), labels=self.class_names, rotation=90, fontsize=8)
        plt.yticks(ticks=np.arange(self.n), labels=self.class_names, fontsize=8)
        
        plt.tight_layout() # Ajusta el padding
        plt.savefig(output_png, dpi=150)
        print(f"[MatrixBuilder] Imagen de la matriz guardada en: {output_png}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python build_structural_matrix.py <INPUT_FILTERED_CSV> <OUTPUT_BASE_NAME>")
        print("Ejemplo: python build_structural_matrix.py jPetStore_fase2_structural_view_filtered.csv jpetstore_structural")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_base_name = sys.argv[2]
    
    # Definir los nombres de los archivos de salida
    output_matrix_csv = f"{output_base_name}_matrix.csv"
    output_png = f"{output_base_name}_matrix.png"

    if not os.path.exists(input_csv):
        print(f"Error: El archivo de entrada no existe: {input_csv}")
        sys.exit(1)

    # --- Ejecutar el Pipeline ---
    builder = MatrixBuilder(filtered_csv_path=input_csv)
    builder.build_matrix()
    builder.symmetrize()
    builder.normalize()
    
    # --- Guardar Resultados ---
    builder.save_matrix_csv(output_matrix_csv)
    builder.save_matrix_png(output_png)
    
    print("\n[Completado] Matriz estructural generada exitosamente.")

if __name__ == '__main__':
    main()