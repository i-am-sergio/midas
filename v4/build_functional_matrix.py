#!/usr/bin/env python3

"""
Fase 2 (Vista Funcional): Construcción de la Matriz Funcional.

Toma el CSV de co-ocurrencias funcionales y construye la
matriz de adyacencia A^(fun) normalizada.

Uso:
    python build_functional_matrix.py <INPUT_CO_OCCURRENCE_CSV> <OUTPUT_BASE_NAME>

Argumentos:
    <INPUT_CO_OCCURRENCE_CSV>: 
        Ruta al CSV generado por extract_functional_view.py
        (ej. jpetstore_fase1_functional_view.csv)
    <OUTPUT_BASE_NAME>: 
        Nombre base para los archivos de salida (ej. jpetstore_functional)

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

class FunctionalMatrixBuilder:
    """
    Construye la matriz funcional A^(fun) a partir de datos de co-ocurrencia.
    """

    def __init__(self, input_csv_path):
        self.df = pd.read_csv(input_csv_path)
        
        # Ordenamos las clases alfabéticamente para garantizar consistencia
        # con las otras vistas (estructural y semántica).
        self.class_names = sorted(list(self.df['class']))
        self.n = len(self.class_names)
        
        # Mapa de 'NombreClase' -> índice_matriz
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        
        # Inicializa la matriz N x N con ceros
        self.matrix = np.zeros((self.n, self.n))
        self.normalized_matrix = None
        
        print(f"[FunBuilder] Iniciando construcción de matriz {self.n}x{self.n}.")

    def build_matrix(self):
        """
        Puebla la matriz desde el CSV.
        Aunque el extractor ya genera datos simétricos, aquí nos aseguramos
        de poblarla correctamente.
        """
        
        for _, row in self.df.iterrows():
            source_class = row['class']
            # El campo 'functional_co_occurrences' es un string que representa un diccionario
            co_occurrences = ast.literal_eval(row['functional_co_occurrences'])
            
            if source_class in self.class_to_index:
                i = self.class_to_index[source_class]
                
                for target_class, weight in co_occurrences.items():
                    if target_class in self.class_to_index:
                        j = self.class_to_index[target_class]
                        self.matrix[i, j] = weight
        
        print("[FunBuilder] Matriz de co-ocurrencia poblada.")

    def normalize(self):
        """
        Normaliza la matriz en el rango [0, 1] (Min-Max Scaling).
        La diagonal principal se mantiene en 0 (una clase no co-ocurre consigo misma
        en el sentido de dependencia funcional externa).
        """
        print("[FunBuilder] Normalizando matriz [0, 1]...")
        
        # Asegurar que la diagonal sea 0
        np.fill_diagonal(self.matrix, 0)
        
        max_val = np.max(self.matrix)
        
        if max_val > 0:
            self.normalized_matrix = self.matrix / max_val
        else:
            self.normalized_matrix = self.matrix

    def save_matrix_csv(self, output_matrix_csv):
        """Guarda la matriz normalizada en un CSV."""
        
        matrix_df = pd.DataFrame(
            self.normalized_matrix,
            index=self.class_names,
            columns=self.class_names
        )
        
        os.makedirs(os.path.dirname(output_matrix_csv), exist_ok=True)
        matrix_df.to_csv(output_matrix_csv)
        print(f"[FunBuilder] Matriz normalizada guardada en: {output_matrix_csv}")
        
    def save_matrix_png(self, output_png):
        """Guarda una visualización (heatmap) de la matriz."""
        
        fig_size = max(8, self.n * 0.4) 
        plt.figure(figsize=(fig_size, fig_size))
        
        # Usamos 'inferno' para distinguir visualmente de la vista estructural/semántica
        plt.imshow(self.normalized_matrix, cmap='inferno', interpolation='nearest')
        
        plt.title(f'Matriz Funcional (A_fun) - {self.n}x{self.n}')
        plt.colorbar(label='Co-ocurrencia Normalizada')
        
        if self.n <= 40:
            plt.xticks(ticks=np.arange(self.n), labels=self.class_names, rotation=90, fontsize=8)
            plt.yticks(ticks=np.arange(self.n), labels=self.class_names, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        print(f"[FunBuilder] Heatmap guardado en: {output_png}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python build_functional_matrix.py <INPUT_CO_OCCURRENCE_CSV> <OUTPUT_BASE_NAME>")
        print("Ejemplo: python build_functional_matrix.py jpetstore_fase1_functional_view.csv jpetstore_functional")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_base_name = sys.argv[2]
    
    if not os.path.exists(input_csv):
        print(f"Error: El archivo de entrada no existe: {input_csv}")
        sys.exit(1)

    output_matrix_csv = f"{output_base_name}_matrix.csv"
    output_png = f"{output_base_name}_matrix.png"

    builder = FunctionalMatrixBuilder(input_csv_path=input_csv)
    builder.build_matrix()
    builder.normalize()
    builder.save_matrix_csv(output_matrix_csv)
    builder.save_matrix_png(output_png)
    
    print("\n[Completado] Matriz Funcional A^(fun) generada exitosamente.")

if __name__ == '__main__':
    main()