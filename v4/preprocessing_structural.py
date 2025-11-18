#!/usr/bin/env python3

"""
Fase 2: Preprocesamiento Estructural Genérico (Filtro por Patrones)

Uso:
    python preprocessing_structural.py <INPUT_CSV> <OUTPUT_FILTERED_CSV>
"""

import sys
import os
import csv
import ast

# -----------------------------------------------------------------------------
# Reglas de Filtro Genéricas
# -----------------------------------------------------------------------------
FILTER_SUFFIXES = [
    'Action', 
    'Form', 
    'Controller', 
    'Validator', 
    'Interceptor', 
    'Client', 
    'Advice', 
    'Session',
    'Servlet'
]

FILTER_PREFIXES = [
    'Base',
    'Secure',
    'MsSql',
    'Oracle',
    'JaxRpc',
    'Mock'
]


class Preprocessor:
    """
    Filtra relaciones estructurales basándose en reglas de patrones
    para descubrir y aislar el núcleo funcional.
    """
    
    # ESTE ES EL __init__ CORRECTO QUE COINCIDE CON main()
    def __init__(self, all_class_data, suffixes, prefixes):
        self.all_data = all_class_data
        self.filter_suffixes = suffixes
        self.filter_prefixes = prefixes
        self.core_classes = self._discover_core_classes()

    def _is_filtered_out(self, class_name):
        """Comprueba si un nombre de clase coincide con alguna regla de filtro."""
        for suffix in self.filter_suffixes:
            if class_name.endswith(suffix):
                return True
        for prefix in self.filter_prefixes:
            if class_name.startswith(prefix):
                return True
        return False

    def _discover_core_classes(self):
        """Genera el conjunto de clases del 'núcleo'."""
        core_set = set()
        total_count = 0
        
        for row in self.all_data:
            total_count += 1
            class_name = row['class']
            if not self._is_filtered_out(class_name):
                core_set.add(class_name)
                
        print(f"[Preprocessor] {total_count} clases de entrada.")
        print(f"[Preprocessor] {len(core_set)} clases descubiertas como 'Núcleo Funcional'.")
        print(f"[Preprocessor] {total_count - len(core_set)} clases filtradas (UI, Framework, etc.).")
        return core_set

    def _filter_relations(self, relations_str):
        """Filtra un diccionario de relaciones."""
        relations_dict = ast.literal_eval(relations_str)
        
        filtered_dict = {
            target_class: score
            for target_class, score in relations_dict.items()
            if target_class in self.core_classes
        }
        
        return str(filtered_dict)

    def process_and_save(self, output_csv_path):
        """
        Procesa la lista completa de clases y guarda solo las filas
        del núcleo, con sus relaciones también filtradas.
        """
        filtered_data = []

        for row in self.all_data:
            source_class = row['class']
            
            if source_class not in self.core_classes:
                continue
            
            relations_str = row['relations']
            filtered_relations_str = self._filter_relations(relations_str)
            
            filtered_data.append({
                'class': source_class,
                'relations': filtered_relations_str
            })

        print(f"Guardando {len(filtered_data)} clases del núcleo en: {output_csv_path}")
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            fieldnames = ['class', 'relations']
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_data)


def main():
    if len(sys.argv) != 3:
        print("Uso: python preprocessing_structural_generic.py <INPUT_CSV> <OUTPUT_FILTERED_CSV>")
        print("Ejemplo: python preprocessing_structural_generic.py jPetStore_fase1_structural_view.csv jPetStore_fase2_structural_view_filtered.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: El archivo de entrada no existe: {input_path}")
        sys.exit(1)

    # Cargamos todos los datos en memoria para poder descubrir el núcleo primero
    all_class_data = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            all_class_data.append(row)
    
    if not all_class_data:
        print(f"Error: El archivo de entrada está vacío: {input_path}")
        sys.exit(1)

    # --- INICIO DE LA CORRECCIÓN ---
    #
    # Corregir la instanciación. 
    # Opción 1 (Posicional):
    preprocessor = Preprocessor(
        all_class_data,
        suffixes=FILTER_SUFFIXES,
        prefixes=FILTER_PREFIXES
    )
    #
    # Opción 2 (Keyword - también correcta):
    # preprocessor = Preprocessor(
    #     all_class_data=all_class_data, # <-- Usando un solo =
    #     suffixes=FILTER_SUFFIXES,
    #     prefixes=FILTER_PREFIXES
    # )
    #
    # --- FIN DE LA CORRECCIÓN ---
    
    preprocessor.process_and_save(output_path)
    
    print("\n[Completado] Preprocesamiento genérico finalizado.")


if __name__ == '__main__':
    main()

# class MatrixBuilder:
#     """
#     Construye la matriz de adyacencia simétrica y normalizada
#     a partir del grafo de relaciones filtrado.
#     """

#     def __init__(self, filtered_csv_path, core_classes):
#         self.df = pd.read_csv(filtered_csv_path)
#         # Ordenamos las clases para tener un orden consistente en la matriz
#         self.class_names = sorted(list(core_classes))
#         self.n = len(self.class_names)
#         # Creamos un mapa de 'NombreClase' -> índice_matriz
#         self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
#         self.matrix = np.zeros((self.n, self.n))
#         self.normalized_matrix = None
#         print(f"[MatrixBuilder] Iniciando construcción de matriz {self.n}x{self.n}.")

#     def build_matrix(self):
#         """Construye la matriz de adyacencia inicial (no simétrica)."""
        
#         # Llenamos la matriz (A)
#         for _, row in self.df.iterrows():
#             source_class = row['class']
#             # Omitimos try-catch como se solicitó
#             relations = ast.literal_eval(row['filtered_relations'])
            
#             # Solo si la clase fuente está en nuestro índice (debería estarlo siempre)
#             if source_class in self.class_to_index:
#                 i = self.class_to_index[source_class]
                
#                 for target_class, score in relations.items():
#                     # Solo si la clase destino está en nuestro índice
#                     if target_class in self.class_to_index:
#                         j = self.class_to_index[target_class]
#                         self.matrix[i, j] = score

#     def symmetrize(self):
#         """
#         Hace la matriz simétrica. La relación (i, j) será la suma
#         de las puntuaciones (i -> j) y (j -> i).
        
#         Esto crea la matriz A^(str) = A + A.T
#         """
#         print("[MatrixBuilder] Simetrizando matriz (A + A.T)...")
#         self.matrix = self.matrix + self.matrix.T
        
#         # Nos aseguramos de que la diagonal sea 0
#         # (Aunque A + A.T ya debería tener 0s si A los tenía)
#         np.fill_diagonal(self.matrix, 0)

#     def normalize(self):
#         """Normaliza la matriz en el rango [0, 1] por Min-Max."""
#         print("[MatrixBuilder] Normalizando matriz [0, 1]...")
#         max_val = np.max(self.matrix)
        
#         if max_val > 0:
#             self.normalized_matrix = self.matrix / max_val
#         else:
#             # Evitar división por cero si la matriz está vacía o es todo 0s
#             self.normalized_matrix = self.matrix
            
#     def save_matrix_csv(self, output_matrix_csv):
#         """Guarda la matriz normalizada en un CSV con cabeceras."""
        
#         # Creamos un DataFrame de Pandas para guardarlo con
#         # los nombres de las clases como cabeceras de fila y columna
#         matrix_df = pd.DataFrame(
#             self.normalized_matrix,
#             index=self.class_names,
#             columns=self.class_names
#         )
        
#         matrix_df.to_csv(output_matrix_csv)
#         print(f"[MatrixBuilder] Matriz normalizada guardada en: {output_matrix_csv}")
        
#     def save_matrix_png(self, output_png):
#         """Guarda una visualización de la matriz como un heatmap PNG."""
        
#         # Ajustar tamaño de la figura basado en N
#         # (para que no sea muy pequeña o muy grande)
#         fig_size = max(8, self.n * 0.5) 
        
#         plt.figure(figsize=(fig_size, fig_size))
        
#         # Usamos 'viridis' como mapa de color
#         plt.imshow(self.normalized_matrix, cmap='viridis', interpolation='nearest')
        
#         plt.title(f'Matriz de Adyacencia Estructural (A_str) - {self.n}x{self.n}')
#         plt.colorbar(label='Puntuación Normalizada')
        
#         # Opcional: Añadir etiquetas si la matriz no es demasiado grande
#         if self.n <= 30:
#             plt.xticks(ticks=np.arange(self.n), labels=self.class_names, rotation=90, fontsize=8)
#             plt.yticks(ticks=np.arange(self.n), labels=self.class_names, fontsize=8)
        
#         plt.tight_layout() # Ajusta el padding
#         plt.savefig(output_png, dpi=150)
#         print(f"[MatrixBuilder] Imagen de la matriz guardada en: {output_png}")


# def main():
#     if len(sys.argv) != 3:
#         print("Uso: python preprocessing_structural.py <INPUT_CSV> <OUTPUT_BASE_NAME>")
#         print("Ejemplo: python preprocessing_structural.py jPetStore_fase1_structural_view.csv jpetstore_structural")
#         sys.exit(1)

#     input_csv = sys.argv[1]
#     output_base_name = sys.argv[2]
    
#     # Definir los nombres de los archivos de salida
#     output_filtered_csv = f"{output_base_name}_filtered.csv"
#     output_matrix_csv = f"{output_base_name}_matrix.csv"
#     output_png = f"{output_base_name}_matrix.png"

#     # --- Fase 1: Limpieza ---
#     cleaner = Cleaner(input_csv_path=input_csv)
#     filtered_csv, core_classes = cleaner.clean_and_save(output_filtered_csv)

#     # --- Fase 2: Construcción de Matriz ---
#     builder = MatrixBuilder(filtered_csv_path=filtered_csv, core_classes=core_classes)
#     builder.build_matrix()
#     builder.symmetrize()
#     builder.normalize()
#     builder.save_matrix_csv(output_matrix_csv)
#     builder.save_matrix_png(output_png)
    
#     print("\n[Completado] Preprocesamiento de vista estructural finalizado.")

# if __name__ == '__main__':
#     main()