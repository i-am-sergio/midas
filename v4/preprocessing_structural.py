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

    preprocessor = Preprocessor(
        all_class_data,
        suffixes=FILTER_SUFFIXES,
        prefixes=FILTER_PREFIXES
    )
    
    preprocessor.process_and_save(output_path)
    
    print("\n[Completado] Preprocesamiento genérico finalizado.")


if __name__ == '__main__':
    main()