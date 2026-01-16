#!/usr/bin/env python3

"""
Extrae la vista funcional de un monolito Java basada en co-ocurrencias 
dentro de los controladores (simulando lógica de endpoints).

Uso:
    python extract_functional_view.py <PATH_SEMANTIC_CSV> <PATH_PROJECT_JAVA> <OUTPUT_CSV>

Argumentos:
    <PATH_SEMANTIC_CSV>: Ruta a un CSV que contenga la lista de clases del núcleo (ej. semantic_view.csv o structural_filtered.csv).
    <PATH_PROJECT_JAVA>: Ruta raíz del proyecto Java.
    <OUTPUT_CSV>: Ruta donde se guardará el CSV de co-ocurrencias funcionales.
"""

import sys
import os
import csv
import re
import itertools
from collections import defaultdict

class FunctionalExtractor:
    def __init__(self, allowed_classes_csv, project_root):
        self.project_root = project_root
        # Cargar el conjunto de clases permitidas (núcleo funcional)
        self.allowed_classes = self._load_allowed_classes(allowed_classes_csv)
        
        # Mapa: "NombreControlador" -> {"ClaseA", "ClaseB", ...}
        self.controller_dependencies = defaultdict(set)
        
        # Matriz de co-ocurrencia: "ClaseA" -> {"ClaseB": 5, ...}
        self.co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
        
        print(f"[Functional] Núcleo cargado: {len(self.allowed_classes)} clases.")

    def _load_allowed_classes(self, csv_path):
        """Carga los nombres de las clases del proyecto desde un CSV existente."""
        allowed = set()
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Detectar nombre de la columna de clases (puede ser 'class' o 'class_name')
            class_col = 'class' if 'class' in reader.fieldnames else 'class_name'
            
            for row in reader:
                allowed.add(row[class_col])
        return allowed

    def _is_controller(self, file_path):
        """
        Determina si un archivo es un controlador/endpoint basándose en su ruta.
        Replica la lógica del código C++: busca en paquetes 'web/spring' o 'web/struts'.
        """
        # Normalizar separadores para consistencia
        path_str = file_path.replace('\\', '/')
        
        # Heurística: Buscar palabras clave en la ruta del paquete
        is_web_package = ('web/spring' in path_str) or ('web/struts' in path_str)
        
        # Heurística adicional: Buscar sufijos comunes si la ruta no es explícita
        is_controller_suffix = path_str.endswith('Controller.java') or path_str.endswith('Action.java')
        
        return is_web_package or is_controller_suffix

    def _remove_comments(self, content):
        """Elimina comentarios de bloque (/* */) y de línea (//)."""
        # Eliminar /* ... */
        content = re.sub(r'/\*[\s\S]*?\*/', '', content)
        # Eliminar // ...
        content = re.sub(r'//.*', '', content)
        return content

    def _find_class_dependencies(self, content, self_name):
        """
        Escanea el contenido en busca de menciones a otras clases del proyecto.
        """
        dependencies = set()
        
        # Regex para encontrar palabras que parecen clases (Capitalizada + alfanumérico)
        # \b asegura límites de palabra completa.
        type_regex = r'\b([A-Z][a-zA-Z0-9_]+)\b'
        
        matches = re.findall(type_regex, content)
        
        for potential_class in matches:
            # Si la palabra encontrada es una de nuestras clases del núcleo
            # Y no es la propia clase que estamos analizando
            if potential_class in self.allowed_classes and potential_class != self_name:
                dependencies.add(potential_class)
                
        return dependencies

    def _process_file(self, file_path):
        """Procesa un único archivo .java."""
        if not self._is_controller(file_path):
            return

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: No se pudo leer {file_path}: {e}")
            return

        content = self._remove_comments(content)
        
        # Obtener nombre de la clase (controlador) actual
        # Asumimos que el nombre del archivo es el nombre de la clase
        controller_name = os.path.basename(file_path).replace('.java', '')
        
        # Buscar dependencias funcionales
        deps = self._find_class_dependencies(content, controller_name)
        
        if deps:
            self.controller_dependencies[controller_name] = deps
            # print(f"  -> Controlador '{controller_name}' conecta: {deps}")

    def find_controller_dependencies(self):
        """Recorre el directorio del proyecto buscando controladores."""
        print(f"[Functional] Escaneando controladores en: {self.project_root}")
        
        for dirpath, _, filenames in os.walk(self.project_root):
            for filename in filenames:
                if filename.endswith('.java'):
                    full_path = os.path.join(dirpath, filename)
                    self._process_file(full_path)
        
        print(f"[Functional] Se encontraron {len(self.controller_dependencies)} controladores con dependencias al núcleo.")

    def calculate_co_occurrences(self):
        """Construye la matriz de co-ocurrencia funcional."""
        print("[Functional] Calculando co-ocurrencias...")
        
        for controller, deps in self.controller_dependencies.items():
            # Generar todos los pares posibles entre las dependencias de este controlador
            # Si ControllerA usa {ClaseX, ClaseY, ClaseZ}, entonces (X,Y), (X,Z), (Y,Z) co-ocurren.
            for class_a, class_b in itertools.combinations(deps, 2):
                # Matriz simétrica
                self.co_occurrence_matrix[class_a][class_b] += 1
                self.co_occurrence_matrix[class_b][class_a] += 1

    def save_results(self, output_path):
        """Guarda los resultados en el formato CSV esperado."""
        
        # Asegurar directorio de salida
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'functional_co_occurrences'])
            
            # Iterar sobre todas las clases del núcleo para asegurar que aparecen en el CSV
            # aunque no tengan co-ocurrencias (fila vacía).
            for class_name in sorted(self.allowed_classes):
                
                co_occurrences = self.co_occurrence_matrix.get(class_name, {})
                
                # Convertir defaultdict a dict normal para impresión limpia
                co_occurrences_dict = dict(co_occurrences)
                
                writer.writerow([class_name, str(co_occurrences_dict)])
                
        print(f"[Functional] Vista funcional guardada en: {output_path}")

    def run(self, output_path):
        self.find_controller_dependencies()
        self.calculate_co_occurrences()
        self.save_results(output_path)


def main():
    if len(sys.argv) != 4:
        print("Uso: python extract_functional_view.py <PATH_ALLOWED_CLASSES_CSV> <PATH_PROJECT_JAVA> <OUTPUT_CSV>")
        print("Ejemplo: python extract_functional_view.py jpetstore_fase2_structural_view_filtered.csv ../monoliths/jPetStore/jpetstore_fase1_functional_view.csv")
        sys.exit(1)

    allowed_classes_csv = sys.argv[1]
    project_root = sys.argv[2]
    output_csv = sys.argv[3]
    
    if not os.path.exists(allowed_classes_csv):
        print(f"Error: El CSV de clases permitidas no existe: {allowed_classes_csv}")
        sys.exit(1)
        
    if not os.path.isdir(project_root):
        print(f"Error: El directorio del proyecto no existe: {project_root}")
        sys.exit(1)

    extractor = FunctionalExtractor(allowed_classes_csv, project_root)
    extractor.run(output_csv)

if __name__ == '__main__':
    main()