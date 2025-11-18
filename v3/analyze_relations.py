#!/usr/bin/env python3

"""
Analiza las relaciones estructurales desde un CSV de vista estructural y
los archivos fuente de Java.

Uso:
    python analyze_relations.py <input_path> <monolith_project> <output_path>

Argumentos:
    <input_path>:         Ruta al 'structural_view.csv' generado en la fase 1.
    <monolith_project>:   Ruta al directorio raíz del proyecto monolítico. 
                          (Requerido por paridad, aunque los paths del CSV 
                           ya son completos).
    <output_path>:        Ruta del archivo CSV donde se guardarán los resultados.
"""

import sys
import os
import csv
import re
from collections import defaultdict

class RelationAnalyzer:
    """
    Carga la vista estructural desde un CSV, lee los archivos Java 
    correspondientes y analiza las relaciones entre clases para
    generar una matriz de adyacencia ponderada.
    """

    # Puntuaciones basadas en el código C++
    INHERITANCE_SCORE = 15
    INSTANTIATION_SCORE = 8
    ATTRIBUTE_SCORE = 5
    METHOD_SIG_SCORE = 3

    def __init__(self, csv_path, project_root):
        """
        Inicializa el analizador.
        
        Args:
            csv_path (str): Ruta al archivo structural_view.csv.
            project_root (str): Ruta al directorio raíz del proyecto Java.
        """
        self.project_root = project_root
        self.classes_ = {}  # Almacenará la info de ClassInfo
        
        self._load_data_from_csv(csv_path)
        self._load_java_file_contents()

    def _load_data_from_csv(self, csv_path):
        """
        Carga los datos iniciales desde el archivo CSV generado.
        Maneja colisiones de nombres (ej. src/main vs src/test)
        dando prioridad a 'src/main'.
        """
        print(f"Loading data from {csv_path}...")
        
        rows_processed = 0
        collisions_ignored = 0
        collisions_overwritten = 0

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_processed += 1
                simple_name = row['class']
                new_filename = row['filename']
                
                new_data = {
                    'filename': new_filename,
                    'simpleName': simple_name,
                    'attributesStr': row['attributes'],
                    'methodsStr': row['methods'],
                    'fullContent': '' # Se cargará después
                }

                if simple_name not in self.classes_:
                    # Es la primera vez que vemos este nombre de clase.
                    self.classes_[simple_name] = new_data
                else:
                    # ¡Colisión! Ya hemos visto esta clase.
                    # Aplicar lógica de priorización.
                    current_filename = self.classes_[simple_name]['filename']
                    
                    # Regla de Prioridad: 'src/main' > cualquier otra cosa
                    # Asumimos que 'test' estará en 'src/test'
                    # (Convertimos a path canónico para asegurar consistencia en / o \)
                    is_current_main = 'src/main' in os.path.normpath(current_filename)
                    is_new_main = 'src/main' in os.path.normpath(new_filename)

                    if is_new_main and not is_current_main:
                        # El nuevo es 'main' y el actual no (ej. es 'test'). Reemplazar.
                        self.classes_[simple_name] = new_data
                        collisions_overwritten += 1
                    else:
                        # El actual es 'main' o ambos son 'test'/'other'.
                        # Nos quedamos con el primero que vimos.
                        collisions_ignored += 1
        
        print(f"CSV loading complete.")
        print(f"  - {rows_processed} rows processed from CSV.")
        print(f"  - {len(self.classes_)} unique classes loaded into memory.")
        if collisions_overwritten > 0:
            print(f"  - {collisions_overwritten} 'main' files replaced 'test' files.")
        if collisions_ignored > 0:
            print(f"  - {collisions_ignored} duplicate rows were ignored (kept first/main).")


    def _load_java_file_contents(self):
        """Lee el contenido completo de cada archivo .java en memoria."""
        print("Loading Java file contents...")
        count = 0
        for info in self.classes_.values():
            file_path = info['filename']
            # Omitimos try-catch como se solicitó
            with open(file_path, 'r', encoding='utf-8') as f:
                info['fullContent'] = f.read()
            count += 1
        print(f"Loaded content for {count} files.")

    def analyze_and_save(self, output_path):
        """
        Ejecuta todos los análisis y guarda los resultados en un CSV.
        
        Args:
            output_path (str): Ruta del archivo CSV de salida.
        """
        print("Analyzing relations...")
        # defaultdict(int) crea un contador (int, default 0) para cada clave
        relation_scores = defaultdict(lambda: defaultdict(int))

        for source_class_name, source_class_info in self.classes_.items():
            self._analyze_inheritance(source_class_name, source_class_info, relation_scores)
            self._analyze_signatures(source_class_name, source_class_info, relation_scores)
            self._analyze_body_instantiations(source_class_name, source_class_info, relation_scores)
        
        print("Analysis complete. Saving results...")
        self._save_results(output_path, relation_scores)

    def _analyze_inheritance(self, source_class, info, scores):
        """Analiza relaciones de 'extends' e 'implements'."""
        # Regex para encontrar 'extends ClassA' o 'implements InterfaceA, InterfaceB'
        regex = r'\b(?:extends|implements)\s+((?:\w+(?:,\s*\w+)*))'
        match = re.search(regex, info['fullContent'])
        
        if match:
            related_classes_str = match.group(1)
            # Divide la lista de clases por coma o espacio
            potential_classes = re.split(r'[,\s]+', related_classes_str)
            
            for class_name in potential_classes:
                if class_name and class_name in self.classes_:
                    scores[source_class][class_name] += self.INHERITANCE_SCORE

    def _analyze_signatures(self, source_class, info, scores):
        """Analiza tipos de atributos y signaturas de métodos."""
        combined_sigs = info['attributesStr'] + " " + info['methodsStr']
        
        # Regex para encontrar palabras en CamelCase (convención de tipos en Java)
        regex = r'\b([A-Z][a-zA-Z0-9_]+)\b'
        potential_types = re.findall(regex, combined_sigs)
        
        # Usamos set() para contar cada tipo único solo una vez
        for potential_class in set(potential_types):
            if potential_class in self.classes_ and potential_class != source_class:
                # Determinar si es atributo o parte de un método
                # Usamos regex con \b (word boundary) para evitar matches parciales
                if re.search(r'\b' + re.escape(potential_class) + r'\b', info['attributesStr']):
                    scores[source_class][potential_class] += self.ATTRIBUTE_SCORE
                else:
                    scores[source_class][potential_class] += self.METHOD_SIG_SCORE

    def _analyze_body_instantiations(self, source_class, info, scores):
        """Analiza instanciaciones 'new ClassName(...)' en el cuerpo del archivo."""
        for target_class in self.classes_:
            if source_class == target_class:
                continue

            # Regex para encontrar 'new TargetClass(...)'
            # re.escape maneja caracteres especiales (ej. en clases internas '$')
            # '.*?' es no-codicioso (non-greedy) para manejar múltiples '()' en una línea
            regex = r'\bnew\s+' + re.escape(target_class) + r'\s*\(.*?\)'
            
            # Contamos todas las instanciaciones
            matches = re.findall(regex, info['fullContent'])
            
            if matches:
                # Acumulamos score por cada instanciación encontrada
                scores[source_class][target_class] += self.INSTANTIATION_SCORE * len(matches)

    def _save_results(self, output_path, scores):
        """
        Guarda el grafo de relaciones ponderado en un archivo CSV.
        Asegura que todas las clases de self.classes_ estén presentes
        en el archivo de salida, incluso si no tienen relaciones.
        """
        
        # Asegurarse de que el directorio de salida exista
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'relations'])
            
            # --- INICIO DE LA MODIFICACIÓN ---
            # Iterar sobre la lista maestra de clases (self.classes_)
            # en lugar de solo las clases en 'scores'.
            for source_class in sorted(self.classes_.keys()):
                # Obtener las relaciones; si no hay, 'scores' 
                # devolverá un defaultdict vacío.
                targets = scores[source_class]
                
                # Convertimos a un dict normal para una impresión limpia
                relations_dict = dict(targets)
                
                # Escribir la fila, incluso si relations_dict está vacío (ej. "{}")
                writer.writerow([source_class, str(relations_dict)])
            # --- FIN DE LA MODIFICACIÓN ---
                
        print(f"Results saved successfully to {output_path}")


def main():
    # Validar que se proporcionen los 3 argumentos (además del nombre del script)
    if len(sys.argv) != 4:
        print("Uso: python analyze_relations.py <input_path> <monolith_project> <output_path>")
        print("Ejemplo: python analyze_relations.py structural_view.csv ../monoliths/jPetStore/ jPetStore_fase1_structural_view.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    project_path = sys.argv[2]
    output_path = sys.argv[3]

    # Validaciones básicas de archivos (opcional, pero buena práctica)
    if not os.path.exists(input_path):
        print(f"Error: El archivo CSV de entrada no existe en {input_path}")
        sys.exit(1)
        
    if not os.path.exists(project_path) or not os.path.isdir(project_path):
        print(f"Error: La ruta del proyecto Java no existe o no es un directorio: {project_path}")
        sys.exit(1)

    # Omitimos try-catch como se solicitó
    
    analyzer = RelationAnalyzer(csv_path=input_path, project_root=project_path)
    analyzer.analyze_and_save(output_path)


if __name__ == '__main__':
    main()