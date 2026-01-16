#!/usr/bin/env python3

import sys
import os
import csv
import javalang
import re
import pandas as pd
from javalang.tree import ClassDeclaration, InterfaceDeclaration, FieldDeclaration, MethodDeclaration
from typing import List, Dict, Set

class SemanticCollector:
    """
    Extrae la vista semántica de un monolito Java, filtrando solo las 
    clases definidas en un archivo CSV de núcleo y aplicando la descomposición léxica.
    """

    def __init__(self, input_path: str, filtered_classes_path: str, output_path: str):
        self.input_directory = input_path
        self.output_file = output_path
        self.output_rows: List[Dict[str, str]] = []
        
        # Cargar la lista de clases permitidas desde el CSV filtrado
        self.allowed_classes: Set[str] = self._load_allowed_classes(filtered_classes_path)
        print(f"[Collector] Núcleo funcional cargado: {len(self.allowed_classes)} clases.")

    def _load_allowed_classes(self, filtered_classes_path: str) -> Set[str]:
        """Carga la columna 'class' del CSV filtrado para usarla como filtro maestro."""
        # Se asume que el CSV filtrado tiene una columna 'class'
        df = pd.read_csv(filtered_classes_path)
        return set(df['class'].tolist())

    def _split_camel_case(self, name: str) -> str:
        """
        Divide cadenas basadas en camelCase, snake_case o números, 
        y convierte el resultado a minúsculas, asegurando que los nombres
        de paquete/métodos se tokenicen correctamente para MPNet.
        """
        # 1. Insertar espacio antes de mayúsculas (camelCase) o números
        s = re.sub('([A-Z])', r' \1', name)
        s = re.sub('([0-9]+)', r' \1', s)
        
        # 2. Reemplazar guiones bajos y puntos por espacios
        s = s.replace('_', ' ').replace('.', ' ')
        
        # 3. Normalizar (quitar espacios duplicados y limpiar extremos) y minúsculas
        s = ' '.join(s.split()).lower()
        return s

    def _extract_semantic_elements(self, content: str) -> Dict[str, str | List[str]]:
        """
        Analiza el código Java usando javalang para extraer el paquete, 
        nombre de la clase/interfaz, atributos y métodos.
        """
        results = {
            'package_name': '',
            'class_name': '',
            'attribute_names': [],
            'method_names': []
        }
        
        # Omitimos try-catch/except de parsing
        tree = javalang.parse.parse(content)
        
        if tree.package:
            results['package_name'] = tree.package.name
            
        for type_declaration in tree.types:
            if isinstance(type_declaration, (ClassDeclaration, InterfaceDeclaration)):
                results['class_name'] = type_declaration.name
                
                # Extraer nombres de atributos
                if isinstance(type_declaration, ClassDeclaration):
                    for member in type_declaration.body:
                        if isinstance(member, FieldDeclaration):
                            for declarator in member.declarators:
                                results['attribute_names'].append(declarator.name)

                # Extraer nombres de métodos
                for member in type_declaration.body:
                    if isinstance(member, MethodDeclaration):
                        results['method_names'].append(member.name)
                
                # Asumimos una clase principal por archivo (sin clases internas anidadas)
                break 

        return results

    def _build_concatenated_string(self, elements: Dict[str, str | List[str]]) -> str:
        """
        Concatena todos los elementos léxicos, aplicando la descomposición léxica.
        """
        text_parts = []
        
        # Aplicar CamelCase Split a todos los tokens
        if elements['package_name']:
            text_parts.append(self._split_camel_case(elements['package_name']))
        
        if elements['class_name']:
            text_parts.append(self._split_camel_case(elements['class_name']))
        
        # Aplicar split a cada nombre en las listas
        for name in elements['attribute_names']:
            text_parts.append(self._split_camel_case(name))
            
        for name in elements['method_names']:
            text_parts.append(self._split_camel_case(name))
        
        return ' '.join(text_parts).strip()

    def _process_file(self, file_path: str):
        """
        Procesa un archivo Java: extrae elementos semánticos, aplica CamelCase Split 
        y filtra por la lista maestra.
        """
        
        # Extraer el nombre simple de la clase desde el path (heurística)
        # Se asume que el nombre del archivo es el nombre de la clase
        filename = os.path.basename(file_path).replace('.java', '')
        
        # --- FILTRO DE INCLUSIÓN MAESTRO ---
        # Solo procesamos si el nombre de la clase está en el núcleo permitido
        if filename not in self.allowed_classes:
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        elements = self._extract_semantic_elements(content)
        
        # El nombre extraído del AST debe coincidir con el nombre del archivo (la clase maestra)
        if elements['class_name'] not in self.allowed_classes:
            # Esto puede ocurrir si el archivo contiene múltiples clases
            # o si el nombre del archivo no es el nombre de la clase pública.
            # En nuestro caso, confiamos en la lista maestra.
            return 

        concatenated_text = self._build_concatenated_string(elements)
        
        self.output_rows.append({
            'class_name': elements['class_name'],
            'concatenated_text': concatenated_text
        })
        
    def _write_to_csv(self):
        """Escribe todos los resultados acumulados al CSV de salida."""
        print(f"[Collector] Escribiendo {len(self.output_rows)} clases al CSV.")
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['class_name', 'concatenated_text']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(self.output_rows)

    def run(self):
        """Recorre el directorio y procesa los archivos Java del núcleo."""
        
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        print(f"[Collector] Buscando archivos en: {self.input_directory}")
        
        for dirpath, _, filenames in os.walk(self.input_directory):
            for filename in filenames:
                if filename.endswith('.java'):
                    full_path = os.path.join(dirpath, filename)
                    self._process_file(full_path)
                    
        self._write_to_csv()
        print(f"[Collector] Proceso completado. Vista semántica guardada en: {self.output_file}")


def main():
    if len(sys.argv) != 4:
        print("Uso: python extract_semantic_view.py <PATH_DEL_PROYECTO_JAVA> <CSV_CLASES_FILTRADAS> <OUTPUT_CSV_PATH>")
        print("Ejemplo: python extract_semantic_view.py ../monoliths/jPetStore/ jpetstore_results/jpetstore_fase2_structural_view_filtered.csv jpetstore_results/jpetstore_fase1_semantic_view.csv")
        sys.exit(1)

    input_path = sys.argv[1] # Ej: ../monoliths/jPetStore/
    filtered_classes_path = sys.argv[2] # Ej: jpetstore_fase2_structural_view_filtered.csv
    output_path = sys.argv[3] # Ej: jpetstore_fase1_semantic_view.csv
    
    if not os.path.isdir(input_path):
        print(f"Error: La ruta proporcionada no es un directorio válido: {input_path}")
        sys.exit(1)

    if not os.path.exists(filtered_classes_path):
        print(f"Error: El archivo de clases filtradas no existe: {filtered_classes_path}")
        sys.exit(1)
        
    # El Preprocesador ahora usa el CSV de clases filtradas como su filtro maestro.
    collector = SemanticCollector(input_path, filtered_classes_path, output_path)
    collector.run()

if __name__ == '__main__':
    main()