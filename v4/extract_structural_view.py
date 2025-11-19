#!/usr/bin/env python3

"""
Extrae la vista estructural (clases, atributos con tipo y métodos) 
de un monolito Java y la guarda en un CSV.

Uso:
    python extract_structural_view.py <ruta_al_directorio_fuente>

Salida:
    Crea un archivo 'structural_view.csv' en el directorio actual.
"""

import sys
import os
import csv
import javalang

class JavaCodeExtractor:
    """
    Encapsula la lógica para parsear archivos Java y extraer su
    estructura (clases, atributos y métodos).
    """
    
    def __init__(self, root_dir, output_csv='structural_view.csv'):
        self.root_dir = root_dir
        self.output_csv = output_csv
        print(f"Iniciando extracción de vista estructural desde: {self.root_dir}")

    @staticmethod
    def _get_method_signature(method_node):
        """Construye una firma de método legible desde un nodo AST."""
        
        # 1. Obtener el tipo de retorno
        return_type = "void"  # Default
        if method_node.return_type:
            return_type = method_node.return_type.name
            if method_node.return_type.dimensions:
                return_type += '[]' * len(method_node.return_type.dimensions)
        
        # 2. Obtener el nombre del método
        method_name = method_node.name
        
        # 3. Construir la lista de parámetros
        params_list = []
        for param in method_node.parameters:
            param_type = param.type.name
            if param.type.dimensions:
                param_type += '[]' * len(param.type.dimensions)
            if param.varargs:
                param_type += '...'
            
            params_list.append(f"{param_type} {param.name}")
        
        params_str = ", ".join(params_list)
        
        return f"{return_type} {method_name}({params_str})"

    def _parse_java_file(self, file_path):
        """
        Analiza un único archivo Java y extrae la información de la clase/interfaz.
        
        Devuelve una lista de tuplas, donde cada tupla contiene:
        (filename, class_name, attributes_str, methods_str)
        """
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Omitimos try/except para errores de parseo como se solicitó
        tree = javalang.parse.parse(content)
        results = []

        for path, node in tree.filter(javalang.tree.TypeDeclaration):
            
            # Solo nos interesan las declaraciones de primer nivel
            if path and len(path) > 1 and isinstance(path[-2], javalang.tree.TypeDeclaration):
                continue
                
            class_name = node.name
            attributes_list = []
            methods_list = []

            # 1. Extraer Atributos (solo para Clases, no Interfaces)
            if isinstance(node, javalang.tree.ClassDeclaration):
                for field in node.body:
                    if isinstance(field, javalang.tree.FieldDeclaration):
                        
                        # ----- INICIO DE LA MODIFICACIÓN -----
                        # Capturar el tipo del atributo
                        field_type = field.type.name
                        if field.type.dimensions:
                            field_type += '[]' * len(field.type.dimensions)
                        
                        # Un FieldDeclaration puede tener múltiples declaradores (ej. int a, b;)
                        for declarator in field.declarators:
                            attr_name = declarator.name
                            # Añadir 'Tipo nombre' a la lista
                            attributes_list.append(f"{field_type} {attr_name}")
                        # ----- FIN DE LA MODIFICACIÓN -----

            # 2. Extraer Métodos (funciona para Clases e Interfaces)
            for method in node.body:
                if isinstance(method, javalang.tree.MethodDeclaration):
                    signature = self._get_method_signature(method)
                    methods_list.append(signature)
            
            # 3. Formatear la salida para que coincida con el ejemplo
            attributes_str = f"[{'; '.join(attributes_list)}]"
            methods_str = f"[{'; '.join(methods_list)}]"
            
            results.append((file_path, class_name, attributes_str, methods_str))

        return results

    def extract_all(self):
        """
        Recorre el directorio raíz, procesa todos los archivos .java
        y guarda los resultados en el archivo CSV de salida.
        """
        
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'class', 'attributes', 'methods'])
            
            file_count = 0
            for dirpath, _, filenames in os.walk(self.root_dir):
                for filename in filenames:
                    if filename.endswith('.java'):
                        full_path = os.path.join(dirpath, filename)
                        
                        # Analizar el archivo y obtener sus datos
                        rows_data = self._parse_java_file(full_path)
                        
                        # Escribir los resultados en el CSV
                        writer.writerows(rows_data)
                            
                        file_count += 1
                        if file_count % 10 == 0:
                            print(f"... {file_count} archivos procesados")

        print(f"\n¡Extracción completada!")
        print(f"Resultados guardados en: {self.output_csv}")

def main():
    if len(sys.argv) != 3:
        print("Uso: python extract_structural_view.py <ruta_al_directorio_fuente> <ruta_carpeta_output>")
        sys.exit(1)

    root_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Crear la carpeta output si no existe (forma concisa)
    os.makedirs(output_dir, exist_ok=True)
    
    # Ruta completa para el archivo CSV
    output_csv_path = os.path.join(output_dir, 'structural_view.csv')
    
    # Instanciar y ejecutar el extractor
    extractor = JavaCodeExtractor(root_dir=root_dir, output_csv=output_csv_path)
    extractor.extract_all()

if __name__ == '__main__':
    main()