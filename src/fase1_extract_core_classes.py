#!/usr/bin/env python3

import sys
import os
import csv
import javalang

# Reglas de Filtrado
FILTER_SUFFIXES = [
    'Action', 'Form', 'Controller', 'Validator', 'Interceptor', 
    'Client', 'Advice', 'Session', 'Servlet', 'Test'
]

FILTER_PREFIXES = [
    'Base', 'Secure', 'MsSql', 'Oracle', 'JaxRpc', 'Mock'
]

class CoreClassExtractor:
    def __init__(self, root_dir, output_all_csv, output_core_csv):
        self.root_dir = root_dir
        self.output_all_csv = output_all_csv
        self.output_core_csv = output_core_csv

    @staticmethod
    def _get_method_signature(method_node):
        return_type = "void"
        if method_node.return_type:
            return_type = method_node.return_type.name
            if method_node.return_type.dimensions:
                return_type += '[]' * len(method_node.return_type.dimensions)
        
        method_name = method_node.name
        params_list = []
        for param in method_node.parameters:
            param_type = param.type.name
            if param.type.dimensions:
                param_type += '[]' * len(param.type.dimensions)
            if param.varargs:
                param_type += '...'
            params_list.append(f"{param_type} {param.name}")
        
        return f"{return_type} {method_name}({', '.join(params_list)})"

    def _parse_java_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        try:
            tree = javalang.parse.parse(content)
        except:
            return []

        results = []
        for path, node in tree.filter(javalang.tree.TypeDeclaration):
            if path and len(path) > 1 and isinstance(path[-2], javalang.tree.TypeDeclaration):
                continue
                
            class_name = node.name
            attributes_list = []
            methods_list = []

            if isinstance(node, javalang.tree.ClassDeclaration):
                for field in node.body:
                    if isinstance(field, javalang.tree.FieldDeclaration):
                        field_type = field.type.name
                        if field.type.dimensions:
                            field_type += '[]' * len(field.type.dimensions)
                        for declarator in field.declarators:
                            attributes_list.append(f"{field_type} {declarator.name}")

            for method in node.body:
                if isinstance(method, javalang.tree.MethodDeclaration):
                    methods_list.append(self._get_method_signature(method))
            
            attributes_str = f"[{'; '.join(attributes_list)}]"
            methods_str = f"[{'; '.join(methods_list)}]"
            
            results.append((file_path, class_name, attributes_str, methods_str))

        return results

    def _is_core_class(self, class_name):
        for suffix in FILTER_SUFFIXES:
            if class_name.endswith(suffix): return False
        for prefix in FILTER_PREFIXES:
            if class_name.startswith(prefix): return False
        return True

    def run(self):
        print(f"Escaneando: {self.root_dir}")
        
        # Crear encabezados
        headers = ['filename', 'class', 'attributes', 'methods']
        
        with open(self.output_all_csv, 'w', newline='', encoding='utf-8') as f_all, \
             open(self.output_core_csv, 'w', newline='', encoding='utf-8') as f_core:
            
            writer_all = csv.writer(f_all)
            writer_core = csv.writer(f_core)
            
            writer_all.writerow(headers)
            writer_core.writerow(headers)
            
            total_count = 0
            core_count = 0

            for dirpath, _, filenames in os.walk(self.root_dir):
                for filename in filenames:
                    if filename.endswith('.java'):
                        full_path = os.path.join(dirpath, filename)
                        rows = self._parse_java_file(full_path)
                        
                        for row in rows:
                            total_count += 1
                            # Escribir en ALL
                            writer_all.writerow(row)
                            
                            # Verificar filtro y escribir en CORE
                            class_name = row[1]
                            if self._is_core_class(class_name):
                                writer_core.writerow(row)
                                core_count += 1

        print(f"Proceso completado.")
        print(f" -> Total Clases Extraidas: {total_count}")
        print(f" -> Clases Core Filtradas: {core_count}")
        print(f" -> CSV Todos: {self.output_all_csv}")
        print(f" -> CSV Core:  {self.output_core_csv}")

def main():
    if len(sys.argv) != 4:
        print("Uso: python fase1_extract_core_classes.py <SOURCE_DIR> <ALL_CLASSES_CSV> <CORE_CLASSES_CSV>")
        sys.exit(1)

    source_dir = sys.argv[1]
    all_csv = sys.argv[2]
    core_csv = sys.argv[3]

    extractor = CoreClassExtractor(source_dir, all_csv, core_csv)
    extractor.run()

if __name__ == '__main__':
    main()