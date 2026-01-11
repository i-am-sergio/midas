# fase1_extract_core_classes.py

import sys
import os
import csv
import javalang
import pandas as pd

# =============================================================================
# REGLAS DE FILTRADO (JPetStore + AcmeAir + DayTrader + Plants)
# =============================================================================

FILTER_SUFFIXES = [
    # JPetStore Ruido
    'Action', 'Form', 'Controller', 'Validator', 'Interceptor', 
    'Client', 'Advice', 'Session', 'Servlet', 'Test',
    # AcmeAir Ruido
    'Test', 'Loader', 'Parser', 'Result', 'Results', 'Stats', 'Totals', 'Runner', 'Main',
    'Config', 'Configuration', 'Constants', 'Factory', 'Manager', 'Locator', 'Generator',
    'Helper', 'Util', 'Utils', 'Utility',
    'Converter', 
    # DayTrader Ruido
    'JSF', 'Listener', 'Producer', 'MDB', 'Filter',
    # Plants Ruido
    'Properties', 'Exception', 'Info',
    # JRideConnect
    'Application', 'Tests',
]

FILTER_PREFIXES = [
    # JPetStore Ruido 
    'Base', 'Secure', 'MsSql', 'Oracle', 'JaxRpc', 'Mock',
    # AcmeAir Ruido 
    'SQL', 'Mongo', 'WXS', 'Jmeter', 'Nmon', 'Rest', 'REST',
    # DayTrader Ruido
    'Ping', 'TradeBuild', 'Explicit', 'RunStats',
    # Plants Ruido
    'Help', 'Populate', 'Reset', 'Populate', 'Validate', 'Help'
]


class CoreClassExtractor:
    def __init__(self, root_dir, output_all_csv, output_core_csv):
        self.root_dir = root_dir
        self.output_all_csv = output_all_csv
        self.output_core_csv = output_core_csv

        # Diccionario para resolución de colisiones: { 'ClassName': row_data }
        self.core_classes_map = {} 
        self.all_rows_buffer = []

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
    
    def _should_replace(self, current_path, new_path):
        """
        Determina si la nueva ruta es una implementación 'mejor' que la actual.
        Prioridad: Morphia/Mongo > JPA > WXS/JDBC
        """
        curr = current_path.lower().replace('\\', '/')
        new_p = new_path.lower().replace('\\', '/')
        
        # Prioridad 1: Morphia (MongoDB) es la implementación canónica en AcmeAir
        if "morphia" in new_p and "morphia" not in curr:
            return True
        if "morphia" in curr:
            return False
            
        # Prioridad 2: JPA sobre JDBC o WXS
        if "jpa" in new_p and "jpa" not in curr:
            return True
            
        # Si no hay preferencia clara, mantener el primero encontrado
        return False

    def run(self):
        print(f"Escaneando y resolviendo colisiones en: {self.root_dir}")
        
        files_scanned = 0
        collisions_found = 0
        
        # PASO 1: Escaneo y Resolución en Memoria
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith('.java'):
                    full_path = os.path.join(dirpath, filename)
                    rows = self._parse_java_file(full_path)
                    
                    for row in rows:
                        self.all_rows_buffer.append(row) # Guardar todo para log
                        
                        class_name = row[1]
                        
                        # Solo procesar lógica de colisión si es Core
                        if self._is_core_class(class_name):
                            if class_name not in self.core_classes_map:
                                # Nueva clase
                                self.core_classes_map[class_name] = row
                            else:
                                # Colisión detectada
                                collisions_found += 1
                                existing_row = self.core_classes_map[class_name]
                                existing_path = existing_row[0]
                                new_path = row[0]
                                
                                if self._should_replace(existing_path, new_path):
                                    # print(f"   [Reemplazo] {class_name}: {os.path.basename(os.path.dirname(existing_path))} -> {os.path.basename(os.path.dirname(new_path))}")
                                    self.core_classes_map[class_name] = row
                    
                    files_scanned += 1
                    if files_scanned % 50 == 0:
                        print(f"... {files_scanned} archivos analizados", end='\r')

        print(f"\n... Escaneo finalizado.")

        # PASO 2: Escritura a Disco
        headers = ['filename', 'class', 'attributes', 'methods']
        
        # CSV ALL (Sin filtrar, con duplicados)
        with open(self.output_all_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(self.all_rows_buffer)

        # CSV CORE (Filtrado y Deduplicado)
        # Ordenamos por nombre de clase para consistencia
        sorted_core_rows = sorted(self.core_classes_map.values(), key=lambda x: x[1])
        
        with open(self.output_core_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(sorted_core_rows)

        print(f"Proceso completado.")
        print(f" -> Total Clases Encontradas: {len(self.all_rows_buffer)}")
        print(f" -> Colisiones Resueltas: {collisions_found}")
        print(f" -> Clases Core Únicas: {len(sorted_core_rows)}")
        print(f" -> CSV Core: {self.output_core_csv}")

        # print files:
        df = pd.read_csv(self.output_core_csv)
        classes = df['class'].tolist()
        print(classes) 

def main():
    if len(sys.argv) != 4:
        print("Uso: python fase1_extract_core_classes.py <SOURCE_DIR> <ALL_CLASSES_CSV> <CORE_CLASSES_CSV>")
        sys.exit(1)

    source_dir = sys.argv[1]
    all_csv = sys.argv[2]
    core_csv = sys.argv[3]

    os.makedirs(os.path.dirname(all_csv), exist_ok=True)

    extractor = CoreClassExtractor(source_dir, all_csv, core_csv)
    extractor.run()

if __name__ == '__main__':
    main()