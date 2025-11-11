import os
import re
import csv
from pathlib import Path

def analyze_java_file(file_path):
    """Analiza un archivo Java y extrae nombre de archivo, clase/interfaz, atributos y métodos"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        filename = Path(file_path).name
        
        # Buscar declaración de clase O interfaz
        class_match = re.search(r'(class|interface|@?interface|enum)\s+(\w+)', content)
        if class_match:
            class_name = class_match.group(2)  # El nombre está en el segundo grupo
        else:
            class_name = "Unknown"
        
        # Buscar atributos (variables de instancia) - patrón mejorado
        attributes = []
        # Patrón para atributos: modificadores + tipo + nombre + ;
        # Maneja generics, arrays, y diferentes modificadores
        attribute_pattern = r'(private|protected|public|static|final)\s+([\w<>\[\],\s]+?)\s+(\w+)\s*[;=]'
        attribute_matches = re.findall(attribute_pattern, content)
        for match in attribute_matches:
            attributes.append(match[2])  # Solo el nombre del atributo
        
        # Buscar métodos - patrón MUY mejorado
        methods = []
        
        # Patrón principal para métodos: maneja diferentes casos
        # Incluye métodos con diferentes modificadores, tipos de retorno complejos, etc.
        method_patterns = [
            # Métodos normales
            r'(public|private|protected|static|final|abstract|synchronized|native)\s+([\w<>\[\],\s]+?)\s+(\w+)\s*\([^)]*\)\s*\{',
            # Métodos en interfaces (sin cuerpo)
            r'(public|private|protected|static|default)\s+([\w<>\[\],\s]+?)\s+(\w+)\s*\([^)]*\)\s*;',
            # Constructores
            r'(public|private|protected)\s+(\w+)\s*\([^)]*\)\s*\{',
            # Métodos con anotaciones
            r'@\w+(?:\([^)]*\))?\s*(?:public|private|protected|static|final|abstract)?\s*([\w<>\[\],\s]+?)\s+(\w+)\s*\([^)]*\)',
        ]
        
        for pattern in method_patterns:
            method_matches = re.findall(pattern, content)
            for match in method_matches:
                if len(match) == 3:  # Para patrones con 3 grupos (modificador, tipo, nombre)
                    methods.append(match[2])
                elif len(match) == 2:  # Para patrones con 2 grupos
                    # Verificar si es un constructor o método con anotación
                    if match[0] in ['public', 'private', 'protected', 'static']:
                        methods.append(match[1])
                    else:
                        methods.append(match[1] if len(match[1].strip()) > 0 else match[0])
        
        # Eliminar duplicados y limpiar
        methods = list(dict.fromkeys(methods))  # Mantener orden y eliminar duplicados
        methods = [m for m in methods if m and not m.isspace()]
        
        return {
            'filename': filename,
            'class': class_name,
            'attributes': attributes,
            'methods': methods
        }
    
    except Exception as e:
        print(f"Error analizando {file_path}: {e}")
        return None

def analyze_java_project(PATH_PROJECT, CSV_PATH):
    """Analiza todos los archivos .java en un proyecto y genera un CSV"""
    
    java_files = []
    
    # Buscar todos los archivos .java recursivamente
    for root, dirs, files in os.walk(PATH_PROJECT):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    print(f"Encontrados {len(java_files)} archivos .java")
    
    # Analizar cada archivo
    results = []
    for java_file in java_files:
        analysis = analyze_java_file(java_file)
        if analysis:
            results.append(analysis)
    
    # Escribir CSV
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'class', 'attributes', 'methods']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            # Convertir listas a strings para el CSV
            csv_result = {
                'filename': result['filename'],
                'class': result['class'],
                'attributes': str(result['attributes']),
                'methods': str(result['methods'])
            }
            writer.writerow(csv_result)
    
    print(f"CSV generado exitosamente en: {CSV_PATH}")
    print(f"Total de archivos procesados: {len(results)}")

# Configuración - MODIFICA ESTAS RUTAS SEGÚN TUS NECESIDADES
PATH_PROJECT = "../monoliths/jPetStore"
CSV_PATH = "fase1.csv"

# Ejecutar análisis
if __name__ == "__main__":
    # Si quieres usar argumentos de línea de comandos:
    import sys
    if len(sys.argv) == 3:
        PATH_PROJECT = sys.argv[1]
        CSV_PATH = sys.argv[2]
    
    analyze_java_project(PATH_PROJECT, CSV_PATH)