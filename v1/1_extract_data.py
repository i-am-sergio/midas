import javalang
import os
import csv
import requests
import re
from typing import List, Dict, Any, Tuple

# =============================================================================
# EXTRACCIÓN ESTRUCTURAL (Codigo Java)
# =============================================================================

def clean_comments(comment_text: str) -> str:
    """
    Limpia comentarios Javadoc removiendo tags y caracteres especiales.
    """
    if not comment_text:
        return ""
    
    # Remover tags Javadoc (@param, @return, etc.)
    cleaned = re.sub(r'@\w+.*?', '', comment_text)
    # Remover asteriscos y espacios múltiples
    cleaned = re.sub(r'\*+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remover HTML tags si existen
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    return cleaned.strip()

def extract_java_elements(java_code: str) -> List[Tuple[str, str, str, str]]:
    """
    Extrae información semánticamente relevante de clases Java.
    """
    tree = javalang.parse.parse(java_code)
    package_name = tree.package.name if tree.package else "default"
    elements = []
    
    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        class_name = node.name
        
        # 1. Extraer COMENTARIOS de clase (Javadoc)
        class_comments = ""
        if node.documentation:
            class_comments = clean_comments(node.documentation)
        
        # 2. Extraer métodos con sus comentarios
        methods_info = []
        for method in node.methods:
            if method.name:
                # Comentarios del método
                method_doc = clean_comments(method.documentation) if method.documentation else ""
                
                # Información semántica del método
                method_info = f"{method.name}: {method_doc}"
                methods_info.append(method_info)
        
        # 3. Extraer campos/atributos con comentarios
        fields_info = []
        if node.fields:
            for field in node.fields:
                field_doc = clean_comments(field.documentation) if field.documentation else ""
                for declarator in field.declarators:
                    field_info = f"{declarator.name}: {field_doc}"
                    fields_info.append(field_info)
        
        # 4. Combinar información semántica (NO código crudo)
        semantic_text = f"Class {class_name}"
        if class_comments:
            semantic_text += f" | Description: {class_comments}"
        if fields_info:
            semantic_text += f" | Fields: {'; '.join(fields_info)}"
        if methods_info:
            semantic_text += f" | Methods: {'; '.join(methods_info)}"
        
        elements.append((package_name, class_name, semantic_text, class_comments))
    
    return elements

def parse_java_project(src_folder: str, output_csv: str = "estructural_data.csv") -> None:
    """
    Recorre un proyecto Java y extrae información estructural detallada.
    """
    all_elements = []
    
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                    elements = extract_java_elements(code)
                    all_elements.extend(elements)
                except Exception as e:
                    print(f"Error procesando {file_path}: {e}")
    
    # Guardar en CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["package", "class_name", "class_elements", "raw_code"])
        writer.writerows(all_elements)
    
    print(f"Extracción estructural completada: {output_csv} con {len(all_elements)} elementos")

# =============================================================================
# EXTRACCIÓN FUNCIONAL (OpenAPI Specification)
# =============================================================================
def get_apis_from_openapi(api_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extrae endpoints y sus detalles de una especificación OpenAPI.
    """
    apis = []
    
    if 'paths' in api_doc:
        for path, methods in api_doc['paths'].items():
            for method, details in methods.items():
                if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    api_info = {
                        'path': path,
                        'method': method.upper(),
                        'summary': details.get('summary', ''),
                        'description': details.get('description', ''),
                        'operation_id': details.get('operationId', ''),
                        'parameters': details.get('parameters', []),  # Mantener como lista, no string
                        'request_body': details.get('requestBody', {}),  # Mantener como dict, no string
                        'responses': details.get('responses', {})  # Mantener como dict, no string
                    }
                    apis.append(api_info)
    
    return apis

def clean_text(text: str) -> str:
    """
    Limpia texto removiendo caracteres especiales y normalizando.
    """
    if not text:
        return ""
    
    # Remover caracteres especiales pero preservar puntuación básica
    cleaned = re.sub(r'[^\w\s.,!?;:-]', ' ', str(text))
    # Normalizar espacios
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remover content types técnicos
    cleaned = re.sub(r'application/(json|xml|x-www-form-urlencoded)', '', cleaned)
    # Remover referencias JSON
    cleaned = re.sub(r'\$ref|#/components/schemas/', '', cleaned)
    
    return cleaned.strip()

def transform_to_semantic(apis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforma datos OpenAPI en texto semántico para modelos BERT.
    """
    semantic_apis = []
    
    for api in apis:
        # Construir texto semántico coherente
        semantic_text = f"{api['method']} {api['path']}"
        
        # Agregar summary y description (limpios)
        if api.get('summary'):
            semantic_text += f" | Purpose: {clean_text(api['summary'])}"
        if api.get('description'):
            semantic_text += f" | Details: {clean_text(api['description'])}"
        
        # Procesar parámetros de manera semántica
        if api.get('parameters') and api['parameters']:
            param_descriptions = []
            for param in api['parameters']:
                param_desc = f"{param.get('name', '')}"
                if param.get('description'):
                    param_desc += f"({clean_text(param['description'])})"
                param_descriptions.append(param_desc)
            
            if param_descriptions:
                semantic_text += f" | Parameters: {', '.join(param_descriptions)}"
        
        # Procesar request body semánticamente
        request_body = api.get('request_body', {})
        if request_body and isinstance(request_body, dict):
            if 'description' in request_body:
                semantic_text += f" | Request: {clean_text(request_body['description'])}"
        
        # Procesar respuestas semánticamente
        responses = api.get('responses', {})
        if responses and isinstance(responses, dict):
            response_descs = []
            for code, details in responses.items():
                if isinstance(details, dict) and 'description' in details:
                    response_descs.append(f"{code}: {clean_text(details['description'])}")
            
            if response_descs:
                semantic_text += f" | Responses: {'; '.join(response_descs)}"
        
        # Crear entrada semántica
        semantic_api = {
            'path': api['path'],
            'method': api['method'],
            'operation_id': api.get('operation_id', ''),
            'semantic_text': semantic_text,
            'raw_summary': api.get('summary', ''),
            'raw_description': api.get('description', '')
        }
        
        semantic_apis.append(semantic_api)
    
    return semantic_apis

def extract_functional_data(url: str, output_csv: str = "functional_data.csv") -> None:
    """
    Extrae y transforma datos funcionales de una especificación OpenAPI.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        api_doc = response.json()
        
        # Extraer APIs en formato crudo
        raw_apis = get_apis_from_openapi(api_doc)
        
        # Transformar a formato semántico
        semantic_apis = transform_to_semantic(raw_apis)
        
        # Guardar en CSV
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            if semantic_apis:
                fieldnames = semantic_apis[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(semantic_apis)
        
        print(f"Extracción funcional completada: {output_csv} con {len(semantic_apis)} endpoints")
        
    except Exception as e:
        print(f"Error en extracción funcional: {e}")

# =============================================================================
# EXTRACCIÓN SEMÁNTICA (Casos de Uso)
# =============================================================================
def clean_use_case_text(text: str) -> str:
    """
    Limpia y normaliza texto de casos de uso.
    """
    if not text or str(text).lower() in ['nan', 'null', 'none', '']:
        return ""
    
    # Limpieza básica
    cleaned = re.sub(r'\s+', ' ', str(text).strip())
    cleaned = re.sub(r'[^\w\s.,;:!?()-]', '', cleaned)  # Remover caracteres especiales excepto puntuación común
    
    return cleaned

def extract_semantic_data(input_csv: str, output_csv: str = "semantic_data.csv") -> None:
    """
    Procesa y limpia casos de uso desde un archivo CSV.
    """
    try:
        cleaned_cases = []
        
        with open(input_csv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Ignorar filas vacías
                    cleaned_text = clean_use_case_text(row[0])
                    if cleaned_text:  # Solo incluir textos no vacíos después de limpieza
                        cleaned_cases.append([cleaned_text])
        
        # Guardar en CSV
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["use_case"])
            writer.writerows(cleaned_cases)
        
        print(f"Extracción semántica completada: {output_csv} con {len(cleaned_cases)} casos de uso")
    except Exception as e:
        print(f"Error en extracción semántica: {e}")

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    OUTPUT_PATH = "results"
    print("Iniciando fase 1: Extracción de datos...")

    # Crear carpeta si no existe
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 1. Extracción Estructural (Java)
    project_path = "../monoliths/jPetStore/src"
    parse_java_project(project_path, "results/data_estructural.csv")
    
    # 2. Extracción Funcional (OpenAPI)
    api_url = "https://petstore3.swagger.io/api/v3/openapi.json"
    extract_functional_data(api_url, "results/data_functional.csv")
    
    extract_semantic_data("jpetstore_use_cases.csv", "results/data_semantic.csv")
    
    print("Fase 1 completada. Archivos generados:")
    print("- estructural_data.csv")
    print("- functional_data.csv")
    print("- semantic_data.csv")