# 2_preprocessing_structural.py
import pandas as pd
import re
import javalang
from typing import Dict
import os

INPUT_PATH = "results/data_estructural.csv"
OUTPUT_PATH = "results/data_cleaned_structural.csv"

def extract_comments_from_code(java_code: str) -> str:
    """
    Extrae comentarios Javadoc del código Java.
    """
    comments = []
    try:
        tree = javalang.parse.parse(java_code)
        for _, node in tree:
            if hasattr(node, "documentation") and node.documentation:
                clean_comment = re.sub(r"/\*\*|\*/|\*", "", node.documentation)
                clean_comment = re.sub(r"\s+", " ", clean_comment).strip()
                if clean_comment and len(clean_comment) > 10:
                    comments.append(clean_comment)
    except:
        pass
    return ". ".join(comments) if comments else ""

def extract_methods_from_code(java_code: str) -> Dict[str, str]:
    """
    Extrae información de métodos del código Java.
    """
    methods_info = {}
    try:
        tree = javalang.parse.parse(java_code)
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            if node.name:
                method_desc = f"returns {node.return_type.name}" if node.return_type else "returns void"
                if node.parameters:
                    params = [f"{param.type.name} {param.name}" for param in node.parameters]
                    method_desc += f" params: {', '.join(params)}"
                methods_info[node.name] = method_desc
    except:
        pass
    return methods_info

def optimize_structural_text(row):
    """
    Optimiza la representación textual de una clase.
    Combina package, class_name, class_elements y raw_code.
    """
    components = []

    # 1. Package y class_name
    package = str(row["package"])
    class_name = str(row["class_name"])
    components.append(f"Package: {package} Class: {class_name}")

    # 2. Extraer info de class_elements
    class_elements = str(row["class_elements"])

    methods_match = re.findall(r"Methods:\s*([^|]+)", class_elements)
    if methods_match:
        methods_text = methods_match[0]
        methods = re.split(r",\s*(?=[A-Za-z])", methods_text)
        for method in methods:
            if method.strip() and len(method.strip()) > 3:
                components.append(f"Method: {method.strip()}")

    fields_match = re.findall(r"Fields:\s*([^|]+)", class_elements)
    if fields_match:
        fields_text = fields_match[0]
        fields = [f.strip() for f in fields_text.split(",") if f.strip()]
        for field in fields:
            if field and len(field) > 2:
                components.append(f"Field: {field}")

    # 3. Extraer información semántica del raw_code
    raw_code = str(row.get("raw_code", ""))
    if raw_code and len(raw_code) > 50:
        comments = extract_comments_from_code(raw_code)
        if comments:
            components.append(f"Comments: {comments}")

        code_methods = extract_methods_from_code(raw_code)
        for method_name, method_info in code_methods.items():
            components.append(f"CodeMethod: {method_name} {method_info}")

    # 4. Combinar y limpiar
    optimized_text = ". ".join(components)
    optimized_text = re.sub(r"\s+", " ", optimized_text)
    optimized_text = re.sub(r"[^\w\s.,;:()<>]", " ", optimized_text)
    return optimized_text.strip()

def main():
    print("=== Preprocesamiento de datos estructurales ===")
    df = pd.read_csv(INPUT_PATH)
    print(f"Datos cargados: {len(df)} clases")

    # Optimizar texto
    df["optimized_text"] = df.apply(optimize_structural_text, axis=1)

    # Construir dataframe limpio
    clean_df = df[["class_name", "optimized_text"]].rename(columns={"class_name": "label"})

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Guardar CSV limpio
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Datos preprocesados guardados en: {OUTPUT_PATH}")

    # Mostrar ejemplos
    print("\nEjemplos de salida:")
    for i, row in clean_df.head(3).iterrows():
        print(f"  {i+1}. Label: {row['label']}")
        print(f"     Optimized: {row['optimized_text'][:120]}...")

if __name__ == "__main__":
    main()