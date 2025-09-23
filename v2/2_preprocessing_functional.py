# 2_preprocessing_functional.py
import pandas as pd
import re
import os

# Configuración
INPUT_PATH = "results/data_functional.csv"
OUTPUT_PATH = "results/data_cleaned_functional.csv"

def optimize_functional_text(row):
    """
    Extrae y optimiza el texto semántico para embeddings funcionales.
    Convierte la información del endpoint en texto natural.
    """
    components = []

    # 1. Método y path
    method_path = f"{row['path']} {row['method']}"
    components.append(method_path)

    # 2. Operation ID
    if pd.notna(row.get('operation_id', '')) and str(row['operation_id']).strip():
        components.append(str(row['operation_id']))

    # 3. Summary + Description
    purpose_parts = []
    if pd.notna(row.get('raw_summary', '')) and str(row['raw_summary']).strip():
        purpose_parts.append(str(row['raw_summary']))
    if pd.notna(row.get('raw_description', '')) and str(row['raw_description']).strip():
        purpose_parts.append(str(row['raw_description']))
    if purpose_parts:
        components.append(' '.join(purpose_parts))

    # 4. Extraer info de responses del semantic_text
    semantic_text = str(row.get('semantic_text', ''))
    responses_match = re.search(r'Responses:\s*([^|]+)', semantic_text)
    if responses_match:
        responses_text = responses_match.group(1)
        response_descriptions = re.findall(r':\s*([^;]+)', responses_text)
        if response_descriptions:
            components.extend([desc.strip() for desc in response_descriptions if desc.strip()])

    # 5. Combinar y limpiar
    optimized_text = ' '.join(components)
    optimized_text = re.sub(r'\b(Purpose|Details|Request|Responses|Parameters):\s*', '', optimized_text)
    optimized_text = re.sub(r'\d{3}:\s*', '', optimized_text)  # eliminar códigos de estado tipo 200:, 404:
    optimized_text = re.sub(r'[|;]\s*', ' ', optimized_text)
    optimized_text = re.sub(r'\s+', ' ', optimized_text).strip()

    return method_path, optimized_text

def main():
    print("=== Preprocesamiento de datos funcionales (Endpoints) ===")
    df = pd.read_csv(INPUT_PATH)
    print(f"Datos cargados: {len(df)} endpoints")

    # Optimizar textos → generar columnas label y optimized_text
    labels = []
    optimized_texts = []
    for _, row in df.iterrows():
        label, opt_text = optimize_functional_text(row)
        labels.append(label)
        optimized_texts.append(opt_text)

    df_cleaned = pd.DataFrame({
        "label": labels,
        "optimized_text": optimized_texts
    })

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Guardar CSV limpio
    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"Datos funcionales preprocesados guardados en: {OUTPUT_PATH}")

    # Mostrar ejemplos
    print("\nEjemplos de preprocesamiento:")
    for i, row in df_cleaned.head(3).iterrows():
        print(f"  {row['label']} → {row['optimized_text'][:120]}...")

if __name__ == "__main__":
    main()
