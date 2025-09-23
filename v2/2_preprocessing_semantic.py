# 2_preprocessing_semantic.py
import pandas as pd
import re
import os

# Configuración
INPUT_PATH = "results/data_semantic.csv"   # CSV con columna: use_case
OUTPUT_PATH = "results/data_cleaned_semantic.csv"

def optimize_semantic_text(text: str) -> str:
    """
    Limpia y optimiza el texto de casos de uso para embeddings.
    """
    if pd.isna(text):
        return ""
    
    # Normalizar espacios
    optimized = text.strip()
    optimized = re.sub(r'\s+', ' ', optimized)
    
    # Eliminar caracteres especiales excesivos (dejamos puntuación básica)
    optimized = re.sub(r'[^\w\s.,;:!?-]', '', optimized)
    
    # Forzar minúsculas (opcional según si quieres preservar nombres propios)
    optimized = optimized.lower()
    
    return optimized

def main():
    print("=== Preprocesamiento de datos semánticos (Casos de uso) ===")
    df = pd.read_csv(INPUT_PATH)
    print(f"Datos cargados: {len(df)} casos de uso")

    if "use_case" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'use_case'")

    # Optimizar textos
    df["optimized_text"] = df["use_case"].apply(optimize_semantic_text)

    # Generar labels
    df["label"] = [f"use_case_{i+1:03d}" for i in range(len(df))]

    # Reordenar columnas
    df_cleaned = df[["label", "optimized_text"]]

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Guardar CSV limpio
    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"Datos semánticos preprocesados guardados en: {OUTPUT_PATH}")

    # Mostrar ejemplos
    print("\nEjemplos de preprocesamiento:")
    for i, row in df_cleaned.head(3).iterrows():
        print(f"  {row['label']}: {row['optimized_text'][:120]}...")

if __name__ == "__main__":
    main()
