# 4_align_multiview.py
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# Paths
# ===============================
STRUCTURAL_PATH = "results/embeddings_structural.csv"
SEMANTIC_PATH   = "results/embeddings_semantic.csv"
FUNCTIONAL_PATH = "results/embeddings_functional.csv"
OUTPUT_PATH     = "results/aligned_multiview.csv"
# ===============================


def load_embeddings(path: str) -> pd.DataFrame:
    """Carga un CSV con embeddings en columna 'embedding' (json string)."""
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))
    return df


def align_views(structural_df, semantic_df, functional_df):
    """
    Para cada nodo estructural:
      - encuentra el caso semántico más similar
      - encuentra el funcional más similar
    Devuelve un DataFrame alineado (1 fila por clase estructural).
    """
    structural_embeddings = np.vstack(structural_df["embedding"].to_numpy())
    semantic_embeddings   = np.vstack(semantic_df["embedding"].to_numpy())
    functional_embeddings = np.vstack(functional_df["embedding"].to_numpy())

    aligned_rows = []

    # Similaridad estructural ↔ semantic
    sim_struct_sem = cosine_similarity(structural_embeddings, semantic_embeddings)
    # Similaridad estructural ↔ functional
    sim_struct_func = cosine_similarity(structural_embeddings, functional_embeddings)

    for i, struct_row in structural_df.iterrows():
        struct_emb = struct_row["embedding"].tolist()

        # Semántico más cercano
        sem_idx = np.argmax(sim_struct_sem[i])
        semantic_row = semantic_df.iloc[sem_idx]
        sem_emb = semantic_row["embedding"].tolist()

        # Funcional más cercano
        func_idx = np.argmax(sim_struct_func[i])
        functional_row = functional_df.iloc[func_idx]
        func_emb = functional_row["embedding"].tolist()

        aligned_rows.append({
            "structural_label": struct_row.get("label", ""),
            "structural_text": struct_row.get("optimized_text", ""),
            "structural_embedding": json.dumps(struct_emb),

            "semantic_label": semantic_row.get("label", ""),
            "semantic_text": semantic_row.get("optimized_text", ""),
            "semantic_embedding": json.dumps(sem_emb),

            "functional_label": functional_row.get("label", ""),
            "functional_text": functional_row.get("optimized_text", ""),
            "functional_embedding": json.dumps(func_emb)
        })

    return pd.DataFrame(aligned_rows)


def main():
    print("=== Alineación Multivista (Structural ↔ Semantic ↔ Functional) ===")

    # Cargar datos
    structural_df = load_embeddings(STRUCTURAL_PATH)
    semantic_df   = load_embeddings(SEMANTIC_PATH)
    functional_df = load_embeddings(FUNCTIONAL_PATH)

    print(f"Estructural: {len(structural_df)} clases")
    print(f"Semántico:   {len(semantic_df)} casos de uso")
    print(f"Funcional:   {len(functional_df)} endpoints")

    # Alinear
    aligned_df = align_views(structural_df, semantic_df, functional_df)

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    aligned_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Alineación completada. Guardado en: {OUTPUT_PATH}")
    print(f"Total filas alineadas: {len(aligned_df)} (1 por clase estructural)")

    # Mostrar ejemplo
    print("\nEjemplo de alineación:")
    print(aligned_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
