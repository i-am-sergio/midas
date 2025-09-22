# 6_plot_clustering.py
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ============================
# Paths
# ============================
INPUT_PATH = "results/aligned_multiview_clustered_k5.csv"
OUTPUT_PLOT = "results/clusters_plot.png"


def parse_embedding(x: str) -> np.ndarray:
    """Convierte string a np.array (acepta JSON o formato con espacios)."""
    try:
        return np.array(json.loads(x))
    except Exception:
        # Limpieza manual si no es JSON válido
        x = x.strip().replace("[", "").replace("]", "")
        parts = [float(v) for v in x.replace("\n", " ").split() if v]
        return np.array(parts)


def load_clustered_csv(path: str) -> pd.DataFrame:
    """Carga el CSV con embeddings y clusters asignados."""
    df = pd.read_csv(path)

    for col in ["structural_embedding", "semantic_embedding", "functional_embedding"]:
        df[col] = df[col].apply(parse_embedding)
    
    return df


def fuse_embeddings(row) -> np.ndarray:
    """Concatena embeddings estructural + semántico + funcional en un solo vector."""
    return np.concatenate([
        row["structural_embedding"],
        row["semantic_embedding"],
        row["functional_embedding"]
    ])


def main():
    print("=== Visualización de Clustering Multivista ===")

    # Cargar datos
    df = load_clustered_csv(INPUT_PATH)
    print(f"Datos cargados: {len(df)} filas")

    # Reconstruir embeddings fusionados
    fused_embeddings = np.vstack(df.apply(fuse_embeddings, axis=1).to_numpy())

    # Reducir a 2D con PCA
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(fused_embeddings)
    df["x"] = reduced[:, 0]
    df["y"] = reduced[:, 1]

    # Graficar
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df["x"], df["y"], c=df["cluster"], cmap="tab10", alpha=0.7, s=50)

    # Etiquetas: usaremos la clase estructural como label de cada punto
    for _, row in df.iterrows():
        plt.text(row["x"] + 0.02, row["y"] + 0.02, row["structural_label"], fontsize=7, alpha=0.8)

    # Leyenda
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)

    plt.title("Visualización de Clustering Multivista (PCA 2D)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()

    # Guardar
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Gráfico guardado en {OUTPUT_PLOT}")
    plt.show()


if __name__ == "__main__":
    main()
