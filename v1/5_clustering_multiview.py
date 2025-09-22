# 5_clustering_multiview.py
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans

INPUT_PATH = "results/aligned_multiview.csv"
K = 5

def load_aligned_csv(path: str) -> pd.DataFrame:
    """Carga el CSV y convierte embeddings JSON a arrays NumPy."""
    df = pd.read_csv(path)

    for col in ["structural_embedding", "semantic_embedding", "functional_embedding"]:
        df[col] = df[col].apply(lambda x: np.array(json.loads(x)))
    
    return df

def fuse_embeddings(row) -> np.ndarray:
    """Concatena embeddings estructural + semántico + funcional en un solo vector."""
    return np.concatenate([
        row["structural_embedding"],
        row["semantic_embedding"],
        row["functional_embedding"]
    ])

def main():
    print("=== Clustering Multivista con KMeans ===")

    df = load_aligned_csv(INPUT_PATH)
    print(f"Datos cargados: {len(df)} filas")

    # Construir matriz de embeddings fusionados
    fused_embeddings = np.vstack(df.apply(fuse_embeddings, axis=1).to_numpy())

    # Clustering
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(fused_embeddings)

    print(f"Clustering completado con K={K}")

    # Mostrar log de cada cluster
    for cluster_id in range(K):
        subset = df[df["cluster"] == cluster_id]

        structural_nodes = subset["structural_label"].unique().tolist()
        semantic_nodes   = subset["semantic_label"].unique().tolist()
        functional_nodes = subset["functional_label"].unique().tolist()

        print("\n--- Cluster", cluster_id, "---")
        print("Clases estructurales:")
        for s in structural_nodes:
            print(f"   - {s}")

        print("Casos de uso (semánticos):")
        for u in semantic_nodes:
            print(f"   - {u}")

        print("Endpoints (funcionales):")
        for f in functional_nodes:
            print(f"   - {f}")

    # Guardar con cluster asignado
    output_path = INPUT_PATH.replace(".csv", f"_clustered_k{K}.csv")
    df.to_csv(output_path, index=False)
    print(f"\nArchivo guardado con clusters en: {output_path}")


if __name__ == "__main__":
    main()
