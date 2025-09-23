# 5_clustering_multiview.py
import pandas as pd
import numpy as np
import json
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# ============================
# Paths y Parámetros
# ============================
INPUT_PATH = "results/aligned_multiview.csv"
OUTPUT_BASE = "results/aligned_multiview_clustered"
K_RANGE = range(2, 11)  # valores de K a evaluar
# ============================


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
    print("=== Clustering Multivista con KMeans y Silhouette Score ===")

    # Cargar datos
    df = load_aligned_csv(INPUT_PATH)
    print(f"Datos cargados: {len(df)} filas")

    # Construir matriz de embeddings fusionados
    fused_embeddings = np.vstack(df.apply(fuse_embeddings, axis=1).to_numpy())

    # Evaluar diferentes valores de K
    best_k = None
    best_score = -1
    best_labels = None

    print("\n--- Evaluación de diferentes K ---")
    for k in K_RANGE:
        # kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        # labels = kmeans.fit_predict(fused_embeddings)
        clustering = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",  # también se puede probar con "rbf"
            assign_labels="kmeans",
            random_state=42
        )
        labels = clustering.fit_predict(fused_embeddings)

        score = silhouette_score(fused_embeddings, labels)
        print(f"K={k} → Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print("\n=== Mejor particionamiento ===")
    print(f"K óptimo: {best_k} con Silhouette Score = {best_score:.4f}")

    # Asignar mejor cluster encontrado
    df["cluster"] = best_labels

    # Mostrar log de cada cluster
    for cluster_id in range(best_k):
        subset = df[df["cluster"] == cluster_id]

        structural_nodes = subset["structural_label"].unique().tolist()
        semantic_nodes   = subset["semantic_label"].unique().tolist()
        functional_nodes = subset["functional_label"].unique().tolist()

        print(f"\n--- Cluster {cluster_id} ---")
        print("Clases estructurales:")
        for s in structural_nodes:
            print(f"   - {s}")

        print("Casos de uso (semánticos):")
        for u in semantic_nodes:
            print(f"   - {u}")

        print("Endpoints (funcionales):")
        for f in functional_nodes:
            print(f"   - {f}")

    # Guardar CSV con el mejor clustering
    output_path = f"{OUTPUT_BASE}_k{best_k}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nArchivo guardado con clusters en: {output_path}")


if __name__ == "__main__":
    main()
