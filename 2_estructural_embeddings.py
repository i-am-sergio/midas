import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/codebert-base"
CSV_PATH = "estructural_data.csv"
OUTPUT_MATRIX_CSV = "structural_similarity_matrix.csv"
OUTPUT_IMAGE_ORIGINAL = "original_structural_view.png"

class CodeBERTEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
    
    def get_embeddings(self, texts: list, batch_size: int = 8) -> np.ndarray:
        """
        Genera embeddings para una lista de textos usando CodeBERT
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenizar el batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(DEVICE)
            
            # Generar embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usar el embedding del token [CLS] como representación de la frase
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)

def load_structural_data(csv_path: str) -> pd.DataFrame:
    """
    Carga y prepara los datos estructurales
    """
    df = pd.read_csv(csv_path)
    print(f"Datos cargados: {len(df)} elementos")
    
    # Combinar información para crear texto significativo
    df['text_for_embedding'] = df.apply(
        lambda row: f"Package: {row['package']} | Class: {row['class_name']} | Elements: {row['class_elements']}",
        axis=1
    )
    
    return df

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de similitud coseno
    """
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix: np.ndarray, labels: list, output_path: str, title: str):
    """
    Plotea y guarda la matriz de similitud
    """
    plt.figure(figsize=(12, 10))
    
    # Crear máscara para la diagonal superior (opcional)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    
    # Plotear heatmap
    sns.heatmap(
        similarity_matrix,
        mask=mask,
        cmap="viridis",
        center=0,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot=False,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matriz guardada como: {output_path}")

def save_similarity_matrix(similarity_matrix: np.ndarray, labels: list, output_path: str):
    """
    Guarda la matriz de similitud en CSV
    """
    # Crear DataFrame con labels como índice y columnas
    matrix_df = pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels
    )
    
    # Guardar con compresión para archivos grandes
    matrix_df.to_csv(output_path, index=True)
    print(f"Matriz de similitud guardada en: {output_path}")
    
    return matrix_df

def main():
    print("Iniciando generación de embeddings estructurales con CodeBERT...")
    print(f"Usando dispositivo: {DEVICE}")
    
    # 1. Cargar datos
    print("Cargando datos estructurales...")
    df = load_structural_data(CSV_PATH)
    
    # Crear labels cortos para visualización
    short_labels = [f"{row['class_name']}\n({row['package'].split('.')[-1]})" 
                   for _, row in df.iterrows()]
    
    # 2. Inicializar CodeBERT
    print("Inicializando modelo CodeBERT...")
    embedder = CodeBERTEmbedder()
    
    # 3. Generar embeddings
    print("Generando embeddings...")
    texts = df['text_for_embedding'].tolist()
    embeddings = embedder.get_embeddings(texts)
    
    print(f"Embeddings generados: {embeddings.shape}")
    
    # 4. Calcular matriz de similitud
    print("Calculando matriz de similitud coseno...")
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    print(f"Matriz de similitud shape: {similarity_matrix.shape}")
    print(f"Rango de similitudes: {similarity_matrix.min():.3f} - {similarity_matrix.max():.3f}")
    
    # 5. Plotear matriz original
    print("Generando visualización de la matriz...")
    plot_similarity_matrix(
        similarity_matrix,
        short_labels,
        OUTPUT_IMAGE_ORIGINAL,
        "Matriz de Similitud Estructural - Vista Original"
    )
    
    # 6. Guardar matriz en CSV
    print("Guardando matriz de similitud...")
    full_labels = [f"{pkg}.{cls}" for pkg, cls in zip(df['package'], df['class_name'])]
    matrix_df = save_similarity_matrix(similarity_matrix, full_labels, OUTPUT_MATRIX_CSV)
    
    # 7. Mostrar estadísticas
    print("\n=== ESTADÍSTICAS ===")
    print(f"Número de elementos: {len(df)}")
    print(f"Dimensión de embeddings: {embeddings.shape[1]}")
    print(f"Similitud promedio: {similarity_matrix.mean():.3f}")
    print(f"Similitud máxima: {similarity_matrix.max():.3f}")
    print(f"Similitud mínima: {similarity_matrix.min():.3f}")
    
    # Mostrar pares más similares
    np.fill_diagonal(similarity_matrix, -1)  # Ignorar diagonal
    max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    max_sim_value = similarity_matrix[max_sim_idx]
    
    print(f"\nPar más similar:")
    print(f"  {full_labels[max_sim_idx[0]]}")
    print(f"  {full_labels[max_sim_idx[1]]}")
    print(f"  Similitud: {max_sim_value:.3f}")

if __name__ == "__main__":
    main()