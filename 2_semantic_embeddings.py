import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # MPNet para casos de uso
CSV_PATH = "semantic_data.csv"
OUTPUT_MATRIX_CSV = "semantic_similarity_matrix.csv"
OUTPUT_IMAGE_ORIGINAL = "original_semantic_view.png"

class SemanticEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        """Inicializa el modelo para embeddings semánticos"""
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.model.eval()
    
    def get_embeddings(self, texts: list, batch_size: int = 16) -> np.ndarray:
        """
        Genera embeddings para casos de uso usando SentenceTransformer
        """
        # Filtrar textos vacíos o muy cortos
        valid_texts = [text for text in texts if text and len(text.strip()) > 10]
        invalid_indices = [i for i, text in enumerate(texts) if not text or len(text.strip()) <= 10]
        
        if invalid_indices:
            print(f"Advertencia: {len(invalid_indices)} textos muy cortos o vacíos serán ignorados")
        
        # Generar embeddings para textos válidos
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        # Crear matriz completa con NaN para textos inválidos
        full_embeddings = np.full((len(texts), embeddings.shape[1]), np.nan)
        valid_count = 0
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 10:
                full_embeddings[i] = embeddings[valid_count]
                valid_count += 1
        
        return full_embeddings

def load_semantic_data(csv_path: str) -> pd.DataFrame:
    """
    Carga y prepara los datos semánticos (casos de uso)
    """
    df = pd.read_csv(csv_path)
    print(f"Casos de uso cargados: {len(df)}")
    
    # Verificar que tenemos la columna use_case
    if 'use_case' not in df.columns:
        raise ValueError("El CSV debe contener la columna 'use_case'")
    
    # Limpieza adicional para casos de uso
    df['cleaned_text'] = df['use_case'].apply(clean_semantic_text)
    
    # Identificar casos de uso cortos
    df['text_length'] = df['cleaned_text'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    short_cases = df[df['text_length'] < 20]
    
    if len(short_cases) > 0:
        print(f"Advertencia: {len(short_cases)} casos de uso muy cortos:")
        for _, row in short_cases.iterrows():
            print(f"  - '{row['cleaned_text']}' (longitud: {row['text_length']})")
    
    return df

def clean_semantic_text(text: str) -> str:
    """
    Limpieza especializada para texto de casos de uso
    """
    if not isinstance(text, str):
        return ""
    
    # Remover números de lista (1., 2., etc.)
    text = re.sub(r'^\d+[\.\)]\s*', '', text)
    # Remover caracteres especiales pero preservar puntuación útil
    text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text.strip())
    # Capitalizar primera letra
    if text and len(text) > 1:
        text = text[0].upper() + text[1:]
    
    return text

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de similitud coseno, manejando NaN values
    """
    # Identificar filas con NaN
    valid_indices = ~np.isnan(embeddings).any(axis=1)
    valid_embeddings = embeddings[valid_indices]
    
    if len(valid_embeddings) == 0:
        raise ValueError("No hay embeddings válidos para calcular similitud")
    
    # Calcular similitud para embeddings válidos
    valid_similarity = cosine_similarity(valid_embeddings)
    
    # Crear matriz completa con NaN para elementos inválidos
    n_total = len(embeddings)
    similarity_matrix = np.full((n_total, n_total), np.nan)
    
    # Mapear índices válidos a la matriz completa
    valid_to_full = np.where(valid_indices)[0]
    for i, idx_i in enumerate(valid_to_full):
        for j, idx_j in enumerate(valid_to_full):
            similarity_matrix[idx_i, idx_j] = valid_similarity[i, j]
    
    return similarity_matrix

def plot_semantic_matrix(similarity_matrix: np.ndarray, labels: list, output_path: str, title: str):
    """
    Plotea y guarda la matriz de similitud semántica
    """
    # Identificar elementos válidos (no NaN)
    valid_indices = ~np.isnan(similarity_matrix).all(axis=1)
    valid_matrix = similarity_matrix[valid_indices][:, valid_indices]
    valid_labels = [labels[i] for i in np.where(valid_indices)[0]]
    
    plt.figure(figsize=(15, 13))
    
    # Plotear heatmap solo para elementos válidos
    heatmap = sns.heatmap(
        valid_matrix,
        cmap="plasma",
        center=0.6,
        square=True,
        xticklabels=valid_labels,
        yticklabels=valid_labels,
        annot=False,
        cbar_kws={"shrink": 0.8, "label": "Similitud Semántica"}
    )
    
    plt.title(title, fontsize=16, pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualización semántica guardada como: {output_path}")

def save_semantic_matrix(similarity_matrix: np.ndarray, use_cases: list, output_path: str):
    """
    Guarda la matriz de similitud semántica en CSV
    """
    # Crear labels compactos
    compact_labels = [f"uc_{i:03d}" for i in range(len(use_cases))]
    
    # Guardar matriz numérica
    matrix_df = pd.DataFrame(
        similarity_matrix,
        index=compact_labels,
        columns=compact_labels
    )
    
    # Guardar metadata con los casos de uso completos
    metadata_df = pd.DataFrame({
        'use_case_id': compact_labels,
        'use_case_text': use_cases,
        'text_length': [len(uc) if isinstance(uc, str) else 0 for uc in use_cases],
        'is_valid': [not np.isnan(similarity_matrix[i]).all() for i in range(len(use_cases))]
    })
    
    # Guardar ambos archivos
    matrix_df.to_csv(output_path, index=True)
    metadata_df.to_csv(output_path.replace('.csv', '_metadata.csv'), index=False)
    
    print(f"Matriz de similitud guardada en: {output_path}")
    print(f"Metadata de casos de uso guardada en: {output_path.replace('.csv', '_metadata.csv')}")
    
    return matrix_df, metadata_df

def analyze_semantic_patterns(similarity_matrix: np.ndarray, use_cases: list):
    """
    Analiza patrones semánticos en los casos de uso
    """
    print("\n=== ANÁLISIS DE PATRONES SEMÁNTICOS ===")
    
    # Identificar elementos válidos
    valid_indices = ~np.isnan(similarity_matrix).all(axis=1)
    valid_matrix = similarity_matrix[valid_indices][:, valid_indices]
    valid_cases = [use_cases[i] for i in np.where(valid_indices)[0]]
    
    if len(valid_cases) < 2:
        print("No hay suficientes casos de uso válidos para análisis")
        return
    
    # Encontrar el par más similar
    np.fill_diagonal(valid_matrix, -1)  # Ignorar diagonal
    max_sim_idx = np.unravel_index(np.argmax(valid_matrix), valid_matrix.shape)
    max_sim_value = valid_matrix[max_sim_idx]
    
    print(f"Par de casos de uso más similares (similitud: {max_sim_value:.3f}):")
    print(f"  • '{valid_cases[max_sim_idx[0]][:80]}...'")
    print(f"  • '{valid_cases[max_sim_idx[1]][:80]}...'")
    
    # Análisis de clusters naturales (similitud > 0.7)
    high_similarity = valid_matrix > 0.7
    np.fill_diagonal(high_similarity, False)
    
    cluster_groups = []
    visited = set()
    for i in range(len(valid_cases)):
        if i not in visited:
            similar_indices = np.where(high_similarity[i])[0]
            if len(similar_indices) > 0:
                cluster = [i] + list(similar_indices)
                cluster_groups.append(cluster)
                visited.update(cluster)
    
    print(f"\nGrupos de casos de uso con alta similitud (>0.7):")
    for i, cluster in enumerate(cluster_groups):
        if len(cluster) > 1:
            print(f"  Grupo {i+1} ({len(cluster)} casos):")
            for idx in cluster[:3]:  # Mostrar primeros 3 de cada grupo
                print(f"    - '{valid_cases[idx][:60]}...'")
            if len(cluster) > 3:
                print(f"    ... y {len(cluster) - 3} más")

def main():
    print("Iniciando generación de embeddings semánticos con MPNet...")
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Modelo: {MODEL_NAME}")
    
    # 1. Cargar datos semánticos
    print("Cargando casos de uso...")
    df = load_semantic_data(CSV_PATH)
    
    use_cases = df['cleaned_text'].tolist()
    
    # 2. Inicializar embedder semántico
    print("Inicializando modelo para embeddings semánticos...")
    embedder = SemanticEmbedder()
    
    # 3. Generar embeddings
    print("Generando embeddings para casos de uso...")
    embeddings = embedder.get_embeddings(use_cases, batch_size=16)
    
    print(f"Embeddings generados: {embeddings.shape}")
    valid_count = np.sum(~np.isnan(embeddings).any(axis=1))
    print(f"Casos de uso válidos: {valid_count}/{len(use_cases)}")
    
    # 4. Calcular matriz de similitud
    print("Calculando matriz de similitud semántica...")
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Estadísticas de la matriz
    valid_values = similarity_matrix[~np.isnan(similarity_matrix)]
    if len(valid_values) > 0:
        print(f"Rango de similitudes: {valid_values.min():.3f} - {valid_values.max():.3f}")
        print(f"Similitud promedio: {valid_values.mean():.3f}")
    
    # 5. Plotear matriz semántica
    print("Generando visualización de la matriz semántica...")
    short_labels = [f"UC_{i}\n{text[:15]}..." if len(text) > 15 else f"UC_{i}\n{text}" 
                   for i, text in enumerate(use_cases)]
    
    plot_semantic_matrix(
        similarity_matrix,
        short_labels,
        OUTPUT_IMAGE_ORIGINAL,
        "Matriz de Similitud Semántica - Vista Original\n(Casos de Uso)"
    )
    
    # 6. Guardar matriz en CSV
    print("Guardando matriz de similitud semántica...")
    matrix_df, metadata_df = save_semantic_matrix(similarity_matrix, use_cases, OUTPUT_MATRIX_CSV)
    
    # 7. Análisis de patrones semánticos
    analyze_semantic_patterns(similarity_matrix, use_cases)
    
    # 8. Mostrar estadísticas generales
    print("\n=== ESTADÍSTICAS SEMÁNTICAS ===")
    print(f"Total de casos de uso: {len(use_cases)}")
    print(f"Casos de uso válidos: {valid_count}")
    print(f"Longitud promedio de texto: {df['text_length'].mean():.1f} caracteres")
    
    # Distribución de longitudes
    length_stats = df['text_length'].describe()
    print(f"Longitud mínima: {length_stats['min']:.0f}")
    print(f"Longitud máxima: {length_stats['max']:.0f}")
    
    # Mostrar ejemplos
    print(f"\nEjemplos de casos de uso procesados:")
    valid_examples = df[df['text_length'] >= 20].head(3)
    for i, (_, row) in enumerate(valid_examples.iterrows()):
        print(f"  {i+1}. '{row['cleaned_text'][:80]}...'")

if __name__ == "__main__":
    main()