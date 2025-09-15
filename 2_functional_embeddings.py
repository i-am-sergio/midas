import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # MPNet para texto natural
CSV_PATH = "functional_data.csv"
OUTPUT_MATRIX_CSV = "functional_similarity_matrix.csv"
OUTPUT_IMAGE_ORIGINAL = "original_functional_view.png"

class MPNetEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
    
    def get_embeddings(self, texts: list, batch_size: int = 8) -> np.ndarray:
        """
        Genera embeddings para una lista de textos usando MPNet
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
                # Mean pooling para obtener embedding de la frase
                embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                embeddings = embeddings.cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Pooling promedio para obtener embedding de la frase
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_functional_data(csv_path: str) -> pd.DataFrame:
    """
    Carga y prepara los datos funcionales
    """
    df = pd.read_csv(csv_path)
    print(f"Datos funcionales cargados: {len(df)} endpoints")
    
    # Verificar que tenemos la columna semantic_text
    if 'semantic_text' not in df.columns:
        raise ValueError("El CSV debe contener la columna 'semantic_text'")
    
    # Limpiar texto adicionalmente
    df['cleaned_text'] = df['semantic_text'].apply(clean_functional_text)
    
    return df

def clean_functional_text(text: str) -> str:
    """
    Limpieza adicional para texto funcional
    """
    if not isinstance(text, str):
        return ""
    
    # Remover prefijos redundantes
    text = re.sub(r'^(GET|POST|PUT|DELETE|PATCH)\s+', '', text)
    # Remover duplicados de Purpose/Details
    text = re.sub(r'Purpose:\s*Details:', 'Purpose:', text)
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de similitud coseno
    """
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix: np.ndarray, labels: list, output_path: str, title: str):
    """
    Plotea y guarda la matriz de similitud para vista funcional
    """
    plt.figure(figsize=(14, 12))
    
    # Crear máscara para la diagonal superior
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    
    # Plotear heatmap con colores más vibrantes
    heatmap = sns.heatmap(
        similarity_matrix,
        mask=mask,
        cmap="viridis",
        center=0.5,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot=False,
        cbar_kws={"shrink": 0.8, "label": "Similitud Coseno"}
    )
    
    plt.title(title, fontsize=16, pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualización guardada como: {output_path}")

def save_similarity_matrix(similarity_matrix: np.ndarray, labels: list, output_path: str):
    """
    Guarda la matriz de similitud en CSV
    """
    # Crear labels compactos para el CSV
    compact_labels = []
    for i, label in enumerate(labels):
        # Usar operation_id si está disponible, sino método + path
        compact_label = f"endpoint_{i:03d}"
        compact_labels.append(compact_label)
    
    # Guardar matriz numérica
    matrix_df = pd.DataFrame(
        similarity_matrix,
        index=compact_labels,
        columns=compact_labels
    )
    
    # Guardar metadata adicional
    metadata_df = pd.DataFrame({
        'endpoint_id': compact_labels,
        'method': [label.split()[0] for label in labels],
        'path': [' '.join(label.split()[1:]) for label in labels],
        'full_label': labels
    })
    
    # Guardar ambos archivos
    matrix_df.to_csv(output_path, index=True)
    metadata_df.to_csv(output_path.replace('.csv', '_metadata.csv'), index=False)
    
    print(f"Matriz de similitud guardada en: {output_path}")
    print(f"Metadata guardada en: {output_path.replace('.csv', '_metadata.csv')}")
    
    return matrix_df, metadata_df

def analyze_functional_patterns(similarity_matrix: np.ndarray, labels: list):
    """
    Analiza patrones en la matriz de similitud funcional
    """
    print("\n=== ANÁLISIS DE PATRONES FUNCIONALES ===")
    
    # Ignorar diagonal para análisis
    np.fill_diagonal(similarity_matrix, -1)
    
    # Encontrar los pares más similares
    max_sim_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    max_sim_value = similarity_matrix[max_sim_indices]
    
    print(f"Par de endpoints más similares:")
    print(f"  • {labels[max_sim_indices[0]]}")
    print(f"  • {labels[max_sim_indices[1]]}")
    print(f"  Similitud: {max_sim_value:.3f}")
    
    # Encontrar similitudes por método HTTP
    methods = [label.split()[0] for label in labels]
    unique_methods = list(set(methods))
    
    print(f"\nSimilitud promedio por método HTTP:")
    for method in unique_methods:
        method_indices = [i for i, m in enumerate(methods) if m == method]
        if len(method_indices) > 1:
            method_sim = similarity_matrix[np.ix_(method_indices, method_indices)]
            avg_sim = method_sim[method_sim != -1].mean()
            print(f"  {method}: {avg_sim:.3f} ({len(method_indices)} endpoints)")

def main():
    print("Iniciando generación de embeddings funcionales con MPNet...")
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Modelo: {MODEL_NAME}")
    
    # 1. Cargar datos
    print("Cargando datos funcionales...")
    df = load_functional_data(CSV_PATH)
    
    # Crear labels para visualización (método + path)
    visual_labels = [f"{row['method']} {row['path']}" for _, row in df.iterrows()]
    
    # 2. Inicializar MPNet
    print("Inicializando modelo MPNet...")
    embedder = MPNetEmbedder()
    
    # 3. Generar embeddings
    print("Generando embeddings para textos funcionales...")
    texts = df['cleaned_text'].tolist()
    embeddings = embedder.get_embeddings(texts, batch_size=16)  # Batch más grande para texto más corto
    
    print(f"Embeddings generados: {embeddings.shape}")
    
    # 4. Calcular matriz de similitud
    print("Calculando matriz de similitud coseno...")
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    print(f"Matriz de similitud shape: {similarity_matrix.shape}")
    print(f"Rango de similitudes: {similarity_matrix.min():.3f} - {similarity_matrix.max():.3f}")
    
    # 5. Plotear matriz original
    print("Generando visualización de la matriz funcional...")
    plot_similarity_matrix(
        similarity_matrix,
        visual_labels,
        OUTPUT_IMAGE_ORIGINAL,
        "Matriz de Similitud Funcional - Vista Original\n(Endpoints API)"
    )
    
    # 6. Guardar matriz en CSV
    print("Guardando matriz de similitud...")
    matrix_df, metadata_df = save_similarity_matrix(similarity_matrix, visual_labels, OUTPUT_MATRIX_CSV)
    
    # 7. Análisis de patrones
    analyze_functional_patterns(similarity_matrix.copy(), visual_labels)
    
    # 8. Mostrar estadísticas generales
    print("\n=== ESTADÍSTICAS GENERALES ===")
    print(f"Número de endpoints: {len(df)}")
    print(f"Dimensión de embeddings: {embeddings.shape[1]}")
    print(f"Similitud promedio: {similarity_matrix.mean():.3f}")
    print(f"Similitud máxima: {similarity_matrix.max():.3f}")
    print(f"Similitud mínima: {similarity_matrix.min():.3f}")
    
    # Mostrar ejemplos de embeddings
    print(f"\nEjemplos de textos procesados:")
    for i in range(min(3, len(texts))):
        print(f"  {i+1}. {texts[i][:100]}...")

if __name__ == "__main__":
    main()