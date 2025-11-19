import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def setup_nltk():
    """Descarga los recursos necesarios de NLTK."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Descargando stopwords de NLTK...")
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Descargando wordnet de NLTK...")
        nltk.download('wordnet')

def preprocess_semantic_text(text):
    """
    Aplica la cadena de preprocesamiento semántico descrita en la tesis.
    """
    
    # 1. Descomposición Léxica (camelCase y snake_case)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text) # camelCase
    text = re.sub(r'_', ' ', text)                  # snake_case
    
    # Convertir a minúsculas
    text = text.lower()
    
    # 2. Remoción de Ruido (stop words y sufijos de software)
    stop_words = set(stopwords.words('english'))
    
    # Sufijos comunes en el desarrollo de software a ignorar
    dev_suffixes = {
        'impl', 'service', 'factory', 'dao', 'dto', 'form', 
        'controller', 'action', 'interceptor', 'validator', 'advice'
    }
    
    all_noise = stop_words.union(dev_suffixes)
    
    # Tokenizar (simple, basado en espacios)
    tokens = text.split()
    
    # 3. Lematización
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in all_noise and len(word) > 1 and word.isalpha()
    ]
    
    return " ".join(lemmatized_tokens)

def get_mpnet_embeddings(texts, model, tokenizer):
    """
    Genera embeddings para una lista de textos usando MPNet.
    Utiliza pooling de media (mean pooling) en la última capa oculta.
    """
    print(f"Generando embeddings para {len(texts)} clases...")
    # Codificar los textos
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    # Generar embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Aplicar Mean Pooling
    # Tomar la última capa oculta y calcular la media a través de la dimensión de tokens (dim=1)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.cpu().numpy()

def process_semantic_view(csv_path, output_csv, output_plot):
    """
    Procesa la vista semántica:
    1. Carga los textos de las clases.
    2. Aplica preprocesamiento semántico.
    3. Genera embeddings con MPNet.
    4. Calcula la matriz de similitud coseno (S_sem).
    5. Guarda la matriz como CSV y como heatmap.
    """
    print("Iniciando procesamiento de la Vista Semántica...")
    
    # 1. Cargar datos
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de entrada: {csv_path}")
        return

    print(f"Cargadas {len(df)} clases para análisis semántico.")

    # 2. Aplicar Preprocesamiento Semántico
    print("Aplicando preprocesamiento de texto (lemmatización, etc.)...")
    df['processed_text'] = df['concatenated_text'].apply(preprocess_semantic_text)
    
    # 3. Cargar modelo MPNet y generar Embeddings
    model_name = "microsoft/mpnet-base"
    print(f"Cargando modelo y tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval() # Poner el modelo en modo de evaluación

    texts_to_embed = df['processed_text'].tolist()
    embeddings = get_mpnet_embeddings(texts_to_embed, model, tokenizer)
    
    print(f"Embeddings generados (Dimensiones: {embeddings.shape})")

    # 4. Calcular Similitud Coseno
    # La similitud coseno ya está normalizada en el rango [-1, 1] (o [0, 1] para texto)
    similarity_matrix_data = cosine_similarity(embeddings)
    
    # Convertir a DataFrame
    all_classes = df['class_name'].tolist()
    similarity_matrix_df = pd.DataFrame(similarity_matrix_data, index=all_classes, columns=all_classes)
    
    # 5. Guardar la matriz normalizada como CSV
    similarity_matrix_df.to_csv(output_csv)
    print(f"Matriz semántica normalizada guardada en: {output_csv}")

    # 6. Guardar el heatmap
    plt.figure(figsize=(14, 11))
    sns.heatmap(similarity_matrix_df, cmap="plasma", xticklabels=False, yticklabels=False)
    plt.title("Heatmap de la Matriz de Similitud Semántica ($S^{(sem)}$)", fontsize=16)
    plt.xlabel("Clases", fontsize=12)
    plt.ylabel("Clases", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Heatmap semántico guardado en: {output_plot}")
    plt.close()

if __name__ == "__main__":
    # Asegurarse de que el directorio de salida exista
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar NLTK
    setup_nltk()

    # Rutas de entrada y salida
    INPUT_FILE = "results/jPetStore_fase1_semantic_view.csv"
    OUTPUT_CSV_FILE = os.path.join(output_dir, "semantic_matrix_normalized.csv")
    OUTPUT_PLOT_FILE = os.path.join(output_dir, "semantic_matrix_heatmap.png")

    process_semantic_view(INPUT_FILE, OUTPUT_CSV_FILE, OUTPUT_PLOT_FILE)
    print("Procesamiento semántico completado.")