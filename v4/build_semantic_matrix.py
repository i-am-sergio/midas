#!/usr/bin/env python3

"""
Fase 2: Construcción de la Matriz Semántica (S^(sem)).

Utiliza el modelo MPNet para generar embeddings de los textos limpios 
y calcula la similitud coseno para formar la matriz S^(sem).

Uso:
    python build_semantic_matrix.py <INPUT_CLEAN_CSV> <OUTPUT_BASE_NAME>
"""

import sys
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Modelo base SBERT basado en MPNet (conocido por su alta calidad en embeddings de código/sentencias)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SemanticMatrixBuilder:
    """
    Genera embeddings usando MPNet y calcula la matriz de similitud S^(sem).
    """
    def __init__(self, input_csv_path: str):
        self.df = pd.read_csv(input_csv_path)
        self.texts = self.df['cleaned_text'].tolist()
        self.class_names = self.df['class_name'].tolist()
        self.n = len(self.class_names)
        
        print(f"[Builder] Inicializando MPNet en dispositivo: {DEVICE}")
        
        # Cargar Tokenizer y Modelo
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Aplica Mean Pooling para obtener un vector de sentencia promedio.
        """
        # model_output[0] es el tensor de embeddings de tokens
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sumar los embeddings y dividir por la suma de la máscara de atención
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def generate_embeddings(self) -> np.ndarray:
        """
        Codifica todos los textos en embeddings usando el modelo MPNet.
        """
        print(f"[Builder] Generando embeddings para {self.n} clases...")
        
        # Tokenizar los textos
        inputs = self.tokenizer(
            self.texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(DEVICE)
        
        # Generar embeddings sin calcular gradientes
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Aplicar Mean Pooling para obtener el vector de sentencia
        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        
        # Mover a CPU y convertir a numpy
        embeddings_np = embeddings.cpu().numpy()
        
        print("[Builder] Embeddings generados.")
        return embeddings_np

    def build_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de similitud coseno S^(sem) a partir de los embeddings.
        """
        print("[Builder] Calculando matriz de Similitud Coseno S^(sem)...")
        # cosine_similarity genera una matriz NxN donde S[i,j] = similitud(i, j)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Asegurarse que la diagonal sea 1 (similitud de un vector consigo mismo)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix

    def save_matrix_csv(self, matrix: np.ndarray, output_matrix_csv: str):
        """Guarda la matriz de similitud en un CSV con cabeceras."""
        matrix_df = pd.DataFrame(
            matrix,
            index=self.class_names,
            columns=self.class_names
        )
        
        matrix_df.to_csv(output_matrix_csv)
        print(f"[Builder] Matriz S^(sem) guardada en: {output_matrix_csv}")
    
    def save_matrix_png(self, matrix: np.ndarray, output_png: str):
        """Guarda una visualización (heatmap) de la matriz S^(sem)."""
        fig_size = max(8, self.n * 0.4) 
        
        plt.figure(figsize=(fig_size, fig_size))
        
        plt.imshow(matrix, cmap='plasma', interpolation='nearest') # Usamos plasma o coolwarm para destacar la similitud
        
        plt.title(f'Matriz Semántica (S_sem) - {self.n}x{self.n}')
        plt.colorbar(label='Similitud Coseno')
        
        if self.n <= 30:
            plt.xticks(ticks=np.arange(self.n), labels=self.class_names, rotation=90, fontsize=8)
            plt.yticks(ticks=np.arange(self.n), labels=self.class_names, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_png, dpi=150)
        print(f"[Builder] Heatmap de S^(sem) guardado en: {output_png}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python build_semantic_matrix.py <INPUT_CLEAN_CSV> <OUTPUT_BASE_NAME>")
        print("Ejemplo: python build_semantic_matrix.py jpetstore_fase2_semantic_clean.csv jpetstore_semantic")
        sys.exit(1)

    input_clean_csv = sys.argv[1]
    output_base_name = sys.argv[2]
    
    if not os.path.exists(input_clean_csv):
        print(f"Error: Archivo de entrada no encontrado: {input_clean_csv}")
        sys.exit(1)

    output_matrix_csv = f"{output_base_name}_matrix.csv"
    output_png = f"{output_base_name}_matrix.png"

    builder = SemanticMatrixBuilder(input_clean_csv)
    
    # 1. Generar embeddings
    embeddings = builder.generate_embeddings()
    
    # 2. Calcular matriz de similitud
    similarity_matrix = builder.build_similarity_matrix(embeddings)
    
    # 3. Guardar resultados
    os.makedirs(os.path.dirname(output_matrix_csv), exist_ok=True)
    builder.save_matrix_csv(similarity_matrix, output_matrix_csv)
    builder.save_matrix_png(similarity_matrix, output_png)
    
    print("\n[Completado] Matriz Semántica S^(sem) generada exitosamente.")

if __name__ == '__main__':
    main()