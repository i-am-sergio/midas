#!/usr/bin/env python3

"""
FASE 2.2: Construcción de la Matriz Semántica (S_sem).

Genera embeddings usando MPNet sobre las descripciones funcionales
y calcula la similitud del coseno entre clases.

Uso:
    python fase2_build_semantic_matrix.py <INPUT_SEMANTIC_CSV> <CORE_CLASSES_CSV> <OUTPUT_BASE_NAME>
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Configuración del Modelo
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cpu" # "cuda"

class SemanticMatrixBuilder:
    def __init__(self, semantic_csv_path, core_csv_path):
        print(f"[Semantic] Cargando datos desde: {semantic_csv_path}")
        
        # Cargar datos semánticos
        self.df_sem = pd.read_csv(semantic_csv_path)
        
        # Cargar núcleo para asegurar orden y consistencia
        self.df_core = pd.read_csv(core_csv_path)
        
        # 1. Alinear: Filtrar y ordenar según el CORE
        # Usamos 'class' del core como la verdad absoluta del orden
        core_classes = self.df_core['class'].tolist()
        
        # Crear un mapa {clase: texto}
        # Asumimos que el input semántico tiene 'class_name' y 'concatenated_text'
        text_map = dict(zip(self.df_sem['class_name'], self.df_sem['concatenated_text']))
        
        self.class_names = []
        self.texts = []
        
        missing_classes = []
        
        for cls in core_classes:
            if cls in text_map:
                self.class_names.append(cls)
                self.texts.append(text_map[cls])
            else:
                missing_classes.append(cls)
                # Fallback: usar el nombre de la clase si falta descripción
                self.class_names.append(cls)
                self.texts.append(cls) 

        self.n = len(self.class_names)
        
        if missing_classes:
            print(f"⚠️ Advertencia: {len(missing_classes)} clases del núcleo no tenían datos semánticos (usando nombre como fallback).")

        print(f"[Semantic] Inicializando MPNet ({DEVICE}) para {self.n} clases...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def generate_embeddings(self):
        print(f"[Semantic] Generando embeddings...")
        
        # Tokenizar
        inputs = self.tokenizer(
            self.texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(DEVICE)
        
        # Inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pooling
        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        embeddings_np = embeddings.cpu().numpy()
        
        print(f"[Semantic] Embeddings generados: {embeddings_np.shape}")
        return embeddings_np

    def build_similarity_matrix(self, embeddings):
        print("[Semantic] Calculando Similitud Coseno...")
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # Diagonal = 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Opcional: Limpieza de ruido (Thresholding suave)
        # similarity_matrix[similarity_matrix < 0.1] = 0
        
        return similarity_matrix

    def save_results(self, matrix, output_base):
        # Rutas de salida
        csv_path = f"{output_base}_matrix.csv"
        png_path = f"{output_base}_matrix.png"
        
        # Guardar CSV
        df_matrix = pd.DataFrame(matrix, index=self.class_names, columns=self.class_names)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_matrix.to_csv(csv_path)
        print(f"[Semantic] Matriz guardada: {csv_path}")
        
        # Guardar PNG
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Similitud Semántica')
        plt.title(f'Matriz Semántica (MPNet) - {self.n}x{self.n}')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[Semantic] Heatmap guardado: {png_path}")

def main():
    if len(sys.argv) != 4:
        print("Uso: python fase2_build_semantic_matrix.py <SEMANTIC_CSV> <CORE_CSV> <OUTPUT_BASE>")
        sys.exit(1)

    semantic_csv = sys.argv[1]
    core_csv = sys.argv[2]
    output_base = sys.argv[3]

    if not os.path.exists(semantic_csv):
        print(f"Error: No existe {semantic_csv}")
        sys.exit(1)

    builder = SemanticMatrixBuilder(semantic_csv, core_csv)
    embeddings = builder.generate_embeddings()
    matrix = builder.build_similarity_matrix(embeddings)
    builder.save_results(matrix, output_base)

if __name__ == '__main__':
    main()