#!/usr/bin/env python3

"""
FASE 2.3: Construcción de la Matriz Funcional (A_fun).

Genera embeddings usando MPNet sobre las descripciones de casos de uso (funcionales)
y calcula la similitud del coseno para crear el grafo funcional.

Uso:
    python fase2_build_functional_matrix.py <INPUT_FUNCTIONAL_CSV> <CORE_CLASSES_CSV> <OUTPUT_BASE_NAME>
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Configuración del Modelo (Mismo que semántico para consistencia espacial)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEVICE = "cpu" # "cuda"

class FunctionalMatrixBuilder:
    def __init__(self, functional_csv_path, core_csv_path):
        print(f"[Functional] Cargando datos desde: {functional_csv_path}")
        
        self.df_fun = pd.read_csv(functional_csv_path)
        self.df_core = pd.read_csv(core_csv_path)
        
        # 1. Alinear con el NÚCLEO (Core Classes)
        core_classes = self.df_core['class'].tolist()
        
        # Mapa {clase: texto_funcional}
        # Nota: fase1_extract_functional_view.py usa 'concatenated_text' como cabecera
        text_map = dict(zip(self.df_fun['class_name'], self.df_fun['concatenated_text']))
        
        self.class_names = []
        self.texts = []
        missing_classes = []
        
        for cls in core_classes:
            if cls in text_map:
                self.class_names.append(cls)
                self.texts.append(text_map[cls])
            else:
                missing_classes.append(cls)
                # Fallback: Si no hay descripción funcional, usamos un texto neutro
                # para que tenga baja similitud con todo, en lugar de romper el script.
                self.class_names.append(cls)
                self.texts.append(f"{cls} represents general utility.") 

        self.n = len(self.class_names)
        
        if missing_classes:
            print(f"⚠️ Advertencia: {len(missing_classes)} clases sin descripción funcional.")

        print(f"[Functional] Inicializando MPNet ({DEVICE}) para {self.n} clases...")
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
        print(f"[Functional] Generando embeddings de comportamiento...")
        
        inputs = self.tokenizer(
            self.texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
        embeddings_np = embeddings.cpu().numpy()
        
        print(f"[Functional] Embeddings generados: {embeddings_np.shape}")
        return embeddings_np

    def build_similarity_matrix(self, embeddings):
        print("[Functional] Calculando Similitud Coseno...")
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # Diagonal = 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Opcional: Limpieza de ruido agresiva para vista funcional
        # Las funciones suelen ser muy distintas, así que similitudes bajas suelen ser ruido puro.
        # similarity_matrix[similarity_matrix < 0.3] = 0
        
        return similarity_matrix

    def save_results(self, matrix, output_base):
        csv_path = f"{output_base}_matrix.csv"
        png_path = f"{output_base}_matrix.png"
        
        # Guardar CSV
        df_matrix = pd.DataFrame(matrix, index=self.class_names, columns=self.class_names)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_matrix.to_csv(csv_path)
        print(f"[Functional] Matriz guardada: {csv_path}")
        
        # Guardar Heatmap
        plt.figure(figsize=(10, 8))
        # Usamos 'inferno' o 'magma' para diferenciar visualmente de la vista semántica
        plt.imshow(matrix, cmap='inferno', interpolation='nearest')
        plt.colorbar(label='Similitud Funcional')
        plt.title(f'Matriz Funcional (MPNet) - {self.n}x{self.n}')
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"[Functional] Heatmap guardado: {png_path}")

def main():
    if len(sys.argv) != 4:
        print("Uso: python fase2_build_functional_matrix.py <FUN_CSV> <CORE_CSV> <OUTPUT_BASE>")
        sys.exit(1)

    fun_csv = sys.argv[1]
    core_csv = sys.argv[2]
    output_base = sys.argv[3]

    if not os.path.exists(fun_csv):
        print(f"Error: No existe {fun_csv}")
        sys.exit(1)

    builder = FunctionalMatrixBuilder(fun_csv, core_csv)
    embeddings = builder.generate_embeddings()
    matrix = builder.build_similarity_matrix(embeddings)
    builder.save_results(matrix, output_base)

if __name__ == '__main__':
    main()