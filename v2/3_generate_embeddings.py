# 3_generate_embeddings_functional.py
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os
import json

# ===============================
# Parámetros
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

INPUT_PATH1 = "results/data_cleaned_functional.csv"
OUTPUT_PATH1 = "results/embeddings_functional.csv"
INPUT_PATH2 = "results/data_cleaned_semantic.csv"
OUTPUT_PATH2 = "results/embeddings_semantic.csv"
INPUT_PATH3 = "results/data_cleaned_structural.csv"
OUTPUT_PATH3 = "results/embeddings_structural.csv"

BATCH_SIZE = 16
MAX_LENGTH = 512
# ===============================


class MPNetEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Cargando modelo: {model_name} en {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts, batch_size=BATCH_SIZE):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.mean_pooling(outputs, inputs["attention_mask"])

            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)


def apply_generate_embeddings(input_path, output_path):
    print("=== Generación de embeddings funcionales ===")
    df = pd.read_csv(input_path)

    if "optimized_text" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'optimized_text'")

    texts = df["optimized_text"].astype(str).tolist()
    print(f"Total textos: {len(texts)}")

    embedder = MPNetEmbedder()
    embeddings = embedder.get_embeddings(texts)

    # Convertir cada vector en string JSON
    df["embedding"] = [json.dumps(vec.tolist()) for vec in embeddings]

    # Crear carpeta si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar CSV con columnas originales + embedding
    df.to_csv(output_path, index=False)

    print(f"Embeddings guardados en: {output_path}")
    print(f"Shape matriz embeddings: {embeddings.shape}")

    # Mostrar ejemplo
    # print("\nEjemplo fila con embedding:")
    # print(df.head(1).to_string(index=False))


if __name__ == "__main__":
    apply_generate_embeddings(INPUT_PATH1, OUTPUT_PATH1)
    apply_generate_embeddings(INPUT_PATH2, OUTPUT_PATH2)
    apply_generate_embeddings(INPUT_PATH3, OUTPUT_PATH3)
