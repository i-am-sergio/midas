#!/usr/bin/env python3

"""
Fase 2: Preprocesamiento de la Vista Semántica.

Realiza la limpieza semántica (eliminación de stop words y lematización)
sobre el texto concatenado para mejorar la calidad de los embeddings.

Uso:
    python preprocessing_semantic.py <INPUT_CSV> <OUTPUT_CLEAN_CSV>
"""

import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Asegurar que los recursos de NLTK estén descargados
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('omw-1.4') # Open Multilingual WordNet

class SemanticPreprocessor:
    """
    Clase para realizar la limpieza de texto y lematización.
    """
    def __init__(self, input_csv_path: str):
        self.df = pd.read_csv(input_csv_path)
        self.lemmatizer = WordNetLemmatizer()
        # Se utilizan stop words de inglés ya que Java/código usa terminología en inglés.
        self.stop_words = set(stopwords.words('english'))
        
        # Añadir stop words específicas de Java/código (crucial para eliminar ruido)
        self.stop_words.update(['get', 'set', 'is', 'has', 'to', 'from', 'dao', 'impl', 'service', 'factory', 'id', 'new'])

    def _clean_and_lemmatize(self, text: str) -> str:
        """
        Limpia el texto: elimina stop words y aplica lematización.
        """
        # El texto ya está en minúsculas y tokenizado por espacios (gracias a extract_semantic_view.py)
        tokens = text.split()
        
        cleaned_tokens = []
        for token in tokens:
            # 1. Eliminar tokens cortos (ej. 'a', 'b', 'c', que suelen ser ruido o variables de bucle)
            if len(token) <= 2:
                continue
            
            # 2. Eliminar Stop Words
            if token not in self.stop_words:
                # 3. Lematizar (reducir a la forma base)
                lemma = self.lemmatizer.lemmatize(token)
                cleaned_tokens.append(lemma)
                
        return ' '.join(cleaned_tokens)

    def process_and_save(self, output_csv_path: str):
        """
        Procesa la columna 'concatenated_text' y guarda el resultado.
        """
        print(f"[Preprocessor] Iniciando limpieza semántica ({len(self.df)} clases)...")
        
        # Aplicar la función de limpieza a la columna de texto
        self.df['cleaned_text'] = self.df['concatenated_text'].apply(self._clean_and_lemmatize)
        
        # Seleccionar las columnas necesarias para la siguiente fase
        output_df = self.df[['class_name', 'cleaned_text']]
        
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        output_df.to_csv(output_csv_path, index=False)
        
        print(f"[Preprocessor] Limpieza completada. Guardado en: {output_csv_path}")


def main():
    if len(sys.argv) != 3:
        print("Uso: python preprocessing_semantic.py <INPUT_CSV> <OUTPUT_CLEAN_CSV>")
        print("Ejemplo: python preprocessing_semantic.py jpetstore_fase1_semantic_view.csv jpetstore_fase2_semantic_clean.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    preprocessor = SemanticPreprocessor(input_path)
    preprocessor.process_and_save(output_path)

if __name__ == '__main__':
    main()