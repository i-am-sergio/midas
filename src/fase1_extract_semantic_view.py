#!/usr/bin/env python3

"""
FASE 1.2: Extracción de Vista Semántica (Enriquecida con LLM).

Utiliza Google Gemini 2.5 Flash para generar descripciones de dominio funcionales
para cada clase del núcleo, eliminando ruido técnico para favorecer el Vertical Slicing.

Uso:
    export GEMINI_API_KEY="tu_api_key"
    python fase1_extract_semantic_view.py <SOURCE_DIR> <CORE_CLASSES_CSV> <OUTPUT_SEMANTIC_CSV>
"""

import sys
import os
import csv
import time
import random
import re
from google import genai
from google.genai import types

# --- CONFIGURACIÓN ---
# Se intenta leer del entorno, sino usa el placeholder (asegúrate de exportarla en tu .bashrc o script)
API_KEY = os.environ.get("GEMINI_API_KEY", "PON_TU_API_KEY_AQUI")
SAFE_DELAY_SECONDS = 7  # Respetar Free Tier (~10-15 RPM)

# Palabras técnicas a eliminar (Stop words de infraestructura) para limpieza agresiva
FORBIDDEN_TERMS = [
    r'\bdao\b', r'\bimplementation\b', r'\bimpl\b', r'\binterface\b', 
    r'\bclass\b', r'\bjava\b', r'\bdatabase\b', r'\bdb\b', r'\bsql\b', 
    r'\bmap\b', r'\bhashmap\b', r'\bstring\b', r'\bint\b', r'\binteger\b',
    r'\bvoid\b', r'\breturn\b', r'\bjdbc\b', r'\bpersistence\b', r'\bquery\b'
]

class SemanticExtractorLLM:
    def __init__(self, source_dir, core_classes_csv, output_csv):
        self.source_dir = source_dir
        self.core_classes_csv = core_classes_csv
        self.output_csv = output_csv
        self.classes_to_process = []
        
        # Inicializar Cliente Gemini
        try:
            if not API_KEY or "PON_TU_API_KEY" in API_KEY:
                raise ValueError("GEMINI_API_KEY no está configurada correctamente en el entorno.")
            self.client = genai.Client(api_key=API_KEY)
        except Exception as e:
            print(f"❌ Error crítico inicializando Gemini: {e}")
            sys.exit(1)

        # Cargar lista de clases del núcleo (Core Classes)
        with open(self.core_classes_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # El CSV de core classes suele tener columna 'class'
            col = 'class' if 'class' in reader.fieldnames else 'class_name'
            for row in reader:
                self.classes_to_process.append(row[col])

    def _find_file(self, class_name):
        """Busca el archivo .java recursivamente en el directorio fuente."""
        for dirpath, _, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename == f"{class_name}.java":
                    return os.path.join(dirpath, filename)
        return None

    def _clean_semantic_text(self, text):
        """
        Elimina palabras técnicas para evitar agrupación por capa técnica.
        Mantiene solo la semántica de negocio.
        """
        if not text: return ""
        
        # 1. Convertir a minúsculas
        clean_text = text.lower()
        
        # 2. Eliminar palabras prohibidas
        for term in FORBIDDEN_TERMS:
            clean_text = re.sub(term, '', clean_text)
            
        # 3. Limpiar espacios extra
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # 4. Restaurar capitalización básica
        return clean_text.capitalize()

    def _generate_summary_with_retry(self, class_name, code_content):
        """
        Genera resumen de dominio usando Gemini con reintentos robustos.
        """
        simple_name = class_name.replace("Dao", "").replace("Impl", "").replace("Service", "")
        
        system_prompt = (
            "You are a Domain-Driven Design expert. "
            "Your task is to identify the REAL-WORLD BUSINESS CONCEPT representing this class. "
            "IGNORE architectural patterns (DAO, Service, Controller)."
        )
        
        user_prompt = f"""
        Analyze the Java class '{class_name}'.
        
        CORE INSTRUCTIONS:
        1. Ignore technical details (SQL, connection, parsing, boilerplate).
        2. If the class is 'AccountDao', describe 'Account'. If it is 'OrderService', describe 'Order'.
        3. What business entity does this represent?
        
        OUTPUT FORMAT:
        - Start the sentence strictly with the business entity name: "{simple_name} represents..."
        - Do NOT use words like: Database, SQL, Interface, Implementation, CRUD, Persist.
        - One short sentence.
        
        Code Snippet:
        ```java
        {code_content[:7000]} 
        ```
        """

        max_retries = 5
        backoff_time = 30 

        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    max_output_tokens=100,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    ]
                )

                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=config
                )
                
                if response.text:
                    return response.text.strip()
                else:
                    print(f"⚠️ [{class_name}] Respuesta vacía. Reintentando...")
                    raise Exception("Empty response text")

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = backoff_time + random.uniform(1, 5)
                    print(f"⏳ [{class_name}] Cuota excedida (429). Esperando {wait:.1f}s...")
                    time.sleep(wait)
                    backoff_time *= 1.5 
                elif "Empty response text" in error_str:
                    time.sleep(2)
                else:
                    print(f"❌ Error no recuperable para {class_name}: {e}")
                    return ""

        return ""
    
    def run(self):
        # Asegurar directorio de salida
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        
        processed_rows = []
        total = len(self.classes_to_process)
        
        print(f"[Semántica] Iniciando extracción con Gemini 2.5 Flash ({total} clases)...")
        
        for i, class_name in enumerate(self.classes_to_process):
            file_path = self._find_file(class_name)
            
            if not file_path:
                print(f"⚠️ Archivo no encontrado: {class_name}")
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # 1. Generar Resumen
            raw_summary = self._generate_summary_with_retry(class_name, code)
            
            if not raw_summary:
                # Fallback: Usar nombre de clase limpio
                clean_name = class_name.replace("Dao", "").replace("Impl", "")
                raw_summary = f"{clean_name} represents the business concept of {clean_name}."

            # 2. Limpieza Técnica
            clean_summary = self._clean_semantic_text(raw_summary)
            
            # 3. Estrategia de Peso (Nombre + Resumen Limpio)
            final_text = f"{class_name} {clean_summary}"
            
            print(f"✅ [{i+1}/{total}] {class_name} -> {final_text[:60]}...")
            
            processed_rows.append({
                'class_name': class_name,
                'concatenated_text': final_text 
            })
            
            # Rate Limiting
            time.sleep(SAFE_DELAY_SECONDS)

        # Guardar CSV
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['class_name', 'concatenated_text'])
            writer.writeheader()
            writer.writerows(processed_rows)
            
        print(f"[Semántica] Vista semántica guardada en: {self.output_csv}")

def main():
    if len(sys.argv) != 4:
        print("Uso: python fase1_extract_semantic_view.py <SOURCE_DIR> <CORE_CLASSES_CSV> <OUTPUT_CSV>")
        sys.exit(1)

    source_dir = sys.argv[1]
    core_classes_csv = sys.argv[2]
    output_csv = sys.argv[3]
    
    if not os.path.exists(core_classes_csv):
        print(f"Error: Archivo de clases core no encontrado: {core_classes_csv}")
        sys.exit(1)

    extractor = SemanticExtractorLLM(source_dir, core_classes_csv, output_csv)
    extractor.run()

if __name__ == '__main__':
    main()