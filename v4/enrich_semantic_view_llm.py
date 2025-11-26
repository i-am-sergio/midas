import sys
import os
import csv
import time
import random
import re  # Importado para limpieza de texto
from google import genai
from google.genai import types

# --- CONFIGURACIÓN ---
API_KEY = os.environ.get("GEMINI_API_KEY", "PON_TU_API_KEY_AQUI")
SAFE_DELAY_SECONDS = 7

# Palabras técnicas a eliminar (Stop words de infraestructura)
FORBIDDEN_TERMS = [
    r'\bdao\b', r'\bimplementation\b', r'\bimpl\b', r'\binterface\b', 
    r'\bclass\b', r'\bjava\b', r'\bdatabase\b', r'\bdb\b', r'\bsql\b', 
    r'\bmap\b', r'\bhashmap\b', r'\bstring\b', r'\bint\b', r'\binteger\b',
    r'\bvoid\b', r'\breturn\b', r'\bjdbc\b', r'\bpersistence\b', r'\bquery\b'
]

class LLMEnricher:
    def __init__(self, input_csv, source_dir, output_csv):
        self.input_csv = input_csv
        self.source_dir = source_dir
        self.output_csv = output_csv
        self.classes_to_process = []
        
        try:
            self.client = genai.Client(api_key=API_KEY)
        except Exception as e:
            print(f"Error inicializando cliente Gemini: {e}")
            sys.exit(1)

        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            col = 'class_name' if 'class_name' in reader.fieldnames else 'class'
            for row in reader:
                self.classes_to_process.append(row[col])

    def _find_file(self, class_name):
        for dirpath, _, filenames in os.walk(self.source_dir):
            for filename in filenames:
                if filename == f"{class_name}.java":
                    return os.path.join(dirpath, filename)
        return None

    def _clean_semantic_text(self, text):
        """
        Elimina palabras técnicas para evitar que el clustering agrupe por 'capa técnica'.
        Mantiene solo la semántica de negocio.
        """
        if not text: return ""
        
        # 1. Convertir a minúsculas para el filtrado
        clean_text = text.lower()
        
        # 2. Eliminar palabras prohibidas usando Regex
        for term in FORBIDDEN_TERMS:
            clean_text = re.sub(term, '', clean_text)
            
        # 3. Limpiar espacios dobles generados por la eliminación
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # 4. Restaurar un poco de formato (opcional, solo capitalización básica)
        return clean_text.capitalize()

    def _generate_summary_with_retry(self, class_name, code_content):
        """
        Prompt enfocado 100% en DOMINIO con manejo robusto de errores y Safety Settings.
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
                # Configuración explícita para evitar bloqueos
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                    max_output_tokens=100, 
                    # Restauramos el thinking_config que funcionaba en tu versión anterior
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    # IMPORTANTE: Desactivar filtros de seguridad para análisis de código
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE"
                        ),
                    ]
                )

                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=user_prompt,
                    config=config
                )
                
                # CORRECCIÓN DEL ERROR: Verificamos si response.text existe antes de usar strip()
                if response.text:
                    return response.text.strip()
                else:
                    # Si no hay texto (pero no dio excepción), puede ser un bloqueo silencioso
                    print(f"⚠️ [{class_name}] Respuesta vacía (posible filtro). Reintentando...")
                    # Lanzamos excepción para activar el mecanismo de retry
                    raise Exception("Empty response text")

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = backoff_time + random.uniform(1, 5)
                    print(f"⏳ [{class_name}] Wait {wait:.1f}s (Rate Limit)...")
                    time.sleep(wait)
                    backoff_time *= 1.5 
                elif "Empty response text" in error_str:
                     # Si fue respuesta vacía, esperamos un poco y reintentamos
                    time.sleep(2)
                else:
                    print(f"Error no recuperable para {class_name}: {e}")
                    return ""

        return ""
    
    def run(self):
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        processed_rows = []
        total = len(self.classes_to_process)
        
        print(f"🚀 Iniciando enriquecimiento (Filtro Anti-Técnico Activado)...")
        
        for i, class_name in enumerate(self.classes_to_process):
            file_path = self._find_file(class_name)
            
            if not file_path:
                print(f"⚠️ Archivo no encontrado: {class_name}")
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # 1. Generar Resumen con LLM
            raw_summary = self._generate_summary_with_retry(class_name, code)
            
            if not raw_summary:
                # Fallback inteligente: Repetir el nombre de la clase limpio
                clean_name = class_name.replace("Dao", "").replace("Impl", "")
                raw_summary = f"{clean_name} represents the business concept of {clean_name}."

            # 2. Limpieza Técnica (Regex)
            clean_summary = self._clean_semantic_text(raw_summary)
            
            # 3. Estrategia de Peso: Repetir el nombre de la clase al inicio
            # Esto ayuda a MPNet a anclar el vector en el nombre.
            # Ejemplo final: "Account Account represents user profile and settings."
            final_text = f"{class_name} {clean_summary}"
            
            print(f"✅ [{i+1}/{total}] {class_name} -> {final_text[:70]}...")
            
            processed_rows.append({
                'class_name': class_name,
                'concatenated_text': final_text 
            })
            
            time.sleep(SAFE_DELAY_SECONDS)

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['class_name', 'concatenated_text'])
            writer.writeheader()
            writer.writerows(processed_rows)
            
        print(f"\n🎉 Procesamiento completado. Archivo generado: {self.output_csv}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Uso: python enrich_semantic_view_llm_v2.py <CSV_CLASES> <SRC_DIR> <OUTPUT_CSV>")
        sys.exit(1)
        
    LLMEnricher(sys.argv[1], sys.argv[2], sys.argv[3]).run()