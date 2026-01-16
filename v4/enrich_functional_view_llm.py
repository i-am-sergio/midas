import sys
import os
import csv
import time
import random
from google import genai
from google.genai import types

# --- CONFIGURACIÓN ---
API_KEY = os.environ.get("GEMINI_API_KEY", "PON_TU_API_KEY_AQUI")
SAFE_DELAY_SECONDS = 7 

class LLMFunctionalEnricher:
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

    def _generate_functional_summary(self, class_name, code_content):
        """
        Prompt enfocado en COMPORTAMIENTO y CASOS DE USO.
        """
        system_prompt = (
            "You are a Software Analyst performing reverse engineering. "
            "Your goal is to identify the USER FEATURES or BUSINESS PROCESSES that trigger this class. "
            "Think: When is this code executed during a user session?"
        )
        
        user_prompt = f"""
        Analyze the Java class '{class_name}'.
        
        TASK:
        Identify the top 1-3 specific User Scenarios or Functional Flows where this class plays a role.
        
        GUIDELINES:
        - Focus on ACTIONS: Browsing, Checkout, Login, Registration, Updating Cart.
        - IGNORE structural definitions (do not say "it stores data").
        - If it's a utility used everywhere, say "General Utility".
        
        OUTPUT FORMAT:
        - A single string joining features with commas.
        - Example: "User Login, Profile Update"
        - Example: "Product Search, Catalog Browsing"
        - Example: "Checkout, Order Confirmation"
        
        Code Snippet:
        ```java
        {code_content[:8000]} 
        ```
        """

        max_retries = 5
        backoff_time = 30 

        for attempt in range(max_retries):
            try:
                # Configuración de seguridad permisiva para análisis de código
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2, # Un poco más de creatividad para deducir flujos
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
                    raise Exception("Empty response text")

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    wait = backoff_time + random.uniform(1, 5)
                    print(f"⏳ [{class_name}] Wait {wait:.1f}s (Rate Limit)...")
                    time.sleep(wait)
                    backoff_time *= 1.5 
                elif "Empty response text" in error_str:
                    time.sleep(2)
                else:
                    print(f"Error no recuperable para {class_name}: {e}")
                    return "General Utility" # Fallback neutral

        return "General Utility"

    def run(self):
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        processed_rows = []
        total = len(self.classes_to_process)
        
        print(f"🚀 Iniciando Extracción de Vista FUNCIONAL...")
        
        for i, class_name in enumerate(self.classes_to_process):
            file_path = self._find_file(class_name)
            
            if not file_path:
                print(f"⚠️ Archivo no encontrado: {class_name}")
                continue
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # Generar descripción funcional
            func_summary = self._generate_functional_summary(class_name, code)
            
            # Formateamos para que MPNet entienda que son acciones
            # "Account participates in User Login, Profile Update"
            final_text = f"{class_name} is used in: {func_summary}"
            
            print(f"⚡ [{i+1}/{total}] {class_name} -> {func_summary}")
            
            processed_rows.append({
                'class_name': class_name,
                'functional_text': final_text 
            })
            
            time.sleep(SAFE_DELAY_SECONDS)

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['class_name', 'functional_text'])
            writer.writeheader()
            writer.writerows(processed_rows)
            
        print(f"\n🎉 Vista Funcional generada: {self.output_csv}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Uso: python enrich_functional_view_llm.py <CSV_CLASES> <SRC_DIR> <OUTPUT_CSV>")
        sys.exit(1)
        
    LLMFunctionalEnricher(sys.argv[1], sys.argv[2], sys.argv[3]).run()