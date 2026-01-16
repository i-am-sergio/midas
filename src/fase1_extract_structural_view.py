# fase1_extract_structural_view.py
import sys
import os
import csv
import re
from collections import defaultdict

class RelationAnalyzer:
    """
    Analiza las relaciones estructurales entre las clases del núcleo funcional.
    """

    # Puntuaciones
    INHERITANCE_SCORE = 15
    INSTANTIATION_SCORE = 8
    ATTRIBUTE_SCORE = 5
    METHOD_SIG_SCORE = 3

    def __init__(self, project_root, input_csv_path):
        self.project_root = project_root
        self.classes_ = {}
        
        # Cargar SOLO las clases listadas en el CSV filtrado (Core Classes)
        self._load_data_from_csv(input_csv_path)
        
        # Cargar contenido de los archivos Java para análisis
        self._load_java_file_contents()

    def _load_data_from_csv(self, csv_path):
        print(f"[Structural] Cargando clases core desde: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # El CSV de entrada (fase1_extract_core_classes.py) tiene:
                # filename, class, attributes, methods
                
                simple_name = row['class']
                
                self.classes_[simple_name] = {
                    'filename': row['filename'],
                    'simpleName': simple_name,
                    'attributesStr': row['attributes'],
                    'methodsStr': row['methods'],
                    'fullContent': '' # Se leerá después
                }
        
        print(f"[Structural] {len(self.classes_)} clases cargadas para análisis.")

    def _load_java_file_contents(self):
        for info in self.classes_.values():
            try:
                with open(info['filename'], 'r', encoding='utf-8', errors='ignore') as f:
                    info['fullContent'] = f.read()
            except Exception as e:
                print(f"Error leyendo {info['filename']}: {e}")

    def analyze_and_save(self, output_path):
        print("[Structural] Analizando relaciones...")
        relation_scores = defaultdict(lambda: defaultdict(int))

        for source_class, info in self.classes_.items():
            self._analyze_inheritance(source_class, info, relation_scores)
            self._analyze_signatures(source_class, info, relation_scores)
            self._analyze_body_instantiations(source_class, info, relation_scores)
        
        self._save_results(output_path, relation_scores)

    # --- Métodos de Análisis (Idénticos a tu versión anterior) ---

    def _analyze_inheritance(self, source_class, info, scores):
        regex = r'\b(?:extends|implements)\s+((?:\w+(?:,\s*\w+)*))'
        match = re.search(regex, info['fullContent'])
        if match:
            related_classes = re.split(r'[,\s]+', match.group(1))
            for related in related_classes:
                if related in self.classes_:
                    scores[source_class][related] += self.INHERITANCE_SCORE

    def _analyze_signatures(self, source_class, info, scores):
        # Combinar atributos y métodos para buscar referencias de tipos
        combined_sigs = info['attributesStr'] + " " + info['methodsStr']
        
        # Buscar cualquier palabra Capitalizada que coincida con una clase del núcleo
        regex = r'\b([A-Z][a-zA-Z0-9_]+)\b'
        potential_types = set(re.findall(regex, combined_sigs))
        
        for target_class in potential_types:
            if target_class in self.classes_ and target_class != source_class:
                # Distinguir si es atributo o parámetro/retorno
                if re.search(r'\b' + re.escape(target_class) + r'\b', info['attributesStr']):
                    scores[source_class][target_class] += self.ATTRIBUTE_SCORE
                else:
                    scores[source_class][target_class] += self.METHOD_SIG_SCORE

    def _analyze_body_instantiations(self, source_class, info, scores):
        for target_class in self.classes_:
            if source_class == target_class: continue
            
            # Buscar 'new TargetClass(...)'
            regex = r'\bnew\s+' + re.escape(target_class) + r'\s*\('
            matches = re.findall(regex, info['fullContent'])
            
            if matches:
                scores[source_class][target_class] += self.INSTANTIATION_SCORE * len(matches)

    def _save_results(self, output_path, scores):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['class', 'relations'])
            
            # Guardar TODAS las clases core, incluso si no tienen relaciones
            for cls in sorted(self.classes_.keys()):
                rels = dict(scores[cls])
                writer.writerow([cls, str(rels)])
                
        print(f"[Structural] Vista guardada en: {output_path}")

def main():
    if len(sys.argv) != 4:
        print("Uso: python fase1_extract_structural_view.py <PROJECT_ROOT> <INPUT_CORE_CSV> <OUTPUT_RELATIONS_CSV>")
        sys.exit(1)

    project_root = sys.argv[1]
    input_csv = sys.argv[2]
    output_csv = sys.argv[3]

    analyzer = RelationAnalyzer(project_root, input_csv)
    analyzer.analyze_and_save(output_csv)

if __name__ == '__main__':
    main()