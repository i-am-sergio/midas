# fase5_evaluation.py

"""
FASE 5: Evaluacion de Metricas de Calidad (SM, ICP, IFN, NED).

Calcula metricas arquitectonicas basadas en el grafo de dependencias original
y la descomposicion propuesta (JSON de clusteres).

Metricas:
1. SM (Structural Modularity): Cohesion vs Acoplamiento. (Mayor es mejor)
2. ICP (Inter Call Percentage): Dependencias externas. (Menor es mejor)
3. IFN (Interface Number): Complejidad de interfaz publica. (Menor es mejor)
4. NED (Non-Extreme Distribution): Balance de tamaño [5, 20]. (Menor es mejor)

Uso:
    python fase5_evaluation.py <CORE_CLASSES_CSV> <CLUSTERS_JSON> <OUTPUT_METRICS_CSV>
"""

import sys
import os
import csv
import json
import ast
import pandas as pd
import numpy as np

class MetricsEvaluator:
    def __init__(self, core_csv, clusters_json):
        self.core_csv = core_csv
        self.clusters_json = clusters_json
        
        # Estructuras de datos
        self.graph = {}       # {source: {target: weight}}
        self.clusters = {}    # {cluster_id: [class_list]}
        self.class_map = {}   # {class_name: cluster_id}
        
        self._load_data()

    def _load_data(self):
        print(f"[Evaluation] Cargando grafo estructural: {self.core_csv}")
        try:
            df = pd.read_csv(self.core_csv)
            for _, row in df.iterrows():
                src = row['class']
                try:
                    rels = ast.literal_eval(row['relations'])
                except:
                    rels = {}
                self.graph[src] = rels
        except Exception as e:
            print(f"!!! Error leyendo CSV: {e}")
            sys.exit(1)

        print(f"[Evaluation] Cargando clusters: {self.clusters_json}")
        try:
            with open(self.clusters_json, 'r') as f:
                self.clusters = json.load(f)
            
            # Crear mapa inverso (clase -> cluster_id)
            for cid, classes in self.clusters.items():
                for cls in classes:
                    self.class_map[cls] = cid
                    
        except Exception as e:
            print(f"!!! Error leyendo JSON: {e}")
            sys.exit(1)

    # --- 1. Structural Modularity (SM) ---
    def calculate_sm(self):
        """
        SM = (1/K * Sum(mu_i / m_i^2)) - (1 / (K*(K-1)/2) * Sum(sigma_ij / 2*m_i*m_j))
        Usa aristas binarias (existencia), no pesos, para medir densidad pura.
        """
        K = len(self.clusters)
        if K < 2: return 0.0 # SM no tiene sentido con 1 solo servicio

        # Termino 1: Cohesion
        cohesion_sum = 0
        for cid, classes in self.clusters.items():
            m_i = len(classes)
            if m_i == 0: continue
            
            # mu_i: Aristas internas
            mu_i = 0
            for src in classes:
                if src not in self.graph: continue
                for tgt in self.graph[src]:
                    if tgt in classes: # Destino esta en el mismo cluster
                        mu_i += 1
            
            # Densidad interna
            cohesion_sum += mu_i / (m_i ** 2)
        
        avg_cohesion = cohesion_sum / K

        # Termino 2: Acoplamiento
        coupling_sum = 0
        cluster_ids = list(self.clusters.keys())
        
        pair_count = 0
        for idx_i in range(len(cluster_ids)):
            for idx_j in range(idx_i + 1, len(cluster_ids)):
                pair_count += 1
                cid_i = cluster_ids[idx_i]
                cid_j = cluster_ids[idx_j]
                
                m_i = len(self.clusters[cid_i])
                m_j = len(self.clusters[cid_j])
                if m_i == 0 or m_j == 0: continue
                
                # sigma_ij: Aristas entre i y j (bidireccional)
                sigma_ij = 0
                
                # De i a j
                for src in self.clusters[cid_i]:
                    if src in self.graph:
                        for tgt in self.graph[src]:
                            if tgt in self.clusters[cid_j]:
                                sigma_ij += 1
                
                # De j a i
                for src in self.clusters[cid_j]:
                    if src in self.graph:
                        for tgt in self.graph[src]:
                            if tgt in self.clusters[cid_i]:
                                sigma_ij += 1
                
                coupling_sum += sigma_ij / (2 * m_i * m_j)

        if pair_count == 0:
            avg_coupling = 0
        else:
            avg_coupling = coupling_sum / pair_count

        return avg_cohesion - avg_coupling

    # --- 2. Inter Call Percentage (ICP) ---
    def calculate_icp(self):
        """
        ICP = (Llamadas externas) / (Total llamadas)
        Usa PESOS para reflejar la intensidad de las llamadas.
        """
        total_calls = 0
        inter_calls = 0
        
        for src, targets in self.graph.items():
            if src not in self.class_map: continue # Clase ignorada/filtrada
            src_cluster = self.class_map[src]
            
            for tgt, weight in targets.items():
                if tgt not in self.class_map: continue
                
                tgt_cluster = self.class_map[tgt]
                
                total_calls += weight
                if src_cluster != tgt_cluster:
                    inter_calls += weight
                    
        if total_calls == 0: return 0.0
        return inter_calls / total_calls

    # --- 3. Interface Number (IFN) ---
    def calculate_ifn(self):
        """
        IFN = Promedio de interfaces por microservicio.
        Una clase es 'interfaz' si recibe llamadas desde OTRO microservicio.
        """
        K = len(self.clusters)
        if K == 0: return 0.0
        
        total_interfaces = 0
        
        for cid, classes in self.clusters.items():
            # Set de clases en este cluster que son llamadas desde fuera
            interfaces_in_cluster = set()
            
            # Verificar quien llama a las clases de este cluster
            # (Ineficiente, pero seguro: iterar todo el grafo)
            for target_class in classes:
                is_interface = False
                # Buscamos si alguien de FUERA llama a target_class
                for src_global, targets_global in self.graph.items():
                    if src_global not in self.class_map: continue
                    
                    # Si el que llama NO esta en este cluster
                    if self.class_map[src_global] != cid:
                        if target_class in targets_global:
                            is_interface = True
                            break
                
                if is_interface:
                    interfaces_in_cluster.add(target_class)
            
            total_interfaces += len(interfaces_in_cluster)
            
        return total_interfaces / K

    # --- 4. Non-Extreme Distribution (NED) ---
    def calculate_ned(self):
        """
        NED = 1 - (Clusters con tamaño apropiado / Total Clusters)
        Tamaño apropiado: [5, 20]
        """
        K = len(self.clusters)
        if K == 0: return 1.0
        
        appropriate_count = 0
        for classes in self.clusters.values():
            size = len(classes)
            if 5 <= size <= 20:
                appropriate_count += 1
            else:
                # Opcional: Log para debug, ya que en sistemas pequeños NED suele ser alto
                # print(f"   -> Cluster size {size} fuera de rango [5, 20]")
                pass
                
        return 1 - (appropriate_count / K)

    def run_all(self, output_file):
        print("[Evaluation] Calculando metricas...")
        
        sm = self.calculate_sm()
        icp = self.calculate_icp()
        ifn = self.calculate_ifn()
        ned = self.calculate_ned()
        
        print(f"   Resultados:")
        print(f"   ---------------------------")
        print(f"   SM  (Higher better): {sm:.4f}")
        print(f"   ICP (Lower better) : {icp:.4f}")
        print(f"   IFN (Lower better) : {ifn:.4f}")
        print(f"   NED (Lower better) : {ned:.4f}")
        print(f"   ---------------------------")
        
        # Guardar en CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['SM', sm])
            writer.writerow(['ICP', icp])
            writer.writerow(['IFN', ifn])
            writer.writerow(['NED', ned])
            
        print(f"[Evaluation] Metricas guardadas en: {output_file}")

def main():
    if len(sys.argv) != 4:
        print("Uso: python fase5_evaluation.py <CORE_CSV> <CLUSTERS_JSON> <OUTPUT_CSV>")
        sys.exit(1)

    core_csv = sys.argv[1]
    clusters_json = sys.argv[2]
    output_csv = sys.argv[3]
    
    if not os.path.exists(core_csv) or not os.path.exists(clusters_json):
        print("Error: Archivos de entrada no encontrados.")
        sys.exit(1)

    evaluator = MetricsEvaluator(core_csv, clusters_json)
    evaluator.run_all(output_csv)

if __name__ == '__main__':
    main()