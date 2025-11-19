import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt

def run_spectral_clustering(fused_matrix_path, output_dir):
    """
    Implementa la Fase 4: Clustering Espectral y Evaluación.
    
    1. Carga la matriz de afinidad fusionada (U).
    2. Convierte U en matriz de distancia (D) para Silhouette.
    3. Itera K de 2 a N-1.
    4. Ejecuta Spectral Clustering y calcula Silhouette Score.
    5. Guarda los puntajes y las asignaciones de clústeres en CSV.
    6. Grafica los puntajes de Silhouette.
    """
    
    print("Iniciando Fase 4: Clustering Espectral y Evaluación...")
    
    # --- 1. Cargar Datos (Matriz de Afinidad U) ---
    try:
        df_fused = pd.read_csv(fused_matrix_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de entrada: {fused_matrix_path}")
        return
    except Exception as e:
        print(f"Error al leer el CSV. Asegúrate de que tenga una 'index_col'. Error: {e}")
        return

    class_names = df_fused.index.tolist()
    n_nodes = len(class_names)
    
    if n_nodes <= 2:
        print(f"Error: Se necesitan al menos 3 nodos para el clustering. Encontrados: {n_nodes}")
        return
        
    print(f"Matriz de afinidad U cargada. Nodos (clases) = {n_nodes}")
    
    affinity_matrix_U = df_fused.values

    # --- CORRECCIÓN 1: Forzar Simetría (para SpectralClustering) ---
    # SpectralClustering espera una matriz simétrica.
    # La advertencia sugería promediar con la transpuesta.
    U_symmetric = (affinity_matrix_U + affinity_matrix_U.T) / 2
    
    # --- 2. Crear Matriz de Distancia (D) para Silhouette ---
    # Usamos la matriz simétrica para consistencia
    distance_matrix_D = 1 - U_symmetric

    # --- CORRECCIÓN 2: Forzar Diagonal a Cero (para Silhouette) ---
    # El error indicó que la diagonal de la matriz de distancia NO era cero.
    # La llenamos con 0.
    np.fill_diagonal(distance_matrix_D, 0)

    # --- 3. Iterar K y Evaluar ---
    k_range = range(2, n_nodes)
    
    silhouette_scores = []
    all_cluster_assignments = [] 

    print(f"Evaluando K desde 2 hasta {n_nodes - 1}...")

    for k in k_range:
        # a. Ejecutar Clustering Espectral
        # Se entrena sobre la matriz de AFINIDAD (U_symmetric)
        model = SpectralClustering(
            n_clusters=k,
            affinity='precomputed',
            assign_labels='kmeans', 
            random_state=42
        )
        
        # Usamos la matriz simétrica para el 'fit'
        model.fit(U_symmetric)
        labels = model.labels_
        
        # --- CORRECCIÓN 3: Capturar Errores Genéricos ---
        try:
            # b. Calcular Puntaje Silhouette
            # Se calcula sobre la matriz de DISTANCIA (D)
            score = silhouette_score(distance_matrix_D, labels, metric='precomputed')
            
            print(f"  K={k}, Puntaje Silhouette: {score:.4f}")
            silhouette_scores.append({'k': k, 'score': score})
            
            # c. Guardar asignaciones de clúster
            for i, class_name in enumerate(class_names):
                all_cluster_assignments.append({
                    'k': k,
                    'class_name': class_name,
                    'cluster_id': labels[i]
                })
                
        except Exception as e: # Cambiado de ValueError a Exception
            # Esto puede pasar si un k es muy alto y crea un clúster con 1 solo miembro
            print(f"  K={k}, No se pudo calcular Silhouette. Error: {e}")
            silhouette_scores.append({'k': k, 'score': np.nan})

    print("Evaluación de Silhouette completada.")

    # --- 4. Guardar Puntajes en CSV ---
    df_scores = pd.DataFrame(silhouette_scores)
    output_scores_csv = os.path.join(output_dir, "silhouette_scores_all_k.csv")
    df_scores.to_csv(output_scores_csv, index=False)
    print(f"Puntajes de Silhouette guardados en: {output_scores_csv}")

    # --- 5. Guardar Asignaciones de Clústeres en CSV ---
    df_clusters = pd.DataFrame(all_cluster_assignments)
    df_clusters = df_clusters.sort_values(by=['k', 'cluster_id', 'class_name'])
    output_clusters_csv = os.path.join(output_dir, "all_cluster_assignments.csv")
    df_clusters.to_csv(output_clusters_csv, index=False)
    print(f"Asignaciones de clústeres guardadas en: {output_clusters_csv}")

    # --- 6. Encontrar y Reportar K Óptimo ---
    if not df_scores.empty and df_scores['score'].notna().any():
        best_k_data = df_scores.loc[df_scores['score'].idxmax()]
        best_k = int(best_k_data['k'])
        best_score = best_k_data['score']
        
        print("\n" + "="*30)
        print("  RESULTADO DE LA FASE 4 (ÓPTIMO)")
        print(f"  El valor K óptimo (máx. Silhouette) es: {best_k}")
        print(f"  Puntaje Silhouette alcanzado: {best_score:.4f}")
        print("="*30 + "\n")

        # --- 7. Guardar Gráfico de Puntajes ---
        output_plot_file = os.path.join(output_dir, "silhouette_scores_plot.png")
        plt.figure(figsize=(12, 7))
        plt.plot(df_scores['k'], df_scores['score'], marker='o', linestyle='--')
        
        plt.axvline(x=best_k, color='red', linestyle=':', 
                    label=f'K Óptimo = {best_k} (Score: {best_score:.4f})')
        
        plt.xlabel("Número de Clústeres (K)")
        plt.ylabel("Coeficiente de Silhouette")
        plt.title("Evaluación de K para Clustering Espectral")
        plt.legend()
        plt.grid(True)
        plt.xticks(range(2, n_nodes, max(1, (n_nodes - 2) // 15))) 
        plt.tight_layout()
        plt.savefig(output_plot_file)
        print(f"Gráfico de puntajes guardado en: {output_plot_file}")
    
    else:
        print("No se pudieron calcular puntajes de Silhouette.")

    print("\nProceso de clustering (Fase 4) completado.")


if __name__ == "__main__":
    # Ruta del archivo de entrada (Fase 3)
    INPUT_FILE = "results_fase3/fused_matrix_0.5_0.5.csv"
    
    # Directorio de salida (Fase 4)
    OUTPUT_DIR = "results_fase4"
    
    # Asegurarse de que el directorio de salida exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    run_spectral_clustering(INPUT_FILE, OUTPUT_DIR)