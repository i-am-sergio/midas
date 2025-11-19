import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ast
import os

def process_structural_view(csv_path, output_csv, output_plot):
    """
    Procesa la vista estructural:
    1. Carga las relaciones de clase desde el CSV.
    2. Construye la matriz de adyacencia (A_str).
    3. Normaliza la matriz al rango [0, 1].
    4. Guarda la matriz como CSV y como heatmap.
    """
    print("Iniciando procesamiento de la Vista Estructural...")
    
    # 1. Cargar datos
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de entrada: {csv_path}")
        return

    # Obtener la lista completa de clases del "núcleo funcional"
    all_classes = sorted(list(df['class']))
    num_classes = len(all_classes)
    
    # Crear un mapeo de nombre de clase a índice para acceso rápido
    class_to_index = {name: i for i, name in enumerate(all_classes)}
    
    # 2. Construir la matriz de adyacencia A_str
    # Inicializar una matriz de ceros
    adj_matrix = np.zeros((num_classes, num_classes))

    for _, row in df.iterrows():
        source_class = row['class']
        
        # Evaluar de forma segura el string que contiene el diccionario
        try:
            relations = ast.literal_eval(row['relations'])
        except (ValueError, SyntaxError):
            print(f"Advertencia: No se pudo parsear relaciones para {source_class}. Omitiendo.")
            continue
            
        if source_class not in class_to_index:
            continue
            
        idx_i = class_to_index[source_class]
        
        for target_class, weight in relations.items():
            if target_class in class_to_index:
                idx_j = class_to_index[target_class]
                
                # Asignar el peso
                adj_matrix[idx_i, idx_j] = weight
                
                # Opcional: hacer la matriz simétrica si las relaciones son bidireccionales
                # adj_matrix[idx_j, idx_i] = weight 

    print(f"Matriz de adyacencia construida (Dimensiones: {adj_matrix.shape})")

    # 3. Normalizar la matriz A_str
    # Usamos MinMaxScaler para escalar todo al rango [0, 1]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(adj_matrix)
    
    # Convertir de nuevo a DataFrame para mejor legibilidad
    normalized_matrix_df = pd.DataFrame(normalized_data, index=all_classes, columns=all_classes)

    # 4. Guardar la matriz normalizada como CSV
    normalized_matrix_df.to_csv(output_csv)
    print(f"Matriz estructural normalizada guardada en: {output_csv}")

    # 5. Guardar el heatmap
    plt.figure(figsize=(14, 11))
    sns.heatmap(normalized_matrix_df, cmap="viridis", xticklabels=False, yticklabels=False)
    plt.title("Heatmap de la Matriz Estructural Normalizada ($A^{(str)}$)", fontsize=16)
    plt.xlabel("Clases", fontsize=12)
    plt.ylabel("Clases", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Heatmap estructural guardado en: {output_plot}")
    plt.close()

if __name__ == "__main__":
    # Asegurarse de que el directorio de salida exista
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Rutas de entrada y salida
    INPUT_FILE = "results/jPetStore_fase1_structural_view.csv"
    OUTPUT_CSV_FILE = os.path.join(output_dir, "fase2_structural_matrix_normalized.csv")
    OUTPUT_PLOT_FILE = os.path.join(output_dir, "fase2_structural_matrix_heatmap.png")

    process_structural_view(INPUT_FILE, OUTPUT_CSV_FILE, OUTPUT_PLOT_FILE)
    print("Procesamiento estructural completado.")