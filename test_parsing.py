# import javalang
# import os

# project_dir = "monoliths/jPetStore/src"

# for root, _, files in os.walk(project_dir):
#     for file in files:
#         if file.endswith(".java"):
#             with open(os.path.join(root, file), "r", encoding="utf-8") as f:
#                 code = f.read()
#             tree = javalang.parse.parse(code)
            
#             # Clases
#             for _, node in tree.filter(javalang.tree.ClassDeclaration):
#                 print("Clase:", node.name)
            
#             # Métodos
#             for _, node in tree.filter(javalang.tree.MethodDeclaration):
#                 print("  Método:", node.name)

import javalang
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------
# Parser de dependencias
# ----------------------
def extract_dependencies(java_code):
    tree = javalang.parse.parse(java_code)
    classes = []
    edges = []

    for path, cls in tree.filter(javalang.tree.ClassDeclaration):
        classes.append(cls.name)

        # Herencia (extends)
        if cls.extends:
            edges.append((cls.name, cls.extends.name))

        # Implementaciones (implements)
        if cls.implements:
            for impl in cls.implements:
                edges.append((cls.name, impl.name))

    return classes, edges


def parse_project(src_folder):
    all_classes = set()
    all_edges = []

    for root, _, files in os.walk(src_folder):
        for f in files:
            if f.endswith(".java"):
                with open(os.path.join(root, f), encoding="utf-8") as fp:
                    code = fp.read()
                try:
                    classes, edges = extract_dependencies(code)
                    all_classes.update(classes)
                    all_edges.extend(edges)
                except Exception as e:
                    print(f"Error en {f}: {e}")

    return sorted(all_classes), all_edges


# ----------------------
# Construir matriz y grafo
# ----------------------
def build_adjacency_matrix(classes, edges):
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((n, n), dtype=int)

    for src, dst in edges:
        if src in idx and dst in idx:
            matrix[idx[src], idx[dst]] = 1

    return matrix, idx


if __name__ == "__main__":
    project_path = "monoliths/jPetStore/src"
    # project_path = "monoliths/sample.plantsbywebsphere/src"

    classes, edges = parse_project(project_path)

    print("Clases encontradas:", len(classes))
    print("Dependencias encontradas:", len(edges))

    # Matriz
    matrix, idx = build_adjacency_matrix(classes, edges)
    df = pd.DataFrame(matrix, index=classes, columns=classes)
    print(df.head())

    # Visualizar matriz
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap="Blues", interpolation="nearest")
    plt.title("Matriz de Adyacencia")
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=90, fontsize=6)
    plt.yticks(range(len(classes)), classes, fontsize=6)
    plt.tight_layout()
    plt.show()

    # Grafo con networkx
    G = nx.DiGraph()
    G.add_nodes_from(classes)
    G.add_edges_from(edges)

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", arrowsize=12, font_size=8)
    plt.title("Grafo de Dependencias entre Clases")
    plt.show()
