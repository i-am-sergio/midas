import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Datos de ejemplo para JPetStore

casos_uso = [
    "Browsing the Product List",
    "Browsing the Catalog",
    "Searching the Catalog",
    "Browsing the items List",
    "Add cart Items",
    "Update items in the Cart",
    "Remove items from cart",
    "View Cart Items",
    "Make payment",
    "List Order Items",
    "View Order Status",
    "Confirm Order",
    "Get Total amount",
    "Change Shipping Info ",
    "Signing Up",
    "Signing In",
    "Signout",
    "Manage Item",
    "Manage Account",
    "Manage Order",
    "Manage product/category"
]

endpoints_openapi = [
    # Autenticación
    "/auth/login",
    "/auth/logout",
    "/auth/register",
    
    # Usuarios
    "/users/{username}",
    "/users/{username}/profile",
    "/users/{username}/addresses",
    "/users/{username}/orders",
    
    # Productos
    "/products",
    "/products/categories",
    "/products/categories/{categoryId}",
    "/products/search",
    "/products/filter",
    
    # Carrito
    "/cart/{userId}",
    "/cart/{userId}/items",
    "/cart/{userId}/items/{itemId}",
    
    # Órdenes
    "/orders",
    "/orders/{orderId}",
    "/orders/{orderId}/status"
]

# 2. Generar embeddings y matrices de similitud
print("Generando embeddings y matrices de similitud...")
modelo = SentenceTransformer('all-mpnet-base-v2')

# Embeddings para casos de uso
embeddings_casos_uso = modelo.encode(casos_uso)
similitud_casos_uso = cosine_similarity(embeddings_casos_uso)

# Embeddings para endpoints
embeddings_endpoints = modelo.encode(endpoints_openapi)
similitud_endpoints = cosine_similarity(embeddings_endpoints)

# Asegurar simetría y diagonal=1
similitud_casos_uso = (similitud_casos_uso + similitud_casos_uso.T) / 2
similitud_endpoints = (similitud_endpoints + similitud_endpoints.T) / 2
np.fill_diagonal(similitud_casos_uso, 1.0)
np.fill_diagonal(similitud_endpoints, 1.0)

print("Dimensiones originales:")
print(f"Casos de uso: {similitud_casos_uso.shape}")
print(f"Endpoints: {similitud_endpoints.shape}")

# 3. Visualizar matrices originales
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Matriz de casos de uso
sns.heatmap(similitud_casos_uso, ax=ax1, cmap='YlOrRd', annot=False,
           xticklabels=[f"CU{i+1}" for i in range(len(casos_uso))],
           yticklabels=[f"CU{i+1}" for i in range(len(casos_uso))])
ax1.set_title('Matriz de Similitud - Casos de Uso')
ax1.tick_params(axis='x', rotation=45)
ax1.tick_params(axis='y', rotation=0)

# Matriz de endpoints
sns.heatmap(similitud_endpoints, ax=ax2, cmap='YlOrRd', annot=False,
           xticklabels=[f"EP{i+1}" for i in range(len(endpoints_openapi))],
           yticklabels=[f"EP{i+1}" for i in range(len(endpoints_openapi))])
ax2.set_title('Matriz de Similitud - Endpoints')
ax2.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig("matrices_similitud_iniciales.png")

# 4. Crear matriz de mapeo entre casos de uso y endpoints
print("\nCreando matriz de mapeo...")
def crear_matriz_mapeo(casos_uso, endpoints):
    modelo = SentenceTransformer('all-mpnet-base-v2')
    emb_casos = modelo.encode(casos_uso)
    emb_endpoints = modelo.encode(endpoints)
    matriz_mapeo = cosine_similarity(emb_casos, emb_endpoints)
    return matriz_mapeo

matriz_mapeo = crear_matriz_mapeo(casos_uso, endpoints_openapi)
print(f"Matriz de mapeo: {matriz_mapeo.shape}")

# Visualizar matriz de mapeo
plt.figure(figsize=(12, 8))
sns.heatmap(matriz_mapeo, cmap='YlOrRd', annot=True, fmt='.2f',
           xticklabels=[f"EP{i+1}" for i in range(len(endpoints_openapi))],
           yticklabels=[f"CU{i+1}" for i in range(len(casos_uso))])
plt.title('Matriz de Mapeo - Similitud entre Casos de Uso y Endpoints')
plt.xlabel('Endpoints')
plt.ylabel('Casos de Uso')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("matriz_mapeo.png")

# 5. Encontrar mapeo óptimo usando algoritmo húngaro
print("\nEncontrando mapeo óptimo...")
def encontrar_mapeo_optimo(matriz_similitud):
    costo = 1 - matriz_similitud
    row_ind, col_ind = linear_sum_assignment(costo)
    mapeo = {}
    for i, j in zip(row_ind, col_ind):
        mapeo[i] = j
    return mapeo

mapeo_optimo = encontrar_mapeo_optimo(matriz_mapeo)
print("Mapeo óptimo encontrado:")
for cu_idx, ep_idx in mapeo_optimo.items():
    print(f"CU{cu_idx+1} -> EP{ep_idx+1}: '{casos_uso[cu_idx]}' ↔ '{endpoints_openapi[ep_idx]}'")

# 6. Expandir matrices al tamaño máximo
print(f"\nExpandiendo matrices al tamaño máximo...")
n_max = max(similitud_casos_uso.shape[0], similitud_endpoints.shape[0])
print(f"Tamaño máximo del grafo unificado: {n_max}")

def expandir_matriz_mejorada(matriz_original, tamaño_destino, mapeo, matriz_mapeo):
    n_original = matriz_original.shape[0]
    matriz_expandida = np.zeros((tamaño_destino, tamaño_destino))
    
    # Llenar con valores de la matriz original para nodos mapeados
    for i in range(n_original):
        for j in range(n_original):
            if i in mapeo and j in mapeo:
                idx_i = mapeo[i]
                idx_j = mapeo[j]
                matriz_expandida[idx_i, idx_j] = matriz_original[i, j]
    
    # Para nodos no mapeados, usar información de la matriz de mapeo
    valor_promedio = np.mean(matriz_original)
    for i in range(tamaño_destino):
        for j in range(tamaño_destino):
            if matriz_expandida[i, j] == 0:
                if i != j:
                    # Buscar similitud en la matriz de mapeo
                    if i < matriz_mapeo.shape[1] and j < matriz_mapeo.shape[1]:
                        # Ambos son endpoints, usar similitud promedio
                        matriz_expandida[i, j] = valor_promedio * 0.7
                    else:
                        matriz_expandida[i, j] = valor_promedio * 0.5
                else:
                    # Diagonal
                    matriz_expandida[i, j] = 1.0
    
    return matriz_expandida

similitud_casos_expandida = expandir_matriz_mejorada(
    similitud_casos_uso, n_max, mapeo_optimo, matriz_mapeo
)
similitud_endpoints_expandida = expandir_matriz_mejorada(
    similitud_endpoints, n_max, 
    {v: k for k, v in mapeo_optimo.items()}, 
    matriz_mapeo.T
)

print(f"Matrices expandidas:")
print(f"Casos de uso expandida: {similitud_casos_expandida.shape}")
print(f"Endpoints expandida: {similitud_endpoints_expandida.shape}")

# 7. Función de fusión autoponderada (la misma que antes)
def fusion_autoponderada(matrices_vistas, max_iter=100, tol=1e-6):
    n_vistas = len(matrices_vistas)
    n = matrices_vistas[0].shape[0]
    
    # Inicializar con más diversidad
    U = np.random.rand(n, n) * 0.1 + np.mean(matrices_vistas, axis=0) * 0.9
    U = (U + U.T) / 2
    np.fill_diagonal(U, 1.0)
    
    # Inicializar pesos con más variación
    pesos = np.random.dirichlet(np.ones(n_vistas))
    
    historial = {'error_total': [], 'pesos': [], 'cambio_U': []}
    
    print("Iniciando fusión autoponderada...")
    print(f"Dimensiones: U[{U.shape}], Vistas: {n_vistas}")
    print(f"Pesos iniciales: {pesos}")
    
    for iteracion in range(max_iter):
        U_prev = U.copy()
        
        # Actualizar pesos
        new_pesos = np.zeros(n_vistas)
        for v in range(n_vistas):
            diff = np.linalg.norm(U - matrices_vistas[v], ord=1)
            if diff > 1e-10:
                new_pesos[v] = 1 / (2 * np.sqrt(diff))
            else:
                new_pesos[v] = 1.0
        
        suma_pesos = np.sum(new_pesos)
        if suma_pesos > 0:
            pesos = new_pesos / suma_pesos
        
        # Actualizar U
        numerador = np.zeros_like(U)
        denominador = np.zeros_like(U)
        
        for v in range(n_vistas):
            diff_matrix = U - matrices_vistas[v]
            weighted_matrix = matrices_vistas[v] + 0.1 * np.tanh(diff_matrix * 10)
            numerador += pesos[v] * weighted_matrix
            denominador += pesos[v]
        
        U_new = np.zeros_like(U)
        mask = denominador > 0
        U_new[mask] = numerador[mask] / denominador[mask]
        
        alpha = 0.7
        U = alpha * U_new + (1 - alpha) * U_prev
        U = (U + U.T) / 2
        np.fill_diagonal(U, 1.0)
        
        # Calcular métricas
        cambio_U = np.linalg.norm(U - U_prev, 'fro')
        error_total = 0
        for v in range(n_vistas):
            error_total += pesos[v] * np.linalg.norm(U - matrices_vistas[v], ord=1)
        
        historial['error_total'].append(error_total)
        historial['pesos'].append(pesos.copy())
        historial['cambio_U'].append(cambio_U)
        
        if iteracion % 5 == 0:
            print(f"Iteración {iteracion + 1}: Error = {error_total:.6f}, Cambio U = {cambio_U:.6f}, Pesos = {pesos}")
        
        if cambio_U < tol and iteracion > 10:
            print(f"Convergencia alcanzada en iteración {iteracion + 1}")
            break
            
    print("Fusión completada")
    print(f"Pesos finales: {pesos}")
    return U, pesos, historial

# 8. Aplicar fusión autoponderada
print("\n" + "=" * 80)
print("APLICANDO FUSIÓN AUTOPONDERADA")
print("=" * 80)

matrices_vistas_aligned = [
    similitud_casos_expandida,
    similitud_endpoints_expandida
]

for i in range(len(matrices_vistas_aligned)):
    matrices_vistas_aligned[i] = (matrices_vistas_aligned[i] + matrices_vistas_aligned[i].T) / 2
    np.fill_diagonal(matrices_vistas_aligned[i], 1.0)
    matrices_vistas_aligned[i] = np.clip(matrices_vistas_aligned[i], 0, 1)

U_unificada, pesos_aprendidos, historial = fusion_autoponderada(matrices_vistas_aligned)

# 9. Visualizar resultados finales
print("\n" + "=" * 80)
print("RESULTADOS FINALES")
print("=" * 80)

print(f"Pesos finales aprendidos:")
print(f"• Vista Casos de Uso: {pesos_aprendidos[0]:.4f}")
print(f"• Vista Endpoints: {pesos_aprendidos[1]:.4f}")

# Visualizar matriz unificada
plt.figure(figsize=(10, 8))
sns.heatmap(U_unificada, cmap='YlOrRd', 
           xticklabels=[f"N{i+1}" for i in range(U_unificada.shape[0])],
           yticklabels=[f"N{i+1}" for i in range(U_unificada.shape[0])])
plt.title('Matriz Unificada U - Fusión Autoponderada')
plt.tight_layout()
plt.savefig("matriz_unificada.png")

# Visualizar convergencia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(historial['error_total'])
ax1.set_title('Evolución del Error Total')
ax1.grid(True, alpha=0.3)

pesos_array = np.array(historial['pesos'])
for i in range(pesos_array.shape[1]):
    ax2.plot(pesos_array[:, i], label=f'Vista {i+1}')
ax2.set_title('Evolución de los Pesos')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("convergencia_fusion.png")




# 10. Aplicar KMeans para clustering de microservicios
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def aplicar_clustering_y_mostrar_resultados(U, casos_uso, endpoints_openapi, mapeo_optimo, n_max):
    """
    Aplica KMeans a la matriz unificada U y muestra los clusters resultantes
    """
    print("\n" + "=" * 80)
    print("CLUSTERING DE MICROSERVICIOS CON KMEANS")
    print("=" * 80)
    
    # Determinar el número óptimo de clusters usando silhouette score
    silhouette_scores = []
    k_range = range(2, min(10, n_max))  # Probar de 2 a 9 clusters
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(U)
        score = silhouette_score(U, clusters)
        silhouette_scores.append(score)
        print(f"K = {k}: Silhouette Score = {score:.4f}")
    
    # Seleccionar el mejor k
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nNúmero óptimo de clusters: {best_k} (Silhouette Score: {max(silhouette_scores):.4f})")
    
    # Aplicar KMeans con el mejor k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(U)
    
    # Crear mapeo inverso para encontrar casos de uso desde endpoints
    mapeo_inverso = {v: k for k, v in mapeo_optimo.items()}
    
    # Organizar elementos por cluster
    clusters_dict = {}
    for cluster_id in range(best_k):
        clusters_dict[cluster_id] = {
            'casos_uso': [],
            'endpoints': [],
            'nodos': []
        }
    
    # Asignar cada nodo a su cluster correspondiente
    for nodo_idx in range(n_max):
        cluster_id = clusters[nodo_idx]
        clusters_dict[cluster_id]['nodos'].append(nodo_idx)
        
        # Verificar si es un caso de uso mapeado
        if nodo_idx in mapeo_inverso:
            cu_idx = mapeo_inverso[nodo_idx]
            clusters_dict[cluster_id]['casos_uso'].append(casos_uso[cu_idx])
        
        # Verificar si es un endpoint (siempre es endpoint porque n_max = 18 = len(endpoints))
        if nodo_idx < len(endpoints_openapi):
            clusters_dict[cluster_id]['endpoints'].append(endpoints_openapi[nodo_idx])
    
    # Mostrar resultados por cluster
    print("\n" + "=" * 80)
    print("PARTICIÓN DE MICROSERVICIOS RESULTANTE")
    print("=" * 80)
    
    for cluster_id in range(best_k):
        print(f"\n🔷 CLUSTER {cluster_id}:")
        print(f"   Nodos asignados: {len(clusters_dict[cluster_id]['nodos'])}")
        
        if clusters_dict[cluster_id]['casos_uso']:
            print(f"   📋 Casos de uso:")
            for caso in clusters_dict[cluster_id]['casos_uso']:
                print(f"      • {caso}")
        
        if clusters_dict[cluster_id]['endpoints']:
            print(f"   🌐 Endpoints:")
            for endpoint in clusters_dict[cluster_id]['endpoints']:
                print(f"      • {endpoint}")
        
        print(f"   🔢 IDs de nodos: {clusters_dict[cluster_id]['nodos']}")
    
    # Visualizar los clusters en la matriz U
    plt.figure(figsize=(12, 10))
    
    # Ordenar la matriz por clusters para mejor visualización
    sorted_indices = np.argsort(clusters)
    U_sorted = U[sorted_indices][:, sorted_indices]
    
    sns.heatmap(U_sorted, cmap='YlOrRd', 
               xticklabels=[f"N{idx+1}" for idx in sorted_indices],
               yticklabels=[f"N{idx+1}" for idx in sorted_indices])
    plt.title(f'Matriz Unificada U Ordenada por Clusters (K={best_k})')
    plt.tight_layout()
    plt.savefig("matriz_clusters.png")
    
    # Visualizar silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Selección del Número Óptimo de Clusters')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Mejor K = {best_k}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("silhouette_scores.png")
    
    return clusters, best_k

# Aplica KMeans a la matriz unificada U y muestra los clusters resultantes, solo para un valor de k
def aplicar_clustering_kmeans(U, casos_uso, endpoints_openapi, mapeo_optimo, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(U)
    
    # Crear mapeo inverso para encontrar casos de uso desde endpoints
    mapeo_inverso = {v: k for k, v in mapeo_optimo.items()}
    
    # Organizar elementos por cluster
    clusters_dict = {}
    for cluster_id in range(k):
        clusters_dict[cluster_id] = {
            'casos_uso': [],
            'endpoints': [],
            'nodos': []
        }
    
    # Asignar cada nodo a su cluster correspondiente
    n_max = U.shape[0]
    for nodo_idx in range(n_max):
        cluster_id = clusters[nodo_idx]
        clusters_dict[cluster_id]['nodos'].append(nodo_idx)
        
        # Verificar si es un caso de uso mapeado
        if nodo_idx in mapeo_inverso:
            cu_idx = mapeo_inverso[nodo_idx]
            clusters_dict[cluster_id]['casos_uso'].append(casos_uso[cu_idx])
        
        # Verificar si es un endpoint (siempre es endpoint porque n_max = 18 = len(endpoints))
        if nodo_idx < len(endpoints_openapi):
            clusters_dict[cluster_id]['endpoints'].append(endpoints_openapi[nodo_idx])
    
    # Mostrar resultados por cluster
    print("\n" + "=" * 80)
    print(f"PARTICIÓN DE MICROSERVICIOS RESULTANTE (K={k})")
    print("=" * 80)
    
    for cluster_id in range(k):
        print(f"\n🔷 CLUSTER {cluster_id}:")
        print(f"   Nodos asignados: {len(clusters_dict[cluster_id]['nodos'])}")
        
        if clusters_dict[cluster_id]['casos_uso']:
            print(f"   📋 Casos de uso:")
            for caso in clusters_dict[cluster_id]['casos_uso']:
                print(f"      • {caso}")
        
        if clusters_dict[cluster_id]['endpoints']:
            print(f"   🌐 Endpoints:")
            for endpoint in clusters_dict[cluster_id]['endpoints']:
                print(f"      • {endpoint}")
        
        print(f"   🔢 IDs de nodos: {clusters_dict[cluster_id]['nodos']}")
    
    # Visualizar los clusters en la matriz U
    plt.figure(figsize=(12, 10))
    # Ordenar la matriz por clusters para mejor visualización
    sorted_indices = np.argsort(clusters)
    U_sorted = U[sorted_indices][:, sorted_indices]
    sns.heatmap(U_sorted, cmap='YlOrRd',
                xticklabels=[f"N{idx+1}" for idx in sorted_indices],
                yticklabels=[f"N{idx+1}" for idx in sorted_indices])
    plt.title(f'Matriz Unificada U Ordenada por Clusters (K={k})')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"matriz_clusters_k{k}.png")

    # Retornar los clusters
    return clusters
    
    

# # Aplicar clustering a la matriz unificada U
# clusters_resultantes, best_k = aplicar_clustering_y_mostrar_resultados(
#     U_unificada, casos_uso, endpoints_openapi, mapeo_optimo, n_max
# )

# aplicar clustering para un valor específico de k
clusters_resultantes = aplicar_clustering_kmeans(
    U_unificada, casos_uso, endpoints_openapi, mapeo_optimo, 5
)
