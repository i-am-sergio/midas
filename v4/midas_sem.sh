#!/bin/bash

# Verificar que se proporcionó el parámetro
if [ $# -ne 1 ]; then
    echo "Uso: $0 <nombre_monolito>"
    echo "Monolitos disponibles: acmeair, jpetstore, daytrader, plants, jrideconnect"
    echo "Ejemplo: $0 jpetstore"
    exit 1
fi

NOMBRE_MONOLITO=$1
MONOLITO_DIR="../monoliths/"
# Ruta absoluta o relativa clara para los resultados
RESULTS_DIR="./${NOMBRE_MONOLITO}_results" 

# Asignar directorio fuente específico
case $NOMBRE_MONOLITO in
    "acmeair") SOURCE_DIR="${MONOLITO_DIR}acmeair/" ;;
    "jpetstore") SOURCE_DIR="${MONOLITO_DIR}jPetStore/" ;;
    "daytrader") SOURCE_DIR="${MONOLITO_DIR}sample.daytrader7/" ;;
    "plants") SOURCE_DIR="${MONOLITO_DIR}sample.plantsbywebsphere/" ;;
    "jrideconnect") SOURCE_DIR="${MONOLITO_DIR}jrideconnect/" ;;
    *)
        echo "Error: Monolito '$NOMBRE_MONOLITO' no reconocido"
        exit 1
        ;;
esac

# Verificar que existe el directorio fuente
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: No se encuentra el directorio fuente $SOURCE_DIR"
    exit 1
fi

# Crear directorio de resultados si no existe
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Creando directorio de resultados: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
fi

echo "=================================================="
echo "Ejecutando MIDAS (Vista Semántica) para: $NOMBRE_MONOLITO"
echo "=================================================="

# --- FASES PRELIMINARES: Obtener la Lista Maestra de Clases del Núcleo (Filtro Estructural) ---
# Necesitamos este archivo para filtrar el SemanticCollector.

STRUCTURAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"
STRUCTURAL_CSV_FILTERED="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural_view_filtered.csv"
STRUCTURAL_MATRIX_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_structural_matrix.csv"

echo "[0/5] PREREQUISITO: Generando lista maestra de clases del núcleo..."
# 1. Extracción Estructural (para obtener el listado de todas las clases)
python extract_structural_view.py "$SOURCE_DIR" "$RESULTS_DIR/"
# 2. Análisis de Relaciones (para eliminar duplicados y obtener el grafo crudo)
python analyze_relations.py "$RESULTS_DIR/structural_view.csv" "$SOURCE_DIR" "$STRUCTURAL_CSV_RAW"
# 3. Filtrado Genérico (para obtener la lista canónica de clases de negocio)
python preprocessing_structural.py "$STRUCTURAL_CSV_RAW" "$STRUCTURAL_CSV_FILTERED"
echo "Maestro de clases de núcleo guardado en: $STRUCTURAL_CSV_FILTERED"

# --- FASE 1: Semantic Extraction ---
SEMANTIC_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_semantic_view.csv"

echo "[1/5] Extracción y Descomposición Léxica (Vista Semántica)..."
# Usamos el CSV filtrado estructural como filtro de inclusión
python extract_semantic_view.py "$SOURCE_DIR" "$STRUCTURAL_CSV_FILTERED" "$SEMANTIC_CSV_RAW"

# --- FASE 2: Semantic Preprocessing and Build Matrix ---
SEMANTIC_CSV_CLEAN="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_semantic_clean.csv"
SEMANTIC_MATRIX_BASE_NAME="$RESULTS_DIR/${NOMBRE_MONOLITO}_semantic"
SEMANTIC_MATRIX_CSV="$SEMANTIC_MATRIX_BASE_NAME"_matrix.csv

echo "[2/5] Preprocesamiento Semántico (Stop Words / Lematización)..."
python preprocessing_semantic.py "$SEMANTIC_CSV_RAW" "$SEMANTIC_CSV_CLEAN"

echo "[2/5] Construcción de Matriz Semántica (MPNet)..."
python build_semantic_matrix.py "$SEMANTIC_CSV_CLEAN" "$SEMANTIC_MATRIX_BASE_NAME"

# --- FASE 3: Multiview Self-Weighted Fusion (Saltado) ---
echo "[3/5] Fusión Multivista (Saltado - Solo Semántica)..."

# --- FASE 4: Clustering ---
SEMANTIC_CLUSTERING_DIR="$RESULTS_DIR/semantic_spectral_clustering_results"
echo "[4/5] Clustering Espectral (Optimizando K) sobre S^(sem)..."
# Pasamos el directorio $RESULTS_DIR para que guarde ahí la carpeta clustering_results_semantic
python optimize_k_spectral.py "$SEMANTIC_MATRIX_CSV" "$SEMANTIC_CLUSTERING_DIR"

# --- FASE 5: Evaluation (Asumimos el mejor K por defecto o K=5 para comparación) ---
BEST_K_JSON="$SEMANTIC_CLUSTERING_DIR/k_5.json"

echo "[5/5] Evaluación de Métricas (para K=5, usando relaciones estructurales)..."

if [ -f "$BEST_K_JSON" ]; then
    # Usamos el JSON del clúster semántico y el CSV de relaciones estructurales crudas
    python calculate_metrics.py "$BEST_K_JSON" "$STRUCTURAL_CSV_FILTERED"
else
    echo "ADVERTENCIA: No se encontró el archivo $BEST_K_JSON. Por favor, revise los resultados de Silhouette."
fi

echo "=================================================="
echo "MIDAS Semántico Completado [MIDAS-sem]"
echo "Resultados en: $RESULTS_DIR"
echo "=================================================="