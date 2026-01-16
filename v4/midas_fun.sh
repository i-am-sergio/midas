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
echo "Ejecutando MIDAS (Vista Funcional) para: $NOMBRE_MONOLITO"
echo "=================================================="

# --- FASES PRELIMINARES: Obtener la Lista Maestra de Clases del Núcleo ---
# La vista funcional necesita saber cuáles son las clases válidas del dominio
# para ignorar librerías externas en los controladores.

STRUCTURAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"
STRUCTURAL_CSV_FILTERED="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural_view_filtered.csv"

echo "[0/5] PREREQUISITO: Generando lista maestra de clases del núcleo..."
# 1. Extracción Estructural
python extract_structural_view.py "$SOURCE_DIR" "$RESULTS_DIR/"
# 2. Análisis de Relaciones
python analyze_relations.py "$RESULTS_DIR/structural_view.csv" "$SOURCE_DIR" "$STRUCTURAL_CSV_RAW"
# 3. Filtrado Genérico
python preprocessing_structural.py "$STRUCTURAL_CSV_RAW" "$STRUCTURAL_CSV_FILTERED"
echo "Maestro de clases de núcleo guardado en: $STRUCTURAL_CSV_FILTERED"

# --- FASE 1: Functional Extraction ---
FUNCTIONAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_functional_view.csv"

echo "[1/5] Extracción de Co-ocurrencias Funcionales..."
# Inputs: CSV de clases permitidas (filtro), Directorio del proyecto, Output
python extract_functional_view.py "$STRUCTURAL_CSV_FILTERED" "$SOURCE_DIR" "$FUNCTIONAL_CSV_RAW"

# --- FASE 2: Build Functional Matrix ---
FUNCTIONAL_MATRIX_BASE_NAME="$RESULTS_DIR/${NOMBRE_MONOLITO}_functional"
FUNCTIONAL_MATRIX_CSV="${FUNCTIONAL_MATRIX_BASE_NAME}_matrix.csv"

echo "[2/5] Construcción de Matriz Funcional A^(fun)..."
# Inputs: CSV de co-ocurrencias, Nombre base de salida
python build_functional_matrix.py "$FUNCTIONAL_CSV_RAW" "$FUNCTIONAL_MATRIX_BASE_NAME"

# --- FASE 3: Multiview Fusion (Saltado) ---
echo "[3/5] Fusión Multivista (Saltado - Solo Funcional)..."

# --- FASE 4: Clustering ---
FUNCTIONAL_CLUSTERING_DIR="$RESULTS_DIR/functional_spectral_clustering_results"

echo "[4/5] Clustering Espectral (Optimizando K) sobre A^(fun)..."
# Inputs: Matriz funcional, Directorio de salida para clustering
python optimize_k_spectral.py "$FUNCTIONAL_MATRIX_CSV" "$FUNCTIONAL_CLUSTERING_DIR"

# --- FASE 5: Evaluation ---
BEST_K_JSON="$FUNCTIONAL_CLUSTERING_DIR/k_5.json"

echo "[5/5] Evaluación de Métricas (para K=5)..."
# Nota: Evaluamos los clústeres generados por la vista funcional
# contra la realidad estructural (dependencias reales) para medir el acoplamiento.

if [ -f "$BEST_K_JSON" ]; then
    python calculate_metrics.py "$BEST_K_JSON" "$STRUCTURAL_CSV_FILTERED"
else
    echo "ADVERTENCIA: No se encontró el archivo $BEST_K_JSON. El clustering funcional puede haber fallado o K óptimo es distinto."
fi

echo "=================================================="
echo "MIDAS Funcional Completado [MIDAS-fun]"
echo "Resultados en: $RESULTS_DIR"
echo "=================================================="