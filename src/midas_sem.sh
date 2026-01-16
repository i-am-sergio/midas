#!/bin/bash

# ==============================================================================
# MIDAS-Sem: Descomposición (Solo Vista Semántica)
# ==============================================================================

# --- 0. CONFIGURACION INICIAL ---

if [ $# -ne 1 ]; then
    echo "Uso: $0 <nombre_monolito>"
    echo "Ejemplo: $0 jpetstore"
    exit 1
fi

NOMBRE_MONOLITO=$1 
MONOLITO_DIR="../monoliths/"
# NOTA: Directorio específico para resultados semánticos
RESULTS_DIR="./${NOMBRE_MONOLITO}_results_SEM"

# Mapeo de directorios y K objetivos
case $NOMBRE_MONOLITO in
    "jpetstore") 
        SOURCE_DIR="${MONOLITO_DIR}jPetStore/" 
        K_TARGET=4
        ;;
    "daytrader") 
        SOURCE_DIR="${MONOLITO_DIR}sample.daytrader7/" 
        K_TARGET=7
        ;;
    "acmeair") 
        SOURCE_DIR="${MONOLITO_DIR}acmeair/"
        K_TARGET=4
        ;;
    "plants") 
        SOURCE_DIR="${MONOLITO_DIR}sample.plantsbywebsphere/" 
        K_TARGET=5
        ;;
    "jrideconnect") 
        SOURCE_DIR="${MONOLITO_DIR}jrideconnect/" 
        K_TARGET=7
        ;;
    *) echo "Error: Monolito no reconocido"; exit 1 ;;
esac

# Verificaciones
if [ ! -d "$SOURCE_DIR" ]; then echo "Error: No existe $SOURCE_DIR"; exit 1; fi
if [ ! -d "$RESULTS_DIR" ]; then mkdir -p "$RESULTS_DIR"; fi

echo "=========================================================="
echo "  MIDAS-Sem PIPELINE: $NOMBRE_MONOLITO (Solo Semántica)"
echo "=========================================================="

# Definición de rutas
ALL_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_all.csv"
CORE_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_filtered.csv"

# NOTA: Necesitamos el CSV estructural RAW para la Fase 5 (Evaluación),
# aunque NO lo usemos para el clustering. Las métricas SM/ICP dependen de las llamadas.
STRUCTURAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"

# CSV Semántico
SEMANTIC_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_semantic_view.csv"

# Matrices
SEM_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_semantic"
FUSION_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase3_fusion" 

CALCULATE_METRICS="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase5_evaluation_metrics.csv"

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 1: EXTRACTION "
echo "=========================================================================="

# 1.0 Extraccion Clases Core
echo "   -> [Core Classes] Extrayendo clases core..."
python fase1_extract_core_classes.py "$SOURCE_DIR" "$ALL_CLASSES_CSV" "$CORE_CLASSES_CSV"

# 1.1 Extracción Estructural (NECESARIA SOLO PARA EVALUACIÓN FINAL)
echo "   -> [Estructural] Extrayendo grafo de llamadas (para métricas)..."
python fase1_extract_structural_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$STRUCTURAL_CSV_RAW"

# 1.2 Extracción Semántica (LA IMPORTANTE AQUÍ)
echo "   -> [Semántica] Extrayendo vista semántica (Tokens & Text)..."
# python fase1_extract_semantic_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$SEMANTIC_CSV_RAW"
# Copiar para evitar llamadas a la api de gemini para code summarization
cp "./${NOMBRE_MONOLITO}_results/${NOMBRE_MONOLITO}_fase1_semantic_view.csv" "$SEMANTIC_CSV_RAW"
# OMITIDO: 1.3 Extracción Funcional

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 2: PREPROCESSING (Semantic Matrix Construction) "
echo "=========================================================================="

# OMITIDO: 2.1 Matriz Estructural

# 2.2 Matriz Semántica (S_sem con MPNet)
echo "   -> [Semántica] Construyendo matriz de similitud MPNet (S_sem)"
python fase2_build_semantic_matrix.py "$SEMANTIC_CSV_RAW" "$CORE_CLASSES_CSV" "$SEM_MATRIX_BASE"

# OMITIDO: 2.3 Matriz Funcional

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 3: FUSION BYPASS (Simulando Fusión con Semántica) "
echo "=========================================================================="

SEM_MATRIX_CSV="${SEM_MATRIX_BASE}_matrix.csv"
FUSION_MATRIX_CSV="${FUSION_MATRIX_BASE}_matrix.csv"

echo "   -> [MIDAS-Sem] Omitiendo algoritmo de fusión ponderada."
echo "   -> [MIDAS-Sem] Copiando Matriz Semántica como Matriz Final..."

# AQUÍ ESTA LA CLAVE DE MIDAS-SEM:
# Copiamos la matriz semántica directamente a la entrada del clustering
cp "$SEM_MATRIX_CSV" "$FUSION_MATRIX_CSV"

if [ -f "$FUSION_MATRIX_CSV" ]; then
    echo "   -> [OK] Matriz Semántica preparada para clustering."
else
    echo "   -> [ERROR] No se pudo generar la matriz semántica."
    exit 1
fi

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 4: CLUSTERING (Spectral Clustering on Semantic View) "
echo "=========================================================================="

CLUSTERING_DIR="${RESULTS_DIR}/${NOMBRE_MONOLITO}_fase4_spectral_clustering"
echo "   -> [Clustering] Ejecutando sobre Matriz Semántica..."

# El script de clustering procesa la matriz que le llega (que ahora es 100% semántica)
python fase4_clustering.py "$FUSION_MATRIX_CSV" "$CLUSTERING_DIR"

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 5: EVALUATION (Quality Evaluation) "
echo "=========================================================================="

# Seleccionamos dinamicamente el K target
TARGET_K_JSON="$CLUSTERING_DIR/clustering_results/k_${K_TARGET}.json"

echo "   -> [Evaluation] Evaluando partición K=$K_TARGET..."

# NOTA: Para evaluar usamos STRUCTURAL_CSV_RAW porque las métricas SM e ICP 
# se calculan sobre las llamadas reales del código, independientemente de cómo 
# se generaron los clusters (semánticamente).
if [ -f "$TARGET_K_JSON" ]; then
    python fase5_evaluation.py "$STRUCTURAL_CSV_RAW" "$TARGET_K_JSON" "$CALCULATE_METRICS"
else
    echo "   -> [ERROR] No se encontró el archivo de clusters: $TARGET_K_JSON"
fi

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  MIDAS-Sem FINALIZADO."
echo "  Resultados en: $RESULTS_DIR"
echo "=========================================================================="
exit 0