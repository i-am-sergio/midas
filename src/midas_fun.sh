#!/bin/bash

# ==============================================================================
# MIDAS-Fun: Descomposición (Solo Vista Funcional)
# ==============================================================================

# --- 0. CONFIGURACION INICIAL ---

if [ $# -ne 1 ]; then
    echo "Uso: $0 <nombre_monolito>"
    echo "Ejemplo: $0 jpetstore"
    exit 1
fi

NOMBRE_MONOLITO=$1 
MONOLITO_DIR="../monoliths/"
# NOTA: Directorio específico para resultados funcionales
RESULTS_DIR="./${NOMBRE_MONOLITO}_results_FUN"

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
echo "  MIDAS-Fun PIPELINE: $NOMBRE_MONOLITO (Solo Funcional)"
echo "=========================================================="

# Definición de rutas
ALL_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_all.csv"
CORE_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_filtered.csv"

# NOTA: Necesitamos el CSV estructural RAW para la Fase 5 (Evaluación)
STRUCTURAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"

# CSV Funcional
FUNCTIONAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_functional_view.csv"

# Matrices
FUN_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_functional"
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

# 1.1 Extracción Estructural (SOLO PARA MÉTRICAS)
echo "   -> [Estructural] Extrayendo grafo de llamadas (para evaluación)..."
python fase1_extract_structural_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$STRUCTURAL_CSV_RAW"

# OMITIDO: 1.2 Extracción Semántica

# 1.3 Extracción Funcional (LA IMPORTANTE AQUÍ)
echo "   -> [Funcional] Extrayendo vista funcional (Endpoints/Controllers)..."
# python fase1_extract_functional_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$FUNCTIONAL_CSV_RAW"

cp "./${NOMBRE_MONOLITO}_results/${NOMBRE_MONOLITO}_fase1_functional_view.csv" "$FUNCTIONAL_CSV_RAW"

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 2: PREPROCESSING (Functional Matrix Construction) "
echo "=========================================================================="

# OMITIDO: 2.1 Matriz Estructural
# OMITIDO: 2.2 Matriz Semántica

# 2.3 Matriz Funcional (A_fun)
echo "   -> [Funcional] Construyendo matriz de co-ocurrencia funcional (A_fun)"
python fase2_build_functional_matrix.py "$FUNCTIONAL_CSV_RAW" "$CORE_CLASSES_CSV" "$FUN_MATRIX_BASE"

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 3: FUSION BYPASS (Simulando Fusión con Funcional) "
echo "=========================================================================="

FUN_MATRIX_CSV="${FUN_MATRIX_BASE}_matrix.csv"
FUSION_MATRIX_CSV="${FUSION_MATRIX_BASE}_matrix.csv"

echo "   -> [MIDAS-Fun] Omitiendo algoritmo de fusión ponderada."
echo "   -> [MIDAS-Fun] Copiando Matriz Funcional como Matriz Final..."

# AQUÍ ESTA LA CLAVE DE MIDAS-FUN:
# Copiamos la matriz funcional directamente a la entrada del clustering
cp "$FUN_MATRIX_CSV" "$FUSION_MATRIX_CSV"

if [ -f "$FUSION_MATRIX_CSV" ]; then
    echo "   -> [OK] Matriz Funcional preparada para clustering."
else
    echo "   -> [ERROR] No se pudo generar la matriz funcional."
    exit 1
fi

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 4: CLUSTERING (Spectral Clustering on Functional View) "
echo "=========================================================================="

CLUSTERING_DIR="${RESULTS_DIR}/${NOMBRE_MONOLITO}_fase4_spectral_clustering"
echo "   -> [Clustering] Ejecutando sobre Matriz Funcional..."

# El script de clustering procesa la matriz que le llega
python fase4_clustering.py "$FUSION_MATRIX_CSV" "$CLUSTERING_DIR"

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 5: EVALUATION (Quality Evaluation) "
echo "=========================================================================="

# Seleccionamos dinamicamente el K target
TARGET_K_JSON="$CLUSTERING_DIR/clustering_results/k_${K_TARGET}.json"

echo "   -> [Evaluation] Evaluando partición K=$K_TARGET..."

# Usamos STRUCTURAL_CSV_RAW para calcular métricas técnicas reales (SM/ICP)
if [ -f "$TARGET_K_JSON" ]; then
    python fase5_evaluation.py "$STRUCTURAL_CSV_RAW" "$TARGET_K_JSON" "$CALCULATE_METRICS"
else
    echo "   -> [ERROR] No se encontró el archivo de clusters: $TARGET_K_JSON"
fi

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  MIDAS-Fun FINALIZADO."
echo "  Resultados en: $RESULTS_DIR"
echo "=========================================================================="
exit 0