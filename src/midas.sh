#!/bin/bash

# ==============================================================================
# MIDAS: Descomposición Automática de Monolitos (Pipeline Completo)
# ==============================================================================

# --- 0. CONFIGURACION INICIAL ---

if [ $# -ne 1 ]; then
    echo "Uso: $0 <nombre_monolito>"
    echo "Ejemplo: $0 jpetstore"
    exit 1
fi

NOMBRE_MONOLITO=$1 # <jpetstore | acmeair | daytrader | plants | jrideconnect>
MONOLITO_DIR="../monoliths/"
RESULTS_DIR="./${NOMBRE_MONOLITO}_results"

# Mapeo de directorios
case $NOMBRE_MONOLITO in
    "acmeair") SOURCE_DIR="${MONOLITO_DIR}acmeair/" ;;
    "jpetstore") SOURCE_DIR="${MONOLITO_DIR}jPetStore/" ;;
    "daytrader") SOURCE_DIR="${MONOLITO_DIR}sample.daytrader7/" ;;
    "plants") SOURCE_DIR="${MONOLITO_DIR}sample.plantsbywebsphere/" ;;
    "jrideconnect") SOURCE_DIR="${MONOLITO_DIR}jrideconnect/" ;;
    *) echo "Error: Monolito no reconocido"; exit 1 ;;
esac

# Verificaciones
if [ ! -d "$SOURCE_DIR" ]; then echo "Error: No existe $SOURCE_DIR"; exit 1; fi
if [ ! -d "$RESULTS_DIR" ]; then mkdir -p "$RESULTS_DIR"; fi

echo "=========================================================="
echo "  MIDAS PIPELINE: $NOMBRE_MONOLITO"
echo "=========================================================="

# Definición de rutas de archivos clave para pasar entre fases
STRUCTURAL_RAW="$RESULTS_DIR/"
ALL_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_all.csv"
CORE_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_filtered.csv"

STRUCTURAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"
SEMANTIC_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_semantic_view.csv"
FUNCTIONAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_functional_view.csv"

STR_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural"
SEM_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_semantic"
FUN_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_functional"
FUSION_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase3_fusion"

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 1: EXTRACTION (View Data Extraction) "
echo "=========================================================================="
# 1.0 Extraccion Clases Core
echo "   -> [Core Classes] Extrayendo clases core..."
python fase1_extract_core_classes.py "$SOURCE_DIR" "$ALL_CLASSES_CSV" "$CORE_CLASSES_CSV"
# 1.1 Extracción Estructural
echo "   -> [Estructural] Extrayendo vista estructural..."
python fase1_extract_structural_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$STRUCTURAL_CSV_RAW"
# 1.2 Extracción Semántica
echo "   -> [Semántica] Extrayendo vista semántica..."
python fase1_extract_semantic_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$SEMANTIC_CSV_RAW"
# 1.3 Extracción Funcional
echo "   -> [Funcional] Extrayendo vista funcional..."
python fase1_extract_functional_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$FUNCTIONAL_CSV_RAW"

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 2: PREPROCESSING AND BUILD MATRIX (Normalization and Construction) "
echo "=========================================================================="
# 2.1 Matriz Estructural (A_str)
echo "   -> [Estructural] Construyendo matriz estructural (A_str)"
python fase2_build_structural_matrix.py "$STRUCTURAL_RAW/structural_view.csv" "$CORE_CLASSES_CSV" "$STR_MATRIX_BASE"
# 2.2 Matriz Semántica (A_sem)
echo "   -> [Semántica] Construyendo matriz semántica (A_sem)"
python fase2_build_semantic_matrix.py "$SEMANTIC_CSV_RAW" "$CORE_CLASSES_CSV" "$SEM_MATRIX_BASE"
# 2.3 Matriz Funcional (A_fun)
echo "   -> [Funcional] Construyendo matriz funcional (A_fun)"
python fase2_build_functional_matrix.py "$FUNCTIONAL_CSV_RAW" "$CORE_CLASSES_CSV" "$FUN_MATRIX_BASE"

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 3: FUSION (Auto-Weighted Multiview Fusion) "
echo "=========================================================================="

STR_MATRIX_CSV="${STR_MATRIX_BASE}_matrix.csv"
SEM_MATRIX_CSV="${SEM_MATRIX_BASE}_matrix.csv"
FUN_MATRIX_CSV="${FUN_MATRIX_BASE}_matrix.csv"
FUSION_MATRIX_CSV="${FUSION_MATRIX_BASE}_matrix.csv"

echo "   -> [Fusion] Ejecutando algoritmo de fusion"
python fase3_multiview_fusion.py "$STR_MATRIX_CSV" "$SEM_MATRIX_CSV" "$FUN_MATRIX_CSV" "$FUSION_MATRIX_CSV"

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 4: CLUSTERING (Spectral Clustering) "
echo "=========================================================================="

CLUSTERING_DIR="${RESULTS_DIR}/${NOMBRE_MONOLITO}_fase4_spectral_clustering"
echo "   -> [Clustering] Optimizando K sobre la Matriz Fusionada..."

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 5: EVALUATION (Quality Evaluation) "
echo "=========================================================================="

TARGET_K_JSON="$CLUSTERING_DIR/k_5.json"
echo "   -> [Evaluation] Calculando Métricas (SM, ICP, IFN, NED)..."

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  MIDAS FINALIZADO."
echo "  Resultados: $RESULTS_DIR"
echo "=========================================================================="