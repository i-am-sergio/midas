#!/bin/bash

# ==============================================================================
# MIDAS: Descomposición Automática de Monolitos (Pipeline Completo)
# ==============================================================================

# --- 0. CONFIGURACIÓN INICIAL ---

if [ $# -ne 1 ]; then
    echo "Uso: $0 <nombre_monolito>"
    echo "Ejemplo: $0 jpetstore"
    exit 1
fi

NOMBRE_MONOLITO=$1
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
STRUCTURAL_RELATIONS="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"
CORE_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural_view_filtered.csv" # Lista Maestra

SEMANTIC_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_semantic_view.csv"
FUNCTIONAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_functional_view.csv"

STR_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural"
SEM_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_semantic"
FUN_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_functional"


# ==============================================================================
# FASE 1: EXTRACTION (Extracción de Datos)
# ==============================================================================
# echo ""
# echo ">>> [FASE 1] Extraction..."
echo ""
echo "=========================================================="
echo "  FASE 1: Extraction "
echo "=========================================================="

# 1.1 Extracción Estructural Base (Necesaria para definir el núcleo)
echo "   -> [Estructural] Extrayendo AST y dependencias..."
python extract_structural_view.py "$SOURCE_DIR" "$STRUCTURAL_RAW"
python analyze_relations.py "$STRUCTURAL_RAW/structural_view.csv" "$SOURCE_DIR" "$STRUCTURAL_RELATIONS"

# 1.2 Generación del Núcleo Funcional (Filtro Maestro)
# Esto técnicamente es un pre-proceso, pero lo hacemos aquí para saber QUÉ extraer en Sem/Fun
echo "   -> [Núcleo] Filtrando clases de framework/utilidad..."
python preprocessing_structural.py "$STRUCTURAL_RELATIONS" "$CORE_CLASSES_CSV"

# 1.3 Extracción Semántica (Usando el Núcleo)
echo "   -> [Semántica] Extrayendo vocabulario y aplicando CamelCase Split..."
python extract_semantic_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$SEMANTIC_CSV_RAW"

# 1.4 Extracción Funcional (Usando el Núcleo)
echo "   -> [Funcional] Escaneando controladores y co-ocurrencias..."
python extract_functional_view.py "$CORE_CLASSES_CSV" "$SOURCE_DIR" "$FUNCTIONAL_CSV_RAW"


# ==============================================================================
# FASE 2: PREPROCESSING AND BUILD MATRIX (Normalización y Construcción)
# ==============================================================================
# echo ""
# echo ">>> [FASE 2] Preprocessing and Build Matrix..."

echo ""
echo "=========================================================="
echo "  FASE 2: Preprocessing and Build Matrices "
echo "=========================================================="

# 2.1 Matriz Estructural (A_str)
echo "   -> Construyendo A_str (Simétrica Normalizada)..."
python build_structural_matrix.py "$CORE_CLASSES_CSV" "$STR_MATRIX_BASE"

# 2.2 Matriz Semántica (S_sem)
SEMANTIC_CLEAN="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_semantic_clean.csv"
echo "   -> Preprocesando Texto (NLP) y Construyendo S_sem (MPNet)..."
python preprocessing_semantic.py "$SEMANTIC_CSV_RAW" "$SEMANTIC_CLEAN"
python build_semantic_matrix.py "$SEMANTIC_CLEAN" "$SEM_MATRIX_BASE"

# 2.3 Matriz Funcional (A_fun)
echo "   -> Construyendo A_fun (Co-ocurrencia Normalizada)..."
python build_functional_matrix.py "$FUNCTIONAL_CSV_RAW" "$FUN_MATRIX_BASE"


# ==============================================================================
# FASE 3: FUSION (Fusión Multivista Auto-Ponderada)
# ==============================================================================
echo ""
echo "=========================================================="
echo "  FASE 3: Fusion "
echo "=========================================================="

STR_MATRIX_CSV="${STR_MATRIX_BASE}_matrix.csv"
SEM_MATRIX_CSV="${SEM_MATRIX_BASE}_matrix.csv"
FUN_MATRIX_CSV="${FUN_MATRIX_BASE}_matrix.csv"
FUSION_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase3"

echo "   -> Ejecutando algoritmo iterativo de consenso..."
# Genera: ..._fused_matrix.csv y ..._fusion_convergence.csv
python multiview_fusion.py "$STR_MATRIX_CSV" "$SEM_MATRIX_CSV" "$FUN_MATRIX_CSV" "$FUSION_BASE"

FUSED_MATRIX_CSV="${FUSION_BASE}_fused_matrix.csv"


# ==============================================================================
# FASE 4: CLUSTERING (Clustering Espectral)
# ==============================================================================
echo ""
echo "=========================================================="
echo "  FASE 4: Clustering "
echo "=========================================================="

CLUSTERING_DIR="$RESULTS_DIR/fusion_spectral_clustering_results"

echo "   -> Optimizando K sobre la Matriz Fusionada..."
python optimize_k_spectral.py "$FUSED_MATRIX_CSV" "$CLUSTERING_DIR"


# ==============================================================================
# FASE 5: EVALUATION (Evaluación de Calidad)
# ==============================================================================
echo ""
echo "=========================================================="
echo "  FASE 5: Evaluation "
echo "=========================================================="

# Nota: Seleccionamos K=5 para el reporte, pero podrías automatizar la elección del mejor K
TARGET_K_JSON="$CLUSTERING_DIR/k_5.json"

echo "   -> Calculando Métricas (SM, ICP, IFN, NED)..."
if [ -f "$TARGET_K_JSON" ]; then
    # Evaluamos el resultado (JSON) contra las dependencias reales (CORE_CLASSES_CSV)
    python calculate_metrics.py "$TARGET_K_JSON" "$CORE_CLASSES_CSV"
else
    echo "⚠️  ADVERTENCIA: No se encontró el resultado para K=5."
    echo "    Revisa $CLUSTERING_DIR/silhouette_scores.csv para ver el mejor K."
fi

echo ""
echo "=========================================================="
echo "  MIDAS FINALIZADO."
echo "  Resultados: $RESULTS_DIR"
echo "=========================================================="