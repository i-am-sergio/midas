#!/bin/bash

# ==============================================================================
# MIDAS-Str: Descomposición (Solo Vista Estructural)
# ==============================================================================

# --- 0. CONFIGURACION INICIAL ---

if [ $# -ne 1 ]; then
    echo "Uso: $0 <nombre_monolito>"
    echo "Ejemplo: $0 jpetstore"
    exit 1
fi

NOMBRE_MONOLITO=$1 
MONOLITO_DIR="../monoliths/"
# NOTA: Cambiamos el directorio para no mezclar con la ejecucion completa
RESULTS_DIR="./${NOMBRE_MONOLITO}_results_STR"

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
echo "  MIDAS-Str PIPELINE: $NOMBRE_MONOLITO (Solo Estructural)"
echo "=========================================================="

# Definición de rutas
ALL_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_all.csv"
CORE_CLASSES_CSV="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view_filtered.csv"
STRUCTURAL_CSV_RAW="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"

# Matrices
STR_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural"
# Usamos el nombre 'fusion' para mantener compatibilidad con Fase 4, 
# aunque en realidad será solo la estructural.
FUSION_MATRIX_BASE="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase3_fusion" 

CALCULATE_METRICS="$RESULTS_DIR/${NOMBRE_MONOLITO}_fase5_evaluation_metrics.csv"

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 1: EXTRACTION (Solo Structural) "
echo "=========================================================================="

# 1.0 Extraccion Clases Core
echo "   -> [Core Classes] Extrayendo clases core..."
python fase1_extract_core_classes.py "$SOURCE_DIR" "$ALL_CLASSES_CSV" "$CORE_CLASSES_CSV"

# 1.1 Extracción Estructural
echo "   -> [Estructural] Extrayendo vista estructural..."
python fase1_extract_structural_view.py "$SOURCE_DIR" "$CORE_CLASSES_CSV" "$STRUCTURAL_CSV_RAW"

# OMITIDO: 1.2 Extracción Semántica
# OMITIDO: 1.3 Extracción Funcional

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 2: PREPROCESSING (Solo Structural Matrix) "
echo "=========================================================================="

# 2.1 Matriz Estructural (A_str)
echo "   -> [Estructural] Construyendo matriz estructural (A_str)"
python fase2_build_structural_matrix.py "$STRUCTURAL_CSV_RAW" "$CORE_CLASSES_CSV" "$STR_MATRIX_BASE"

# OMITIDO: 2.2 Matriz Semántica
# OMITIDO: 2.3 Matriz Funcional

# ==============================================================================

echo ""
echo "=========================================================================="
echo "  FASE 3: FUSION BYPASS (Simulando Fusión) "
echo "=========================================================================="

STR_MATRIX_CSV="${STR_MATRIX_BASE}_matrix.csv"
FUSION_MATRIX_CSV="${FUSION_MATRIX_BASE}_matrix.csv"

echo "   -> [MIDAS-Str] Omitiendo algoritmo de fusión ponderada."
echo "   -> [MIDAS-Str] Copiando Matriz Estructural como Matriz Final..."

# AQUÍ ESTA LA CLAVE DE MIDAS-STR:
# Copiamos la matriz estructural directamente a la entrada del clustering
cp "$STR_MATRIX_CSV" "$FUSION_MATRIX_CSV"

if [ -f "$FUSION_MATRIX_CSV" ]; then
    echo "   -> [OK] Matriz preparada para clustering."
else
    echo "   -> [ERROR] No se pudo generar la matriz para clustering."
    exit 1
fi

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 4: CLUSTERING (Spectral Clustering on Structural View) "
echo "=========================================================================="

CLUSTERING_DIR="${RESULTS_DIR}/${NOMBRE_MONOLITO}_fase4_spectral_clustering"
echo "   -> [Clustering] Ejecutando sobre Matriz Estructural..."

# El script de clustering no sabe que es solo estructural, solo procesa la matriz que le llega
python fase4_clustering.py "$FUSION_MATRIX_CSV" "$CLUSTERING_DIR"

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  FASE 5: EVALUATION (Quality Evaluation) "
echo "=========================================================================="

# Seleccionamos dinamicamente el K target definido arriba
TARGET_K_JSON="$CLUSTERING_DIR/clustering_results/k_${K_TARGET}.json"

echo "   -> [Evaluation] Evaluando partición K=$K_TARGET..."

if [ -f "$TARGET_K_JSON" ]; then
    python fase5_evaluation.py "$STRUCTURAL_CSV_RAW" "$TARGET_K_JSON" "$CALCULATE_METRICS"
else
    echo "   -> [ERROR] No se encontró el archivo de clusters: $TARGET_K_JSON"
fi

# ==============================================================================
echo ""
echo "=========================================================================="
echo "  MIDAS-Str FINALIZADO."
echo "  Resultados en: $RESULTS_DIR"
echo "=========================================================================="
exit 0