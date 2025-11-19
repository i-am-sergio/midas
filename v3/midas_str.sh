#!/bin/bash

# Verificar que se proporcionó el parámetro
if [ $# -ne 1 ]; then
    echo "Uso: $0 <nombre_monolito>"
    echo "Monolitos disponibles: acmeair, jpetstore, daytrader, plants, jrideconnect"
    echo "Ejemplo: $0 jpetstore"
    echo "Ejemplo: $0 daytrader"
    exit 1
fi

NOMBRE_MONOLITO=$1
MONOLITO_DIR="../monoliths/"
RESULTS_DIR="${NOMBRE_MONOLITO}_results"

# Asignar directorio fuente específico según el monolito
case $NOMBRE_MONOLITO in
    "acmeair")
        SOURCE_DIR="${MONOLITO_DIR}acmeair/"
        ;;
    "jpetstore")
        SOURCE_DIR="${MONOLITO_DIR}jPetStore/"
        ;;
    "daytrader")
        SOURCE_DIR="${MONOLITO_DIR}sample.daytrader7/"
        ;;
    "plants")
        SOURCE_DIR="${MONOLITO_DIR}sample.plantsbywebsphere/"
        ;;
    "jrideconnect")
        SOURCE_DIR="${MONOLITO_DIR}jrideconnect/"
        ;;
    *)
        echo "Error: Monolito '$NOMBRE_MONOLITO' no reconocido"
        echo "Monolitos disponibles: acmeair, jpetstore, daytrader, plants, jrideconnect"
        exit 1
        ;;
esac

# Verificar que existe el directorio fuente
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: No se encuentra el directorio fuente $SOURCE_DIR"
    exit 1
fi

echo "=================================================="
echo "Ejecutando MIDAS para: $NOMBRE_MONOLITO"
echo "Directorio fuente: $SOURCE_DIR"
echo "Directorio resultados: $RESULTS_DIR"
echo "=================================================="

# FASE 1: Extraction 
echo "FASE 1: Extraction"
python extract_structural_view.py "$SOURCE_DIR" "$RESULTS_DIR/"
python analyze_relations.py "$RESULTS_DIR/structural_view.csv" "$SOURCE_DIR" "$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv"

# FASE 2: Preprocessing and Build Matrices
echo "FASE 2: Preprocessing and Build Matrices"
python preprocessing_structural.py "$RESULTS_DIR/${NOMBRE_MONOLITO}_fase1_structural_view.csv" "$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural_view_filtered.csv"
python build_structural_matrix.py "$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural_view_filtered.csv" "$RESULTS_DIR/${NOMBRE_MONOLITO}_structural"

# FASE 3: Multiview Self-Weighted Fusion (pendiente)
# echo "FASE 3: Multiview Self-Weighted Fusion"
# python multiview_fusion.py "$RESULTS_DIR/${NOMBRE_MONOLITO}_structural_matrix.csv" 

# FASE 4: Clustering
echo "FASE 4: Clustering"
python optimize_k_spectral.py "$RESULTS_DIR/${NOMBRE_MONOLITO}_structural_matrix.csv" "$RESULTS_DIR/spectral_clustering_results"

# FASE 5: Evaluation
echo "FASE 5: Evaluation"
python calculate_metrics.py "$RESULTS_DIR/spectral_clustering_results/k_5.json" "$RESULTS_DIR/${NOMBRE_MONOLITO}_fase2_structural_view_filtered.csv"

echo "=================================================="
echo "Proceso completado para: $NOMBRE_MONOLITO"
echo "Resultados en: $RESULTS_DIR"
echo "=================================================="