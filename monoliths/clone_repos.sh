#!/bin/bash
# =========================================================
# Script para clonar proyectos en la carpeta actual
# =========================================================

# Lista de repositorios a clonar
REPOS=(
    "https://github.com/KimJongSung/jPetStore.git"
    "https://github.com/acmeair/acmeair.git"
    "https://github.com/WASdev/sample.daytrader7.git"
    "https://github.com/WASdev/sample.plantsbywebsphere.git"
)

# Iterar y clonar cada repo
for REPO in "${REPOS[@]}"; do
    echo "Clonando $REPO ..."
    git clone "$REPO"
    echo "--------------------------------------------"
done

echo "✅ Todos los repositorios han sido clonados."

