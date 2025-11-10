#!/bin/bash

# Requirements
## Para cpr y nlohmann-json (functional view fase1)
# sudo apt install libcpr-dev nlohmann-json3-dev -y

# si no existe la carpeta results, crearla
if [ ! -d "results" ]; then
  mkdir results
fi


# FASE 1: Extraction
echo "----------------- FASE 1: Extraction -----------------"
## Estructural View
g++ extract_structural_view.cpp -o extract_structural_view 
./extract_structural_view ../monoliths/jPetStore

g++ analyze_relations.cpp -o analyze_relations
./analyze_relations results/jPetStore_fase1.csv ../monoliths/jPetStore/

## Semantic View
g++ extract_semantic_view.cpp -o extract_semantic_view
./extract_semantic_view ../monoliths/jPetStore

## Functional View
g++ extract_functional_view.cpp -o extract_functional_view
./extract_functional_view results/jPetStore_fase1_semantic_view.csv ../monoliths/jPetStore/


# FASE 2: Preprocessing and Matrix Generation
echo "---------- FASE 2: Preprocessing and Matrix Generation ----------"
# g++ preprocess_data.cpp -o preprocess_data
# ./preprocess_data ../monoliths/jPetStore

# FASE 3: Multiview Fusion

# FASE 4: Clustering











# La lista de 24 clases que usará el script es:

# Account
# AccountActionForm
# AccountDao
# AccountForm
# BaseAction
# BaseActionForm
# Cart
# CartActionForm
# CartItem
# Category
# CategoryDao
# Item
# ItemDao
# LineItem
# Order
# OrderActionForm
# OrderDao
# OrderForm
# OrderService
# PetStoreFacade
# PetStoreImpl
# Product
# ProductDao
# UserSession