# **MIDAS: A Multiview Graph-Based Approach for Automatic Microservice Extraction Enhanced by Domain Knowledge Using BERT Models and Self-Weighted Clustering**

## Prerequisites
- **Environment:** SO Linux/Debian 13
```sh
# neofetch
       _,met$$$$$gg.          heros@debian 
    ,g$$$$$$$$$$$$$$$P.       ------------ 
  ,g$$P"     """Y$$.".        OS: Debian GNU/Linux 13 (trixie) x86_64 
 ,$$P'              `$$$.     Host: HP Laptop 15-gw0xxx 
',$$P       ,ggs.     `$$b:   Kernel: 6.12.30-amd64 
`d$$'     ,$P"'   .    $$$    Uptime: 22 hours, 57 mins 
 $$P      d$'     ,    $$P    Packages: 2309 (dpkg) 
 $$:      $$.   -    ,d$$'    Shell: bash 5.2.37 
 $$;      Y$b._   _,d$P'      Resolution: 1366x768 
 Y$$.    `.`"Y$$$$P"'         DE: Cinnamon 6.4.10 
 `$$b      "-.__              WM: Mutter (Muffin) 
  `Y$$                        WM Theme: cinnamon (Default) 
   `Y$$.                      Theme: Adwaita-dark [GTK2/3] 
     `$$b.                    Icons: mate [GTK2/3] 
       `Y$$b.                 Terminal: WarpTerminal 
          `"Y$b._             CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx (8) @ 2.100GHz 
              `"""            GPU: AMD ATI Radeon Vega Series / Radeon Vega Mobile Series 
                              Memory: 7813MiB / 13925MiB 
```

- **Python Version:**
```sh
# python --version
Python 3.9.18
```

- **GCC Version:**
```sh
# g++ --version
g++ (Debian 14.2.0-19) 14.2.0
Copyright (C) 2024 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

---

## Dependencies

Para instalar todas las librerías necesarias en tu entorno (Google Colab, Jupyter Notebook o local con Python 3.x), ejecuta:

```bash
# Parser de código Java en Python
pip install javalang

# Modelos preentrenados (Hugging Face Transformers)
pip install transformers

# PyTorch (para Transformers y deep learning)
pip install torch

# Scikit-learn (para preprocesamiento, métricas y modelos ML)
pip install scikit-learn

# Visualización de datos
pip install matplotlib seaborn

# Manejo numérico y algebra lineal
pip install numpy

# Librerías para grafos
pip install networkx pygraphvizV
```

## Approach

### Q1: Complete Approach 

```sh
# FASE 1: Extraction 
python extract_structural_view.py ../monoliths/jPetStore/ jpetstore_results/
python analyze_relations.py jpetstore_results/structural_view.csv ../monoliths/jPetStore/ jpetstore_results/jPetStore_fase1_structural_view.csv
# FASE 2: Preprocessing and Build Matrices
python preprocessing_structural.py jpetstore_results/jPetStore_fase1_structural_view.csv jpetstore_results/jPetStore_fase2_structural_view_filtered.csv
python build_structural_matrix.py jpetstore_results/jPetStore_fase2_structural_view_filtered.csv jpetstore_results/jpetstore_structural
# FASE 3: Multiview Fusion
python self_weighted_fusion.py jpetstore_results/jpetstore_structural_matrix.csv jpetstore_results/jpetstore_fused_matrix.csv
# FASE 4: Clustering
python optimize_k_spectral.py jpetstore_results/jpetstore_structural_matrix.csv jpetstore_results/spectral_clustering_results
# FASE 5: Evaluation
python calculate_metrics.py jpetstore_results/spectral_clustering_results/k_5.json jpetstore_results/jPetStore_fase2_structural_view_filtered.csv
```

```sh
# Ejecutar para cada monolito
./midas.sh acmeair
./midas.sh jpetstore
./midas.sh daytrader
./midas.sh plants
./midas.sh jrideconnect
```


### Q2: Evaluation of monoview versions of MIDAS
- **MIDAS-Str**

```sh
# Ejecutar para cada monolito
./midas_str.sh acmeair
./midas_str.sh jpetstore
./midas_str.sh daytrader
./midas_str.sh plants
./midas_str.sh jrideconnect
```
- **MIDAS-Sem**

```sh
# Ejecutar para cada monolito
./midas_sem.sh acmeair
./midas_sem.sh jpetstore
./midas_sem.sh daytrader
./midas_sem.sh plants
./midas_sem.sh jrideconnect
```

- **MIDAS-Fun**

```sh
# Ejecutar para cada monolito
./midas_fun.sh acmeair
./midas_fun.sh jpetstore
./midas_fun.sh daytrader
./midas_fun.sh plants
./midas_fun.sh jrideconnect
```

### Q3: Best weighted combination of views

```sh
# Ejecutar para cada monolito
./midas_weighted.sh acmeair
./midas_weighted.sh jpetstore
./midas_weighted.sh daytrader
./midas_weighted.sh plants
./midas_weighted.sh jrideconnect
```

<!-- 1. **Extraction**
```sh
python 1_extract_data.py # write data in csvs 
```

2. **Embeddings Generation and Build Graph**
```sh
python 2_estructural_embeddings.py # CodeBERT and Similarity Matrix
python 2_functional_embeddings.py # MPNet and Similarity Matrix
python 2_semantic_embeddings.py # MPNet and Similarity Matrix
```

3. **Multiview Mapping and Self-Weighted Fusion**
```sh
python 3_fusion.py # fusion of matrix
```

4. **Clustering**
```sh
python 4_clustering.py # Kmeans
```

<!-- 5. **Evaluation** -->
