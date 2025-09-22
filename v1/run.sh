#!/bin/bash

python 1_extract_data.py 
python 2_preprocessing_structural.py  
python 2_preprocessing_functional.py  
python 2_preprocessing_semantic.py    
python 3_generate_embeddings.py    
python 4_build_multiview_graph.py  
python 5_clustering_multiview.py 
python 6_plot_clustering.py