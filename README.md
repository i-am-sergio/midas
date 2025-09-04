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