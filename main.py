# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:56:30 2026

@author: gurpi
"""

import numpy as np
from pathlib import Path

from src.metriques import calcular_metriques,imprimir_metriques
from src.preprocessament_3 import preprocessa_imatge,visualitza_preprocessament


# BLOC 2 - METRIQUES

mascara_gt_prova=np.array([
    [0,0,1,1],
    [0,0,1,1],
    [0,0,0,0],
    [1,1,0,0]
])

mascara_predita_prova=np.array([
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [1,0,0,0]
])

metriques=calcular_metriques(mascara_predita_prova,mascara_gt_prova)

imprimir_metriques("Prova bloc 2",metriques)


# BLOC 3 - PREPROCESSAMENT


# Ruta de la carpeta principal del projecte
ruta_projecte=Path(__file__).resolve().parent

# Ruta de la carpeta on tenim les imatges
ruta_dades=ruta_projecte/"data_samples"


# Posem les 3 imatges i els seus Ground Truth
imatges=[
    ("Exudate_JSIEC_002.jpg","Exudate_JSIEC_002_Groundtruth.jpg"),
    ("Exudate_DR2019_001.jpg","Exudate_DR2019_001_Groundtruth.jpg"),
    ("Exudate_DR2019_003.jpg","Exudate_DR2019_003_Groundtruth.jpg")
]


# Provem les 3 imatges una per una
for nom_imatge,nom_gt in imatges:

    print("\n------------------------------")
    print("Imatge:",nom_imatge)
    print("------------------------------")

    # Creem les rutes completes
    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # Fem tot el preprocessament
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,ruta_gt
    )

    # Mostrem les dimensions per comprovar que tot coincideix
    print("Mida imatge:",imatge.shape)
    print("Mida Ground Truth:",mascara_gt.shape)
    print("Mida mascara retina:",mascara_retina.shape)
    print("Nombre de canals:",len(canals))

    # Comprovem que podem comparar pixel a pixel
    if imatge.shape[:2]==mascara_gt.shape:
        print("OK: imatge i Ground Truth tenen la mateixa mida")
    else:
        print("ERROR: les mides no coincideixen")

    # Mostrem els resultats del preprocessament
    visualitza_preprocessament(
        imatge,
        mascara_gt,
        mascara_retina,
        canals
    )