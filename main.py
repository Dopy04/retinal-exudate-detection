# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:56:30 2026

@author: gurpi
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.metriques import calcular_metriques,imprimir_metriques,comparar_metodes
from src.preprocessament_3 import preprocessa_imatge,visualitza_preprocessament
from src.segmentacio import obtenir_mascares_baseline

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
    
# ============================================================
# BLOC 4 - SEGMENTACIO BASELINE
# ============================================================

print("\n==============================")
print("BLOC 4 - SEGMENTACIO BASELINE")
print("==============================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 4:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # Tornem a fer el preprocessament per obtenir
    # els canals i la mascara de retina
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,ruta_gt
    )

    # Utilitzem el canal G
    canal_g=canals[1]

    # Executem tots els metodes de segmentacio baseline
    mascares=obtenir_mascares_baseline(
        canal_g,
        mascara_retina
    )

    # Comparem totes les mascares amb el Ground Truth
    resultats=comparar_metodes(
        mascares,
        mascara_gt
    )

    # Mostrem les metriques de cada metode
    for nom_metode in resultats:

        imprimir_metriques(
            nom_metode,
            resultats[nom_metode]
        )


    # --------------------------------------------------------
    # VISUALITZACIO
    # --------------------------------------------------------

    plt.figure(figsize=(14,8))

    plt.suptitle(
        "PROVA BLOC 4 - "+nom_imatge+" - Canal G",
        fontsize=16
    )

    # Imatge original
    plt.subplot(2,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    # Canal G
    plt.subplot(2,4,2)
    plt.imshow(canal_g,cmap="gray")
    plt.title("Canal G")
    plt.axis("off")

    # Ground Truth
    plt.subplot(2,4,3)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    # Otsu
    plt.subplot(2,4,4)
    plt.imshow(mascares["Otsu"],cmap="gray")
    plt.title("Otsu")
    plt.axis("off")

    # Llindar fix
    plt.subplot(2,4,5)
    plt.imshow(mascares["Llindar_fix"],cmap="gray")
    plt.title("Llindar fix")
    plt.axis("off")

    # Adaptatiu
    plt.subplot(2,4,6)
    plt.imshow(mascares["Adaptatiu"],cmap="gray")
    plt.title("Adaptatiu")
    plt.axis("off")

    # Sauvola
    plt.subplot(2,4,7)
    plt.imshow(mascares["Sauvola"],cmap="gray")
    plt.title("Sauvola")
    plt.axis("off")

    # K-means
    plt.subplot(2,4,8)
    plt.imshow(mascares["Kmeans"],cmap="gray")
    plt.title("K-means")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()