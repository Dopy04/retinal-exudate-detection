# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:27:08 2026

@author: gurpi
"""

"""
Bloc 8:Watershed.

En aquest bloc utilitzem Watershed per separar blobs d'exsudats que
han quedat units després de la segmentacio i la morfologia.

Watershed no busca exsudats nous ni modifica les regions detectades,
simplement intenta separar una mateixa regio connectada en diferents blobs
per poder analitzar-los individualment després.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import feature,segmentation


def transformar_distancia(mascara):
    """
    Calculem la distancia de cada pixel blanc fins al pixel de fons
    més proper.
    Els pixels que estan al centre dels blobs tindran una distancia
    més gran i ens ajudaran després a trobar el centre de cada regio.
    """

    mascara=mascara.astype(bool)

    #Calculem el mapa de distancies de tota la mascara
    distancia=ndi.distance_transform_edt(mascara)

    return distancia


def trobar_marcadors(distancia,mascara,distancia_minima=5):
    """
    Busquem els maxims locals del mapa de distancies.

    Aquests maxims representen possibles centres dels blobs i els
    utilitzarem com a punts inicials per aplicar Watershed.La 
    distancia_minima indica la separacio minima que hi ha d'haver
    entre dos marcadors diferents.
    """
    #Busquem els punts amb una distancia local més gran dins de la mascara
    coordenades=feature.peak_local_max(
        distancia,
        labels=mascara,
        min_distance=distancia_minima)

    #Creem una imatge buida amb la mateixa mida que la mascara
    marcadors=np.zeros(mascara.shape,dtype=int)

    #Assignem una etiqueta diferent a cada maxim que hem trobat
    for i in range(len(coordenades)):
        fila,columna=coordenades[i]
        marcadors[fila,columna]=i+1

    return marcadors


def aplicar_watershed(mascara,distancia_minima=5):
    """
    Separem els blobs connectats utilitzant Watershed.

    Primer calculem la transformada de distancia, després busquem
    els marcadors i finalment apliquem Watershed per separar les regions.

    La sortida ja no és una mascara True/False sinó una imatge
    d'etiquetes on:
        0=fons
        1=blob 1
        2=blob 2
        3=blob 3...
    """

    mascara=mascara.astype(bool)

    #Calculem la distancia dels pixels respecte el fons
    distancia=transformar_distancia(mascara)

    #Busquem els centres que utilitzarem com a marcadors
    marcadors=trobar_marcadors(distancia,mascara,distancia_minima)

    #Utilitzem -distancia perquè Watershed comenci pels centres
    #dels blobs i vagi expandint cada regio
    etiquetes=segmentation.watershed(-distancia,marcadors,mask=mascara)

    return etiquetes


if __name__=="__main__":

    #Prova Bloc 8: apliquem Watershed sobre una mascara real per veure
    #com separa els diferents blobs que estan connectats

    from pathlib import Path
    import matplotlib.pyplot as plt
    import B2_preprocessament as prep
    from B3_segmentacio import segmentar_otsu
    from B7_postprocessament import netejar_mascara
    from B1_metriques import calcular_metriques,imprimir_metriques

    #Imatge que utilitzem per la prova
    #Per provar una altra només cal canviar aquests dos noms
    nom_imatge="Exudate_DR2019_003.jpg"
    nom_gt="Exudate_DR2019_003_Groundtruth.jpg"

    ruta_projecte=Path(__file__).resolve().parent.parent
    ruta_dades=ruta_projecte/"data_samples"

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    #Fem el preprocessament i obtenim el canal verd
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=prep.preprocessa_imatge(
        ruta_imatge,ruta_gt)

    canal_g=canals[1]

    #Fem una segmentacio inicial amb Otsu i després la netegem
    #amb la morfologia abans d'aplicar Watershed
    mascara_otsu=segmentar_otsu(canal_g,mascara_retina)
    mascara_morfologia=netejar_mascara(mascara_otsu)

    #Apliquem Watershed i obtenim una etiqueta diferent per cada blob
    etiquetes=aplicar_watershed(mascara_morfologia)

    #Per calcular les metriques tornem a convertir les etiquetes
    #en una mascara binaria
    mascara_watershed=etiquetes>0

    metriques_abans=calcular_metriques(mascara_morfologia,mascara_gt)
    metriques_despres=calcular_metriques(mascara_watershed,mascara_gt)

    print("\n======================================")
    print("PROVA BLOC 8")
    print("Imatge:",nom_imatge)
    print("Nombre de blobs:",np.max(etiquetes))
    print("======================================")

    imprimir_metriques("Abans Watershed",metriques_abans)
    imprimir_metriques("Despres Watershed",metriques_despres)

    #Mostrem el resultat abans i després de separar els blobs
    plt.figure(figsize=(16,7))
    plt.suptitle("PROVA BLOC 8 - "+nom_imatge,fontsize=16)

    plt.subplot(1,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(mascara_morfologia,cmap="gray")
    plt.title("Abans Watershed")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(etiquetes,cmap="nipy_spectral")
    plt.title("Watershed")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()