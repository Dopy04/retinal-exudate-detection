# -*- coding: utf-8 -*-

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
    coordenades=feature.peak_local_max(distancia,labels=mascara,
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
    from B4_segmentacio import segmentar_otsu
    from B7_postprocessament import netejar_mascara
    from B1_metriques import calcular_metriques,imprimir_metriques

    #Escollim el dataset i la imatge que volem provar
    #Per provar una altra només cal canviar aquests dos valors
    dataset="DR2019"
    numero=3

    ruta_projecte=Path(__file__).resolve().parent.parent
    ruta_dades=ruta_projecte/"data_samples"

    #Obtenim automaticament la imatge i el seu Ground Truth
    ruta_imatge,ruta_gt=prep.obtenir_rutes_dataset(
        ruta_dades,dataset,numero)

    #Fem el preprocessament i obtenim el canal verd
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=prep.preprocessa_imatge(
        ruta_imatge,ruta_gt)

    canal_g=canals[1]

    #Fem primer una segmentacio amb Otsu i apliquem la morfologia
    #per obtenir la mascara que entrarà al Watershed
    mascara_otsu=segmentar_otsu(canal_g,mascara_retina)

    mascara_morfologia=netejar_mascara(mascara_otsu)

    #Apliquem Watershed per separar els blobs que estan connectats
    etiquetes=aplicar_watershed(mascara_morfologia)

    #Cada valor diferent de zero representa un blob
    nombre_blobs=np.max(etiquetes)

    #Per comparar amb el Ground Truth tornem les etiquetes
    #a una mascara binaria
    mascara_watershed=etiquetes>0

    metriques_abans=calcular_metriques(mascara_morfologia,mascara_gt)

    metriques_despres=calcular_metriques(mascara_watershed,mascara_gt)

    print("\n======================================")
    print("PROVA BLOC 8")
    print("Dataset:",dataset)
    print("Imatge:",str(numero).zfill(3))
    print("Nombre de blobs:",nombre_blobs)
    print("======================================")

    imprimir_metriques("Abans Watershed",metriques_abans)
    imprimir_metriques("Despres Watershed",metriques_despres)

    #Mostrem la mascara abans del Watershed i els blobs
    #separats amb una etiqueta diferent
    plt.figure(figsize=(16,7))
    plt.suptitle(
        "PROVA BLOC 8 - "+dataset+" "+str(numero).zfill(3),
        fontsize=16)

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
    
# Ús d'IA:
# Hem utilitzat IA com a suport per entendre millor el funcionament de Watershed,
# sobretot la transformada de distancia i l'ús dels marcadors per separar blobs.
# A partir d'aixo hem adaptat el metode a les mascares del nostre projecte
# i hem comprovat que la separació de regions funcionés correctament.