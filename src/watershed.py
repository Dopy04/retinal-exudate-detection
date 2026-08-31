# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:27:08 2026

@author: gurpi
"""


"""
Watershed per separar blobs d'exsudats que han quedat units.

La mascara que rebem ja ve de la segmentacio i de la morfologia.
Watershed no busca exsudats nous, nomes intenta separar regions
que estan connectades.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage import feature,segmentation,measure


def transformar_distancia(mascara):
    """
    Calculem la distancia de cada pixel blanc fins al fons.

    Els pixels que estan al centre dels blobs tindran
    una distancia mes gran.
    """

    mascara=mascara.astype(bool)

    distancia=ndi.distance_transform_edt(mascara)

    return distancia


def trobar_marcadors(distancia,mascara,distancia_minima=5):
    """
    Busquem els maxims locals del mapa de distancies.

    Aquests punts els utilitzarem com a centres inicials
    dels diferents blobs.

    distancia_minima:
        separacio minima entre dos maxims.
        Si es molt petita podem detectar massa marcadors.
    """

    mascara=mascara.astype(bool)

    coordenades=feature.peak_local_max(
        distancia,
        labels=mascara,
        min_distance=distancia_minima,
        exclude_border=False
    )

    # Creem una imatge buida per guardar els marcadors
    marcadors=np.zeros(mascara.shape,dtype=int)

    # Cada maxim tindra una etiqueta diferent:
    # primer maxim=1, segon=2, tercer=3...
    for i in range(len(coordenades)):
        fila,columna=coordenades[i]
        marcadors[fila,columna]=i+1

    # Comprovem que cada component connectat tingui
    # com a minim un marcador.
    components=measure.label(
        mascara,
        connectivity=2
    )

    seguent_marcador=np.max(marcadors)+1

    for etiqueta in range(1,np.max(components)+1):

        component=components==etiqueta

        # Si aquest component ja te algun marcador no fem res
        if np.any(marcadors[component]>0):
            continue

        # Si no te marcador busquem el pixel mes interior
        distancia_component=np.where(
            component,
            distancia,
            -1
        )

        posicio=np.unravel_index(
            np.argmax(distancia_component),
            distancia.shape
        )

        marcadors[posicio]=seguent_marcador

        seguent_marcador=seguent_marcador+1

    return marcadors


def aplicar_watershed(mascara,distancia_minima=5):
    """
    Separem els blobs connectats utilitzant watershed.

    Passos:
        1. transformada de distancia
        2. trobar marcadors
        3. aplicar watershed

    La sortida no es una mascara True/False.
    Es una imatge d'etiquetes:
        0=fons
        1=blob 1
        2=blob 2
        3=blob 3...
    """

    mascara=mascara.astype(bool)

    # Si la mascara esta buida no podem aplicar watershed
    if not np.any(mascara):
        return np.zeros(mascara.shape,dtype=int)

    # Calculem la transformada de distancia
    distancia=transformar_distancia(mascara)

    # Busquem els centres dels diferents blobs
    marcadors=trobar_marcadors(
        distancia,
        mascara,
        distancia_minima
    )

    # Watershed treballa com si fos un relleu.
    # Posem -distancia perque els centres dels blobs
    # es comportin com les zones des d'on comenca la segmentacio.
    etiquetes=segmentation.watershed(
        -distancia,
        marcadors,
        mask=mascara
    )

    return etiquetes