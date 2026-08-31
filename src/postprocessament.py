# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:08:50 2026

@author: gurpi
"""

"""
Morfologia matematica per netejar les mascares dels exsudats.

Les mascares que rebem ja venen amb un algoritme aplicat com el
Otsu,Sauvola,K-means etc.En aquest apartarr intentarem netejar la mascara inicial.
"""

import numpy as np

from skimage import morphology
from skimage.measure import label,regionprops


def eliminar_objectes_petits(mascara,mida_minima=20):
    """
    Eliminem blobs molt petits.

    Aquests blobs poden ser soroll o falsos positius
    generats durant la segmentacio. La mida mínima es el nombre
    mínim de pixels que ha de tenir l'objecte per mantenir-ho.
    """

    mascara=mascara.astype(bool)

    mascara_neta=morphology.remove_small_objects(
        mascara,
        min_size=mida_minima
    )

    return mascara_neta


def tancar_mascara(mascara,radi=2):
    """
    Fem una operacio de closing.
    El closing es basa en la diltaació+erosió, serveix per tancar 
    petits espais i unir zones molt properes dins d'un mateix blob.
    """

    element=morphology.disk(radi)

    mascara_tancada=morphology.closing(
        mascara,
        element
    )

    return mascara_tancada


def obrir_mascara(mascara,radi=1):
    """
    Fem una operacio d'opening.

    L'opening es erosio + dilatacio, serveix per eliminar petits punts o irregularitats.

    Aquesta operació s'ha de tenir compte perque pot eliminar petits exsudats.
    """

    element=morphology.disk(radi)

    mascara_oberta=morphology.opening(
        mascara,
        element
    )

    return mascara_oberta


def eliminar_forats_petits(mascara,mida_forat=30):
    """
    Omplim petits forats que hi hagi dins dels blobs. Per exemple, 
    si un exsudat queda detectat com una regio
    blanca amb un petit forat negre al mig, podem omplir-lo.
    """

    mascara_neta=morphology.remove_small_holes(
        mascara,
        area_threshold=mida_forat
    )

    return mascara_neta


def eliminar_artefactes_vora(mascara,mascara_retina,radi_vora=3):
    """
    Eliminem objectes grans que estan enganxats al limit de la retina.

    En algunes imatges apareixen reflexos o zones molt brillants
    al contorn de la retina que Otsu o INRG poden detectar com exsudats.

    No eliminem objectes petits de la vora per evitar perdre
    possibles exsudats periferics.
    """

    if mascara_retina is None:
        return mascara

    if mascara.shape!=mascara_retina.shape:
        raise ValueError(
            "La mascara i la mascara de retina no tenen la mateixa mida"
        )

    mascara=mascara.astype(bool)

    # Creem una zona estreta corresponent al contorn de la retina
    retina_interior=morphology.erosion(
        mascara_retina,
        morphology.disk(radi_vora)
    )

    vora_retina=mascara_retina&(~retina_interior)

    etiquetes=label(mascara)

    mascara_neta=np.copy(mascara)

    # Considerem gran un objecte que ocupa mes del 0.5%
    # de tota la zona de retina
    area_retina=np.sum(mascara_retina)

    area_gran=0.005*area_retina

    for regio in regionprops(etiquetes):

        mascara_regio=etiquetes==regio.label

        toca_vora=np.any(
            mascara_regio&vora_retina
        )

        if toca_vora and regio.area>area_gran:
            mascara_neta[mascara_regio]=False

    return mascara_neta


def netejar_mascara(mascara,mascara_retina=None,mida_minima=20,
                    mida_forat=30,eliminar_vora=False):
    """
    Funcio que aplica el proces de morfologia complet.
    
    El procediment sera primer eliminar els objectes petits, desprès aplicar el closing
    i finalment eliminar els forats petits.
    De moment no posem opening automaticament perque podria eliminar exsudats petits.
    
    Si eliminar_vora=True també eliminem objectes grans que estan enganxats
    al contorn de la retina. Ho utilitzarem sobretot amb Otsu, ja que amb INRG
    podem tenir regions molt grans connectades i podriem eliminar exsudats reals.
    """

    mascara=mascara.astype(bool)

    # Tanquem petites separacions dins dels blobs
    mascara=tancar_mascara(mascara,radi=2)

    # Eliminem petits blobs que poden ser soroll
    mascara=eliminar_objectes_petits(mascara,mida_minima)

    # Omplim petits forats
    mascara=eliminar_forats_petits(mascara,mida_forat)

    # Eliminem grans artefactes del contorn nomes si ho volem
    if eliminar_vora:
        mascara=eliminar_artefactes_vora(
            mascara,
            mascara_retina
        )

    # Ens assegurem que no queda res fora de la retina
    if mascara_retina is not None:
        mascara=mascara&mascara_retina

    return mascara