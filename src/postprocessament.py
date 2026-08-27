# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:08:50 2026

@author: gurpi
"""

"""
Morfologia matematica per netejar les mascares dels exsudats.

La mascara que rebem ja ve d'un metode de segmentacio
com Otsu, Sauvola, K-means o NICK/INRG.

Aqui no detectem els exsudats des de zero.
El que fem es intentar netejar la mascara inicial.
"""

from skimage import morphology


def eliminar_objectes_petits(mascara,mida_minima=50):
    """
    Eliminem blobs molt petits.

    Aquests blobs poden ser soroll o falsos positius
    generats durant la segmentacio.

    mida_minima:
        nombre minim de pixels que ha de tenir un objecte
        per mantenir-lo.
    """

    mascara=mascara.astype(bool)

    mascara_neta=morphology.remove_small_objects(
        mascara,min_size=mida_minima
    )

    return mascara_neta


def tancar_mascara(mascara,radi=2):
    """
    Fem una operacio de closing.

    El closing es:
        dilatacio + erosio

    Serveix per tancar petits espais i unir zones
    molt properes dins d'un mateix blob.
    """

    element=morphology.disk(radi)

    mascara_tancada=morphology.binary_closing(
        mascara,element
    )

    return mascara_tancada


def obrir_mascara(mascara,radi=1):
    """
    Fem una operacio d'opening.

    L'opening es:
        erosio + dilatacio

    Serveix per eliminar petits punts o irregularitats.

    Hem de vigilar amb aquesta operacio perque
    els exsudats petits tambe poden desapareixer.
    """

    element=morphology.disk(radi)

    mascara_oberta=morphology.binary_opening(
        mascara,element
    )

    return mascara_oberta


def eliminar_forats_petits(mascara,mida_forat=50):
    """
    Omplim petits forats que hi hagi dins dels blobs.

    Per exemple, si un exsudat queda detectat com una regio
    blanca amb un petit forat negre al mig, podem omplir-lo.
    """

    mascara_neta=morphology.remove_small_holes(
        mascara,area_threshold=mida_forat
    )

    return mascara_neta


def netejar_mascara(mascara,mida_minima=50,mida_forat=50):
    """
    Funcio que aplica el proces de morfologia complet.

    Ordre:
        1. eliminar objectes petits
        2. closing
        3. eliminar forats petits

    De moment no posem opening automaticament
    perque podria eliminar exsudats petits.
    """

    mascara=mascara.astype(bool)

    # Eliminem petits blobs que poden ser soroll
    mascara=eliminar_objectes_petits(mascara,mida_minima)

    # Tanquem petites separacions dins dels blobs
    mascara=tancar_mascara(mascara,radi=2)

    # Omplim petits forats
    mascara=eliminar_forats_petits(mascara,mida_forat)

    return mascara