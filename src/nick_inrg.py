# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:42:52 2026

@author: gurpi
"""

"""
NICK i INRG per la segmentacio d'exsudats.

NICK calcula un llindar local per cada pixel.
INRG aplica NICK diverses vegades per intentar completar
les regions candidates.

Com que els exsudats son brillants, primer invertim el canal
per convertir-los en regions fosques.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def invertir_canal(canal):
    """
    Invertim la intensitat del canal.

    Treballem suposant que el canal esta normalitzat entre 0 i 1.

    Exemple:
        pixel brillant=0.9
        pixel invertit=1-0.9=0.1

    Per tant els exsudats, que son brillants, passen a ser foscos.
    """

    canal_invertit=1-canal

    return canal_invertit


def segmentar_nick(canal,mascara_retina=None,mida_finestra=95,k=-0.2):
    """
    Fem una segmentacio amb NICK thresholding.

    NICK calcula un llindar diferent per cada zona de la imatge,
    utilitzant la mitjana i la variacio local.

    canal:
        canal invertit on els exsudats son foscos

    mida_finestra:
        mida de la zona local que mirem per calcular el llindar

    k:
        parametre del metode NICK
        al paper s'utilitza aproximadament k=-0.2
    """

    # Calculem la mitjana local de cada zona
    mitjana=uniform_filter(canal,size=mida_finestra)

    # Per calcular la variacio necessitem tambe la mitjana de I^2
    mitjana_quadrat=uniform_filter(canal**2,size=mida_finestra)

    # Variancia local:
    # E[I^2]-E[I]^2
    variancia=mitjana_quadrat-mitjana**2

    # Poden sortir petits valors negatius per errors numerics
    # per aixo els posem com a minim a 0
    variancia=np.maximum(variancia,0)

    # Desviacio local
    desviacio=np.sqrt(variancia)

    # Llindar local de NICK
    llindar=mitjana+k*desviacio

    # Com que hem invertit la imatge, els exsudats son foscos
    # per tant busquem pixels per sota del llindar
    mascara=canal<llindar

    # Si tenim mascara de retina, eliminem el fons exterior
    if mascara_retina is not None:
        mascara=mascara&mascara_retina

    return mascara


def segmentar_inrg(canal,mascara_retina=None,mida_finestra=95,k=-0.2,
                   iteracions=3,delta=0.1):
    """
    Apliquem INRG.

    La idea es:
        1. invertim el canal
        2. fem NICK
        3. modifiquem la intensitat de la regio detectada
        4. tornem a aplicar NICK
        5. parem si la regio gairebe ja no creix

    iteracions:
        nombre maxim de vegades que repetirem el metode

    delta:
        canvi minim de l'area per continuar iterant
    """

    # Primer invertim el canal perquè els exsudats siguin foscos
    imatge=1-canal

    # Al principi encara no tenim cap regio detectada
    mascara_anterior=np.zeros(canal.shape,dtype=bool)

    for i in range(iteracions):

        # Fem NICK sobre la imatge actual
        mascara=segmentar_nick(imatge,mascara_retina,mida_finestra,k)

        # Calculem l'area detectada abans i ara
        area_anterior=np.sum(mascara_anterior)
        area_actual=np.sum(mascara)

        # A partir de la segona iteracio podem mirar quant ha canviat
        if area_anterior>0:

            creixement=abs(area_actual-area_anterior)/area_anterior

            # Si la regio canvia menys que delta, parem
            if creixement<delta:
                break

        # Si hem detectat pixels, calculem la seva intensitat mitjana
        if area_actual>0:

            valor_mitja=np.mean(imatge[mascara])

            # Substituim els pixels detectats per aquest valor mitja
            # i tornarem a aplicar NICK en la seguent iteracio
            imatge[mascara]=valor_mitja

        mascara_anterior=mascara.copy()

    return mascara