# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:42:52 2026

@author: gurpi
"""

"""
NICK i INRG per la segmentacio d'exsudats.

Amb el algoritme NICK calculem un llindar local  per cada pixel i
posteriorment el INRG aplica NICK divereses interacions fins que no hi ha
hagui un canvi notable

El NICK estava fet en principi per les hemorragies pero aplicant una inversió
de canal podem detectar exsudats, ja que aquests són brillants i per aquest
metode volem que siguin regions fosques.
"""

import numpy as np
from scipy.ndimage import uniform_filter


def invertir_canal(canal):
    """
    Invertim la intensitat del canal suposant que el canal esta normalitzat
    entre 0 o 1, amb això els exsudats passen a ser foscos
    Invertim la intensitat del canal.
    """
    canal_invertit=1-canal

    return canal_invertit


def segmentar_nick(canal,mascara_retina=None,mida_finestra=95,k=-0.10):
    """
    Fem una segmentacio amb NICK thresholding.
    
    NICK calcula un llindar diferent per cada  area de la imatge 
    utilitzant la mitjana i la variació local de pixels. Utilizem el 
    canal invertit en aquesta funció i definim la mida de la finestra que
    és la zona local que mirem per calcular el llindar i  k que és el parametre 
    del  metode NICK(-0,2 al paper)
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

    El metode comença amb una mascara NICK inicial.
    Despres modifica la intensitat de les regions detectades
    i torna a aplicar NICK.

    Les noves regions es van acumulant amb les regions
    detectades anteriorment.
    """

    # Invertim el canal perquè els exsudats siguin foscos
    imatge=1-canal

    # Primera segmentacio NICK
    mascara=segmentar_nick(
        imatge,
        mascara_retina,
        mida_finestra,
        k
    )

    # A partir de la segona iteracio
    for i in range(1,iteracions):

        mascara_anterior=mascara.copy()

        area_anterior=np.sum(mascara_anterior)

        # Si no hi ha cap regio, no podem continuar
        if area_anterior==0:
            break

        # Intensitat mitjana de la regio detectada
        valor_mitja=np.mean(
            imatge[mascara_anterior]
        )

        # Substituim la regio per la seva intensitat mitjana
        imatge[mascara_anterior]=valor_mitja

        # Tornem a aplicar NICK
        mascara_nova=segmentar_nick(
            imatge,
            mascara_retina,
            mida_finestra,
            k
        )

        # INRG acumula la regio anterior i la nova
        mascara=mascara_anterior | mascara_nova

        area_actual=np.sum(mascara)

        creixement=(area_actual-area_anterior)/area_anterior

        # Si la regio creix menys que delta, parem
        if creixement<delta:
            break

    return mascara