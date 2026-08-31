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

    canal=np.asarray(
        canal,
        dtype=np.float64
    )

    canal=np.clip(
        canal,
        0,
        1
    )

    canal_invertit=1-canal

    return canal_invertit


def calcula_estadistiques_locals(
    canal,
    mascara_retina,
    mida_finestra
):
    """
    Calculem la mitjana i la variacio local.

    Si tenim mascara de retina fem els calculs nomes amb
    els pixels que estan dins de la retina per evitar que
    el fons exterior afecti NICK.
    """

    # La finestra ha de ser senar
    if mida_finestra%2==0:
        mida_finestra=mida_finestra+1

    # Evitem que sigui mes gran que la imatge
    mida_max=min(
        canal.shape
    )

    if mida_max%2==0:
        mida_max=mida_max-1

    mida_finestra=min(
        mida_finestra,
        mida_max
    )


    # --------------------------------------------------------
    # SENSE MASCARA DE RETINA
    # --------------------------------------------------------

    if mascara_retina is None:

        mitjana=uniform_filter(
            canal,
            size=mida_finestra
        )

        mitjana_quadrat=uniform_filter(
            canal**2,
            size=mida_finestra
        )


    # --------------------------------------------------------
    # AMB MASCARA DE RETINA
    # --------------------------------------------------------

    else:

        if canal.shape!=mascara_retina.shape:
            raise ValueError(
                "El canal i la mascara de retina no tenen la mateixa mida"
            )

        mascara_float=mascara_retina.astype(
            np.float64
        )


        # Numero de pixels valids de la retina
        pes=uniform_filter(
            mascara_float,
            size=mida_finestra
        )


        # Mitjana local nomes dins de la retina
        suma=uniform_filter(
            canal*mascara_float,
            size=mida_finestra
        )

        mitjana=suma/(
            pes+1e-8
        )


        # Mitjana local dels quadrats
        suma_quadrat=uniform_filter(
            canal**2*mascara_float,
            size=mida_finestra
        )

        mitjana_quadrat=suma_quadrat/(
            pes+1e-8
        )


    # Variancia local:
    # E[I^2]-E[I]^2

    variancia=(
        mitjana_quadrat
        -mitjana**2
    )


    # Poden sortir petits valors negatius
    # per errors numerics

    variancia=np.maximum(
        variancia,
        0
    )


    desviacio=np.sqrt(
        variancia
    )


    return mitjana,desviacio


def segmentar_nick(
    canal,
    mascara_retina=None,
    mida_finestra=95,
    k=-0.2
):
    """
    Fem una segmentacio amb NICK thresholding.
    
    NICK calcula un llindar diferent per cada  area de la imatge 
    utilitzant la mitjana i la variació local de pixels. Utilizem el 
    canal invertit en aquesta funció i definim la mida de la finestra que
    és la zona local que mirem per calcular el llindar i  k que és el parametre 
    del  metode NICK(-0,2 al paper)
    """

    canal=np.asarray(
        canal,
        dtype=np.float64
    )


    # Calculem la mitjana i desviacio local
    mitjana,desviacio=calcula_estadistiques_locals(
        canal,
        mascara_retina,
        mida_finestra
    )


    # Llindar local de NICK
    llindar=(
        mitjana
        +k*desviacio
    )


    # Com que hem invertit la imatge, els exsudats son foscos
    # per tant busquem pixels per sota del llindar
    mascara=canal<llindar


    # Si tenim mascara de retina, eliminem el fons exterior
    if mascara_retina is not None:

        mascara=(
            mascara
            &mascara_retina
        )


    return mascara


def segmentar_inrg(
    canal,
    mascara_retina=None,
    mida_finestra=95,
    k=-0.2,
    iteracions=3,
    delta=0.1
):
    """
    Apliquem INRG.

    El metode comença amb una mascara NICK inicial.
    Despres modifica la intensitat de les regions detectades
    i torna a aplicar NICK.

    Les noves regions es van acumulant amb les regions
    detectades anteriorment.
    """

    canal=np.asarray(
        canal,
        dtype=np.float64
    )


    # Invertim el canal perquè els exsudats siguin foscos
    imatge=invertir_canal(
        canal
    )


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


        area_anterior=np.sum(
            mascara_anterior
        )


        # Si no hi ha cap regio, no podem continuar
        if area_anterior==0:
            break


        # Intensitat mitjana de la regio detectada
        valor_mitja=np.mean(
            imatge[mascara_anterior]
        )


        # Substituim la regio per la seva intensitat mitjana
        imatge[
            mascara_anterior
        ]=valor_mitja


        # Tornem a aplicar NICK
        mascara_nova=segmentar_nick(
            imatge,
            mascara_retina,
            mida_finestra,
            k
        )


        # INRG acumula la regio anterior i la nova
        mascara=(
            mascara_anterior
            |mascara_nova
        )


        area_actual=np.sum(
            mascara
        )


        creixement=(
            area_actual-area_anterior
        )/area_anterior


        # Si la regio creix menys que delta, parem
        if creixement<delta:
            break


    return mascara