# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:02:02 2026

@author: gurpi
"""

"""Bloc 4: segmentacio basica.

Aqui fem els primers metodes simples per detectar possibles exsudats.
Aquests metodes ens serviran com a baseline, o sigui, com a comparacio
amb el metode final que farem despres.

Entrada:
    un canal de la imatge, per exemple G, B, gris o HSV_V

Sortida:
    una mascara binaria
    True  = possible exsudat
    False = fons
"""

import numpy as np
from skimage import filters
from sklearn.cluster import KMeans


def aplicar_mascara_retina(mascara,mascara_retina):
    """
    Aquesta funcio serveix per quedar-nos nomes amb la zona de la retina.

    La part negra de fora de la retina no ens interessa.
    Si no posem aquesta mascara, alguns metodes poden detectar coses al fons.
    """

    if mascara_retina is None:
        return mascara

    mascara=mascara & mascara_retina

    return mascara


def segmentar_otsu(canal,mascara_retina=None):
    """
    Segmentacio amb Otsu.

    Otsu calcula automaticament un llindar global.
    Com que els exsudats son zones brillants, ens quedem amb els pixels
    que estan per sobre del llindar.

    Problema:
        Si la imatge te il.luminacio desigual, Otsu pot marcar massa zones
        o pot perdre exsudats petits.
    """

    # Si tenim mascara de retina, calculem el llindar nomes dins la retina
    if mascara_retina is not None:
        valors=canal[mascara_retina]
        llindar=filters.threshold_otsu(valors)
    else:
        llindar=filters.threshold_otsu(canal)

    # Els exsudats son brillants, per aixo fem canal>llindar
    mascara=canal>llindar

    # Eliminem el fons exterior si tenim mascara de retina
    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_llindar_fix(canal,llindar=0.7,mascara_retina=None):
    """
    Segmentacio amb un llindar fix.

    Aquest metode es molt simple:
        si el pixel es mes gran que el llindar -> possible exsudat

    No esperem que sigui el millor metode, pero ens serveix com a comparacio.
    """

    mascara=canal>llindar

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_adaptatiu(canal,mascara_retina=None,mida_bloc=51,offset=0):
    """
    Segmentacio amb threshold adaptatiu/local.

    En comptes de buscar un unic llindar per tota la imatge,
    calcula un llindar diferent segons la zona local.

    Aixo te sentit en imatges de retina perque la il.luminacio no es uniforme.
    """

    # El block size ha de ser senar. Si ens passen un valor parell, el corregim.
    if mida_bloc%2==0:
        mida_bloc=mida_bloc+1

    llindar_local=filters.threshold_local(canal,block_size=mida_bloc,offset=offset)

    # Exsudats brillants -> pixels per sobre del llindar local
    mascara=canal>llindar_local

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_sauvola(canal,mascara_retina=None,mida_finestra=51):
    """
    Segmentacio amb Sauvola.

    Sauvola tambe es un threshold local.
    Es una opcio semblant a Niblack/NICK, per tant es coherent amb el paper.

    La idea es provar si aquest metode local funciona millor que Otsu.
    """

    if mida_finestra%2==0:
        mida_finestra=mida_finestra+1

    llindar=filters.threshold_sauvola(canal,window_size=mida_finestra)

    mascara=canal>llindar

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_kmeans_canal(canal,mascara_retina=None,n_clusters=3):
    """
    Segmentacio amb K-means utilitzant un sol canal.

    K-means separa els pixels en grups segons la intensitat.
    Com que els exsudats son brillants, ens quedem amb el cluster
    que tingui el centre mes alt.

    Exemple:
        cluster fosc     -> fons / vasos
        cluster mig      -> retina normal
        cluster brillant -> possibles exsudats
    """

    alcada,amplada=canal.shape

    # Preparem les dades per K-means.
    # K-means necessita una matriu de forma:
    # files = pixels
    # columnes = caracteristiques
    if mascara_retina is not None:
        dades=canal[mascara_retina].reshape(-1,1)
    else:
        dades=canal.reshape(-1,1)

    model=KMeans(n_clusters=n_clusters,random_state=0,n_init=10)
    etiquetes=model.fit_predict(dades)

    # Centres dels clusters.
    # Com nomes tenim un canal, cada centre es una intensitat.
    centres=model.cluster_centers_.ravel()

    # Ens quedem amb el cluster mes brillant
    cluster_brilant=np.argmax(centres)

    # Reconstruim la mascara amb la mida original
    mascara=np.zeros((alcada,amplada),dtype=bool)

    if mascara_retina is not None:
        mascara[mascara_retina]=etiquetes==cluster_brilant
    else:
        mascara=(etiquetes==cluster_brilant).reshape(alcada,amplada)

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def obtenir_mascares_baseline(canal,mascara_retina=None):
    """
    Aquesta funcio executa tots els metodes basics de cop.

    Retorna un diccionari amb totes les mascares.
    Aixo despres ho podrem passar directament al bloc de metriques.
    """

    mascara_otsu=segmentar_otsu(canal,mascara_retina)
    mascara_fix=segmentar_llindar_fix(canal,llindar=0.7,mascara_retina=mascara_retina)
    mascara_adaptativa=segmentar_adaptatiu(canal,mascara_retina,mida_bloc=51,offset=0)
    mascara_sauvola=segmentar_sauvola(canal,mascara_retina,mida_finestra=51)
    mascara_kmeans=segmentar_kmeans_canal(canal,mascara_retina,n_clusters=3)

    mascares={
        "Otsu":mascara_otsu,
        "Llindar_fix":mascara_fix,
        "Adaptatiu":mascara_adaptativa,
        "Sauvola":mascara_sauvola,
        "Kmeans":mascara_kmeans
    }

    return mascares


if __name__=="__main__":

    # Prova petita artificial per veure si el fitxer funciona.
    # En el projecte real, el canal vindra de preprocessing.py.

    canal=np.array([
        [0.1,0.2,0.3,0.8,0.9],
        [0.1,0.2,0.4,0.7,0.8],
        [0.1,0.2,0.3,0.4,0.5],
        [0.2,0.3,0.8,0.9,0.4],
        [0.1,0.2,0.3,0.4,0.5]
    ])

    mascara_retina=np.array([
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]
    ],dtype=bool)

    mascares=obtenir_mascares_baseline(canal,mascara_retina)

    for nom in mascares:
        print("\nMetode:",nom)
        print(mascares[nom].astype(int))