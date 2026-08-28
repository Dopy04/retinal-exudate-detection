# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:02:02 2026

@author: gurpi
"""

"""Bloc 4: segmentacio basica.

Aqui fem els primers metodes simples per detectar possibles exsudats.
Aquests metodes ens serviran com a base, o sigui, com a comparació
amb el metode final que farem despres.

Rebem com a entrada un canal de la imatge (RGB,gris,HSV_V) i obtenim a la sortida
una mascara binaria on el True és un possible exsudat i el False es el fons

"""

import numpy as np
from skimage import filters
from sklearn.cluster import KMeans


def aplicar_mascara_retina(mascara,mascara_retina):
    """
    Aquesta funció fa una mica de preprocessament ja que ens permet només quedarnos
    amb la zona de la retina que és la que ens interresa. Si per un cas no usem
    aquesta mascara es detectarien coses en el fons i no tindriem una segmentació correcte
    """
    
    if mascara_retina is None:
        return mascara

    mascara=mascara & mascara_retina

    return mascara


def segmentar_otsu(canal,mascara_retina=None):
    """
    Segmentacio amb Otsu.
    Aquesta funció ens calcula en base al algoritme OTSU, un algoritme de segmentació
    basada en un llindar global. En el nostre cas com els exsudats són zones brillants ens
    quedem amb els pixels per sobre del llindar.
    
    Aquest metode ens donara errors o problemes en cas de exsudats petits o zones 
    amb una iluminació no uniforme
    """

    # Si tenim mascara de retina, calculem el llindar nomes dins la retina
    if mascara_retina is not None:
        valors=canal[mascara_retina]
        llindar=filters.threshold_otsu(valors)
    else:
        llindar=filters.threshold_otsu(canal)

    # Els exsudats son brillants, per aixo fem canal>llindar
    mascara=canal>llindar

    # Eliminem el fons exterior si tenim mascara de retina per quedarnos només
    #amb les zona de la retina
    
    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_llindar_fix(canal,llindar=0.7,mascara_retina=None):
    """
    Segmentacio amb un llindar fix.
    
    Aquest metode té un funcionament molt simple, sempre que algun pixel superi
    el llindar el classfiquem com un possible exsudat. No obtindrem la mascara 
    més acurada pero ens permet fer una bona comparació.
    """
    
    mascara=canal>llindar

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_adaptatiu(canal,mascara_retina=None,mida_bloc=51,offset=0):
    """
    Segmentacio amb threshold adaptatiu/local.

    En aquest cas busquem calcular un llindar per cada zona per no fer una 
    generalització i així podem millorar la detecció de exsudats en zones
    amb iluminació no uniforme.
    """

    # El block size ha de ser senar. Si ens passen un valor parell, el corregim.
    if mida_bloc%2==0:
        mida_bloc=mida_bloc+1

    llindar_local=filters.threshold_local(canal,block_size=mida_bloc,offset=offset)

    #Els pixels per sobre del llindar local seràn els exudats perquè són els que 
    #més brillen
    
    mascara=canal>llindar_local

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_sauvola(canal,mascara_retina=None,mida_finestra=51):
    """
    Segmentacio amb Sauvola.

    Sauvola és un tipus de algoritme de segmentació que funciona com un llindar local
    , és un metode com el OTSU per amb un llindar local, i l'utilitzem per veure si funciona
    millor que l'OTSU.
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
    
    Els K-means ens permetran separar els clusters fosc en fons, el cluster mig
    en retina normal i el cluster brillant en un possible exsudat.
    
    """

    alcada,amplada=canal.shape

    # Preparem les dades per K-means.
    #Els K-means necessiten una matriu on les files siguin els pixels i
    #les columnes les caracteristiques.


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
    Aquesta funcio executa totes les funcions anterios.

    Retorna un diccionari amb totes les mascares, que ens servira per 
    passar-ho directament a les metriques.
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


"""if __name__=="__main__":     ESTO NO SIRVE ERA PA PROBAR ALGO

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
"""