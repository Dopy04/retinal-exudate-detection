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

    mascara=mascara.astype(bool)
    
    if mascara_retina is None:
        return mascara

    mascara_retina=mascara_retina.astype(bool)

    if mascara.shape!=mascara_retina.shape:
        raise ValueError(
            "La mascara i la mascara de retina no tenen la mateixa mida"
        )

    mascara=mascara & mascara_retina

    return mascara


def preparar_canal_local(canal,mascara_retina):
    """
    Preparem el canal abans d'utilitzar metodes de llindar local.

    El problema es que fora de la retina tenim pixels negres i aquests
    pixels podrien afectar el calcul dels llindars prop del contorn.

    Per evitar-ho substituim temporalment el fons exterior per la mediana
    dels pixels de la retina.
    """

    canal_preparat=np.copy(canal).astype(np.float64)

    if mascara_retina is None:
        return canal_preparat

    if canal.shape!=mascara_retina.shape:
        raise ValueError(
            "El canal i la mascara de retina no tenen la mateixa mida"
        )

    valors=canal[mascara_retina]

    if valors.size==0:
        return canal_preparat

    valor_fons=np.median(valors)

    canal_preparat[~mascara_retina]=valor_fons

    return canal_preparat


def segmentar_otsu(canal,mascara_retina=None):
    """
    Segmentacio amb Otsu.
    Aquesta funció ens calcula en base al algoritme OTSU, un algoritme de segmentació
    basada en un llindar global. En el nostre cas com els exsudats són zones brillants ens
    quedem amb els pixels per sobre del llindar.
    
    Aquest metode ens donara errors o problemes en cas de exsudats petits o zones 
    amb una iluminació no uniforme
    """

    canal=np.asarray(canal,dtype=np.float64)

    # Si tenim mascara de retina, calculem el llindar nomes dins la retina
    if mascara_retina is not None:

        if canal.shape!=mascara_retina.shape:
            raise ValueError(
                "El canal i la mascara de retina no tenen la mateixa mida"
            )

        valors=canal[mascara_retina]

    else:
        valors=canal.ravel()

    # Evitem errors si per algun problema no tenim pixels
    if valors.size==0:
        return np.zeros(canal.shape,dtype=bool)

    # Si tots els pixels tenen practicament el mateix valor
    # Otsu no pot separar correctament dues classes
    if np.max(valors)-np.min(valors)<1e-8:
        return np.zeros(canal.shape,dtype=bool)

    llindar=filters.threshold_otsu(valors)

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


def segmentar_adaptatiu(canal,mascara_retina=None,mida_bloc=101,offset=-0.15):
    """
    Segmentacio amb threshold adaptatiu/local.

    En aquest cas busquem calcular un llindar per cada zona per no fer una 
    generalització i així podem millorar la detecció de exsudats en zones
    amb iluminació no uniforme.
    """

    canal_preparat=preparar_canal_local(
        canal,
        mascara_retina
    )

    # El block size ha de ser senar. Si ens passen un valor parell, el corregim.
    if mida_bloc%2==0:
        mida_bloc=mida_bloc+1

    # Evitem que la finestra sigui mes gran que la imatge
    mida_max=min(canal.shape)

    if mida_max%2==0:
        mida_max=mida_max-1

    mida_bloc=min(
        mida_bloc,
        mida_max
    )

    if mida_bloc<3:
        mida_bloc=3

    llindar_local=filters.threshold_local(
        canal_preparat,
        block_size=mida_bloc,
        offset=offset
    )

    #Els pixels per sobre del llindar local seràn els exudats perquè són els que 
    #més brillen
    
    mascara=canal_preparat>llindar_local

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_sauvola(canal,mascara_retina=None,mida_finestra=51,k=0.2):
    """
    Segmentacio amb Sauvola.

    Sauvola és un tipus de algoritme de segmentació que funciona com un llindar local
    , és un metode com el OTSU per amb un llindar local, i l'utilitzem per veure si funciona
    millor que l'OTSU.
    """

    canal_preparat=preparar_canal_local(
        canal,
        mascara_retina
    )

    if mida_finestra%2==0:
        mida_finestra=mida_finestra+1

    # Evitem que la finestra sigui mes gran que la imatge
    mida_max=min(canal.shape)

    if mida_max%2==0:
        mida_max=mida_max-1

    mida_finestra=min(
        mida_finestra,
        mida_max
    )

    if mida_finestra<3:
        mida_finestra=3

    llindar=filters.threshold_sauvola(
        canal_preparat,
        window_size=mida_finestra,
        k=k
    )

    mascara=canal_preparat>llindar

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

        if canal.shape!=mascara_retina.shape:
            raise ValueError(
                "El canal i la mascara de retina no tenen la mateixa mida"
            )

        dades=canal[mascara_retina].reshape(-1,1)

    else:
        dades=canal.reshape(-1,1)

    # Evitem errors si no tenim pixels
    if dades.size==0:
        return np.zeros((alcada,amplada),dtype=bool)

    # Comprovem quants valors diferents tenim
    valors_unics=np.unique(dades)

    # Si tots els pixels tenen el mateix valor
    # no té sentit aplicar K-means
    if len(valors_unics)<2:
        return np.zeros((alcada,amplada),dtype=bool)

    # Si tenim menys intensitats diferents que clusters
    # reduim el nombre de clusters automaticament
    n_clusters_real=min(
        n_clusters,
        len(valors_unics)
    )

    model=KMeans(
        n_clusters=n_clusters_real,
        random_state=0,
        n_init=10
    )

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

    mascara_otsu=segmentar_otsu(
        canal,
        mascara_retina
    )

    mascara_fix=segmentar_llindar_fix(
        canal,
        llindar=0.7,
        mascara_retina=mascara_retina
    )

    mascara_adaptativa=segmentar_adaptatiu(
        canal,
        mascara_retina
    )

    mascara_sauvola=segmentar_sauvola(
        canal,
        mascara_retina
    )

    mascara_kmeans=segmentar_kmeans_canal(
        canal,
        mascara_retina,
        n_clusters=3
    )

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