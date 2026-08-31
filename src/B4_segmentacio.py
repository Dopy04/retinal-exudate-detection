# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:02:02 2026

@author: gurpi
"""

"""
Bloc 4:Segmentacio basica.

Aqui fem els primers metodes simples per detectar possibles exsudats,
aquests metodes ens serviran com a base, o sigui, com a comparació
amb el metode final que farem despres.

Rebem com a entrada un canal de la imatge (RGB,gris,HSV_V) i obtenim a la sortida
una mascara binaria on True és un possible exsudat i False és el fons.
"""

import numpy as np
from skimage import filters
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt

def aplicar_mascara_retina(mascara,mascara_retina):
    """
    Aquesta funció ens permet quedarnos només amb la zona de la retina,
    que és la part de la imatge que ens interessa analitzar.

    Si no apliquessim aquesta mascara podriem detectar pixels del fons
    com a possibles exsudats.
    """

    mascara=mascara.astype(bool)

    #Si no tenim mascara de retina retornem directament la mascara original
    if mascara_retina is None:
        return mascara

    mascara_retina=mascara_retina.astype(bool)

    #Comprovem que les dues mascares tinguin la mateixa mida
    if mascara.shape!=mascara_retina.shape:
        raise ValueError(
            "La mascara i la mascara de retina no tenen la mateixa mida")

    #Amb AND només ens quedem amb els pixels detectats dins de la retina
    mascara=mascara&mascara_retina

    return mascara


def preparar_canal_local(canal,mascara_retina):
    """
    Preparem el canal abans d'utilitzar metodes de llindar local.

    Fora de la retina tenim pixels negres i aquests podrien afectar
    el calcul del llindar en zones properes al contorn. Per evitar-ho
    substituim temporalment el fons exterior per la mitjana
    dels pixels que estan dins de la retina.
    
    """

    canal_preparat=np.copy(canal).astype(np.float64)

    if mascara_retina is None:
        return canal_preparat

    if canal.shape!=mascara_retina.shape:
        raise ValueError(
            "El canal i la mascara de retina no tenen la mateixa mida")

    #Agafem només els valors que estan dins de la retina
    valors=canal[mascara_retina]

    #Si per algun problema la retina esta buida retornem el canal sense modificar
    if valors.size==0:
        return canal_preparat

    #Utilitzem la mitjana perquè és menys sensible a zones molt brillants
    valor_fons=np.median(valors)

    canal_preparat[~mascara_retina]=valor_fons

    return canal_preparat


def segmentar_otsu(canal,mascara_retina=None):
    """
    Segmentacio amb Otsu.

    Otsu calcula automaticament un llindar global intentant separar els pixels
    en dues classes. Com els exsudats són zones brillants ens quedem amb els
    pixels que estan per sobre del llindar.

    Aquest metode pot tenir problemes amb exsudats petits o amb una
    iluminació no uniforme.
    """

    canal=np.asarray(canal,dtype=np.float64)

    #Si tenim mascara de retina calculem Otsu només amb els pixels de la retina
    if mascara_retina is not None:

        if canal.shape!=mascara_retina.shape:
            raise ValueError(
                "El canal i la mascara de retina no tenen la mateixa mida")

        valors=canal[mascara_retina]

    else:
        valors=canal.ravel()

    #Evitem errors si per algun problema no tenim pixels
    if valors.size==0:
        return np.zeros(canal.shape,dtype=bool)

    #Si tots els pixels tenen practicament el mateix valor
    #Otsu no pot separar correctament les dues classes
    if np.max(valors)-np.min(valors)<1e-8:
        return np.zeros(canal.shape,dtype=bool)

    llindar=filters.threshold_otsu(valors)

    #Els exsudats son brillants,per aixo seleccionem els pixels superiors al llindar
    mascara=canal>llindar

    #Eliminem qualsevol detecció que hagi quedat fora de la retina
    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_llindar_fix(canal,llindar=0.7,mascara_retina=None):
    """
    Segmentacio amb un llindar fix.

    El funcionament és molt simple, qualsevol pixel que superi el llindar
    es considera un possible exsudat. No obtindrem la mascara més acurada
    pero ens permet tenir un metode senzill de comparació.
    """

    #Classifiquem com a exsudat els pixels que superen el llindar
    mascara=canal>llindar

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_adaptatiu(canal,mascara_retina=None,mida_bloc=101,offset=-0.15):
    """
    Segmentacio amb threshold adaptatiu/local.

    En aquest cas calculem un llindar diferent per cada zona de la imatge.
    Això ens permet adaptar la segmentació a zones amb una iluminació
    diferent en comptes d'utilitzar un únic llindar global.
    """

    canal_preparat=preparar_canal_local(canal,mascara_retina)

    #La mida del bloc ha de ser senar
    if mida_bloc%2==0:
        mida_bloc=mida_bloc+1

    #Evitem que la finestra sigui més gran que la imatge
    mida_max=min(canal.shape)

    if mida_max%2==0:
        mida_max=mida_max-1

    mida_bloc=min(mida_bloc,mida_max)

    #La finestra ha de tenir com a minim una mida de 3
    if mida_bloc<3:
        mida_bloc=3

    #Calculem un llindar local per cada zona de la imatge
    llindar_local=filters.threshold_local(canal_preparat,
                                          block_size=mida_bloc,offset=offset)

    #Els pixels més brillants que el seu llindar local seran possibles exsudats
    mascara=canal_preparat>llindar_local

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_sauvola(canal,mascara_retina=None,mida_finestra=51,k=0.2):
    """
    Segmentacio amb Sauvola.

    Sauvola també és un metode de llindar local, per tant el llindar
    depèn de la zona de la imatge. L'utilitzem com una altra comparació
    per veure si funciona millor que Otsu.
    """

    canal_preparat=preparar_canal_local(canal,mascara_retina)
    
    #La finestra ha de ser senar
    if mida_finestra%2==0:
        mida_finestra=mida_finestra+1

    #Evitem que la finestra sigui més gran que la imatge
    mida_max=min(canal.shape)

    if mida_max%2==0:
        mida_max=mida_max-1

    mida_finestra=min(mida_finestra,mida_max)

    if mida_finestra<3:
        mida_finestra=3

    #Calculem el llindar de Sauvola per cada zona
    llindar=filters.threshold_sauvola(
        canal_preparat,window_size=mida_finestra,k=k)

    #Ens quedem amb els pixels que superen el seu llindar
    mascara=canal_preparat>llindar

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def segmentar_kmeans_canal(canal,mascara_retina=None,n_clusters=3):
    """
    Segmentacio amb K-means utilitzant un sol canal.

    K-means separa els pixels en grups segons la seva intensitat.
    En aquest cas utilitzem tres clusters per intentar separar fons,
    retina normal i zones brillants.

    Com els exsudats són brillants ens quedem amb el cluster que
    tingui el centre amb una intensitat més alta.
    """

    alcada,amplada=canal.shape

    # K-means necessita una matriu on cada fila sigui un pixel
    # i cada columna una caracteristica,en aquest cas només la intensitat
    if mascara_retina is not None:
        if canal.shape!=mascara_retina.shape:
            raise ValueError(
                "El canal i la mascara de retina no tenen la mateixa mida")

        dades=canal[mascara_retina].reshape(-1,1)

    else:
        dades=canal.reshape(-1,1)

    #Evitem errors si no tenim cap pixel
    if dades.size==0:
        return np.zeros((alcada,amplada),dtype=bool)

    #Comprovem quants valors d'intensitat diferents tenim
    valors_unics=np.unique(dades)

    #Si tots els pixels tenen el mateix valor no té sentit aplicar K-means
    if len(valors_unics)<2:
        return np.zeros((alcada,amplada),dtype=bool)

    #Si tenim menys valors diferents que clusters reduim el nombre de clusters
    n_clusters_real=min(n_clusters,len(valors_unics))

    #Apliquem K-means sobre les intensitats dels pixels
    model=KMeans(n_clusters=n_clusters_real,random_state=0,n_init=10)

    etiquetes=model.fit_predict(dades)

    # Cada centre representa la intensitat mitjana d'un cluster
    centres=model.cluster_centers_.ravel()

    # Seleccionem el cluster amb el centre més brillant
    cluster_brilant=np.argmax(centres)

    # Reconstruim la mascara amb la mida de la imatge original
    mascara=np.zeros((alcada,amplada),dtype=bool)

    if mascara_retina is not None:
        mascara[mascara_retina]=etiquetes==cluster_brilant
    else:
        mascara=(etiquetes==cluster_brilant).reshape(alcada,amplada)

    mascara=aplicar_mascara_retina(mascara,mascara_retina)

    return mascara


def obtenir_mascares_baseline(canal,mascara_retina=None):
    """
    Aquesta funció executa tots els metodes de segmentació anteriors
    sobre el mateix canal.

    Retornem un diccionari amb totes les mascares per poder passar-les
    directament al bloc de metriques i comparar els resultats.
    """

    # Apliquem tots els metodes sobre exactament la mateixa entrada
    mascara_otsu=segmentar_otsu(canal,mascara_retina)

    mascara_fix=segmentar_llindar_fix(canal,llindar=0.7,
        mascara_retina=mascara_retina)

    mascara_adaptativa=segmentar_adaptatiu(canal,mascara_retina)

    mascara_sauvola=segmentar_sauvola(canal,mascara_retina)

    mascara_kmeans=segmentar_kmeans_canal(canal,mascara_retina,n_clusters=3)

    # Guardem cada mascara amb el nom del metode que l'ha generat
    mascares={
        "Otsu":mascara_otsu,
        "Llindar_fix":mascara_fix,
        "Adaptatiu":mascara_adaptativa,
        "Sauvola":mascara_sauvola,
        "Kmeans":mascara_kmeans}

    return mascares

if __name__=="__main__":

    #Prova Bloc 4: utilitzem una imatge del dataset per comprovar
    #els diferents metodes de segmentacio i comparar les seves metriques

    from pathlib import Path
    import matplotlib.pyplot as plt
    import B2_preprocessament as prep
    from B1_metriques import comparar_metodes,imprimir_metriques

    #Imatge que utilitzem per la prova
    #Per provar una altra només cal canviar aquests dos noms
    nom_imatge="Exudate_DR2019_003.jpg"
    nom_gt="Exudate_DR2019_003_Groundtruth.jpg"

    ruta_projecte=Path(__file__).resolve().parent.parent
    ruta_dades=ruta_projecte/"data_samples"

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    #Fem el preprocessament per obtenir la imatge, el Ground Truth,
    #la mascara de retina i els diferents canals
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=prep.preprocessa_imatge(
        ruta_imatge,ruta_gt)

    #Utilitzem el canal G que és el que ens ha donat millors
    #resultats en la segmentacio basica
    canal_g=canals[1]

    #Apliquem tots els metodes del bloc sobre la mateixa imatge
    mascares=obtenir_mascares_baseline(canal_g,mascara_retina)

    #Calculem les metriques de cada metode respecte el Ground Truth
    resultats=comparar_metodes(mascares,mascara_gt)

    print("\n======================================")
    print("PROVA BLOC 4")
    print("Imatge:",nom_imatge)
    print("======================================")

    for nom_metode in resultats:
        imprimir_metriques(nom_metode,resultats[nom_metode])

    #Mostrem la imatge original, el Ground Truth i les segmentacions
    #per poder comparar visualment els diferents metodes
    plt.figure(figsize=(16,8))
    plt.suptitle("PROVA BLOC 4 - "+nom_imatge,fontsize=16)

    plt.subplot(2,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,4,2)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2,4,3)
    plt.imshow(mascares["Otsu"],cmap="gray")
    plt.title("Otsu")
    plt.axis("off")

    plt.subplot(2,4,4)
    plt.imshow(mascares["Llindar_fix"],cmap="gray")
    plt.title("Llindar fix")
    plt.axis("off")

    plt.subplot(2,4,5)
    plt.imshow(mascares["Adaptatiu"],cmap="gray")
    plt.title("Adaptatiu")
    plt.axis("off")

    plt.subplot(2,4,6)
    plt.imshow(mascares["Sauvola"],cmap="gray")
    plt.title("Sauvola")
    plt.axis("off")

    plt.subplot(2,4,7)
    plt.imshow(mascares["Kmeans"],cmap="gray")
    plt.title("K-means")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.94])
    plt.show()