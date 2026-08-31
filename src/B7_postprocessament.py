# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:08:50 2026

@author: gurpi
"""

"""
Bloc 7:Morfologia matematica.

En aquest bloc intentem netejar les mascares que obtenim després
d'aplicar els diferents metodes de segmentacio.

Utilitzem operacions morfologiques per eliminar petits objectes,
unir zones properes i omplir petits forats que poden haver quedat
dins de les regions detectades.
"""

from skimage import morphology
from pathlib import Path
import matplotlib.pyplot as plt

def eliminar_objectes_petits(mascara,mida_minima=50):
    """
    Eliminem els blobs molt petits que apareixen a la mascara.

    Aquests blobs poden ser soroll o falsos positius generats durant
    la segmentacio.La mida_minima indica el nombre minim de pixels que
    ha de tenir un objecte per mantenir-lo.
    """

    mascara=mascara.astype(bool)

    #Eliminem els objectes que tinguin menys pixels que la mida minima
    mascara_neta=morphology.remove_small_objects(mascara,
                                                 min_size=mida_minima)
    return mascara_neta


def tancar_mascara(mascara,radi=2):
    """
    Apliquem una operacio de closing.

    El closing consisteix en una dilatacio seguida d'una erosio
    i ens permet tancar petits espais i unir zones molt properes
    dins d'un mateix blob.
    """

    mascara=mascara.astype(bool)

    # Creem un element circular que utilitzarem per aplicar el closing
    element=morphology.disk(radi)

    mascara_tancada=morphology.binary_closing(mascara,element)

    return mascara_tancada


def obrir_mascara(mascara,radi=1):
    """
    Apliquem una operacio d'opening.

    L'opening consisteix en una erosio seguida d'una dilatacio
    i serveix per eliminar petits punts o irregularitats.

    Aquesta operacio s'ha d'utilitzar amb compte perquè també
    podria eliminar exsudats petits.
    """
    mascara=mascara.astype(bool)

    #Creem un element circular amb el radi indicat
    #que utilitzarem per aplicar l'opening
    element=morphology.disk(radi)

    #Apliquem opening per eliminar petits punts o irregularitats
    #mantenint la forma general dels blobs més grans
    mascara_oberta=morphology.binary_opening(mascara,element)

    return mascara_oberta

def eliminar_forats_petits(mascara,mida_forat=50):
    """
    Omplim petits forats que poden quedar dins dels blobs.

    Per exemple, si un exsudat queda detectat com una regio blanca
    amb un petit forat negre al mig, aquesta funcio ens permet omplir-lo.
    """

    mascara=mascara.astype(bool)

    # Omplim només els forats que siguin més petits que mida_forat
    mascara_neta=morphology.remove_small_holes(mascara,
                                               area_threshold=mida_forat)

    return mascara_neta


def netejar_mascara(mascara,mida_minima=50,mida_forat=50):
    """
    Aquesta funcio aplica tot el proces de morfologia sobre una mascara.

    Primer eliminem els objectes petits, després apliquem closing
    per unir zones properes i finalment omplim els petits forats.

    No apliquem opening automaticament perquè podria eliminar
    alguns exsudats petits.
    """

    mascara=mascara.astype(bool)

    # Eliminem petits blobs que poden ser soroll
    mascara=eliminar_objectes_petits(mascara,mida_minima)

    # Tanquem petites separacions que hi hagi dins dels blobs
    mascara=tancar_mascara(mascara,radi=2)

    # Omplim els petits forats que hagin quedat dins de les regions
    mascara=eliminar_forats_petits(mascara,mida_forat)

    return mascara

if __name__=="__main__":

    #Prova Bloc 7:Apliquem la morfologia sobre una mascara obtinguda
    #amb Otsu per veure com canvia abans i després de la neteja

    import B2_preprocessament as prep
    from B3_segmentacio import segmentar_otsu
    from B1_metriques import calcular_metriques,imprimir_metriques

    #Imatge que utilitzem per la prova
    #Per provar una altra només cal canviar aquests dos noms
    nom_imatge="Exudate_DR2019_003.jpg"
    nom_gt="Exudate_DR2019_003_Groundtruth.jpg"

    ruta_projecte=Path(__file__).resolve().parent.parent
    ruta_dades=ruta_projecte/"data_samples"

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    #Fem el preprocessament i agafem el canal G
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=prep.preprocessa_imatge(
        ruta_imatge,ruta_gt)

    canal_g=canals[1]

    #Obtenim una mascara inicial amb Otsu
    mascara_otsu=segmentar_otsu(canal_g,mascara_retina)

    #Apliquem el nostre proces de morfologia
    mascara_morfologia=netejar_mascara(mascara_otsu)

    #Calculem les metriques abans i després de la morfologia
    metriques_abans=calcular_metriques(mascara_otsu,mascara_gt)
    metriques_despres=calcular_metriques(mascara_morfologia,mascara_gt)

    print("\n======================================")
    print("PROVA BLOC 7")
    print("Imatge:",nom_imatge)
    print("======================================")

    imprimir_metriques("Otsu abans morfologia",metriques_abans)
    imprimir_metriques("Otsu despres morfologia",metriques_despres)

    #Mostrem visualment l'efecte de la morfologia
    plt.figure(figsize=(14,6))
    plt.suptitle("PROVA BLOC 7 - "+nom_imatge,fontsize=16)

    plt.subplot(1,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(mascara_otsu,cmap="gray")
    plt.title("Abans morfologia")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(mascara_morfologia,cmap="gray")
    plt.title("Despres morfologia")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()