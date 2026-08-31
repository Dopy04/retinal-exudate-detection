# -*- coding: utf-8 -*-
"""
Bloc 1:Metriques per avaluar la segmentacio d'exsudats.

En aquest bloc comparem la mascara que obtenim amb els nostres algoritmes
amb el Ground Truth, que és la mascara correcte feta pels experts.

Considerarem True un esxsaudat i fals els fons.

Les metriques principals que utilitzarem seran Precision, Recall, Dice i IoU.
També calculem F1 i Accuracy com informació complementaria.
"""

import numpy as np


def passar_a_gris(imatge):
    """
    Si la imatge té tres canals RGB la passem a escala de grisos
    utilitzant la formula de luminancia, en el cas que sigui 2D no fem
    cap conversió.
    """
    imatge=np.asarray(imatge)

    if imatge.ndim==3:
        r=imatge[:,:,0]
        g=imatge[:,:,1]
        b=imatge[:,:,2]

        imatge_gris=0.299*r+0.587*g+0.114*b

    else:
        imatge_gris=imatge

    return imatge_gris


def binaritzar_mascara(mascara):
    """
    Convertim qualsevol mascara a tipus boolea.
    Les mascares poden venir en RGB, en escala de grisos,
    amb valors entre 0-1 o amb valors entre 0-255.
    
    Aquesta funció ens permet classficar com True els exsudats i
    com False el fons
    """

    mascara=passar_a_gris(mascara)

    # Si la mascara ja és booleana no necessitem aplicar cap llindar
    if mascara.dtype==bool:
        return mascara

    mascara=mascara.astype(np.float64)

    # Si tenim valors de 0 a 255 utilitzem 128 com a llindar
    # i si estan normalitzats entre 0 i 1 utilitzem 0.5
    
    if np.max(mascara)>1:
        llindar=128
    else:
        llindar=0.5

    mascara_binaria=mascara>llindar

    return mascara_binaria


def divisio_segura(numerador,denominador):
    """
    Aquesta funció ens permet evitar divisions entre zero
    quan calculem les metriques. En el cas que el denominador
    sigui zero retornem zero directament.
    """

    if denominador==0:
        return 0.0

    return numerador/denominador


def calcular_confusio(mascara_predita,mascara_gt,mascara_avaluacio=None):
    """
    Calculem TP,FP,FN i TN comparant pixel a pixel la nostra
    predicció amb el Ground Truth.
    
    TP: detectem exsudat i realment és exsudat
    FP: detectem exsudat pero realment és fons
    FN: detectem fons pero realment és exsudat
    TN: detectem fons i realment és fons

    La funció mascara avaluació ens permet aplicar les metriques només 
    dins d'una zona es una funció auxiliar per dir-ho d'alguna forma.
    """

    pred=binaritzar_mascara(mascara_predita)
    gt=binaritzar_mascara(mascara_gt)

    #Això ens permet tenir una seguretat que siguien de la mateixa mida
    if pred.shape!=gt.shape: raise ValueError(
    "ERROR: la mascara predita i el Ground Truth no tenen la mateixa mida")

    # Si tenim una mascara d'avaluació ens quedem només amb aquesta zona
    if mascara_avaluacio is not None:
        zona=binaritzar_mascara(mascara_avaluacio)

        if zona.shape!=gt.shape:
            raise ValueError(
                "ERROR: la mascara d'avaluacio no te la mateixa mida")
        pred=pred[zona]
        gt=gt[zona]

    # Calculem els quatre casos possibles comparant les dues mascares
    
    TP=np.logical_and(pred,gt).sum()
    FP=np.logical_and(pred,~gt).sum()
    FN=np.logical_and(~pred,gt).sum()
    TN=np.logical_and(~pred,~gt).sum()

    return int(TP),int(FP),int(FN),int(TN)


def calcular_metriques(mascara_predita,mascara_gt,mascara_avaluacio=None):
    """
    Calculem les principals metriques de segmentacio comparant
    la mascara que hem obtingut amb el Ground Truth.
    """

    TP,FP,FN,TN=calcular_confusio(mascara_predita,mascara_gt,
                                  mascara_avaluacio)

    #Precision:
    #De tot el que hem detectat com exsudat, quina part realment ho era
    precisio=divisio_segura(TP,TP+FP)

    # Recall:
    #De tots els exsudats reals, quants hem aconseguit detectar
    recall=divisio_segura(TP,TP+FN)

    #F1:
    #Combina Precision i Recall en una sola metrica
    f1=divisio_segura(2*precisio*recall,precisio+recall)

    #Dice:
    #Mesura el solapament entre la nostra mascara i el Ground Truth
    dice=divisio_segura(2*TP,2*TP+FP+FN)

    #IoU:
    #Calcula la interseccio entre les dues mascares respecte la seva unio
    iou=divisio_segura(TP,TP+FP+FN)

    #Accuracy:
    #Percentatge total de pixels classificats correctament
    #no serà la principal perquè tenim molts més pixels de fons
    accuracy=divisio_segura(TP+TN,TP+FP+FN+TN)

    metriques={
        "TP":TP,
        "FP":FP,
        "FN":FN,
        "TN":TN,
        "precisio":precisio,
        "recall":recall,
        "f1":f1,
        "dice":dice,
        "iou":iou,
        "accuracy":accuracy}

    return metriques


def comparar_metodes(diccionari_mascares,mascara_gt,mascara_avaluacio=None):
    """
    Aquesta funció ens permet comparar diferents metodes de segmentació
    utilitzant exactament el mateix Ground Truth.
    """

    resultats={}

    # Recorrem totes les mascares que tenim guardades al diccionari.
    # En cada iteracio obtenim el nom del metode i la seva mascara predita.
    for nom_metode,mascara_predita in diccionari_mascares.items():

    # Calculem les metriques d'aquest metode comparant la seva mascara
    # amb el mateix Ground Truth i guardem el resultat amb el nom del metode.
        resultats[nom_metode]=calcular_metriques(
        mascara_predita,
        mascara_gt,
        mascara_avaluacio
        )

    # Retornem un diccionari amb les metriques de tots els metodes
    # per poder comparar-los posteriorment.
    return resultats


def imprimir_metriques(nom_metode,metriques):
    """
    Mostrem per pantalla totes les metriques que hem calculat
    per un metode concret.
    """

    print("\nMetode:",nom_metode)
    print("TP:",metriques["TP"])
    print("FP:",metriques["FP"])
    print("FN:",metriques["FN"])
    print("TN:",metriques["TN"])
    print("Precisio:",round(metriques["precisio"],4))
    print("Recall:",round(metriques["recall"],4))
    print("F1:",round(metriques["f1"],4))
    print("Dice:",round(metriques["dice"],4))
    print("IoU:",round(metriques["iou"],4))
    print("Accuracy:",round(metriques["accuracy"],4))


def resum_dataset(llista_metriques):
    """
    Quan tinguem totes les imatges del dataset calculem la mitjana
    i la desviacio estandard de les metriques.
    Això ens permetrà obtenir un resultat general del metode en comptes
    de mirar només el resultat d'una imatge concreta.
    """

    noms=[
        "precisio",
        "recall",
        "f1",
        "dice",
        "iou",
        "accuracy"]

    resum={}

    for nom in noms:

    # Agafem els valors d'aquesta metrica de totes les imatges
    # i els guardem en un array per poder calcular estadistiques.
        valors=np.array(
            [m[nom] for m in llista_metriques],
            dtype=float)
        
        # Guardem la mitjana i la desviacio estandard
        # per veure el resultat general del metode.
        resum[nom]={
            "mitjana":np.mean(valors),
            "desviacio":np.std(valors)}

    # Retornem el resum de totes les metriques
    return resum


def imprimir_resum_dataset(nom_metode,resum,nombre_imatges):
    """
    Mostrem la mitjana i la desviacio estandard de les metriques
    de totes les imatges del dataset.
    """

    print("\n======================================")
    print("RESULTATS DATASET:",nom_metode)
    print("Nombre d'imatges:",nombre_imatges)
    print("======================================")

    noms=["precisio","recall","dice","iou","accuracy"]

    # Recorrem les metriques que volem mostrar en el resum final
    for nom in noms:

        # Recuperem la mitjana i la desviacio estandard
        # que hem calculat anteriorment per cada metrica
        mitjana=resum[nom]["mitjana"]
        desviacio=resum[nom]["desviacio"]

        # Mostrem els resultats arrodonits a 4 decimals
        print(nom,":",round(mitjana,4),"+/-",round(desviacio,4))


if __name__=="__main__":

    # Prova Bloc 1: creem dues mascares petites artificials per comprovar
    # que el calcul de les metriques funciona correctament

    mascara_gt=np.array([
        [0,0,1,1],
        [0,0,1,1],
        [0,0,0,0],
        [1,1,0,0]
    ])

    mascara_predita=np.array([
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [1,0,0,0]
    ])

    metriques=calcular_metriques(
        mascara_predita,
        mascara_gt
    )

    imprimir_metriques(
        "Prova Bloc 1",
        metriques
    )