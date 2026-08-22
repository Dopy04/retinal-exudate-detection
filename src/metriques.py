# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 2026

@author: gurpi

Fitxer de metriques per comparar la mascara que dona el nostre algoritme
amb el ground truth dels experts.

1=True  -> exsudat
0=False -> fons
"""

import numpy as np

eps=1e-8


def passar_a_gris(imatge):
    """
    Si la imatge ve en RGB, la passem a gris amb la formula de luma.
    Si ja ve en gris, la deixem igual.
    """

    if imatge.ndim==3:
        R=imatge[:,:,0]
        G=imatge[:,:,1]
        B=imatge[:,:,2]

        imatge_gris=0.299*R+0.587*G+0.114*B

    else:
        imatge_gris=imatge

    return imatge_gris


def binaritzar_mascara(mascara):
    """
    Passem una mascara a binaria.

    Els ground truth poden venir en RGB, gris, 0-255 o 0-1.
    Nosaltres volem una mascara booleana:
        True=exsudat
        False=fons
    """

    mascara=passar_a_gris(mascara)

    if np.max(mascara)>1:
        llindar=128
    else:
        llindar=0.5

    mascara_binaria=mascara>llindar

    return mascara_binaria


def calcular_confusio(mascara_predita,mascara_gt):
    """
    Calculem TP, FP, FN i TN comparant pixel a pixel.

    TP: prediccio=exsudat i ground truth=exsudat
    FP: prediccio=exsudat pero ground truth=fons
    FN: prediccio=fons pero ground truth=exsudat
    TN: prediccio=fons i ground truth=fons
    """

    pred=binaritzar_mascara(mascara_predita)
    gt=binaritzar_mascara(mascara_gt)

    if pred.shape!=gt.shape:
        raise ValueError("ERROR: la mascara predita i el ground truth no tenen la mateixa mida")

    TP=np.logical_and(pred==True,gt==True).sum()
    FP=np.logical_and(pred==True,gt==False).sum()
    FN=np.logical_and(pred==False,gt==True).sum()
    TN=np.logical_and(pred==False,gt==False).sum()

    return TP,FP,FN,TN


def calcular_metriques(mascara_predita,mascara_gt):
    """
    Calculem les metriques principals de segmentacio.
    Aquestes metriques comparen la nostra mascara amb el ground truth.
    """

    TP,FP,FN,TN=calcular_confusio(mascara_predita,mascara_gt)

    # Precisio:
    # de tot el que hem marcat com exsudat, quina part ho era realment
    precisio=TP/(TP+FP+eps)

    # Recall:
    # de tots els exsudats reals, quants hem trobat
    recall=TP/(TP+FN+eps)

    # F1:
    # barreja precisio i recall en un sol valor
    f1=2*precisio*recall/(precisio+recall+eps)

    # Dice:
    # molt usada en segmentacio medica
    dice=2*TP/(2*TP+FP+FN+eps)

    # IoU:
    # interseccio entre unio, mes estricta que Dice
    iou=TP/(TP+FP+FN+eps)

    # Accuracy:
    # percentatge total de pixels ben classificats
    # no sera la mes important perque hi ha molt fons
    accuracy=(TP+TN)/(TP+FP+FN+TN+eps)

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
        "accuracy":accuracy
    }

    return metriques


def comparar_metodes(diccionari_mascares,mascara_gt):
    """
    Compara diversos metodes de segmentacio amb el mateix ground truth.

    Exemple:
        {
            "Otsu":mascara_otsu,
            "Kmeans":mascara_kmeans,
            "Final":mascara_final
        }
    """

    resultats={}

    for nom_metode in diccionari_mascares:
        mascara_predita=diccionari_mascares[nom_metode]
        resultats[nom_metode]=calcular_metriques(mascara_predita,mascara_gt)

    return resultats


def imprimir_metriques(nom_metode,metriques):
    """
    Imprimeix les metriques d'un metode de forma clara.
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


if __name__=="__main__":

    # Prova petita per veure si les metriques funcionen.
    # 1=exsudat, 0=fons

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

    metriques=calcular_metriques(mascara_predita,mascara_gt)

    imprimir_metriques("prova",metriques)