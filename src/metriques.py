# -*- coding: utf-8 -*-
"""
Metriques per avaluar la segmentacio d'exsudats.

Comparem la mascara predita amb el Ground Truth:

    True  = exsudat
    False = fons

Les metriques principals seran:
    - Precision
    - Recall
    - Dice
    - IoU

També calculem F1 i Accuracy com a informacio complementaria.
"""

import numpy as np


# -------------------------------------------------------------------------
# 1. CONVERTIR IMATGES A MASCARES BINARIES
# -------------------------------------------------------------------------

def passar_a_gris(imatge):
    """
    Si la imatge te tres canals, la passem a escala de grisos
    utilitzant la formula de luminancia.

    Si ja es una imatge 2D, la deixem igual.
    """

    imatge = np.asarray(imatge)

    if imatge.ndim == 3:

        r = imatge[:, :, 0]
        g = imatge[:, :, 1]
        b = imatge[:, :, 2]

        imatge_gris = (
            0.299*r
            + 0.587*g
            + 0.114*b
        )

    else:

        imatge_gris = imatge

    return imatge_gris


def binaritzar_mascara(mascara):
    """
    Converteix una mascara a tipus boolea.

    Admet:
        - mascares booleanes
        - imatges entre 0 i 1
        - imatges entre 0 i 255
        - mascares RGB

    Retorna:
        True  = exsudat
        False = fons
    """

    mascara = passar_a_gris(
        mascara
    )


    # Si ja es booleana no cal aplicar cap llindar.
    if mascara.dtype == bool:

        return mascara


    mascara = mascara.astype(
        np.float64
    )


    # Detectem automaticament l'escala.
    if np.max(mascara) > 1:

        llindar = 128

    else:

        llindar = 0.5


    mascara_binaria = (
        mascara > llindar
    )

    return mascara_binaria


# -------------------------------------------------------------------------
# 2. DIVISIO SEGURA
# -------------------------------------------------------------------------

def divisio_segura(
    numerador,
    denominador
):
    """
    Evita divisions entre zero.

    Si el denominador es zero,
    retornem 0.
    """

    if denominador == 0:

        return 0.0

    return numerador / denominador


# -------------------------------------------------------------------------
# 3. MATRIU DE CONFUSIO
# -------------------------------------------------------------------------

def calcular_confusio(
    mascara_predita,
    mascara_gt,
    mascara_avaluacio=None
):
    """
    Calcula TP, FP, FN i TN pixel a pixel.

    TP:
        prediem exsudat i realment es exsudat

    FP:
        prediem exsudat pero es fons

    FN:
        prediem fons pero realment es exsudat

    TN:
        prediem fons i realment es fons


    mascara_avaluacio:
        opcionalment permet calcular les metriques
        nomes dins d'una determinada regio,
        per exemple dins de la retina.
    """

    pred = binaritzar_mascara(
        mascara_predita
    )

    gt = binaritzar_mascara(
        mascara_gt
    )


    if pred.shape != gt.shape:

        raise ValueError(
            "ERROR: la mascara predita i el Ground Truth "
            "no tenen la mateixa mida"
        )


    # --------------------------------------------------------
    # OPCIONAL: AVALUAR NOMES UNA REGIO
    # --------------------------------------------------------

    if mascara_avaluacio is not None:

        zona = binaritzar_mascara(
            mascara_avaluacio
        )

        if zona.shape != gt.shape:

            raise ValueError(
                "ERROR: la mascara d'avaluacio "
                "no te la mateixa mida"
            )


        pred = pred[zona]
        gt = gt[zona]


    # --------------------------------------------------------
    # CONFUSIO
    # --------------------------------------------------------

    TP = np.logical_and(
        pred,
        gt
    ).sum()


    FP = np.logical_and(
        pred,
        ~gt
    ).sum()


    FN = np.logical_and(
        ~pred,
        gt
    ).sum()


    TN = np.logical_and(
        ~pred,
        ~gt
    ).sum()


    return (
        int(TP),
        int(FP),
        int(FN),
        int(TN)
    )


# -------------------------------------------------------------------------
# 4. METRIQUES
# -------------------------------------------------------------------------

def calcular_metriques(
    mascara_predita,
    mascara_gt,
    mascara_avaluacio=None
):
    """
    Calcula les principals metriques
    de segmentacio.
    """

    TP,FP,FN,TN = calcular_confusio(
        mascara_predita,
        mascara_gt,
        mascara_avaluacio
    )


    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------
    #
    # De tot el que hem detectat com exsudat,
    # quina part realment era exsudat?

    precisio = divisio_segura(
        TP,
        TP + FP
    )


    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------
    #
    # De tots els exsudats reals,
    # quants hem aconseguit detectar?

    recall = divisio_segura(
        TP,
        TP + FN
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = divisio_segura(
        2*precisio*recall,
        precisio + recall
    )


    # --------------------------------------------------------
    # DICE
    # --------------------------------------------------------
    #
    # Mesura el solapament entre la prediccio
    # i el Ground Truth.

    dice = divisio_segura(
        2*TP,
        2*TP + FP + FN
    )


    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------
    #
    # Intersection over Union.

    iou = divisio_segura(
        TP,
        TP + FP + FN
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------
    #
    # No sera la nostra metrica principal,
    # ja que tenim molt mes fons que exsudat.

    accuracy = divisio_segura(
        TP + TN,
        TP + FP + FN + TN
    )


    metriques = {

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


# -------------------------------------------------------------------------
# 5. COMPARAR DIFERENTS METODES
# -------------------------------------------------------------------------

def comparar_metodes(
    diccionari_mascares,
    mascara_gt,
    mascara_avaluacio=None
):
    """
    Compara diferents metodes utilitzant
    exactament el mateix Ground Truth.

    Exemple:

        {
            "Otsu":mascara_otsu,
            "INRG":mascara_inrg,
            "Final":mascara_final
        }
    """

    resultats = {}


    for nom_metode,mascara_predita in diccionari_mascares.items():

        resultats[nom_metode] = calcular_metriques(
            mascara_predita,
            mascara_gt,
            mascara_avaluacio
        )


    return resultats


# -------------------------------------------------------------------------
# 6. MOSTRAR METRIQUES
# -------------------------------------------------------------------------

def imprimir_metriques(
    nom_metode,
    metriques
):
    """
    Mostra les metriques d'un metode.
    """

    print(
        "\nMetode:",
        nom_metode
    )

    print(
        "TP:",
        metriques["TP"]
    )

    print(
        "FP:",
        metriques["FP"]
    )

    print(
        "FN:",
        metriques["FN"]
    )

    print(
        "TN:",
        metriques["TN"]
    )

    print(
        "Precisio:",
        round(
            metriques["precisio"],
            4
        )
    )

    print(
        "Recall:",
        round(
            metriques["recall"],
            4
        )
    )

    print(
        "F1:",
        round(
            metriques["f1"],
            4
        )
    )

    print(
        "Dice:",
        round(
            metriques["dice"],
            4
        )
    )

    print(
        "IoU:",
        round(
            metriques["iou"],
            4
        )
    )

    print(
        "Accuracy:",
        round(
            metriques["accuracy"],
            4
        )
    )


# -------------------------------------------------------------------------
# 7. RESUM DE TOT EL DATASET
# -------------------------------------------------------------------------

def resum_dataset(
    llista_metriques
):
    """
    Calcula la mitjana i la desviacio estandard
    de les metriques de totes les imatges.

    Aquesta funcio sera util per obtenir
    els resultats finals del projecte.
    """

    noms = [
        "precisio",
        "recall",
        "f1",
        "dice",
        "iou",
        "accuracy"
    ]


    resum = {}


    for nom in noms:

        valors = np.array(
            [
                m[nom]
                for m in llista_metriques
            ],
            dtype=float
        )


        resum[nom] = {

            "mitjana":np.mean(
                valors
            ),

            "desviacio":np.std(
                valors
            )
        }


    return resum


# -------------------------------------------------------------------------
# 8. MOSTRAR RESUM DEL DATASET
# -------------------------------------------------------------------------

def imprimir_resum_dataset(
    nom_metode,
    resum,
    nombre_imatges
):
    """
    Mostra mitjana +- desviacio estandard
    de totes les imatges.
    """

    print("\n======================================")

    print(
        "RESULTATS DATASET:",
        nom_metode
    )

    print(
        "Nombre d'imatges:",
        nombre_imatges
    )

    print("======================================")


    noms = [
        "precisio",
        "recall",
        "dice",
        "iou",
        "accuracy"
    ]


    for nom in noms:

        mitjana = resum[
            nom
        ]["mitjana"]

        desviacio = resum[
            nom
        ]["desviacio"]


        print(
            nom,
            ":",
            round(mitjana,4),
            "+/-",
            round(desviacio,4)
        )


# -------------------------------------------------------------------------
# 9. PROVA DEL BLOC
# -------------------------------------------------------------------------

if __name__ == "__main__":


    mascara_gt = np.array([

        [0,0,1,1],

        [0,0,1,1],

        [0,0,0,0],

        [1,1,0,0]
    ])


    mascara_predita = np.array([

        [0,0,1,0],

        [0,0,1,1],

        [0,1,0,0],

        [1,0,0,0]
    ])


    metriques = calcular_metriques(
        mascara_predita,
        mascara_gt
    )


    imprimir_metriques(
        "Prova Bloc 2",
        metriques
    )