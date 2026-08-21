# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 23:41:49 2026

@author: gurpi
"""

"""
Mòdul de mètriques per al projecte de segmentació d'exsudats.

Aquest fitxer serveix per comparar una màscara predita pel nostre algoritme
amb la màscara "ground truth" marcada per experts.

En el nostre cas:
    1 = píxel d'exsudat
    0 = píxel de fons / no exsudat
Les mètriques principals que utilitzarem són:
    - Precisió
    - Recall / Sensibilitat
    - F1-score
    - Dice
    - IoU
Aquestes mètriques ens permeten saber objectivament si la segmentació
és bona o dolenta.
"""

import csv
import numpy as np

EPS = 1e-8
# EPS és un valor molt petit que posem als denominadors per evitar divisions per zero.

# 1. FUNCIONS AUXILIARS

def binaritza_mascara(mascara, llindar=None):
    """
    Converteix una màscara qualsevol a binària.

    Paràmetres
    ----------
    mascara : np.ndarray
        Pot ser una imatge booleana, uint8 [0,255] o float [0,1].

    llindar : float o None
        Valor de tall per decidir què és positiu.
        Si és None, el codi decideix automàticament:
            - si la màscara té valors grans, usa 128
            - si la màscara està normalitzada, usa 0.5

    Retorna
    -------
    mascara_binaria : np.ndarray de bool
        True  = exsudat
        False = fons
    """

    mascara = np.asarray(mascara)

    # Si la màscara té tres canals, ens quedem amb el primer.
    # Això pot passar si el ground truth està guardat com una imatge RGB.
    if mascara.ndim == 3:
        mascara = mascara[:, :, 0]

    # Si ja és booleana, no cal fer res més.
    if mascara.dtype == bool:
        return mascara

    # Si no ens donen un llindar, triem un llindar automàtic.
    if llindar is None:
        if np.nanmax(mascara) > 1.5:
            llindar = 128
        else:
            llindar = 0.5

    mascara_binaria = mascara > llindar

    return mascara_binaria


def comprova_dimensions(mascara_predita, mascara_gt):
    """
    Comprova que la màscara predita i el ground truth tinguin la mateixa mida.

    Això és important perquè les mètriques es calculen píxel a píxel.
    Si les dues imatges tenen dimensions diferents, no es poden comparar directament.
    """

    if mascara_predita.shape != mascara_gt.shape:
        raise ValueError(
            f"Les dimensions no coincideixen: "
            f"predita={mascara_predita.shape}, ground_truth={mascara_gt.shape}"
        )


def prepara_mascares(mascara_predita, mascara_gt, mascara_zona_valida=None):
    """
    Prepara les màscares abans de calcular les mètriques.

    Fa tres coses:
        1. Binaritza la màscara predita.
        2. Binaritza la màscara ground truth.
        3. Aplica una màscara de zona vàlida si volem ignorar el fons negre exterior.

    Paràmetres
    ----------
    mascara_predita : np.ndarray
        Màscara generada pel nostre algoritme.

    mascara_gt : np.ndarray
        Màscara ground truth marcada per experts.

    mascara_zona_valida : np.ndarray o None
        Màscara opcional de la retina.
        True  = zona que volem tenir en compte.
        False = zona que volem ignorar, per exemple fons negre exterior.

    Retorna
    -------
    pred : np.ndarray de bool
        Màscara predita ja binaritzada.

    gt : np.ndarray de bool
        Ground truth ja binaritzat.

    zona_valida : np.ndarray de bool
        Zona on es calcularan les mètriques.
    """

    pred = binaritza_mascara(mascara_predita)
    gt = binaritza_mascara(mascara_gt)

    comprova_dimensions(pred, gt)

    if mascara_zona_valida is None:
        zona_valida = np.ones_like(gt, dtype=bool)
    else:
        zona_valida = binaritza_mascara(mascara_zona_valida)
        comprova_dimensions(zona_valida, gt)

    return pred, gt, zona_valida


# ============================================================
# 2. CÀLCUL DE TP, FP, FN, TN
# ============================================================

def calcula_confusio(mascara_predita, mascara_gt, mascara_zona_valida=None):
    """
    Calcula TP, FP, FN i TN comparant la predicció amb el ground truth.

    Definicions
    -----------
    TP = True Positive
        El nostre algoritme diu que és exsudat i el ground truth també.

    FP = False Positive
        El nostre algoritme diu que és exsudat, però el ground truth diu que és fons.

    FN = False Negative
        El nostre algoritme diu que és fons, però el ground truth diu que era exsudat.

    TN = True Negative
        El nostre algoritme diu que és fons i el ground truth també.

    Retorna
    -------
    TP, FP, FN, TN : int
        Nombre de píxels de cada tipus.
    """

    pred, gt, zona_valida = prepara_mascares(
        mascara_predita,
        mascara_gt,
        mascara_zona_valida
    )

    # Ens quedem només amb els píxels de la zona vàlida.
    # Això evita que el fons negre exterior domini les mètriques.
    pred = pred[zona_valida]
    gt = gt[zona_valida]

    TP = np.logical_and(pred == True, gt == True).sum()
    FP = np.logical_and(pred == True, gt == False).sum()
    FN = np.logical_and(pred == False, gt == True).sum()
    TN = np.logical_and(pred == False, gt == False).sum()

    return int(TP), int(FP), int(FN), int(TN)


# ============================================================
# 3. MÈTRIQUES PRINCIPALS
# ============================================================

def calcula_metriques(mascara_predita, mascara_gt, mascara_zona_valida=None):
    """
    Calcula totes les mètriques importants de segmentació.

    Aquest és el bloc principal del fitxer.

    Paràmetres
    ----------
    mascara_predita : np.ndarray
        Màscara generada pel nostre mètode.

    mascara_gt : np.ndarray
        Màscara correcta feta per experts.

    mascara_zona_valida : np.ndarray o None
        Opcional. Serveix per ignorar zones que no volem avaluar,
        com el fons negre exterior de la imatge retinal.

    Retorna
    -------
    metriques : dict
        Diccionari amb TP, FP, FN, TN i totes les mètriques.
    """

    TP, FP, FN, TN = calcula_confusio(
        mascara_predita,
        mascara_gt,
        mascara_zona_valida
    )

    # ------------------------------------------------------------
    # Precisió
    # ------------------------------------------------------------
    # De tot el que el nostre algoritme marca com a exsudat,
    # quina part realment és exsudat?
    #
    # Precisió alta = pocs falsos positius.
    # Precisió baixa = el mètode detecta massa coses que no són exsudats.
    precisio = TP / (TP + FP + EPS)

    # ------------------------------------------------------------
    # Recall / Sensibilitat
    # ------------------------------------------------------------
    # De tots els exsudats reals que hi havia segons el ground truth,
    # quina part ha trobat el nostre algoritme?
    #
    # Recall alt = trobem molts exsudats reals.
    # Recall baix = ens deixem molts exsudats sense detectar.
    sensibilitat = TP / (TP + FN + EPS)

    # També guardem el mateix valor amb el nom "recall",
    # perquè és el terme que apareix sovint als papers.
    recall = sensibilitat

    # ------------------------------------------------------------
    # Especificitat
    # ------------------------------------------------------------
    # De tot el fons real, quina part hem classificat correctament com a fons?
    #
    # No serà la nostra mètrica principal perquè en imatges mèdiques
    # hi ha moltíssim fons i pot donar valors molt alts encara que
    # la segmentació dels exsudats no sigui bona.
    especificitat = TN / (TN + FP + EPS)

    # ------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------
    # Percentatge total de píxels classificats correctament.
    #
    # Compte: pot ser enganyosa perquè gairebé tota la imatge és fons.
    # Per això al projecte donarem més importància a Dice, IoU, F1,
    # precisió i recall.
    accuracy = (TP + TN) / (TP + FP + FN + TN + EPS)

    # ------------------------------------------------------------
    # F1-score
    # ------------------------------------------------------------
    # Mitjana harmònica entre precisió i recall.
    #
    # Serveix per tenir una sola mètrica que equilibri:
    #   - no tenir massa falsos positius
    #   - no perdre massa exsudats reals
    f1 = 2 * precisio * recall / (precisio + recall + EPS)

    # ------------------------------------------------------------
    # Dice
    # ------------------------------------------------------------
    # Mètrica molt utilitzada en segmentació mèdica.
    #
    # Mesura el solapament entre la màscara predita i el ground truth.
    # Dice = 1 significa segmentació perfecta.
    # Dice = 0 significa cap solapament.
    dice = 2 * TP / (2 * TP + FP + FN + EPS)

    # ------------------------------------------------------------
    # IoU
    # ------------------------------------------------------------
    # Intersection over Union.
    #
    # També mesura el solapament, però és més estricta que Dice.
    # IoU = intersecció / unió
    iou = TP / (TP + FP + FN + EPS)

    metriques = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precisio": precisio,
        "recall": recall,
        "sensibilitat": sensibilitat,
        "especificitat": especificitat,
        "accuracy": accuracy,
        "f1": f1,
        "dice": dice,
        "iou": iou
    }

    return metriques


# ============================================================
# 4. AVALUAR DIVERSOS MÈTODES SOBRE UNA IMATGE
# ============================================================

def avalua_metodes(mascares_metodes, mascara_gt, mascara_zona_valida=None):
    """
    Avalua diversos mètodes de segmentació sobre una mateixa imatge.

    Paràmetres
    ----------
    mascares_metodes : dict
        Diccionari on:
            clau  = nom del mètode
            valor = màscara predita per aquell mètode

        Exemple:
            {
                "Otsu": mascara_otsu,
                "Kmeans": mascara_kmeans,
                "CLAHE_adaptatiu": mascara_adaptativa,
                "Final": mascara_final
            }

    mascara_gt : np.ndarray
        Ground truth de la imatge.

    mascara_zona_valida : np.ndarray o None
        Zona on volem calcular les mètriques.

    Retorna
    -------
    resultats : dict
        Diccionari amb les mètriques de cada mètode.
    """

    resultats = {}

    for nom_metode, mascara_predita in mascares_metodes.items():
        resultats[nom_metode] = calcula_metriques(
            mascara_predita,
            mascara_gt,
            mascara_zona_valida
        )

    return resultats


# ============================================================
# 5. PASSAR RESULTATS A FORMAT TAULA
# ============================================================

def resultats_a_files(nom_imatge, resultats_metodes):
    """
    Converteix els resultats d'una imatge a una llista de files.

    Això és útil per després guardar-ho en un CSV.

    Cada fila tindrà:
        imatge, metode, TP, FP, FN, TN, precisio, recall, f1, dice, iou...
    """

    files = []

    for nom_metode, metriques in resultats_metodes.items():
        fila = {
            "imatge": nom_imatge,
            "metode": nom_metode,
            "TP": metriques["TP"],
            "FP": metriques["FP"],
            "FN": metriques["FN"],
            "TN": metriques["TN"],
            "precisio": metriques["precisio"],
            "recall": metriques["recall"],
            "sensibilitat": metriques["sensibilitat"],
            "especificitat": metriques["especificitat"],
            "accuracy": metriques["accuracy"],
            "f1": metriques["f1"],
            "dice": metriques["dice"],
            "iou": metriques["iou"]
        }

        files.append(fila)

    return files


def desa_resultats_csv(files_resultats, ruta_sortida):
    """
    Desa els resultats de mètriques en un fitxer CSV.

    Aquest CSV després servirà per:
        - fer taules al paper
        - comparar mètodes
        - calcular mitjanes
        - justificar conclusions
    """

    if len(files_resultats) == 0:
        print("No hi ha resultats per guardar.")
        return

    camps = list(files_resultats[0].keys())

    with open(ruta_sortida, mode="w", newline="", encoding="utf-8") as fitxer:
        writer = csv.DictWriter(fitxer, fieldnames=camps)
        writer.writeheader()
        writer.writerows(files_resultats)


# ============================================================
# 6. RESUMIR RESULTATS DE MOLTES IMATGES
# ============================================================

def resumeix_resultats(files_resultats):
    """
    Calcula la mitjana de les mètriques per cada mètode.

    Entrada
    -------
    files_resultats : list of dict
        Llista de files com les que genera resultats_a_files().

    Retorna
    -------
    resum : dict
        Diccionari amb la mitjana de cada mètrica per mètode.

    Exemple de sortida:
        {
            "Otsu": {
                "precisio": 0.25,
                "recall": 0.70,
                "f1": 0.36,
                "dice": 0.36,
                "iou": 0.22
            },
            "Final": {
                "precisio": 0.55,
                "recall": 0.62,
                "f1": 0.58,
                "dice": 0.58,
                "iou": 0.41
            }
        }
    """

    metriques_numeriques = [
        "precisio",
        "recall",
        "sensibilitat",
        "especificitat",
        "accuracy",
        "f1",
        "dice",
        "iou"
    ]

    # Primer agrupem totes les files segons el mètode.
    dades_per_metode = {}

    for fila in files_resultats:
        metode = fila["metode"]

        if metode not in dades_per_metode:
            dades_per_metode[metode] = []

        dades_per_metode[metode].append(fila)

    # Ara calculem la mitjana de cada mètrica per cada mètode.
    resum = {}

    for metode, files in dades_per_metode.items():
        resum[metode] = {}

        for metrica in metriques_numeriques:
            valors = [float(fila[metrica]) for fila in files]
            resum[metode][metrica] = float(np.mean(valors))

    return resum


def imprimeix_resum(resum):
    """
    Imprimeix per pantalla un resum ordenat dels resultats.

    Això és útil mentre esteu provant el codi a Spyder.
    """

    print("\n===== RESUM DE MÈTRIQUES PER MÈTODE =====\n")

    for metode, metriques in resum.items():
        print(f"Mètode: {metode}")
        print(f"  Precisió:     {metriques['precisio']:.4f}")
        print(f"  Recall:       {metriques['recall']:.4f}")
        print(f"  F1-score:     {metriques['f1']:.4f}")
        print(f"  Dice:         {metriques['dice']:.4f}")
        print(f"  IoU:          {metriques['iou']:.4f}")
        print(f"  Accuracy:     {metriques['accuracy']:.4f}")
        print("")


# ============================================================
# 7. PROVA RÀPIDA DEL MÒDUL
# ============================================================

if __name__ == "__main__":
    """
    Aquesta part només s'executa si obrim directament metrics.py.

    Serveix per comprovar que les mètriques funcionen abans d'integrar
    aquest fitxer amb la resta del projecte.
    """

    # Exemple petit artificial.
    # 1 significa exsudat i 0 significa fons.

    mascara_gt_exemple = np.array([
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
    ])

    mascara_predita_exemple = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
    ])

    metriques = calcula_metriques(
        mascara_predita_exemple,
        mascara_gt_exemple
    )

    print("Resultats de la prova ràpida:")
    for nom, valor in metriques.items():
        print(f"{nom}: {valor}")