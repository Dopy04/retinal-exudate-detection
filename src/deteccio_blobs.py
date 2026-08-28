# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 21:00:06 2026

@author: pablo
"""


"""
Bloc 10 - Extraccio de caracteristiques de blobs.

Aquest bloc agafa una mascara candidata d'exsudats i calcula
features de cada regio blanca de la mascara.

La idea es preparar una taula per al bloc 11, on podrem entrenar
un classificador amb SVM.

Entrada principal:
    - canals del preprocessament
    - canal usat per segmentar
    - mascara candidata final
    - mascara ground truth

Sortida:
    - una taula on cada fila es un blob
"""

import numpy as np
import pandas as pd

from skimage.measure import label, regionprops


# -------------------------------------------------------------------------
# 1. Funcions auxiliars
# -------------------------------------------------------------------------

def calcula_circularitat(area, perimetre):
    """
    Calcula la circularitat d'un blob.

    circularitat = 4*pi*area / perimetre^2

    Si el perimetre es zero, retornem zero per evitar errors.
    """
    if perimetre == 0:
        return 0

    circularitat = 4 * np.pi * area / (perimetre ** 2)

    return circularitat


def calcula_mitjana(canal, mascara_blob):
    """
    Calcula la mitjana d'un canal dins d'un blob.
    """
    valors = canal[mascara_blob]

    if len(valors) == 0:
        return 0

    mitjana = np.mean(valors)

    return mitjana


def calcula_desviacio(canal, mascara_blob):
    """
    Calcula la desviacio estandard d'un canal dins d'un blob.
    """
    valors = canal[mascara_blob]

    if len(valors) == 0:
        return 0

    desviacio = np.std(valors)

    return desviacio


def calcula_solapament_gt(mascara_blob, mascara_gt):
    """
    Calcula quin percentatge del blob cau dins de la ground truth.

    solapament = area(blob i GT) / area(blob)
    """
    area_blob = np.sum(mascara_blob)

    if area_blob == 0:
        return 0

    area_solapament = np.sum(mascara_blob & mascara_gt)

    solapament = area_solapament / area_blob

    return solapament


def etiqueta_blob(solapament, llindar_solapament=0.3):
    """
    Etiqueta un blob segons el seu solapament amb la ground truth.

    etiqueta = 1  -> exsudat real
    etiqueta = 0  -> fals positiu
    """
    if solapament >= llindar_solapament:
        etiqueta = 1
    else:
        etiqueta = 0

    return etiqueta


# -------------------------------------------------------------------------
# 2. Features d'un blob
# -------------------------------------------------------------------------

def extreu_features_blob(
    regio,
    etiquetes,
    canals,
    canal_segmentacio,
    mascara_gt,
    nom_imatge,
    nom_canal,
    nom_metode,
    llindar_solapament=0.3,
):
    """
    Extreu les features d'una regio connectada.

    regio ve de regionprops.
    etiquetes es la imatge etiquetada amb label().
    """

    mascara_blob = etiquetes == regio.label

    # Canals del preprocessament
    r = canals[0]
    g = canals[1]
    b = canals[2]
    luma = canals[3]
    h = canals[4]
    s = canals[5]
    v = canals[6]

    # Features de forma
    area = regio.area
    perimetre = regio.perimeter
    circularitat = calcula_circularitat(area, perimetre)

    eccentricitat = regio.eccentricity
    solidesa = regio.solidity
    extent = regio.extent

    centroid_y = regio.centroid[0]
    centroid_x = regio.centroid[1]

    min_row, min_col, max_row, max_col = regio.bbox
    bbox_alt = max_row - min_row
    bbox_ample = max_col - min_col

    # Features de color i intensitat
    mean_r = calcula_mitjana(r, mascara_blob)
    mean_g = calcula_mitjana(g, mascara_blob)
    mean_b = calcula_mitjana(b, mascara_blob)

    mean_luma = calcula_mitjana(luma, mascara_blob)
    mean_h = calcula_mitjana(h, mascara_blob)
    mean_s = calcula_mitjana(s, mascara_blob)
    mean_v = calcula_mitjana(v, mascara_blob)

    std_luma = calcula_desviacio(luma, mascara_blob)
    std_v = calcula_desviacio(v, mascara_blob)

    # Tambe guardem la intensitat del canal concret amb el qual hem segmentat.
    # Aixi podem comparar V, V CLAHE, V KMMRC, etc.
    mean_canal_segmentacio = calcula_mitjana(canal_segmentacio, mascara_blob)
    std_canal_segmentacio = calcula_desviacio(canal_segmentacio, mascara_blob)

    # Solapament amb ground truth
    solapament_gt = calcula_solapament_gt(
        mascara_blob,
        mascara_gt,
    )

    etiqueta = etiqueta_blob(
        solapament_gt,
        llindar_solapament,
    )

    fila = [
        nom_imatge,
        nom_canal,
        nom_metode,
        regio.label,

        area,
        perimetre,
        circularitat,
        eccentricitat,
        solidesa,
        extent,

        centroid_y,
        centroid_x,
        bbox_alt,
        bbox_ample,

        mean_r,
        mean_g,
        mean_b,
        mean_luma,
        mean_h,
        mean_s,
        mean_v,

        std_luma,
        std_v,

        mean_canal_segmentacio,
        std_canal_segmentacio,

        solapament_gt,
        etiqueta,
    ]

    return fila


# -------------------------------------------------------------------------
# 3. Taula de features
# -------------------------------------------------------------------------

def crea_taula_features(
    canals,
    canal_segmentacio,
    mascara_candidats,
    mascara_gt,
    nom_imatge="",
    nom_canal="",
    nom_metode="",
    llindar_solapament=0.3,
):
    """
    Crea una taula de features a partir d'una mascara candidata.

    Cada regio blanca separada de la mascara es considera un blob.
    """

    mascara_candidats = mascara_candidats.astype(bool)
    mascara_gt = mascara_gt.astype(bool)

    if mascara_candidats.shape != mascara_gt.shape:
        raise ValueError("La mascara candidata i la ground truth no tenen la mateixa mida.")

    if canal_segmentacio.shape != mascara_candidats.shape:
        raise ValueError("El canal de segmentacio i la mascara no tenen la mateixa mida.")

    # Separem els blobs de la mascara.
    etiquetes = label(mascara_candidats)
    regions = regionprops(etiquetes)

    columnes = [
        "nom_imatge",
        "nom_canal",
        "nom_metode",
        "blob_id",

        "area",
        "perimetre",
        "circularitat",
        "eccentricitat",
        "solidesa",
        "extent",

        "centroid_y",
        "centroid_x",
        "bbox_alt",
        "bbox_ample",

        "mean_r",
        "mean_g",
        "mean_b",
        "mean_luma",
        "mean_h",
        "mean_s",
        "mean_v",

        "std_luma",
        "std_v",

        "mean_canal_segmentacio",
        "std_canal_segmentacio",

        "solapament_gt",
        "etiqueta",
    ]

    files = []

    for regio in regions:

        fila = extreu_features_blob(
            regio,
            etiquetes,
            canals,
            canal_segmentacio,
            mascara_gt,
            nom_imatge,
            nom_canal,
            nom_metode,
            llindar_solapament,
        )

        files.append(fila)

    taula = pd.DataFrame(
        files,
        columns=columnes,
    )

    return taula


# -------------------------------------------------------------------------
# 4. Resum de la taula
# -------------------------------------------------------------------------

def imprimeix_resum_features(taula, nom=""):
    """
    Imprimeix un resum simple de la taula de features.
    """

    print("\n====================================")
    print("RESUM FEATURES:", nom)
    print("====================================")

    print("Nombre de blobs:", len(taula))

    if len(taula) == 0:
        return

    n_exsudats = np.sum(taula["etiqueta"] == 1)
    n_falsos = np.sum(taula["etiqueta"] == 0)

    print("Blobs exsudat real:", n_exsudats)
    print("Blobs fals positiu:", n_falsos)

    print("Area mitjana:", round(np.mean(taula["area"]), 3))
    print("Solapament GT mitja:", round(np.mean(taula["solapament_gt"]), 3))
    print("Mean V mitja:", round(np.mean(taula["mean_v"]), 3))
    print("Mean canal segmentacio:", round(np.mean(taula["mean_canal_segmentacio"]), 3))


def ajunta_taules(llista_taules):
    """
    Junta diferents taules de features en una sola taula.
    """

    if len(llista_taules) == 0:
        return pd.DataFrame()

    taula_total = pd.concat(
        llista_taules,
        ignore_index=True,
    )

    return taula_total