# -*- coding: utf-8 -*-
"""
Prova completa dels blocs 2 a 11.

Aquest fitxer s'ha de posar al mateix nivell que main.py, src/ i data_samples/.

Estructura esperada:

retina-exudates-piva/
├── main_blocs_2_11_net.py
├── main.py
├── src/
└── data_samples/
    ├── Exudate_JSIEC_002.jpg
    ├── Exudate_JSIEC_002_Groundtruth.jpg
    ├── Exudate_DR2019_001.jpg
    ├── Exudate_DR2019_001_Groundtruth.jpg
    ├── Exudate_DR2019_003.jpg
    └── Exudate_DR2019_003_Groundtruth.jpg
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.metriques import calcular_metriques, imprimir_metriques, comparar_metodes
from src.preprocessament_3 import preprocessa_imatge, visualitza_preprocessament
from src.segmentacio import obtenir_mascares_baseline, segmentar_otsu
from src.mejora_contraste_5 import millora_contrast, visualitza_contrast
from src.nick_inrg import invertir_canal, segmentar_nick, segmentar_inrg
from src.disc_optic import detecta_disc_optic, elimina_disc_optic
from src.postprocessament import netejar_mascara
from src.watershed import transformar_distancia, trobar_marcadors, aplicar_watershed

from src.deteccio_blobs import (
    crea_taula_features,
    imprimeix_resum_features,
    ajunta_taules,
)

from src.classificador_SVM import (
    separa_train_test_per_imatge,
    entrena_svm,
    imprimeix_features_seleccionades,
    prediu_taula,
    avalua_prediccio_blobs,
    reconstrueix_mascara_svm,
)


# ============================================================
# CONFIGURACIO GENERAL
# ============================================================

ruta_projecte = Path(__file__).resolve().parent
ruta_dades = ruta_projecte / "data_samples"

imatges = [
    ("Exudate_JSIEC_002.jpg", "Exudate_JSIEC_002_Groundtruth.jpg"),
    ("Exudate_DR2019_001.jpg", "Exudate_DR2019_001_Groundtruth.jpg"),
    ("Exudate_DR2019_003.jpg", "Exudate_DR2019_003_Groundtruth.jpg"),
]

# Configuracio del SVM.
# Si vols reproduir el codi antic, pots posar llindar_positiu_svm = 0.30.
k_chi2_svm = 5
llindar_positiu_svm = 0.25
llindar_negatiu_svm = 0.05

# Posa False si vols que el codi calculi sense obrir tantes figures.
mostrar_figures = True


# ============================================================
# FUNCIONS AUXILIARS
# ============================================================

def comprova_fitxers():
    """Comprova que les imatges de prova existeixen."""

    print("Carpeta del projecte:", ruta_projecte)
    print("Carpeta de dades:", ruta_dades)

    for nom_imatge, nom_gt in imatges:
        ruta_imatge = ruta_dades / nom_imatge
        ruta_gt = ruta_dades / nom_gt

        if not ruta_imatge.exists():
            raise FileNotFoundError("No existeix la imatge: " + str(ruta_imatge))

        if not ruta_gt.exists():
            raise FileNotFoundError("No existeix la Ground Truth: " + str(ruta_gt))


def calcula_mitjana_metriques(llista_metriques):
    """Calcula la mitjana de les metriques d'una llista."""

    claus = ["precisio", "recall", "f1", "dice", "iou", "accuracy"]
    mitjana = {}

    for clau in claus:
        valors = []

        for metriques in llista_metriques:
            valors.append(metriques[clau])

        mitjana[clau] = np.mean(valors)

    return mitjana


def imprimeix_mitjana_metriques(nom, metriques):
    """Imprimeix les metriques mitjanes."""

    print("\n====================================")
    print(nom)
    print("====================================")
    print("Precisio:", round(metriques["precisio"], 4))
    print("Recall:", round(metriques["recall"], 4))
    print("F1:", round(metriques["f1"], 4))
    print("Dice:", round(metriques["dice"], 4))
    print("IoU:", round(metriques["iou"], 4))
    print("Accuracy:", round(metriques["accuracy"], 4))


def crea_mascara_otsu(canal_g, canal_r, mascara_retina):
    """Pipeline A: G original -> Otsu -> disc optic -> morfologia -> watershed."""

    mascara_otsu = segmentar_otsu(
        canal_g,
        mascara_retina,
    )

    _, _, disc_r = detecta_disc_optic(
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3,
    )

    mascara_sense_disc = elimina_disc_optic(
        mascara_otsu,
        disc_r,
    )

    mascara_morfologia = netejar_mascara(
        mascara_sense_disc,
        mida_minima=50,
        mida_forat=50,
    )

    etiquetes = aplicar_watershed(
        mascara_morfologia,
        distancia_minima=5,
    )

    mascara_candidats = etiquetes > 0

    return mascara_candidats, etiquetes, disc_r


def crea_mascara_inrg(canal_g, canal_r, mascara_retina):
    """Pipeline B: G original -> INRG -> disc optic -> morfologia -> watershed."""

    mascara_inrg = segmentar_inrg(
        canal_g,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
        iteracions=3,
        delta=0.1,
    )

    _, _, disc_r = detecta_disc_optic(
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3,
    )

    mascara_sense_disc = elimina_disc_optic(
        mascara_inrg,
        disc_r,
    )

    mascara_morfologia = netejar_mascara(
        mascara_sense_disc,
        mida_minima=50,
        mida_forat=50,
    )

    etiquetes = aplicar_watershed(
        mascara_morfologia,
        distancia_minima=5,
    )

    mascara_candidats = etiquetes > 0

    return mascara_candidats, etiquetes, disc_r


def afegeix_resultat(resum_final, imatge, pipeline, abans, despres):
    """Guarda un resultat per calcular mitjanes finals."""

    resum_final.append({
        "imatge": imatge,
        "pipeline": pipeline,
        "abans": abans,
        "despres": despres,
    })


def prova_svm_pipeline(taula_total, dades, nom_test, nom_pipeline):
    """Entrena el SVM deixant una imatge com a test."""

    print("\n------------------------------------------")
    print(nom_pipeline)
    print("Imatge test:", nom_test)
    print("------------------------------------------")

    taula_train, taula_test = separa_train_test_per_imatge(
        taula_total,
        [nom_test],
    )

    print("Blobs train:", len(taula_train))
    print("Blobs test:", len(taula_test))
    print("Positius train:", np.sum(taula_train["etiqueta"] == 1))
    print("Negatius train:", np.sum(taula_train["etiqueta"] == 0))
    print("Ambigus train:", np.sum(taula_train["etiqueta"] == -1))
    print("Positius test:", np.sum(taula_test["etiqueta"] == 1))
    print("Negatius test:", np.sum(taula_test["etiqueta"] == 0))
    print("Ambigus test:", np.sum(taula_test["etiqueta"] == -1))

    try:
        escalador, selector, model = entrena_svm(
            taula_train,
            k_chi2=k_chi2_svm,
        )
    except ValueError as error:
        print("No s'ha pogut entrenar el SVM:", error)
        return None

    imprimeix_features_seleccionades(selector)

    taula_predita = prediu_taula(
        taula_test,
        escalador,
        selector,
        model,
    )

    avalua_prediccio_blobs(taula_predita)

    mascara_abans = dades[nom_test]["candidats"]
    mascara_gt = dades[nom_test]["gt"]

    mascara_svm = reconstrueix_mascara_svm(
        mascara_abans,
        taula_predita,
    )

    metriques_abans = calcular_metriques(
        mascara_abans,
        mascara_gt,
    )

    metriques_svm = calcular_metriques(
        mascara_svm,
        mascara_gt,
    )

    imprimir_metriques(nom_pipeline + " abans SVM", metriques_abans)
    imprimir_metriques(nom_pipeline + " despres SVM", metriques_svm)

    return {
        "mascara_abans": mascara_abans,
        "mascara_svm": mascara_svm,
        "metriques_abans": metriques_abans,
        "metriques_svm": metriques_svm,
    }


def mostra_comparacio_svm(nom_imatge, imatge, mascara_gt, resultat_otsu, resultat_inrg):
    """Mostra la comparacio final dels dos pipelines."""

    if not mostrar_figures:
        return

    plt.figure(figsize=(16, 9))

    plt.suptitle(
        "COMPARACIO PIPELINES - " + nom_imatge,
        fontsize=16,
    )

    plt.subplot(2, 3, 1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(mascara_gt, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(mascara_gt, cmap="Reds", alpha=0.5)
    plt.imshow(resultat_otsu["mascara_abans"], cmap="Greens", alpha=0.5)
    plt.title("Otsu abans SVM")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(mascara_gt, cmap="Reds", alpha=0.5)
    plt.imshow(resultat_otsu["mascara_svm"], cmap="Greens", alpha=0.5)
    plt.title("Otsu despres SVM")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(mascara_gt, cmap="Reds", alpha=0.5)
    plt.imshow(resultat_inrg["mascara_abans"], cmap="Greens", alpha=0.5)
    plt.title("INRG abans SVM")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(mascara_gt, cmap="Reds", alpha=0.5)
    plt.imshow(resultat_inrg["mascara_svm"], cmap="Greens", alpha=0.5)
    plt.title("INRG despres SVM")
    plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ============================================================
# BLOC 2 - METRIQUES
# ============================================================

print("\n====================")
print("BLOC 2 - METRIQUES")
print("====================")

comprova_fitxers()

mascara_gt_prova = np.array([
    [0, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 0, 0, 0],
    [1, 1, 0, 0],
])

mascara_predita_prova = np.array([
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
])

metriques = calcular_metriques(
    mascara_predita_prova,
    mascara_gt_prova,
)

imprimir_metriques("Prova bloc 2", metriques)


# ============================================================
# BLOC 3 - PREPROCESSAMENT
# ============================================================

print("\n==========================")
print("BLOC 3 - PREPROCESSAMENT")
print("==========================")

for nom_imatge, nom_gt in imatges:

    print("\n------------------------------")
    print("Imatge:", nom_imatge)
    print("------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    print("Mida imatge:", imatge.shape)
    print("Mida Ground Truth:", mascara_gt.shape)
    print("Mida mascara retina:", mascara_retina.shape)
    print("Nombre de canals:", len(canals))

    if imatge.shape[:2] == mascara_gt.shape:
        print("OK: imatge i Ground Truth tenen la mateixa mida")
    else:
        print("ERROR: les mides no coincideixen")

    if mostrar_figures:
        visualitza_preprocessament(
            imatge,
            mascara_gt,
            mascara_retina,
            canals,
        )


# ============================================================
# BLOC 4 - SEGMENTACIO BASELINE
# ============================================================

print("\n==============================")
print("BLOC 4 - SEGMENTACIO BASELINE")
print("==============================")

for nom_imatge, nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 4:", nom_imatge)
    print("----------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    canal_g = canals[1]

    mascares = obtenir_mascares_baseline(
        canal_g,
        mascara_retina,
    )

    resultats = comparar_metodes(
        mascares,
        mascara_gt,
    )

    for nom_metode in resultats:
        imprimir_metriques(
            nom_metode,
            resultats[nom_metode],
        )

    if mostrar_figures:
        plt.figure(figsize=(14, 8))
        plt.suptitle("PROVA BLOC 4 - " + nom_imatge + " - Canal G", fontsize=16)

        plt.subplot(2, 4, 1)
        plt.imshow(imatge)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 4, 2)
        plt.imshow(canal_g, cmap="gray")
        plt.title("Canal G")
        plt.axis("off")

        plt.subplot(2, 4, 3)
        plt.imshow(mascara_gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(2, 4, 4)
        plt.imshow(mascares["Otsu"], cmap="gray")
        plt.title("Otsu")
        plt.axis("off")

        plt.subplot(2, 4, 5)
        plt.imshow(mascares["Llindar_fix"], cmap="gray")
        plt.title("Llindar fix")
        plt.axis("off")

        plt.subplot(2, 4, 6)
        plt.imshow(mascares["Adaptatiu"], cmap="gray")
        plt.title("Adaptatiu")
        plt.axis("off")

        plt.subplot(2, 4, 7)
        plt.imshow(mascares["Sauvola"], cmap="gray")
        plt.title("Sauvola")
        plt.axis("off")

        plt.subplot(2, 4, 8)
        plt.imshow(mascares["Kmeans"], cmap="gray")
        plt.title("K-means")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
# BLOC 5 - MILLORA DE CONTRAST
# ============================================================

print("\n============================")
print("BLOC 5 - MILLORA DE CONTRAST")
print("============================")

for nom_imatge, nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 5:", nom_imatge)
    print("----------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    contrast = millora_contrast(
        imatge_retina,
        mascara_retina,
        canals,
    )

    g_clahe = contrast[2]
    v_clahe = contrast[4]
    v_kmmrc = contrast[5]
    v_kmmrc_clahe = contrast[6]
    v_kmmrc_clahe_filtrat = contrast[7]

    print("Mida G CLAHE:", g_clahe.shape)
    print("Mida V CLAHE:", v_clahe.shape)
    print("Mida V KMMRC:", v_kmmrc.shape)
    print("Mida KMMRC + CLAHE:", v_kmmrc_clahe.shape)
    print("Mida resultat final:", v_kmmrc_clahe_filtrat.shape)

    if v_kmmrc_clahe_filtrat.shape == mascara_gt.shape:
        print("OK: el resultat del bloc 5 te la mateixa mida que el Ground Truth")
    else:
        print("ERROR: les mides no coincideixen")

    if mostrar_figures:
        visualitza_contrast(
            imatge_retina,
            canals,
            contrast,
            "PROVA BLOC 5 - " + nom_imatge,
        )


# ============================================================
# BLOC 5B - CONTRAST + OTSU
# ============================================================

print("\n================================")
print("PROVA BLOC 5B - CONTRAST + OTSU")
print("================================")

for nom_imatge, nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 5B:", nom_imatge)
    print("----------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    contrast = millora_contrast(
        imatge_retina,
        mascara_retina,
        canals,
    )

    g_original = canals[1]
    g_clahe = contrast[2]
    luma_clahe = contrast[1]
    v_clahe = contrast[4]

    mascara_g_original = segmentar_otsu(g_original, mascara_retina)
    mascara_g_clahe = segmentar_otsu(g_clahe, mascara_retina)
    mascara_luma_clahe = segmentar_otsu(luma_clahe, mascara_retina)
    mascara_v_clahe = segmentar_otsu(v_clahe, mascara_retina)

    mascares_5b = {
        "G original + Otsu": mascara_g_original,
        "G CLAHE + Otsu": mascara_g_clahe,
        "Luma CLAHE + Otsu": mascara_luma_clahe,
        "V CLAHE + Otsu": mascara_v_clahe,
    }

    resultats_5b = comparar_metodes(
        mascares_5b,
        mascara_gt,
    )

    for nom_metode in resultats_5b:
        imprimir_metriques(
            nom_metode,
            resultats_5b[nom_metode],
        )

    if mostrar_figures:
        plt.figure(figsize=(14, 8))
        plt.suptitle("PROVA BLOC 5B - " + nom_imatge, fontsize=16)

        plt.subplot(2, 4, 1)
        plt.imshow(mascara_gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(2, 4, 2)
        plt.imshow(g_original, cmap="gray")
        plt.title("G original")
        plt.axis("off")

        plt.subplot(2, 4, 3)
        plt.imshow(g_clahe, cmap="gray")
        plt.title("G CLAHE")
        plt.axis("off")

        plt.subplot(2, 4, 4)
        plt.imshow(luma_clahe, cmap="gray")
        plt.title("Luma CLAHE")
        plt.axis("off")

        plt.subplot(2, 4, 5)
        plt.imshow(mascara_g_original, cmap="gray")
        plt.title("G original + Otsu")
        plt.axis("off")

        plt.subplot(2, 4, 6)
        plt.imshow(mascara_g_clahe, cmap="gray")
        plt.title("G CLAHE + Otsu")
        plt.axis("off")

        plt.subplot(2, 4, 7)
        plt.imshow(mascara_luma_clahe, cmap="gray")
        plt.title("Luma CLAHE + Otsu")
        plt.axis("off")

        plt.subplot(2, 4, 8)
        plt.imshow(mascara_v_clahe, cmap="gray")
        plt.title("V CLAHE + Otsu")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
# BLOC 6 - NICK / INRG
# ============================================================

print("\n============================")
print("BLOC 6 - NICK / INRG")
print("============================")

for nom_imatge, nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 6:", nom_imatge)
    print("----------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    g_original = canals[1]

    contrast = millora_contrast(
        imatge_retina,
        mascara_retina,
        canals,
    )

    g_clahe = contrast[2]

    g_invertit = invertir_canal(g_original)
    g_clahe_invertit = invertir_canal(g_clahe)

    mascara_nick_g = segmentar_nick(
        g_invertit,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
    )

    mascara_nick_clahe = segmentar_nick(
        g_clahe_invertit,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
    )

    mascara_inrg_g = segmentar_inrg(
        g_original,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
        iteracions=3,
        delta=0.1,
    )

    mascara_inrg_clahe = segmentar_inrg(
        g_clahe,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
        iteracions=3,
        delta=0.1,
    )

    mascares_6 = {
        "G original + NICK": mascara_nick_g,
        "G CLAHE + NICK": mascara_nick_clahe,
        "G original + INRG": mascara_inrg_g,
        "G CLAHE + INRG": mascara_inrg_clahe,
    }

    resultats_6 = comparar_metodes(
        mascares_6,
        mascara_gt,
    )

    for nom_metode in resultats_6:
        imprimir_metriques(
            nom_metode,
            resultats_6[nom_metode],
        )

    if mostrar_figures:
        plt.figure(figsize=(14, 8))
        plt.suptitle("PROVA BLOC 6 - " + nom_imatge, fontsize=16)

        plt.subplot(2, 4, 1)
        plt.imshow(mascara_gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(2, 4, 2)
        plt.imshow(g_original, cmap="gray")
        plt.title("G original")
        plt.axis("off")

        plt.subplot(2, 4, 3)
        plt.imshow(g_invertit, cmap="gray")
        plt.title("G invertit")
        plt.axis("off")

        plt.subplot(2, 4, 4)
        plt.imshow(g_clahe, cmap="gray")
        plt.title("G CLAHE")
        plt.axis("off")

        plt.subplot(2, 4, 5)
        plt.imshow(mascara_nick_g, cmap="gray")
        plt.title("G + NICK")
        plt.axis("off")

        plt.subplot(2, 4, 6)
        plt.imshow(mascara_nick_clahe, cmap="gray")
        plt.title("G CLAHE + NICK")
        plt.axis("off")

        plt.subplot(2, 4, 7)
        plt.imshow(mascara_inrg_g, cmap="gray")
        plt.title("G + INRG")
        plt.axis("off")

        plt.subplot(2, 4, 8)
        plt.imshow(mascara_inrg_clahe, cmap="gray")
        plt.title("G CLAHE + INRG")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2)
        plt.show()


# ============================================================
# BLOC 7 - ELIMINACIO DEL DISC OPTIC
# ============================================================

print("\n================================")
print("BLOC 7 - ELIMINACIO DISC OPTIC")
print("================================")

for nom_imatge, nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 7:", nom_imatge)
    print("----------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    canal_r = canals[0]
    canal_g = canals[1]
    canal_v = canals[6]
    canal_rv = (canal_r + canal_v) / 2

    r_suau, r_candidats, disc_r = detecta_disc_optic(
        canal_r,
        mascara_retina,
    )

    v_suau, v_candidats, disc_v = detecta_disc_optic(
        canal_v,
        mascara_retina,
    )

    rv_suau, rv_candidats, disc_rv = detecta_disc_optic(
        canal_rv,
        mascara_retina,
    )

    mascara_otsu = segmentar_otsu(
        canal_g,
        mascara_retina,
    )

    mascara_otsu_sense_r = elimina_disc_optic(mascara_otsu, disc_r)
    mascara_otsu_sense_v = elimina_disc_optic(mascara_otsu, disc_v)
    mascara_otsu_sense_rv = elimina_disc_optic(mascara_otsu, disc_rv)

    mascares_7 = {
        "Otsu original": mascara_otsu,
        "Otsu - disc R": mascara_otsu_sense_r,
        "Otsu - disc V": mascara_otsu_sense_v,
        "Otsu - disc R+V": mascara_otsu_sense_rv,
    }

    resultats_7 = comparar_metodes(
        mascares_7,
        mascara_gt,
    )

    for nom_metode in resultats_7:
        imprimir_metriques(
            nom_metode,
            resultats_7[nom_metode],
        )

    print("\nPixels disc R:", disc_r.sum())
    print("Pixels disc V:", disc_v.sum())
    print("Pixels disc R+V:", disc_rv.sum())

    if mostrar_figures:
        plt.figure(figsize=(14, 11))
        plt.suptitle("PROVA BLOC 7 - " + nom_imatge, fontsize=16)

        plt.subplot(3, 4, 1)
        plt.imshow(imatge)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(3, 4, 2)
        plt.imshow(mascara_gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(3, 4, 3)
        plt.imshow(canal_r, cmap="gray")
        plt.title("Canal R")
        plt.axis("off")

        plt.subplot(3, 4, 4)
        plt.imshow(canal_v, cmap="gray")
        plt.title("Canal V")
        plt.axis("off")

        plt.subplot(3, 4, 5)
        plt.imshow(r_candidats, cmap="gray")
        plt.title("Candidats disc R")
        plt.axis("off")

        plt.subplot(3, 4, 6)
        plt.imshow(disc_r, cmap="gray")
        plt.title("Disc detectat R")
        plt.axis("off")

        plt.subplot(3, 4, 7)
        plt.imshow(disc_v, cmap="gray")
        plt.title("Disc detectat V")
        plt.axis("off")

        plt.subplot(3, 4, 8)
        plt.imshow(disc_rv, cmap="gray")
        plt.title("Disc detectat R+V")
        plt.axis("off")

        plt.subplot(3, 4, 9)
        plt.imshow(mascara_otsu, cmap="gray")
        plt.title("Otsu original")
        plt.axis("off")

        plt.subplot(3, 4, 10)
        plt.imshow(mascara_otsu_sense_r, cmap="gray")
        plt.title("Otsu - disc R")
        plt.axis("off")

        plt.subplot(3, 4, 11)
        plt.imshow(mascara_otsu_sense_v, cmap="gray")
        plt.title("Otsu - disc V")
        plt.axis("off")

        plt.subplot(3, 4, 12)
        plt.imshow(mascara_otsu_sense_rv, cmap="gray")
        plt.title("Otsu - disc R+V")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2)
        plt.show()


# ============================================================
# BLOC 8 - MORFOLOGIA MATEMATICA
# ============================================================

print("\n==============================")
print("BLOC 8 - MORFOLOGIA MATEMATICA")
print("==============================")

for nom_imatge, nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 8:", nom_imatge)
    print("----------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    canal_r = canals[0]
    canal_g = canals[1]

    mascara_otsu = segmentar_otsu(
        canal_g,
        mascara_retina,
    )

    _, _, disc_r = detecta_disc_optic(
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3,
    )

    mascara_sense_disc = elimina_disc_optic(
        mascara_otsu,
        disc_r,
    )

    mascara_morfologia = netejar_mascara(
        mascara_sense_disc,
        mida_minima=50,
        mida_forat=50,
    )

    mascares_8 = {
        "Otsu original": mascara_otsu,
        "Otsu - disc": mascara_sense_disc,
        "Otsu - disc + morfologia": mascara_morfologia,
    }

    resultats_8 = comparar_metodes(
        mascares_8,
        mascara_gt,
    )

    for nom_metode in resultats_8:
        imprimir_metriques(
            nom_metode,
            resultats_8[nom_metode],
        )

    if mostrar_figures:
        plt.figure(figsize=(14, 8))
        plt.suptitle("PROVA BLOC 8 - " + nom_imatge, fontsize=16)

        plt.subplot(2, 3, 1)
        plt.imshow(imatge)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(mascara_gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(mascara_otsu, cmap="gray")
        plt.title("Otsu")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(disc_r, cmap="gray")
        plt.title("Disc optic")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(mascara_sense_disc, cmap="gray")
        plt.title("Otsu - disc")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(mascara_morfologia, cmap="gray")
        plt.title("Otsu - disc + morfologia")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2)
        plt.show()


# ============================================================
# BLOC 9 - WATERSHED
# ============================================================

print("\n====================")
print("BLOC 9 - WATERSHED")
print("====================")

for nom_imatge, nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 9:", nom_imatge)
    print("----------------------------------")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    canal_r = canals[0]
    canal_g = canals[1]

    mascara_otsu = segmentar_otsu(
        canal_g,
        mascara_retina,
    )

    _, _, disc_r = detecta_disc_optic(
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3,
    )

    mascara_sense_disc = elimina_disc_optic(
        mascara_otsu,
        disc_r,
    )

    mascara_morfologia = netejar_mascara(
        mascara_sense_disc,
        mida_minima=50,
        mida_forat=50,
    )

    distancia = transformar_distancia(
        mascara_morfologia,
    )

    marcadors = trobar_marcadors(
        distancia,
        mascara_morfologia,
        distancia_minima=5,
    )

    etiquetes = aplicar_watershed(
        mascara_morfologia,
        distancia_minima=5,
    )

    mascara_watershed = etiquetes > 0
    nombre_blobs = np.max(etiquetes)

    print("Nombre de blobs despres de Watershed:", nombre_blobs)

    mascares_9 = {
        "Otsu - disc + morfologia": mascara_morfologia,
        "Watershed binari": mascara_watershed,
    }

    resultats_9 = comparar_metodes(
        mascares_9,
        mascara_gt,
    )

    for nom_metode in resultats_9:
        imprimir_metriques(
            nom_metode,
            resultats_9[nom_metode],
        )

    if mostrar_figures:
        plt.figure(figsize=(14, 8))
        plt.suptitle("PROVA BLOC 9 - " + nom_imatge, fontsize=16)

        plt.subplot(2, 3, 1)
        plt.imshow(mascara_gt, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(mascara_morfologia, cmap="gray")
        plt.title("Mascara morfologia")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(distancia, cmap="gray")
        plt.title("Transformada distancia")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(marcadors, cmap="nipy_spectral")
        plt.title("Marcadors")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(etiquetes, cmap="nipy_spectral")
        plt.title("Watershed - blobs")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(mascara_watershed, cmap="gray")
        plt.title("Watershed binari")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2)
        plt.show()


# ============================================================
# BLOC 10 - FEATURES DELS BLOBS
# ============================================================

print("\n==========================================")
print("BLOC 10 - FEATURES DELS DOS PIPELINES")
print("==========================================")

taules_otsu = []
taules_inrg = []

dades_otsu = {}
dades_inrg = {}

for nom_imatge, nom_gt in imatges:

    print("\n==========================================")
    print("PROVA BLOC 10:", nom_imatge)
    print("==========================================")

    ruta_imatge = ruta_dades / nom_imatge
    ruta_gt = ruta_dades / nom_gt

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt,
    )

    canal_r = canals[0]
    canal_g = canals[1]

    # Pipeline A: Otsu
    mascara_otsu, etiquetes_otsu, disc_otsu = crea_mascara_otsu(
        canal_g,
        canal_r,
        mascara_retina,
    )

    taula_otsu = crea_taula_features(
        imatge,
        canals,
        mascara_otsu,
        mascara_gt,
        nom_imatge=nom_imatge,
        nom_canal="G original",
        nom_metode="Otsu + disc + morfologia + watershed",
        llindar_positiu=llindar_positiu_svm,
        llindar_negatiu=llindar_negatiu_svm,
    )

    print("\n---------- PIPELINE A: OTSU ----------")
    print("Blobs Watershed:", np.max(etiquetes_otsu))
    imprimeix_resum_features(taula_otsu, nom_imatge + " - Otsu")

    imprimir_metriques(
        "PIPELINE A - Otsu abans SVM",
        calcular_metriques(mascara_otsu, mascara_gt),
    )

    taules_otsu.append(taula_otsu)

    dades_otsu[nom_imatge] = {
        "imatge": imatge,
        "gt": mascara_gt,
        "candidats": mascara_otsu,
    }

    # Pipeline B: INRG
    mascara_inrg, etiquetes_inrg, disc_inrg = crea_mascara_inrg(
        canal_g,
        canal_r,
        mascara_retina,
    )

    taula_inrg = crea_taula_features(
        imatge,
        canals,
        mascara_inrg,
        mascara_gt,
        nom_imatge=nom_imatge,
        nom_canal="G original",
        nom_metode="INRG + disc + morfologia + watershed",
        llindar_positiu=llindar_positiu_svm,
        llindar_negatiu=llindar_negatiu_svm,
    )

    print("\n---------- PIPELINE B: INRG ----------")
    print("Blobs Watershed:", np.max(etiquetes_inrg))
    imprimeix_resum_features(taula_inrg, nom_imatge + " - INRG")

    imprimir_metriques(
        "PIPELINE B - INRG abans SVM",
        calcular_metriques(mascara_inrg, mascara_gt),
    )

    taules_inrg.append(taula_inrg)

    dades_inrg[nom_imatge] = {
        "imatge": imatge,
        "gt": mascara_gt,
        "candidats": mascara_inrg,
    }


taula_total_otsu = ajunta_taules(taules_otsu)
taula_total_inrg = ajunta_taules(taules_inrg)

print("\n==========================================")
print("RESUM BLOC 10")
print("==========================================")

print("\nPIPELINE A - OTSU")
print("Blobs totals:", len(taula_total_otsu))
print("Positius:", np.sum(taula_total_otsu["etiqueta"] == 1))
print("Negatius:", np.sum(taula_total_otsu["etiqueta"] == 0))
print("Ambigus:", np.sum(taula_total_otsu["etiqueta"] == -1))

print("\nPIPELINE B - INRG")
print("Blobs totals:", len(taula_total_inrg))
print("Positius:", np.sum(taula_total_inrg["etiqueta"] == 1))
print("Negatius:", np.sum(taula_total_inrg["etiqueta"] == 0))
print("Ambigus:", np.sum(taula_total_inrg["etiqueta"] == -1))


# ============================================================
# BLOC 11 - SVM DELS DOS PIPELINES
# ============================================================

print("\n==========================================")
print("BLOC 11 - SVM DELS DOS PIPELINES")
print("==========================================")

resum_final = []

for nom_test, nom_gt in imatges:

    print("\n\n##########################################")
    print("IMATGE DE TEST:", nom_test)
    print("##########################################")

    resultat_otsu = prova_svm_pipeline(
        taula_total_otsu,
        dades_otsu,
        nom_test,
        "PIPELINE A - OTSU",
    )

    resultat_inrg = prova_svm_pipeline(
        taula_total_inrg,
        dades_inrg,
        nom_test,
        "PIPELINE B - INRG",
    )

    if resultat_otsu is None or resultat_inrg is None:
        continue

    afegeix_resultat(
        resum_final,
        nom_test,
        "OTSU",
        resultat_otsu["metriques_abans"],
        resultat_otsu["metriques_svm"],
    )

    afegeix_resultat(
        resum_final,
        nom_test,
        "INRG",
        resultat_inrg["metriques_abans"],
        resultat_inrg["metriques_svm"],
    )

    imatge_original = dades_otsu[nom_test]["imatge"]
    mascara_gt = dades_otsu[nom_test]["gt"]

    mostra_comparacio_svm(
        nom_test,
        imatge_original,
        mascara_gt,
        resultat_otsu,
        resultat_inrg,
    )


# ============================================================
# MITJANES FINALS
# ============================================================

print("\n==========================================")
print("MITJANES FINALS BLOC 11")
print("==========================================")

metriques_otsu_abans = []
metriques_otsu_svm = []
metriques_inrg_abans = []
metriques_inrg_svm = []

for resultat in resum_final:

    if resultat["pipeline"] == "OTSU":
        metriques_otsu_abans.append(resultat["abans"])
        metriques_otsu_svm.append(resultat["despres"])

    if resultat["pipeline"] == "INRG":
        metriques_inrg_abans.append(resultat["abans"])
        metriques_inrg_svm.append(resultat["despres"])

if len(metriques_otsu_abans) > 0:
    imprimeix_mitjana_metriques(
        "MITJANA OTSU ABANS SVM",
        calcula_mitjana_metriques(metriques_otsu_abans),
    )

    imprimeix_mitjana_metriques(
        "MITJANA OTSU DESPRES SVM",
        calcula_mitjana_metriques(metriques_otsu_svm),
    )

if len(metriques_inrg_abans) > 0:
    imprimeix_mitjana_metriques(
        "MITJANA INRG ABANS SVM",
        calcula_mitjana_metriques(metriques_inrg_abans),
    )

    imprimeix_mitjana_metriques(
        "MITJANA INRG DESPRES SVM",
        calcula_mitjana_metriques(metriques_inrg_svm),
    )

print("\nProva completa acabada.")
