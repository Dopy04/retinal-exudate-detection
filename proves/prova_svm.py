# -*- coding: utf-8 -*-
"""
Prova del Bloc 10 + Bloc 11.

Aquest fitxer va al mateix nivell que main.py.

Estructura esperada:

retina-exudates-piva/
│
├── prova_svm.py
├── main.py
├── src/
├── Exudate-DR2019/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── Exudate-DR2019-Groundtruth/
    ├── 001.jpg
    ├── 002.jpg
    └── ...

Les imatges i les Ground Truth tenen el mateix nom.
Exemple:
    Exudate-DR2019/001.jpg
    Exudate-DR2019-Groundtruth/001.jpg
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.metriques import calcular_metriques, imprimir_metriques
from src.preprocessament_3 import preprocessa_imatge
from src.segmentacio import segmentar_otsu
from src.disc_optic import detecta_disc_optic, elimina_disc_optic
from src.postprocessament import netejar_mascara
from src.watershed import aplicar_watershed

from src.deteccio_blobs import crea_taula_features
from src.deteccio_blobs import imprimeix_resum_features
from src.deteccio_blobs import ajunta_taules

from src.classificador_SVM import separa_train_test_per_imatge
from src.classificador_SVM import entrena_svm
from src.classificador_SVM import imprimeix_features_seleccionades
from src.classificador_SVM import prediu_taula
from src.classificador_SVM import avalua_prediccio_blobs
from src.classificador_SVM import reconstrueix_mascara_svm


# ============================================================
# FUNCIONS PETITES PER A LA PROVA
# ============================================================

def busca_parelles_imatges(ruta_imatges, ruta_gt):
    """
    Busca parelles imatge-GT.

    En aquest cas es molt simple:
        imatge = 001.jpg
        GT     = 001.jpg
    """

    llista_imatges = sorted(ruta_imatges.glob("*.jpg"))

    parelles = []

    for ruta_imatge in llista_imatges:

        ruta_mascara = ruta_gt / ruta_imatge.name

        if ruta_mascara.exists():
            parelles.append((ruta_imatge, ruta_mascara))

    return parelles


def calcula_mitjana_metriques(llista_metriques):
    """
    Calcula la mitjana de les metriques de moltes imatges.
    """

    mitjana = {}

    claus = [
        "precisio",
        "recall",
        "f1",
        "dice",
        "iou",
        "accuracy",
    ]

    for clau in claus:

        valors = []

        for metriques in llista_metriques:
            valors.append(metriques[clau])

        mitjana[clau] = np.mean(valors)

    return mitjana


def imprimeix_mitjana_metriques(nom, metriques):
    """
    Imprimeix les metriques mitjanes.
    """

    print("\n====================================")
    print(nom)
    print("====================================")

    print("Precisio:", round(metriques["precisio"], 4))
    print("Recall:", round(metriques["recall"], 4))
    print("F1:", round(metriques["f1"], 4))
    print("Dice:", round(metriques["dice"], 4))
    print("IoU:", round(metriques["iou"], 4))
    print("Accuracy:", round(metriques["accuracy"], 4))


def crea_mascara_candidata(imatge, mascara_gt, mascara_retina, canals):
    """
    Crea la mascara candidata que passarem al SVM.

    Fem servir la branca simple:
        G original -> Otsu -> eliminar disc optic -> morfologia -> watershed

    Aquesta mascara encara NO es el resultat final.
    Nomes son candidats que el SVM intentara filtrar.
    """

    canal_r = canals[0]
    canal_g = canals[1]

    # Segmentacio inicial amb Otsu sobre el canal G
    mascara_otsu = segmentar_otsu(
        canal_g,
        mascara_retina
    )

    # Detectem el disc optic amb el canal R
    r_suau, r_candidats, disc_r = detecta_disc_optic(
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3
    )

    # Eliminem el disc optic de la mascara candidata
    mascara_sense_disc = elimina_disc_optic(
        mascara_otsu,
        disc_r
    )

    # Neteja morfologica
    mascara_morfologia = netejar_mascara(
        mascara_sense_disc,
        mida_minima=50,
        mida_forat=50
    )

    # Watershed separa alguns blobs units
    etiquetes_ws = aplicar_watershed(
        mascara_morfologia,
        distancia_minima=5
    )

    mascara_watershed = etiquetes_ws > 0

    return mascara_watershed, disc_r


def processa_imatge(ruta_imatge, ruta_gt):
    """
    Processa una imatge i retorna tot el necessari per al SVM.
    """

    imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
        ruta_imatge,
        ruta_gt
    )

    mascara_candidata, disc_r = crea_mascara_candidata(
        imatge,
        mascara_gt,
        mascara_retina,
        canals
    )

    taula = crea_taula_features(
        imatge,
        canals,
        mascara_candidata,
        mascara_gt,
        nom_imatge=ruta_imatge.name,
        nom_canal="G original",
        nom_metode="Otsu + disc + morfologia + watershed",
        llindar_positiu=0.30,
        llindar_negatiu=0.05
    )

    return imatge, mascara_gt, mascara_candidata, disc_r, taula


# ============================================================
# CONFIGURACIO
# ============================================================

print("\n====================================")
print("PROVA SVM DR2019")
print("====================================")

# Aquest fitxer esta al mateix nivell que main.py i src.
ruta_projecte = Path(__file__).resolve().parent

ruta_imatges = ruta_projecte / "Exudate-DR2019"
ruta_gt = ruta_projecte / "Exudate-DR2019-Groundtruth"

print("Carpeta imatges:", ruta_imatges)
print("Carpeta Ground Truth:", ruta_gt)

# Si alguna imatge te el disc optic molt mal detectat, posa-la aqui.
# Exemple:
# imatges_descartades = ["012.jpg", "044.jpg"]
imatges_descartades = [
    "013.jpg",
    "015.jpg",
    "016.jpg",
    "025.jpg",
    "038.jpg",
    "041.jpg",
    "043.jpg",
    "046.jpg",
    "048.jpg",
    "057.jpg",
    "064.jpg",
    "068.jpg",
    "073.jpg",
    "074.jpg",
    "099.jpg",
    "101.jpg",
    "106.jpg",
    "26.jpg",
    "27.jpg",
    "060.jpg",
    "083.jpg"]

parelles = busca_parelles_imatges(
    ruta_imatges,
    ruta_gt
)

# Eliminem les imatges descartades, pero sense borrar-les de la carpeta.
parelles_bones = []

for ruta_imatge, ruta_mascara in parelles:

    if ruta_imatge.name not in imatges_descartades:
        parelles_bones.append((ruta_imatge, ruta_mascara))

parelles = parelles_bones

print("Nombre de parelles trobades:", len(parelles))

if len(parelles) == 0:
    raise ValueError("No s'han trobat imatges. Revisa les carpetes i els noms.")


# ------------------------------------------------------------
# Primeres 50 per test i ultimes 50 per train
# ------------------------------------------------------------

parelles_test = parelles[:50]
parelles_train = parelles[-50:]

# Si el dataset te menys de 100 imatges, pot haver-hi solapament.
# Per evitar fer trampa, traiem del train les que ja estan al test.
noms_test = []

for ruta_imatge, ruta_mascara in parelles_test:
    noms_test.append(ruta_imatge.name)

parelles_train_sense_solapament = []

for ruta_imatge, ruta_mascara in parelles_train:

    if ruta_imatge.name not in noms_test:
        parelles_train_sense_solapament.append((ruta_imatge, ruta_mascara))

parelles_train = parelles_train_sense_solapament

print("Imatges test:", len(parelles_test))
print("Imatges train:", len(parelles_train))


# ============================================================
# BLOC 10: CREEM FEATURES
# ============================================================

taules_features = []

mascares_candidats = {}
mascares_gt = {}
imatges_originals = {}
discs_detectats = {}

parelles_total = parelles_train + parelles_test

for ruta_imatge, ruta_mascara in parelles_total:

    print("\n------------------------------------")
    print("Processant:", ruta_imatge.name)
    print("------------------------------------")

    imatge, mascara_gt, mascara_candidata, disc_r, taula = processa_imatge(
        ruta_imatge,
        ruta_mascara
    )

    imprimeix_resum_features(
        taula,
        ruta_imatge.name
    )

    taules_features.append(taula)

    mascares_candidats[ruta_imatge.name] = mascara_candidata
    mascares_gt[ruta_imatge.name] = mascara_gt
    imatges_originals[ruta_imatge.name] = imatge
    discs_detectats[ruta_imatge.name] = disc_r


taula_total = ajunta_taules(
    taules_features
)

print("\n====================================")
print("TAULA TOTAL FEATURES")
print("====================================")

print("Nombre total de blobs:", len(taula_total))
print("Positius totals:", np.sum(taula_total["etiqueta"] == 1))
print("Negatius totals:", np.sum(taula_total["etiqueta"] == 0))
print("Ambigus totals:", np.sum(taula_total["etiqueta"] == -1))


# ============================================================
# BLOC 11: SEPARACIO TRAIN / TEST
# ============================================================

taula_train, taula_test = separa_train_test_per_imatge(
    taula_total,
    noms_test
)

print("\n====================================")
print("SEPARACIO TRAIN / TEST")
print("====================================")

print("Blobs train:", len(taula_train))
print("Blobs test:", len(taula_test))

print("Train positius:", np.sum(taula_train["etiqueta"] == 1))
print("Train negatius:", np.sum(taula_train["etiqueta"] == 0))
print("Train ambigus:", np.sum(taula_train["etiqueta"] == -1))

print("Test positius:", np.sum(taula_test["etiqueta"] == 1))
print("Test negatius:", np.sum(taula_test["etiqueta"] == 0))
print("Test ambigus:", np.sum(taula_test["etiqueta"] == -1))


# ============================================================
# ENTRENAMENT SVM
# ============================================================

print("\n====================================")
print("ENTRENANT SVM")
print("====================================")

escalador, selector, model = entrena_svm(
    taula_train,
    k_chi2=5
)

imprimeix_features_seleccionades(
    selector
)


# ============================================================
# AVALUACIO A NIVELL DE BLOBS
# ============================================================

taula_test_predita = prediu_taula(
    taula_test,
    escalador,
    selector,
    model
)

avalua_prediccio_blobs(
    taula_test_predita
)


# ============================================================
# RECONSTRUCCIO DE MASCARES I METRIQUES
# ============================================================

metriques_abans_llista = []
metriques_despres_llista = []

max_figures = 20
figures_mostrades = 0

for nom_test in noms_test:

    print("\n------------------------------------")
    print("RESULTAT:", nom_test)
    print("------------------------------------")

    mascara_candidata = mascares_candidats[nom_test]
    mascara_gt = mascares_gt[nom_test]
    imatge = imatges_originals[nom_test]

    taula_imatge = taula_test_predita[
        taula_test_predita["nom_imatge"] == nom_test
    ]

    mascara_svm = reconstrueix_mascara_svm(
        mascara_candidata,
        taula_imatge
    )

    metriques_abans = calcular_metriques(
        mascara_candidata,
        mascara_gt
    )

    metriques_despres = calcular_metriques(
        mascara_svm,
        mascara_gt
    )

    imprimir_metriques(
        "Abans SVM",
        metriques_abans
    )

    imprimir_metriques(
        "Despres SVM",
        metriques_despres
    )

    metriques_abans_llista.append(metriques_abans)
    metriques_despres_llista.append(metriques_despres)

    # Mostrem nomes unes quantes figures per no saturar Spyder.
    if figures_mostrades < max_figures:

        plt.figure(figsize=(14, 5))

        plt.suptitle(
            "SVM - " + nom_test,
            fontsize=16
        )

        plt.subplot(1, 3, 1)
        plt.imshow(imatge)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mascara_gt, cmap="gray", alpha=1)
        plt.imshow(mascara_candidata, cmap="Greens", alpha=0.5)
        plt.title("Abans SVM")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(mascara_gt, cmap="gray", alpha=1)
        plt.imshow(mascara_svm, cmap="Greens", alpha=0.5)
        plt.title("Despres SVM")
        plt.axis("off")

        plt.tight_layout(
            rect=[0, 0, 1, 0.90]
        )

        plt.show()

        figures_mostrades = figures_mostrades + 1


# ============================================================
# MITJANES FINALS
# ============================================================

mitjana_abans = calcula_mitjana_metriques(
    metriques_abans_llista
)

mitjana_despres = calcula_mitjana_metriques(
    metriques_despres_llista
)

imprimeix_mitjana_metriques(
    "MITJANA METRIQUES ABANS SVM",
    mitjana_abans
)

imprimeix_mitjana_metriques(
    "MITJANA METRIQUES DESPRES SVM",
    mitjana_despres
)

print("\nProva acabada.")
