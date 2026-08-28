from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 0. Imports des de proves/
# ------------------------------------------------------------

print("Python esta buscant des de:")
print(Path.cwd())

# Com el fitxer esta dins de proves/, pugem una carpeta i entrem a src/
sys.path.append("../src")


from preprocessament_3 import preprocessa_imatge
from mejora_contraste_5 import millora_contrast
from segmentacio import obtenir_mascares_baseline
from postprocessament import netejar_mascara
from metriques import calcular_metriques, imprimir_metriques


# ------------------------------------------------------------
# 1. Rutes de la imatge i de la ground truth
# ------------------------------------------------------------

# Canvia aquestes dues rutes per provar una altra imatge.
ruta_imatge = Path("../data_samples/Exudate_DR2019_001.jpg")
ruta_mascara = Path("../data_samples/Exudate_DR2019_001_Groundtruth.jpg")

print("Existeix imatge:", ruta_imatge.exists())
print("Existeix mascara:", ruta_mascara.exists())

if not ruta_imatge.exists() or not ruta_mascara.exists():
    raise FileNotFoundError("No trobe la imatge o la mascara. Revisa la ruta.")


# ------------------------------------------------------------
# 2. Bloc 3: preprocessament
# ------------------------------------------------------------

imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
    ruta_imatge,
    ruta_mascara,
)

print("Mida imatge:", imatge.shape)
print("Mida ground truth:", mascara_gt.shape)
print("Mida mascara retina:", mascara_retina.shape)


# ------------------------------------------------------------
# 3. Bloc 5: millora de contrast
# ------------------------------------------------------------

contrast = millora_contrast(
    imatge_retina,
    mascara_retina,
    canals,
)

# Canal V original del preprocessament
v = canals[6]

# Canals millorats del bloc 5
v_clahe = contrast[4]
v_kmmrc = contrast[5]
v_kmmrc_clahe = contrast[6]


# ------------------------------------------------------------
# 4. Canals que volem provar
# ------------------------------------------------------------

canals_prova = [
    ("V original", v),
    ("V CLAHE", v_clahe),
    ("V KMMRC", v_kmmrc),
    ("V KMMRC + CLAHE", v_kmmrc_clahe),
]


# Aquí guardarem totes les màscares finals per comparar després.
mascares_finals = {}


# ------------------------------------------------------------
# 5. Segmentació per cada canal
# ------------------------------------------------------------

for nom_canal, canal in canals_prova:

    print("\n====================================")
    print("Canal:", nom_canal)
    print("====================================")

    # Obtenim totes les màscares base:
    # Otsu, Llindar_fix, Adaptatiu, Sauvola, Kmeans
    mascares = obtenir_mascares_baseline(
        canal,
        mascara_retina,
    )

    # Netejem totes les màscares
    mascares_netes = {}

    for nom_metode in mascares:

        mascara = mascares[nom_metode]

        mascara_neta = netejar_mascara(
            mascara,
            mida_minima=50,
            mida_forat=50,
        )

        mascares_netes[nom_metode] = mascara_neta

        # Guardem la màscara final amb un nom complet:
        # per exemple: "V CLAHE - Otsu"
        nom_resultat = nom_canal + " - " + nom_metode
        mascares_finals[nom_resultat] = mascara_neta

    # --------------------------------------------------------
    # Representació de les màscares netes d'aquest canal
    # --------------------------------------------------------

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 4, 1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(mascara_gt, cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(canal, cmap="gray")
    plt.title(nom_canal)
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(mascara_retina, cmap="gray")
    plt.title("Mascara retina")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(mascares_netes["Otsu"], cmap="gray")
    plt.title("Otsu net")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(mascares_netes["Llindar_fix"], cmap="gray")
    plt.title("Llindar fix net")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(mascares_netes["Adaptatiu"], cmap="gray")
    plt.title("Adaptatiu net")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(mascares_netes["Sauvola"], cmap="gray")
    plt.title("Sauvola net")
    plt.axis("off")

    plt.suptitle("Segmentacio neta - " + nom_canal)
    plt.tight_layout()
    plt.show()

    # Com que en una figura 2x4 no cabia K-means amb tot l'anterior,
    # fem una segona figura curta per veure K-means i la seua superposicio.
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(canal, cmap="gray")
    plt.title(nom_canal)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mascares_netes["Kmeans"], cmap="gray")
    plt.title("K-means net")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(imatge)
    plt.imshow(mascares_netes["Kmeans"], cmap="Reds", alpha=0.40)
    plt.title("K-means sobre original")
    plt.axis("off")

    plt.suptitle("K-means - " + nom_canal)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. Comparació de totes les màscares finals amb la GT
# ------------------------------------------------------------

print("\n\n====================================")
print("RESULTATS FINALS COMPARATS AMB GT")
print("====================================")

resultats_metriques = []

for nom_resultat in mascares_finals:

    mascara_final = mascares_finals[nom_resultat]

    metriques = calcular_metriques(
        mascara_final,
        mascara_gt,
    )

    imprimir_metriques(
        nom_resultat,
        metriques,
    )

    resultats_metriques.append(
        (
            nom_resultat,
            metriques["precisio"],
            metriques["recall"],
            metriques["f1"],
            metriques["dice"],
            metriques["iou"],
            metriques["accuracy"],
        )
    )


# ------------------------------------------------------------
# 7. Resum ordenat per F1
# ------------------------------------------------------------

resultats_metriques = sorted(
    resultats_metriques,
    key=lambda x: x[3],
    reverse=True,
)

print("\n\n====================================")
print("RESUM ORDENAT PER F1")
print("====================================")

for resultat in resultats_metriques:

    nom_resultat = resultat[0]
    precisio = resultat[1]
    recall = resultat[2]
    f1 = resultat[3]
    dice = resultat[4]
    iou = resultat[5]
    accuracy = resultat[6]

    print("\n", nom_resultat)
    print("Precisio:", round(precisio, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("Dice:", round(dice, 4))
    print("IoU:", round(iou, 4))
    print("Accuracy:", round(accuracy, 4))


# ------------------------------------------------------------
# 8. Representem les millors màscares
# ------------------------------------------------------------

millors = resultats_metriques[:4]

plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(imatge)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(mascara_gt, cmap="gray")
plt.title("Ground truth")
plt.axis("off")

for i in range(len(millors)):

    nom_resultat = millors[i][0]
    mascara = mascares_finals[nom_resultat]

    plt.subplot(2, 3, i + 3)
    plt.imshow(imatge)
    plt.imshow(mascara, cmap="Reds", alpha=0.40)
    plt.title(nom_resultat)
    plt.axis("off")

plt.suptitle("Millors resultats segons F1")
plt.tight_layout()
plt.show()
