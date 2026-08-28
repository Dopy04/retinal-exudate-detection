from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. Imports des de proves/
# ------------------------------------------------------------

print("Python esta buscant des de:")
print(Path.cwd())

sys.path.append("../src")

from preprocessament_3 import preprocessa_imatge
from mejora_contraste_5 import millora_contrast
from segmentacio import obtenir_mascares_baseline
from nick_inrg import invertir_canal, segmentar_nick, segmentar_inrg
from postprocessament import netejar_mascara
from disc_optic import detecta_disc_optic, elimina_disc_optic


# ------------------------------------------------------------
# 1. Rutes
# ------------------------------------------------------------

# Canvia aquestes dues rutes
ruta_imatge = Path("../data_samples/Exudate_DR2019_001.jpg")
ruta_mascara = Path("../data_samples/Exudate_DR2019_001_Groundtruth.jpg")

print("Existeix imatge:", ruta_imatge.exists())
print("Existeix mascara:", ruta_mascara.exists())

if not ruta_imatge.exists() or not ruta_mascara.exists():
    raise FileNotFoundError("No trobe la imatge o la mascara. Revisa la ruta.")


# ------------------------------------------------------------
# 2. Preprocessament
# ------------------------------------------------------------

imatge, mascara_gt, mascara_retina, imatge_retina, canals = preprocessa_imatge(
    ruta_imatge,
    ruta_mascara,
)


# ------------------------------------------------------------
# 3. Millora de contrast
# ------------------------------------------------------------

contrast = millora_contrast(
    imatge_retina,
    mascara_retina,
    canals,
)

r = canals[0]
v = canals[6]

v_original = canals[6]
v_clahe = contrast[4]
v_kmmrc = contrast[5]
v_kmmrc_clahe = contrast[6]


# ------------------------------------------------------------
# 4. Detectem el disc optic una sola vegada
# ------------------------------------------------------------

# Fem servir R + V per detectar el disc optic
canal_disc = 0.5 * r + 0.5 * v
canal_disc = np.clip(canal_disc, 0.0, 1.0)

resultat_disc = detecta_disc_optic(
    canal_disc,
    mascara_retina,
    percentil=98,
    sigma=2,
)

mascara_disc = resultat_disc[2]


# ------------------------------------------------------------
# 5. Canals que volem provar
# ------------------------------------------------------------

llista_canals = [
    ("V original", v_original),
    ("V CLAHE", v_clahe),
    ("V KMMRC", v_kmmrc),
    ("V KMMRC + CLAHE", v_kmmrc_clahe),
]


# ------------------------------------------------------------
# 6. Per cada canal: 7 mètodes + neteja + eliminar disc optic
# ------------------------------------------------------------

for nom_canal, canal_segmentacio in llista_canals:

    # --------------------------------------------------------
    # 6.1 Màscares baseline del bloc 4
    # --------------------------------------------------------

    mascares = obtenir_mascares_baseline(
        canal_segmentacio,
        mascara_retina,
    )

    # --------------------------------------------------------
    # 6.2 Afegim NICK i INRG
    # --------------------------------------------------------

    canal_invertit = invertir_canal(canal_segmentacio)

    mascara_nick = segmentar_nick(
        canal_invertit,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
    )

    mascara_inrg = segmentar_inrg(
        canal_segmentacio,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
        iteracions=3,
        delta=0.1,
    )

    mascares["NICK"] = mascara_nick
    mascares["INRG"] = mascara_inrg

    # --------------------------------------------------------
    # 6.3 Neteja + eliminació del disc optic
    # --------------------------------------------------------

    mascares_finals = {}

    for nom_metode in mascares:

        mascara = mascares[nom_metode]

        mascara_neta = netejar_mascara(
            mascara,
            mida_minima=50,
            mida_forat=50,
        )

        mascara_sense_disc = elimina_disc_optic(
            mascara_neta,
            mascara_disc,
        )

        mascares_finals[nom_metode] = mascara_sense_disc

    # --------------------------------------------------------
    # 6.4 Representació
    # --------------------------------------------------------

    plt.figure(figsize=(16, 8))

    # 1) Canal sobre el qual estem aplicant els mètodes
    plt.subplot(2, 4, 1)
    plt.imshow(canal_segmentacio, cmap="gray")
    plt.title("Canal: " + nom_canal)
    plt.axis("off")

    # 2) Otsu
    plt.subplot(2, 4, 2)
    
    plt.imshow(mascara_gt, cmap="gray", alpha=1)
    plt.imshow(mascares_finals["Otsu"], cmap="Greens", alpha=0.5)
    plt.title("Otsu")
    plt.axis("off")

    # 3) Llindar fix
    plt.subplot(2, 4, 3)
    
    plt.imshow(mascara_gt, cmap="gray", alpha=1)
    plt.imshow(mascares_finals["Llindar_fix"], cmap="Greens", alpha=0.5)
    plt.title("Llindar fix")
    plt.axis("off")

    # 4) Adaptatiu
    plt.subplot(2, 4, 4)
    
    plt.imshow(mascara_gt, cmap="gray", alpha=1)
    plt.imshow(mascares_finals["Adaptatiu"], cmap="Greens", alpha=0.5)
    plt.title("Adaptatiu")
    plt.axis("off")

    # 5) Sauvola
    plt.subplot(2, 4, 5)
    
    plt.imshow(mascara_gt, cmap="gray", alpha=1)
    plt.imshow(mascares_finals["Sauvola"], cmap="Greens", alpha=0.5)
    plt.title("Sauvola")
    plt.axis("off")

    # 6) K-means
    plt.subplot(2, 4, 6)
    
    plt.imshow(mascara_gt, cmap="gray", alpha=1)
    plt.imshow(mascares_finals["Kmeans"], cmap="Greens", alpha=0.5)
    plt.title("K-means")
    plt.axis("off")

    # 7) NICK
    plt.subplot(2, 4, 7)
    
    plt.imshow(mascara_gt, cmap="gray", alpha=1)
    plt.imshow(mascares_finals["NICK"], cmap="Greens", alpha=0.5)
    plt.title("NICK")
    plt.axis("off")

    # 8) INRG
    plt.subplot(2, 4, 8)
    
    plt.imshow(mascara_gt, cmap="gray", alpha=1)
    plt.imshow(mascares_finals["INRG"], cmap="Greens", alpha=0.5)
    plt.title("INRG")
    plt.axis("off")

    plt.suptitle("Resultats finals - " + nom_canal)
    plt.tight_layout()
    plt.show()