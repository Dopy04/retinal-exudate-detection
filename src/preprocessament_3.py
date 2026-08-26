from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk


# -------------------------------------------------------------------------
# 1. Carrega d'imatges
# -------------------------------------------------------------------------

def carrega_imatge_rgb(ruta):
    """
    Carrega una imatge RGB i la normalitza a float entre 0 i 1.
    """
    imatge = imread(ruta)

    if imatge.ndim != 3:
        raise ValueError("La imatge hauria de tenir 3 canals RGB.")

    imatge = imatge[:, :, :3]
    imatge = imatge.astype(np.float64)

    if imatge.max() > 1.0:
        imatge = imatge / 255.0

    return np.clip(imatge, 0.0, 1.0)



def carrega_mascara_gt(ruta, llindar=0.5):
    """
    Carrega la màscara ground truth i la passa a booleana.
    """
    mascara = imread(ruta)

    if mascara.ndim == 3:
        mascara = rgb2gray(mascara[:, :, :3])
    else:
        mascara = mascara.astype(np.float64)
        if mascara.max() > 1.0:
            mascara = mascara / 255.0

    mascara = mascara > llindar

    return mascara


# -------------------------------------------------------------------------
# 2. Preprocessament bàsic
# -------------------------------------------------------------------------

def calcula_luma(imatge):
    """
    Calcula la imatge en escala de grisos amb la fórmula de luminància.
    """
    r = imatge[:, :, 0]
    g = imatge[:, :, 1]
    b = imatge[:, :, 2]

    luma = 0.299 * r + 0.587 * g + 0.114 * b

    return luma



def crea_mascara_retina(imatge, llindar=0.05):
    """
    Crea una màscara per separar la retina del fons negre.
    """
    luma = calcula_luma(imatge)

    mascara_retina = luma > llindar
    mascara_retina = remove_small_objects(mascara_retina, min_size=500)
    mascara_retina = binary_closing(mascara_retina, disk(5))
    mascara_retina = remove_small_holes(mascara_retina, area_threshold=500)

    return mascara_retina



def aplica_mascara_retina(imatge, mascara_retina):
    """
    Posa a negre els píxels que queden fora de la retina.
    """
    resultat = np.copy(imatge)

    # Amb ~ invertim la màscara:
    # dins de la retina (True) passa a False
    # fora de la retina (False) passa a True
    resultat[~mascara_retina] = 0

    return resultat


# -------------------------------------------------------------------------
# 3. Extracció de canals
# -------------------------------------------------------------------------

def extreu_canals_rgb(imatge):
    """
    Extreu els canals R, G i B.
    """
    r = imatge[:, :, 0]
    g = imatge[:, :, 1]
    b = imatge[:, :, 2]

    return r, g, b



def extreu_canals_hsv(imatge):
    """
    Passa la imatge de RGB a HSV i retorna H, S i V.
    """
    hsv = rgb2hsv(imatge)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    return h, s, v



def extreu_canals_lab(imatge):
    """
    Passa la imatge de RGB a Lab i retorna L, a i b.
    """
    lab = rgb2lab(imatge)

    l_lab = lab[:, :, 0]
    a_lab = lab[:, :, 1]
    b_lab = lab[:, :, 2]

    return l_lab, a_lab, b_lab



def extreu_canals(imatge):
    """
    Extreu els canals principals que farem servir després.

    Ordre de sortida:
    [0] r
    [1] g
    [2] b
    [3] luma
    [4] h
    [5] s
    [6] v
    [7] l_lab
    [8] a_lab
    [9] b_lab
    """
    r, g, b = extreu_canals_rgb(imatge)
    luma = calcula_luma(imatge)
    h, s, v = extreu_canals_hsv(imatge)
    l_lab, a_lab, b_lab = extreu_canals_lab(imatge)

    return r, g, b, luma, h, s, v, l_lab, a_lab, b_lab


# -------------------------------------------------------------------------
# 4. Funció principal del bloc 3
# -------------------------------------------------------------------------

def preprocessa_imatge(ruta_imatge, ruta_mascara):
    """
    Carrega una imatge i la seua màscara, crea la màscara de retina,
    aplica el fons negre i extreu els canals.

    Retorna, en aquest ordre:
    [0] imatge original normalitzada
    [1] mascara ground truth
    [2] mascara de retina
    [3] imatge amb el fons fora de la retina a negre
    [4] canals
    """
    imatge = carrega_imatge_rgb(ruta_imatge)
    mascara_gt = carrega_mascara_gt(ruta_mascara)

    if imatge.shape[:2] != mascara_gt.shape:
        print("Atenció: la imatge i la màscara no tenen la mateixa mida.")

    mascara_retina = crea_mascara_retina(imatge)
    imatge_retina = aplica_mascara_retina(imatge, mascara_retina)
    canals = extreu_canals(imatge_retina)

    return imatge, mascara_gt, mascara_retina, imatge_retina, canals


# -------------------------------------------------------------------------
# 5. Visualització bàsica
# -------------------------------------------------------------------------

def visualitza_preprocessament(imatge, mascara_gt, mascara_retina, canals):
    """
    Mostra una visualització bàsica per comprovar que el preprocessament va bé.
    """
    g = canals[1]
    luma = canals[3]
    v = canals[6]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(mascara_gt, cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(mascara_retina, cmap="gray")
    plt.title("Mascara retina")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(g, cmap="gray")
    plt.title("Canal G")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(luma, cmap="gray")
    plt.title("Luma")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(v, cmap="gray")
    plt.title("Canal V")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Exemple d'ús en el main
# -------------------------------------------------------------------------
# ruta_imatge = Path("Exudate-JSIEC/001.jpg")
# ruta_mascara = Path("Exudate-JSIEC-Groundtruth/001.jpg")
#
# resultat = preprocessa_imatge(ruta_imatge, ruta_mascara)
#
# imatge = resultat[0]
# mascara_gt = resultat[1]
# mascara_retina = resultat[2]
# imatge_retina = resultat[3]
# canals = resultat[4]
#
# visualitza_preprocessament(imatge, mascara_gt, mascara_retina, canals)
