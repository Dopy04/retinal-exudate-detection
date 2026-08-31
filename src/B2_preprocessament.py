
"""
Bloc 2: Preprocessament de les imatges de retina.

En aquest bloc carreguem la imatge i el seu Ground Truth, eliminem el fons
negre que queda fora de la retina i extraiem diferents canals de color.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
from skimage.transform import resize


# Carrega d'imatges  i preprocessament bàsic:


def carrega_imatge_rgb(ruta):
    """
    Carrega una imatge RGB i la normalitza a float entre 0 i 1.
    """
    imatge = imread(ruta)

    if imatge.ndim != 3:
        raise ValueError("La imatge hauria de tenir 3 canals RGB.")
        
    # Ens quedem només amb els tres canals RGB per si la imatge
    # conté també un canal alfa.
    
    imatge = imatge[:, :, :3]
    imatge = imatge.astype(np.float64)

    #passem la imatge a float i la deixem entre 0 i 1
    if imatge.max() > 1.0:
        imatge = imatge / 255.0

    return np.clip(imatge, 0.0, 1.0)



def carrega_mascara_gt(ruta, llindar=0.5):
    """
    Carrega la màscara Ground Truth i la passa a booleana.
    """
    mascara = imread(ruta)

    # Si el Ground Truth és RGB el passem a gris.
    if mascara.ndim == 3:
        mascara = rgb2gray(mascara[:, :, :3])
    else:
        mascara = mascara.astype(np.float64)
        if mascara.max() > 1.0:
            mascara = mascara / 255.0

    #Definim un llindar per convertir la imatge a booleana, valors<llindar són True
    # i  valors>llindar són False
    
    mascara = mascara > llindar

    return mascara



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
    
    # Amb un llindar baix separem la zona il·luminada de la retina del fons negre
    mascara_retina = luma > llindar
    
    #busquem uan forma continua així que eliminem defectes
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


# Extracció de canals


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

    L'ordre és:
    R, G, B, Luma, H, S, V, L, a, b.
    """
    
    r, g, b = extreu_canals_rgb(imatge)
    luma = calcula_luma(imatge)
    h, s, v = extreu_canals_hsv(imatge)
    l_lab, a_lab, b_lab = extreu_canals_lab(imatge)

    return r, g, b, luma, h, s, v, l_lab, a_lab, b_lab



def preprocessa_imatge(ruta_imatge, ruta_mascara):
    """
    Fem tot el preprocessament inicial d'una imatge
    
    Retornem la imatge, el Ground Truth, la màscara de retina,
    la imatge sense el fons i els canals.
    """
    imatge = carrega_imatge_rgb(ruta_imatge)
    mascara_gt = carrega_mascara_gt(ruta_mascara)

    # Ens hem trobat casos on la imatge i el Ground Truth tenen dimensions
    # diferents. Adaptem la imatge a la mida de la màscara.
    if imatge.shape[:2] != mascara_gt.shape:

        imatge = resize(imatge, (mascara_gt.shape[0], mascara_gt.shape[1], 3), preserve_range=True, anti_aliasing=True)
        # Redimensionem la imatge al tamany de la màscara Groundtruth
        imatge = np.clip(imatge, 0.0, 1.0)
        
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
    
"""Prova de redimensió imatge DR2019

ruta_imatge = Path("../data_samples/Exudate_DR2019_001.jpg")
ruta_mascara = Path("../data_samples/Exudate_DR2019_001_Groundtruth.jpg")


resultat = preprocessa_imatge(ruta_imatge, ruta_mascara)

imatge = resultat[0]
mascara_gt = resultat[1]
mascara_retina = resultat[2]
imatge_retina = resultat[3]
canals = resultat[4]

if imatge.shape[:2] == mascara_gt.shape:
    print("TRue",imatge.shape, mascara_gt.shape)
     
visualitza_preprocessament(imatge, mascara_gt, mascara_retina, canals)
"""