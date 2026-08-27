"""
Bloc 5 - Millora de contrast del projecte PIVA Retina.

Objectiu:
    - Aplicar equalitzacio d'histograma i CLAHE a canals 2D.
    - Aplicar una versio de KMMRC basada en el paper de 2022.
    - Deixar canals millorats preparats per al bloc de segmentacio.
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.color import rgb2hsv, hsv2rgb
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter


# -------------------------------------------------------------------------
# 1. Funcions auxiliars per aplicar contrast nomes dins la retina
# -------------------------------------------------------------------------

def prepara_canal(canal, mascara_retina):
    """
    Prepara un canal abans d'aplicar contrast.

    Fora de la retina no deixem zeros, perque el fons negre podria afectar
    l'equalitzacio. Posem la mitjana dels pixels que estan dins la retina.
    """
    canal_preparat = np.copy(canal).astype(np.float64)

    valor_fons = np.median(canal_preparat[mascara_retina])
    canal_preparat[~mascara_retina] = valor_fons
    
    canal_preparat = np.clip(canal_preparat, 0.0, 1.0)
    
    return canal_preparat


# -------------------------------------------------------------------------
# 2. Equalitzacio i CLAHE
# -------------------------------------------------------------------------

def equalitza_histograma(canal, mascara_retina):
    """
    Aplica equalitzacio global d'histograma a un canal.

    Es una prova senzilla per comparar amb CLAHE. Despres tornem a posar
    el fons exterior de la retina a zero.
    """
    canal_preparat = prepara_canal(canal, mascara_retina)

    canal_eq = equalize_hist(canal_preparat, mask=mascara_retina)
    canal_eq[~mascara_retina] = 0

    return np.clip(canal_eq, 0.0, 1.0)



def aplica_clahe(canal, mascara_retina, clip_limit=0.01):
    """
    Aplica CLAHE a un canal 2D.

    CLAHE millora el contrast de forma local, i per a imatges de retina
    pot ajudar quan la il·luminacio no es uniforme.
    """
    canal_preparat = prepara_canal(canal, mascara_retina)

    canal_clahe = equalize_adapthist(
        canal_preparat,
        clip_limit=clip_limit,
    )

    canal_clahe[~mascara_retina] = 0

    return np.clip(canal_clahe, 0.0, 1.0)


# -------------------------------------------------------------------------
# 3. KMMRC basat en el paper de 2022
# -------------------------------------------------------------------------

def aplica_kmmrc(imatge_retina, mascara_retina, k=5, beta=1.5):
    """
    Aplica KMMRC sobre el canal V de HSV.

    Idea del paper:
        1. Convertir RGB a HSV.
        2. Treballar amb el canal V, que representa la intensitat/brillantor.
        3. Dividir V en k regions amb K-means.
        4. Millorar cada regio amb una transformacio lineal al voltant
           de la seua mitjana.
        5. Reconstruir la imatge RGB amb H, S i el nou V.

    Retorna:
        v_kmmrc, imatge_kmmrc
    """
    hsv = rgb2hsv(imatge_retina)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    v_nou = np.zeros_like(v)

    # Apliquem K-means nomes als pixels de dins de la retina.
    
    valors_v = v[mascara_retina].reshape(-1, 1)
    #El -1 significa: calcula automáticament quantes files necessitem.
    #El 1 significa: una sola característica per píxel.

    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    etiquetes = model.fit_predict(valors_v)

    valors_v = valors_v.ravel()
    #Tornem a posar valors_v com un vector 1D.
    valors_v_nous = np.copy(valors_v)

    for i in range(k):
        posicions = etiquetes == i
        valors_regio = valors_v[posicions]

        mitjana = np.mean(valors_regio)

        # Realç lineal inspirat en el pseudocodi KMMRC:
        # C = M + beta*(V_i - M)
        # V_i_nou = V_i + C
        contrast = mitjana + beta * (valors_regio - mitjana)
        valors_v_nous[posicions] = valors_regio + contrast

    valors_v_nous = np.clip(valors_v_nous, 0.0, 1.0)

    v_nou[mascara_retina] = valors_v_nous
    v_nou[~mascara_retina] = 0

    hsv_nou = np.zeros_like(hsv)
    hsv_nou[:, :, 0] = h
    hsv_nou[:, :, 1] = s
    hsv_nou[:, :, 2] = v_nou

    imatge_kmmrc = hsv2rgb(hsv_nou)
    imatge_kmmrc[~mascara_retina] = 0

    return v_nou, np.clip(imatge_kmmrc, 0.0, 1.0)



# -------------------------------------------------------------------------
# 5. Funcio principal del bloc 5
# -------------------------------------------------------------------------

def millora_contrast(imatge_retina, mascara_retina, canals):
    """
    Aplica els metodes principals de millora de contrast.

    canals ve del bloc 3 amb aquest ordre:
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

    Retorna, en aquest ordre:
        [0] luma_eq
        [1] luma_clahe
        [2] g_clahe
        [3] b_clahe
        [4] v_clahe
        [5] v_kmmrc
        [6] v_kmmrc_clahe
        [7] v_kmmrc_clahe_filtrat
        [8] imatge_kmmrc
    """
    g = canals[1]
    b = canals[2]
    luma = canals[3]
    v = canals[6]

    luma_eq = equalitza_histograma(luma, mascara_retina)
    luma_clahe = aplica_clahe(luma, mascara_retina)

    g_clahe = aplica_clahe(g, mascara_retina)
    b_clahe = aplica_clahe(b, mascara_retina)
    v_clahe = aplica_clahe(v, mascara_retina)

    v_kmmrc, imatge_kmmrc = aplica_kmmrc(imatge_retina, mascara_retina)

    v_kmmrc_clahe = aplica_clahe(v_kmmrc, mascara_retina)


    return (luma_eq, luma_clahe, g_clahe, b_clahe, v_clahe, v_kmmrc, v_kmmrc_clahe, imatge_kmmrc)


# -------------------------------------------------------------------------
# 6. Visualitzacio basica
# -------------------------------------------------------------------------

def visualitza_contrast(imatge_retina, canals, contrast):
    """
    Mostra una comparacio visual dels canals originals i millorats.
    """
    luma = canals[3]
    v = canals[6]

    luma_clahe = contrast[1]
    g_clahe = contrast[2]
    b_clahe = contrast[3]
    v_clahe = contrast[4]
    v_kmmrc = contrast[5]
    v_kmmrc_clahe = contrast[6]

    plt.figure(figsize=(13, 8))

    plt.subplot(2, 4, 1)
    plt.imshow(imatge_retina)
    plt.title("Imatge retina")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(luma, cmap="gray")
    plt.title("Luma original")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(luma_clahe, cmap="gray")
    plt.title("Luma CLAHE")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(g_clahe, cmap="gray")
    plt.title("G CLAHE")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(b_clahe, cmap="gray")
    plt.title("B CLAHE")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(v, cmap="gray")
    plt.title("V original")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(v_clahe, cmap="gray")
    plt.title("V CLAHE")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(v_kmmrc_clahe, cmap="gray")
    plt.title("KMMRC + CLAHE")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------
# Exemple d'us en el main
# -------------------------------------------------------------------------
# resultat = preprocessa_imatge(ruta_imatge, ruta_mascara)
#
# imatge = resultat[0]
# mascara_gt = resultat[1]
# mascara_retina = resultat[2]
# imatge_retina = resultat[3]
# canals = resultat[4]
#
# contrast = millora_contrast(imatge_retina, mascara_retina, canals)
#
# luma_eq = contrast[0]
# luma_clahe = contrast[1]
# g_clahe = contrast[2]
# b_clahe = contrast[3]
# v_clahe = contrast[4]
# v_kmmrc = contrast[5]
# v_kmmrc_clahe = contrast[6]
# v_kmmrc_clahe_filtrat = contrast[7]
# imatge_kmmrc = contrast[8]
#
# visualitza_contrast(imatge_retina, canals, contrast)
