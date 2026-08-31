# -*- coding: utf-8 -*-
"""
Bloc 3 - Preprocessament de les imatges de retina.

Aquest bloc:
    - carrega les imatges
    - carrega el Ground Truth
    - iguala les dimensions
    - detecta la retina
    - elimina el fons exterior
    - extreu canals de color
    - corregeix variacions d'il·luminacio
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter

from skimage.io import imread
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.transform import resize
from skimage.measure import label
from skimage.morphology import closing, disk, remove_small_holes


# -------------------------------------------------------------------------
# 1. CARREGA D'IMATGES
# -------------------------------------------------------------------------

def carrega_imatge_rgb(ruta):
    """
    Carrega una imatge RGB i la normalitza entre 0 i 1.
    """

    imatge=imread(ruta)

    if imatge.ndim!=3:
        raise ValueError(
            "La imatge hauria de tenir 3 canals RGB."
        )

    # Si hi ha canal alpha, ens quedem nomes amb RGB
    imatge=imatge[:,:,:3]

    imatge=imatge.astype(np.float64)

    if imatge.max()>1:
        imatge=imatge/255.0

    imatge=np.clip(
        imatge,
        0,
        1
    )

    return imatge


def carrega_mascara_gt(ruta,llindar=0.5):
    """
    Carrega el Ground Truth i el passa a mascara booleana.

    True  = exsudat
    False = fons
    """

    mascara=imread(ruta)

    if mascara.ndim==3:

        mascara=rgb2gray(
            mascara[:,:,:3]
        )

    else:

        mascara=mascara.astype(
            np.float64
        )

        if mascara.max()>1:
            mascara=mascara/255.0

    mascara=mascara>llindar

    return mascara


# -------------------------------------------------------------------------
# 2. IGUALAR DIMENSIONS
# -------------------------------------------------------------------------

def ajusta_mida_imatge(imatge,mascara_gt):
    """
    Si la imatge i el Ground Truth tenen mides diferents,
    redimensionem la imatge a la mida del Ground Truth.
    """

    if imatge.shape[:2]!=mascara_gt.shape:

        imatge=resize(
            imatge,
            (
                mascara_gt.shape[0],
                mascara_gt.shape[1],
                3
            ),
            preserve_range=True,
            anti_aliasing=True
        )

    imatge=np.clip(
        imatge,
        0,
        1
    )

    return imatge


# -------------------------------------------------------------------------
# 3. LUMA
# -------------------------------------------------------------------------

def calcula_luma(imatge):
    """
    Calcula la luminancia de la imatge.
    """

    r=imatge[:,:,0]
    g=imatge[:,:,1]
    b=imatge[:,:,2]

    luma=(
        0.299*r
        +0.587*g
        +0.114*b
    )

    return luma


# -------------------------------------------------------------------------
# 4. MASCARA RETINA
# -------------------------------------------------------------------------

def crea_mascara_retina(imatge,llindar=0.05):
    """
    Separa la retina del fons negre.
    Ens quedem amb el component connectat mes gran.
    """

    luma=calcula_luma(
        imatge
    )

    mascara_inicial=luma>llindar

    etiquetes=label(
        mascara_inicial,
        connectivity=2
    )

    if etiquetes.max()==0:
        raise ValueError(
            "No s'ha pogut detectar la retina."
        )

    arees=np.bincount(
        etiquetes.ravel()
    )

    # El 0 es el fons
    arees[0]=0

    etiqueta_retina=np.argmax(
        arees
    )

    mascara_retina=(
        etiquetes==etiqueta_retina
    )

    # Tanquem petites discontinuïtats
    mascara_retina=closing(
        mascara_retina,
        disk(5)
    )

    # Omplim petits forats
    mascara_retina=remove_small_holes(
        mascara_retina,
        area_threshold=500
    )

    return mascara_retina


# -------------------------------------------------------------------------
# 5. APLICAR MASCARA RETINA
# -------------------------------------------------------------------------

def aplica_mascara_retina(imatge,mascara_retina):
    """
    Posa a zero els pixels fora de la retina.
    """

    resultat=np.copy(
        imatge
    )

    resultat[
        ~mascara_retina
    ]=0

    return resultat


# -------------------------------------------------------------------------
# 6. CORRECCIO D'IL.LUMINACIO
# -------------------------------------------------------------------------
def corregeix_iluminacio(canal,mascara_retina):
    """
    Corregeix variacions lentes d'il·luminacio
    dins de la retina.
    """

    canal=canal.astype(np.float64)

    # Finestra gran proporcional a la mida de la imatge
    mida=int(round(0.10*min(canal.shape)))

    if mida<3:
        mida=3

    if mida%2==0:
        mida+=1


    # Mascara en format numeric
    mascara_float=mascara_retina.astype(np.float64)


    # Calculem el fons evitant que el negre exterior
    # de la retina afecti el filtre
    numerador=uniform_filter(
        canal*mascara_float,
        size=mida
    )

    denominador=uniform_filter(
        mascara_float,
        size=mida
    )

    fons=numerador/(denominador+1e-8)


    # Intensitat mitjana de la retina
    mitjana=np.mean(
        canal[mascara_retina]
    )


    # Correccio de la il·luminacio
    canal_corregit=canal-fons+mitjana


    canal_corregit=np.clip(
        canal_corregit,
        0,
        1
    )


    # Eliminem el fons exterior
    canal_corregit[
        ~mascara_retina
    ]=0


    return canal_corregit

# -------------------------------------------------------------------------
# 7. CANALS RGB
# -------------------------------------------------------------------------

def extreu_canals_rgb(imatge):
    """
    Extreu R, G i B.
    """

    r=imatge[:,:,0]
    g=imatge[:,:,1]
    b=imatge[:,:,2]

    return r,g,b


# -------------------------------------------------------------------------
# 8. CANALS HSV
# -------------------------------------------------------------------------

def extreu_canals_hsv(imatge):
    """
    Converteix RGB a HSV.
    """

    hsv=rgb2hsv(
        imatge
    )

    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]

    return h,s,v


# -------------------------------------------------------------------------
# 9. CANALS LAB
# -------------------------------------------------------------------------

def extreu_canals_lab(imatge):
    """
    Converteix RGB a Lab.
    """

    lab=rgb2lab(
        imatge
    )

    l_lab=lab[:,:,0]
    a_lab=lab[:,:,1]
    b_lab=lab[:,:,2]

    return l_lab,a_lab,b_lab


# -------------------------------------------------------------------------
# 10. TOTS ELS CANALS
# -------------------------------------------------------------------------

def extreu_canals(imatge):
    """
    Ordre dels canals:

    [0] R
    [1] G
    [2] B
    [3] Luma
    [4] H
    [5] S
    [6] V
    [7] L*
    [8] a*
    [9] b*
    """

    r,g,b=extreu_canals_rgb(
        imatge
    )

    luma=calcula_luma(
        imatge
    )

    h,s,v=extreu_canals_hsv(
        imatge
    )

    l_lab,a_lab,b_lab=extreu_canals_lab(
        imatge
    )

    return (
        r,
        g,
        b,
        luma,
        h,
        s,
        v,
        l_lab,
        a_lab,
        b_lab
    )


# -------------------------------------------------------------------------
# 11. FUNCIO PRINCIPAL
# -------------------------------------------------------------------------

def preprocessa_imatge(ruta_imatge,ruta_mascara):
    """
    Executa tot el preprocessament.

    Retorna:

        imatge
        mascara_gt
        mascara_retina
        imatge_retina
        canals
    """

    imatge=carrega_imatge_rgb(
        ruta_imatge
    )

    mascara_gt=carrega_mascara_gt(
        ruta_mascara
    )

    # Igualem dimensions
    imatge=ajusta_mida_imatge(
        imatge,
        mascara_gt
    )

    # Creem mascara retina
    mascara_retina=crea_mascara_retina(
        imatge
    )

    # Eliminem fons exterior
    imatge_retina=aplica_mascara_retina(
        imatge,
        mascara_retina
    )

    # Extreiem canals
    canals=extreu_canals(
        imatge_retina
    )

    return (
        imatge,
        mascara_gt,
        mascara_retina,
        imatge_retina,
        canals
    )


# -------------------------------------------------------------------------
# 12. VISUALITZACIO
# -------------------------------------------------------------------------

def visualitza_preprocessament(
    imatge,
    mascara_gt,
    mascara_retina,
    canals
):
    """
    Mostra les principals sortides del Bloc 3.
    """

    g=canals[1]
    luma=canals[3]
    v=canals[6]

    # Canal G amb correccio d'il·luminacio
    g_corregit=corregeix_iluminacio(
        g,
        mascara_retina
    )


    plt.figure(figsize=(16,8))


    # Original
    plt.subplot(2,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")


    # Ground Truth
    plt.subplot(2,4,2)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")


    # Mascara retina
    plt.subplot(2,4,3)
    plt.imshow(mascara_retina,cmap="gray")
    plt.title("Mascara retina")
    plt.axis("off")


    # Canal G original
    plt.subplot(2,4,4)
    plt.imshow(g,cmap="gray")
    plt.title("Canal G")
    plt.axis("off")


    # Canal G corregit
    plt.subplot(2,4,5)
    plt.imshow(g_corregit,cmap="gray")
    plt.title("G corregit")
    plt.axis("off")


    # Luma
    plt.subplot(2,4,6)
    plt.imshow(luma,cmap="gray")
    plt.title("Luma")
    plt.axis("off")


    # Canal V
    plt.subplot(2,4,7)
    plt.imshow(v,cmap="gray")
    plt.title("Canal V")
    plt.axis("off")


    plt.tight_layout()
    plt.show()