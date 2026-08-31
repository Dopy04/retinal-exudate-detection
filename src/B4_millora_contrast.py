"""
Bloc 4 - Millora de contrast del projecte PIVA Retina.

En aquest bloc apliquem diferents metodes per millorar el contrast
dels canals que hem obtingut durant el preprocessament.

Provarem equalitzacio d'histograma, CLAHE i una versio de KMMRC
aplicada sobre el canal V. Els canals resultants quedaran preparats
per utilitzar-los posteriorment en la segmentacio.

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.exposure import equalize_hist,equalize_adapthist
from skimage.color import rgb2hsv,hsv2rgb
from sklearn.cluster import KMeans
from scipy.ndimage import uniform_filter



def prepara_canal(canal,mascara_retina):
    """
    Prepara un canal abans d'aplicar metodes de  contrast.

    Fora de la retina no deixem zeros, perque el fons negre podria
    afectar l'equalitzacio.

    Per tant els substituim temporalment per la mitjana dels pixels
    que estan dins de la retina.
    """

    canal_preparat = np.copy(canal).astype(np.float64)

    valor_fons = np.median(canal_preparat[mascara_retina])

    canal_preparat[~mascara_retina] = valor_fons

    canal_preparat=np.clip(canal_preparat, 0.0, 1.0)

    return canal_preparat




def equalitza_histograma(canal,mascara_retina):
    """
    Aplica equalitzacio global d'histograma a un canal.

    Despres tornem a posar el fons exterior de la retina a zero.
    """

    canal_preparat = prepara_canal(canal,mascara_retina)

    canal_eq = equalize_hist( canal_preparat, mask=mascara_retina)

    # Una vegada aplicat el contrast tornem a posar el fons a 0.
    canal_eq[~mascara_retina]=0

    return np.clip(canal_eq,0.0,1.0)


def aplica_clahe(canal,mascara_retina,clip_limit=0.01):
    """
    Aplica CLAHE, el qual millora el contrast de forma local.
    Pot ser util quan la iluminacio de la retina no es uniforme.
    """

    canal_preparat = prepara_canal(canal,mascara_retina)

    canal_clahe = equalize_adapthist(canal_preparat,clip_limit=clip_limit)

    canal_clahe[~mascara_retina]=0

    return np.clip(canal_clahe,0.0,1.0)


def filtrar_mitjana(canal,mascara_retina,mida=5):
    """
    Aplica un filtre de mitjana per suavitzar el canal.

     Ajuda a reduir soroll i petits canvis
    d'intensitat després del contrast.
    """

    canal_preparat=prepara_canal(canal,mascara_retina)

    canal_filtrat=uniform_filter(canal_preparat, size=mida)

    canal_filtrat[~mascara_retina]=0

    return np.clip(canal_filtrat,0.0,1.0)




def aplica_kmmrc(imatge_retina, mascara_retina, k=5, beta=1.5):
    """
    Aplica la millora KMMRC sobre el canal V de HSV.

    Ve definit per:
        1. Convertir RGB a HSV.
        2. Dividir V en k regions amb K-means.
        3. Millorar el contrast de cada regio.
        4. Reconstruir la imatge RGB amb el nou canal V.

    """

    hsv=rgb2hsv(imatge_retina)

    h=hsv[:,:,0]
    s=hsv[:,:,1]
    v=hsv[:,:,2]

    v_nou=np.zeros_like(v)

    # Agafem nomes els pixels que estan dins de la retina, i els convrtim en un vector columna
    valors_v=v[mascara_retina].reshape(-1,1)

    # K-means separa les intensitats en k grups
    model=KMeans(n_clusters=k,random_state=0,n_init=10)

    etiquetes=model.fit_predict(valors_v)

    # Tornem a convertir els valors en vector 1D
    valors_v=valors_v.ravel()

    valors_v_nous=np.copy(valors_v)

    # Processem cada cluster per separat, 
    #cada cluster representa una zona amb intensitats semblants.
    # El realç es calcula respecte a la mitjana de cada grup.
    for i in range(k):

        posicions=etiquetes==i
        valors_regio=valors_v[posicions]

        mitjana=np.mean(valors_regio)

        # Realç lineal inspirat en KMMRC
        contrast=mitjana+beta*(valors_regio-mitjana)

        valors_v_nous[posicions]=valors_regio+contrast

    # Limitem els valors entre 0 i 1
    valors_v_nous=np.clip(valors_v_nous,0.0,1.0)

    # Tornem a col·locar els pixels dins la imatge
    v_nou[mascara_retina]=valors_v_nous
    v_nou[~mascara_retina]=0

    # Reconstruim HSV
    hsv_nou=np.zeros_like(hsv)

    hsv_nou[:,:,0]=h
    hsv_nou[:,:,1]=s
    hsv_nou[:,:,2]=v_nou

    # Tornem a RGB
    imatge_kmmrc=hsv2rgb(hsv_nou)
    imatge_kmmrc[~mascara_retina]=0

    return v_nou,np.clip(imatge_kmmrc,0.0,1.0)



def millora_contrast(imatge_retina,mascara_retina,canals):
    """
    Apliquem els diferents metodes de millora sobre els canals

    L'ordre dels canals rebuts del preprocessament és:
    R, G, B, Luma, H, S, V, L, a, b.

    Retornem els canals millorats i la imatge reconstruida amb KMMRC.
    """
    g=canals[1]
    b=canals[2]
    luma=canals[3]
    v=canals[6]

    # Equalitzacio global i CLAHE sobre luma
    luma_eq=equalitza_histograma(luma,mascara_retina)
    luma_clahe=aplica_clahe(luma,mascara_retina)
    
    
    # Apliquem CLAHE també sobre alguns canals originals per poder
    # comparar quin d'ells facilita més la deteccio dels exsudats.
    
    g_clahe=aplica_clahe(g,mascara_retina)
    b_clahe=aplica_clahe(b,mascara_retina)
    v_clahe=aplica_clahe(v,mascara_retina)


    # KMMRC sobre V i després provem d'afegir CLAHE
    # i un suavitzat final al resultat.
    
    v_kmmrc,imatge_kmmrc=aplica_kmmrc(imatge_retina,mascara_retina)
    v_kmmrc_clahe=aplica_clahe(v_kmmrc,mascara_retina)
    v_kmmrc_clahe_filtrat=filtrar_mitjana(v_kmmrc_clahe,mascara_retina,mida=5)

    return (luma_eq,luma_clahe,g_clahe,b_clahe,v_clahe,v_kmmrc,
            v_kmmrc_clahe,v_kmmrc_clahe_filtrat,imatge_kmmrc)


# Visualitzacio


def visualitza_contrast(imatge_retina,canals,contrast,titol):
    """
    Mostra una comparacio visual dels canals originals i millorats.
    """

    g=canals[1]
    luma=canals[3]
    v=canals[6]
    
    
    luma_eq=contrast[0]
    luma_clahe=contrast[1]
    
    g_clahe=contrast[2]
    v_clahe=contrast[4]
    
    v_kmmrc=contrast[5]
    v_kmmrc_clahe=contrast[6]
    v_kmmrc_clahe_filtrat=contrast[7]
    imatge_kmmrc=contrast[8]
    
    
    plt.figure(figsize=(14,10))

    plt.suptitle(titol,fontsize=16)

    plt.subplot(3,4,1)
    plt.imshow(imatge_retina)
    plt.title("Imatge retina")
    plt.axis("off")

    plt.subplot(3,4,2)
    plt.imshow(imatge_kmmrc)
    plt.title("Imatge KMMRC")
    plt.axis("off")

    plt.subplot(3,4,3)
    plt.imshow(g,cmap="gray")
    plt.title("G original")
    plt.axis("off")

    plt.subplot(3,4,4)
    plt.imshow(g_clahe,cmap="gray")
    plt.title("G CLAHE")
    plt.axis("off")

    plt.subplot(3,4,5)
    plt.imshow(luma,cmap="gray")
    plt.title("Luma original")
    plt.axis("off")

    plt.subplot(3,4,6)
    plt.imshow(luma_eq,cmap="gray")
    plt.title("Luma equalitzada")
    plt.axis("off")

    plt.subplot(3,4,7)
    plt.imshow(luma_clahe,cmap="gray")
    plt.title("Luma CLAHE")
    plt.axis("off")

    plt.subplot(3,4,8)
    plt.imshow(v,cmap="gray")
    plt.title("V original")
    plt.axis("off")

    plt.subplot(3,4,9)
    plt.imshow(v_clahe,cmap="gray")
    plt.title("V CLAHE")
    plt.axis("off")

    plt.subplot(3,4,10)
    plt.imshow(v_kmmrc,cmap="gray")
    plt.title("V KMMRC")
    plt.axis("off")

    plt.subplot(3,4,11)
    plt.imshow(v_kmmrc_clahe,cmap="gray")
    plt.title("KMMRC + CLAHE")
    plt.axis("off")

    plt.subplot(3,4,12)
    plt.imshow(v_kmmrc_clahe_filtrat,cmap="gray")
    plt.title("KMMRC + CLAHE + filtre")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

