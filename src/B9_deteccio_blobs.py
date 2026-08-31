"""
Bloc 10 - Extraccio de features de blobs.

Aquest bloc agafa una mascara candidata d'exsudats i calcula
caracteristiques de cada blob.

Cada blob sera una fila de la taula.

La ground truth nomes s'utilitza per posar l'etiqueta:
    1  -> exsudat
    0  -> fals positiu
   -1  -> ambigu

"""

import numpy as np
import pandas as pd

from skimage.measure import label, regionprops
from skimage.color import rgb2ycbcr


# Funcions auxiliars

def calcula_circularitat(area, perimetre):
    """
    Calcula si el blob es mes o menys circular.
    
    Un valor proper a 1 indica una forma semblant a un cercle,
    """

    if perimetre == 0:
        return 0

    # La formula compara l'area real amb un cercle
    # amb un perimetre equivalent.
    
    circularitat = 4 * np.pi * area / (perimetre ** 2)

    return circularitat


def calcula_ratio_eixos(eix_major, eix_menor):
    """
    Calcula la relacio entre l'eix major i l'eix menor del blob.
    """
    
    #Evitem dividir entre 0
    if eix_menor == 0:
        return 0

    ratio = eix_major / eix_menor

    return ratio


def mitjana_top25(canal, mascara_blob):
    """
    Calcula la mitjana del 25% de pixels mes intensos del blob.
    """

    valors = canal[mascara_blob]

    # El percentil 75 separa el 25% de valors amb major intensitat.
    q75 = np.percentile(valors, 75)
    valors_alts = valors[valors >= q75]

    mitjana = np.mean(valors_alts)

    return mitjana


def calcula_y(imatge):
    """
    Calcula el canal Y de YCbCr i el normalitza a [0, 1].
    """

    ycbcr = rgb2ycbcr(imatge)

    y = ycbcr[:, :, 0]

    # En skimage, Y queda aproximadament entre 16 i 235, per normalitzar:
    y = (y - 16) / 219
    y = np.clip(y, 0.0, 1.0)

    return y


def normalitza_l(l_lab):
    """
    Normalitza el canal L de Lab, que sol estar entre 0 i 100.
    """

    l_norm = l_lab / 100
    l_norm = np.clip(l_norm, 0.0, 1.0)

    return l_norm


def normalitza_bstar(b_lab):
    """
    Normalitza b* procedent del canal Lab.

    b* pot ser negatiu, per això el desplacem a valors positius.
    """

    bstar = (b_lab + 128) / 255
    bstar = np.clip(bstar, 0.0, 1.0)

    return bstar


def calcula_solapament(mascara_blob, mascara_gt):
    """
    Calcula quina part del blob coincideix amb un exudat del Groundtruth.
    """

    area_blob = np.sum(mascara_blob)
    area_solapament = np.sum(mascara_blob & mascara_gt)

    solapament = area_solapament / area_blob

    return solapament


def calcula_etiqueta(solapament, llindar_positiu=0.30, llindar_negatiu=0.05):
    """
    Dona una etiqueta al blob segons el solapament amb la Groundtruth.
    
    Considerem +1 un blob amb prou solapament, -1 un pràcticament fora 
    del Ground Truth i ambigu els casos intermedis.
    """

    if solapament >= llindar_positiu:
        etiqueta = 1

    elif solapament <= llindar_negatiu:
        etiqueta = 0

    else:
        etiqueta = -1

    return etiqueta


# -------------------------------------------------------------------------
# 2. Features d'un blob
# -------------------------------------------------------------------------

def extreu_features_blob(regio,etiquetes,canals,y,l_norm,bstar,mascara_gt, 
                         nom_imatge,nom_canal,nom_metode,llindar_positiu=0.30, 
                         llindar_negatiu=0.05): 
    """ 
    Calculem totes les features corresponents a un únic blob.
    """ 
    # Seleccionem només els pixels que pertanyen al blob actual. 
    mascara_blob=etiquetes==regio.label

    # Canals que venen del preprocessament.
    g = canals[1]
    b = canals[2]
    gray = canals[3]

 
    # Calculem tots els parametres de forma:


    area = regio.area
    perimetre = regio.perimeter

    circularitat=calcula_circularitat(area,perimetre)
    

    solidesa = regio.solidity
    eccentricitat = regio.eccentricity
    extent = regio.extent

    eix_major = regio.major_axis_length
    eix_minor = regio.minor_axis_length

    ratio_eixos=calcula_ratio_eixos(eix_major,eix_minor)


    #Calculem tots els paramteres de color i intensitat:

    # Per cada canal guardem tant la mitjana com la desviacio estandard, 
    # ja que ens interessa la intensitat però també la seva variacio.
    mean_g = np.mean(g[mascara_blob])
    std_g = np.std(g[mascara_blob])

    mean_b = np.mean(b[mascara_blob])
    std_b = np.std(b[mascara_blob])

    mean_y = np.mean(y[mascara_blob])
    std_y = np.std(y[mascara_blob])

    mean_gray = np.mean(gray[mascara_blob])
    std_gray = np.std(gray[mascara_blob])

    mean_l = np.mean(l_norm[mascara_blob])
    std_l = np.std(l_norm[mascara_blob])

    mean_bstar = np.mean(bstar[mascara_blob])
    std_bstar = np.std(bstar[mascara_blob])


    # Afegim percentils per representar millor la distribucio 
    #d'intensitats dins del blob i no dependre només de la mitjana.

    q25_g = np.percentile(g[mascara_blob], 25)
    q75_g = np.percentile(g[mascara_blob], 75)
    top25_g = mitjana_top25(g, mascara_blob)

    q25_b = np.percentile(b[mascara_blob], 25)
    q75_b = np.percentile(b[mascara_blob], 75)
    top25_b = mitjana_top25(b, mascara_blob)

   

    # Etiqueta del blob 
    solapament=calcula_solapament(mascara_blob,mascara_gt) 
    etiqueta=calcula_etiqueta(solapament,llindar_positiu,llindar_negatiu)

    fila = [
        nom_imatge,
        nom_canal,
        nom_metode,
        regio.label,

        area,
        perimetre,
        circularitat,
        solidesa,
        eccentricitat,
        extent,
        eix_major,
        eix_minor,
        ratio_eixos,

        mean_g,
        std_g,
        mean_b,
        std_b,
        mean_y,
        std_y,
        mean_gray,
        std_gray,
        mean_l,
        std_l,
        mean_bstar,
        std_bstar,

        q25_g,
        q75_g,
        top25_g,
        q25_b,
        q75_b,
        top25_b,

        solapament,
        etiqueta,
    ]

    return fila


# -------------------------------------------------------------------------
# 3. Funcio principal del bloc 10
# -------------------------------------------------------------------------

def crea_taula_features(
    imatge,
    canals,
    mascara_candidats,
    mascara_gt,
    nom_imatge="",
    nom_canal="",
    nom_metode="",
    llindar_positiu=0.30,
    llindar_negatiu=0.05,
):
    """
    Crea una taula de features.

    Cada component connectada de la mascara candidata es un blob.
    """

    mascara_candidats = mascara_candidats.astype(bool)
    mascara_gt = mascara_gt.astype(bool)

   
    # Canals que utilitzarem per les features.
    # Les features es calculen sobre la imatge original/preprocessada,
    # no sobre KMMRC ni CLAHE.
    l_lab = canals[7]
    b_lab = canals[9]

    y = calcula_y(imatge)
    l_norm = normalitza_l(l_lab)
    bstar = normalitza_bstar(b_lab)

    # Convertim la mascara binaria en components connectades.
    # connectivity=2 vol dir que les diagonals tambe compten com connexio.
    #(Connectivity = 1 només comptaria dalt baix esquerra i dreta com connexions)
    
    etiquetes = label(
        mascara_candidats,
        connectivity=2,
    )

    regions = regionprops(etiquetes)

    columnes = [
        "nom_imatge",
        "nom_canal",
        "nom_metode",
        "blob_id",

        "area",
        "perimetre",
        "circularitat",
        "solidesa",
        "eccentricitat",
        "extent",
        "eix_major",
        "eix_minor",
        "ratio_eixos",

        "mean_g",
        "std_g",
        "mean_b",
        "std_b",
        "mean_y",
        "std_y",
        "mean_gray",
        "std_gray",
        "mean_l",
        "std_l",
        "mean_bstar",
        "std_bstar",

        "q25_g",
        "q75_g",
        "top25_g",
        "q25_b",
        "q75_b",
        "top25_b",

        "solapament_gt",
        "etiqueta",
    ]

    files = []

    for regio in regions:

        fila = extreu_features_blob(
            regio,
            etiquetes,
            canals,
            y,
            l_norm,
            bstar,
            mascara_gt,
            nom_imatge,
            nom_canal,
            nom_metode,
            llindar_positiu,
            llindar_negatiu,
        )

        files.append(fila)

    taula = pd.DataFrame(
        files,
        columns=columnes,
    )

    return taula


# -------------------------------------------------------------------------
# 4. Resum
# -------------------------------------------------------------------------

def imprimeix_resum_features(taula, nom=""):
    """
    Imprimeix un resum curt de les features.
    """

    print("\n====================================")
    print("RESUM FEATURES:", nom)
    print("====================================")

    print("Nombre de blobs:", len(taula))

    if len(taula) == 0:
        return

    print("Positius:", np.sum(taula["etiqueta"] == 1))
    print("Negatius:", np.sum(taula["etiqueta"] == 0))
    print("Ambigus:", np.sum(taula["etiqueta"] == -1))

    print("Area mitjana:", round(np.mean(taula["area"]), 3))
    print("Solapament mitja:", round(np.mean(taula["solapament_gt"]), 3))
    print("Mean G mitja:", round(np.mean(taula["mean_g"]), 3))
    print("Mean B mitja:", round(np.mean(taula["mean_b"]), 3))


def ajunta_taules(llista_taules):
    """
    Junta diferents taules de features.
    """

    taula_total = pd.concat(
        llista_taules,
        ignore_index=True,
    )

    return taula_total


def columnes_svm():
    """
    Columnes que podrem usar com X en el bloc 11.

    No posem:
        - nom_imatge
        - nom_canal
        - nom_metode
        - blob_id
        - solapament_gt
        - etiqueta
    """

    columnes = [
        "area",
        "perimetre",
        "circularitat",
        "solidesa",
        "eccentricitat",
        "extent",
        "eix_major",
        "eix_minor",
        "ratio_eixos",

        "mean_g",
        "std_g",
        "mean_b",
        "std_b",
        "mean_y",
        "std_y",
        "mean_gray",
        "std_gray",
        "mean_l",
        "std_l",
        "mean_bstar",
        "std_bstar",

        "q25_g",
        "q75_g",
        "top25_g",
        "q25_b",
        "q75_b",
        "top25_b",
    ]

    return columnes