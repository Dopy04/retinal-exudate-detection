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

        # Guardem totes les caracteristiques calculades del blob en una fila
    fila=[nom_imatge,nom_canal,nom_metode,regio.label]

    fila.extend([area,perimetre,circularitat,solidesa,eccentricitat,
                 extent,eix_major,eix_minor,ratio_eixos])

    fila.extend([mean_g,std_g,mean_b,std_b,mean_y,std_y,
                 mean_gray,std_gray,mean_l,std_l,mean_bstar,std_bstar])

    fila.extend([q25_g,q75_g,top25_g,q25_b,q75_b,top25_b])

    # Al final guardem el solapament amb el Ground Truth i
    # l'etiqueta que utilitzarem posteriorment per entrenar el SVM
    fila.extend([solapament,etiqueta])

    return fila


# -------------------------------------------------------------------------
# Funcio principal del Bloc 10
# -------------------------------------------------------------------------

def crea_taula_features(imatge,canals,mascara_candidats,mascara_gt,
                        nom_imatge="",nom_canal="",nom_metode="",
                        llindar_positiu=0.30,llindar_negatiu=0.05):
    """
    Creem una taula amb les features de tots els blobs detectats.

    Cada fila de la taula correspon a un blob diferent.
    """

    mascara_candidats=mascara_candidats.astype(bool)
    mascara_gt=mascara_gt.astype(bool)

    # Agafem els canals que necessitem per calcular algunes
    # de les caracteristiques de color.
    l_lab=canals[7]
    b_lab=canals[9]

    y=calcula_y(imatge)
    l_norm=normalitza_l(l_lab)
    bstar=normalitza_bstar(b_lab)

    # Separem la mascara candidata en diferents blobs.
    # Amb connectivity=2 considerem connectats també els pixels
    # que es toquen diagonalment.
    etiquetes=label(mascara_candidats,connectivity=2)
    regions=regionprops(etiquetes)

    # Noms de les caracteristiques que tindrà cada columna de la taula.
    columnes=[
        "nom_imatge","nom_canal","nom_metode","blob_id",

        "area","perimetre","circularitat","solidesa","eccentricitat",
        "extent","eix_major","eix_minor","ratio_eixos",

        "mean_g","std_g","mean_b","std_b","mean_y","std_y",
        "mean_gray","std_gray","mean_l","std_l","mean_bstar","std_bstar",

        "q25_g","q75_g","top25_g","q25_b","q75_b","top25_b",

        "solapament_gt","etiqueta"
    ]

    files=[]

    # Recorrem tots els blobs i calculem una fila de features
    # per cadascun d'ells.
    for regio in regions:
        fila=extreu_features_blob(regio,etiquetes,canals,y,l_norm,bstar,
                                  mascara_gt,nom_imatge,nom_canal,nom_metode,
                                  llindar_positiu,llindar_negatiu)

        files.append(fila)

    # Amb totes les files construim finalment la taula de features.
    taula=pd.DataFrame(files,columns=columnes)

    return taula



# Resum de les features


def imprimeix_resum_features(taula,nom=""):
    """
    Mostrem un resum dels blobs que hem obtingut.
    """

 
    print("RESUM FEATURES:",nom)


    print("Nombre de blobs:",len(taula))

    if len(taula)==0:
        return

    print("Positius:",np.sum(taula["etiqueta"]==1))
    print("Negatius:",np.sum(taula["etiqueta"]==0))
    print("Ambigus:",np.sum(taula["etiqueta"]==-1))

    print("Area mitjana:",round(np.mean(taula["area"]),3))
    print("Solapament mitja:",round(np.mean(taula["solapament_gt"]),3))
    print("Mean G mitja:",round(np.mean(taula["mean_g"]),3))
    print("Mean B mitja:",round(np.mean(taula["mean_b"]),3))


def ajunta_taules(llista_taules):
    """
    Ajuntem les taules de features de diferents imatges
    en una única taula.
    """

    taula_total=pd.concat(llista_taules,ignore_index=True)

    return taula_total


def columnes_svm():
    """
    Indiquem les features que utilitzarem com entrada del SVM.

    No utilitzem el nom de la imatge, metode o blob perquè només
    serveixen per identificar-lo. Tampoc utilitzem el solapament
    ni l'etiqueta, ja que aquests valors provenen del Ground Truth.
    """

    columnes=["area","perimetre","circularitat","solidesa","eccentricitat",
        "extent","eix_major","eix_minor","ratio_eixos",

        "mean_g","std_g","mean_b","std_b","mean_y","std_y",
        "mean_gray","std_gray","mean_l","std_l","mean_bstar","std_bstar",

        "q25_g","q75_g","top25_g","q25_b","q75_b","top25_b"]

    return columnes
    
#Visualitzacio

if __name__=="__main__":

    #Prova Bloc 9: utilitzem una imatge del dataset per comprovar
    #visualment com una mascara candidata es separa en blobs
    #i com aquests blobs queden etiquetats segons el Ground Truth

    from pathlib import Path
    import matplotlib.pyplot as plt

    import B2_preprocessament as prep
    from B4_millora_contrast import millora_contrast
    from B4_segmentacio import obtenir_mascares_baseline
    from B6_disc_optic import detecta_disc_optic,elimina_disc_optic

    #Escollim el dataset i la imatge que volem provar
    #Per provar una altra només cal canviar aquests dos valors
    dataset="DR2019"
    numero=3

    ruta_projecte=Path(__file__).resolve().parent.parent
    ruta_dades=ruta_projecte/"data_samples"

    #Obtenim automaticament la imatge i el seu Ground Truth
    ruta_imatge,ruta_gt=prep.obtenir_rutes_dataset(
        ruta_dades,dataset,numero)

    #Fem el preprocessament i obtenim la mascara de retina,
    #la imatge sense fons i els canals originals
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=prep.preprocessa_imatge(
        ruta_imatge,ruta_gt)

    #Apliquem la millora de contrast del Bloc 4
    contrast=millora_contrast(
        imatge_retina,
        mascara_retina,
        canals)

    #Agafem el canal V CLAHE perquè és el que hem vist que funciona bé
    #per generar candidats d'exsudats
    canal_v_clahe=contrast[4]

    #Detectem el disc optic per eliminar-lo de la mascara candidata
    canal_r=canals[0]
    canal_v=canals[6]

    canal_disc=0.5*canal_r+0.5*canal_v
    canal_disc=np.clip(canal_disc,0.0,1.0)

    canal_suau,mascara_disc_optic,mascara_disc=detecta_disc_optic(
        canal_disc,
        mascara_retina,
        percentil=98,
        sigma=2)

    #Fem una segmentacio candidata.
    #Utilitzem Adaptatiu sobre V CLAHE perquè és una de les combinacions
    #que millor resultat ens ha donat després amb el SVM
    mascares=obtenir_mascares_baseline(
        canal_v_clahe,
        mascara_retina)

    mascara_candidata=mascares["Adaptatiu"]

    #Eliminem el disc optic de la mascara candidata
    mascara_candidata=elimina_disc_optic(
        mascara_candidata,
        mascara_disc)

    #Creem la taula de features.
    #Cada fila de la taula correspon a un blob diferent
    taula=crea_taula_features(
        imatge,
        canals,
        mascara_candidata,
        mascara_gt,
        nom_imatge=str(numero).zfill(3)+".jpg",
        nom_canal="V CLAHE",
        nom_metode="Adaptatiu",
        llindar_positiu=0.25,
        llindar_negatiu=0.05)

    #Separem la mascara candidata en blobs per poder visualitzar-los
    etiquetes=label(mascara_candidata,connectivity=2)

    #Preparem tres mascares noves per separar visualment
    #els blobs positius, negatius i ambigus
    mascara_positius=np.zeros_like(mascara_candidata,dtype=bool)
    mascara_negatius=np.zeros_like(mascara_candidata,dtype=bool)
    mascara_ambigus=np.zeros_like(mascara_candidata,dtype=bool)

    for i in range(len(taula)):

        blob_id=int(taula.iloc[i]["blob_id"])
        etiqueta=int(taula.iloc[i]["etiqueta"])

        if etiqueta==1:
            mascara_positius=mascara_positius | (etiquetes==blob_id)

        elif etiqueta==0:
            mascara_negatius=mascara_negatius | (etiquetes==blob_id)

        else:
            mascara_ambigus=mascara_ambigus | (etiquetes==blob_id)

    print("\n======================================")
    print("PROVA BLOC 9")
    print("Dataset:",dataset)
    print("Imatge:",str(numero).zfill(3))
    print("======================================")

    imprimeix_resum_features(
        taula,
        "V CLAHE + Adaptatiu")

    #Mostrem la mascara candidata, els blobs separats i les etiquetes
    #que venen del solapament amb el Ground Truth
    plt.figure(figsize=(16,8))
    plt.suptitle(
        "PROVA BLOC 9 - "+dataset+" "+str(numero).zfill(3),
        fontsize=16)

    plt.subplot(2,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,4,2)
    plt.imshow(mascara_candidata,cmap="gray")
    plt.title("Mascara candidata")
    plt.axis("off")

    plt.subplot(2,4,3)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2,4,4)
    plt.imshow(etiquetes,cmap="nipy_spectral")
    plt.title("Blobs etiquetats")
    plt.axis("off")

    plt.subplot(2,4,5)
    plt.imshow(mascara_positius,cmap="gray")
    plt.title("Blobs positius")
    plt.axis("off")

    plt.subplot(2,4,6)
    plt.imshow(mascara_negatius,cmap="gray")
    plt.title("Blobs negatius")
    plt.axis("off")

    plt.subplot(2,4,7)
    plt.imshow(mascara_ambigus,cmap="gray")
    plt.title("Blobs ambigus")
    plt.axis("off")

    plt.subplot(2,4,8)
    plt.imshow(imatge)
    plt.imshow(mascara_gt,cmap="Reds",alpha=0.45)
    plt.imshow(mascara_candidata,cmap="Greens",alpha=0.45)
    plt.title("GT + candidata")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()
    
    
# Comentari IA:
# Hem utilitzat IA com a suport per organitzar aquesta part del projecte,
# ja que treballem amb una gran quantitat d'informacio per cada blob:
# features de forma, color, intensitat, etiquetes i dades de diferents canals.
#
# L'hem fet servir especialment per entendre com estructurar aquestes dades
# en taules, mantenir el mateix ordre entre files i columnes i assegurar que
# cada feature correspon al canal i al blob correcte durant tot el proces.
#
# També ens ha servit de guia per connectar aquesta taula de features amb
# els blocs posteriors, especialment amb el SVM i la reconstruccio de les
# mascares a partir dels blobs classificats.