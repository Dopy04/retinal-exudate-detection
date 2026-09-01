# -- coding: utf-8 --
"""
Bloc 6: Deteccio i eliminacio del disc optic.

En aquest bloc detectem el disc optic a partir de les zones més brillants
de la retina. Provarem diferents canals perquè el disc optic destaca
especialment en els canals R i V.

La mascara obtinguda es podrà utilitzar posteriorment per eliminar
el disc optic de les regions candidates a exsudats.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects,remove_small_holes,closing,dilation,disk
from skimage.measure import label,regionprops


#Aquests paràmetres a la funció s'han triat de manera experimental i a partir
#de paràmetres propers al q podriem esperar
def detecta_disc_optic(canal_od, mascara_retina, percentil=98,
                       area_minima=600, radi_tancament=8,
                       radi_dilatacio=12, sigma=2):
    """
    Detecta una possible mascara del disc optic a partir d'un canal.

   Primer suavitzem la imatge i seleccionem els pixels per sobre
    d'un percentil alt. Després netegem la mascara i escollim
    la regio gran i compacta que més s'assembla al disc optic.
    """

    canal_od = canal_od.astype(np.float64)
    canal_od = np.clip(canal_od, 0.0, 1.0)

    # Suavitzem una mica el canal perque els vasos sanguinis que travessen 
    #no divideixin tant la zona brillant del disc optic.
    if sigma > 0:
        canal_suau = gaussian_filter(canal_od, sigma=sigma)
    else:
        canal_suau = np.copy(canal_od)

    
    
    # Calculem el percentil només amb els pixels de la retina per evitar
    # que el fons negre afecti el valor del llindar.
    canal_suau[~mascara_retina] = 0
    
    valors_retina = canal_suau[mascara_retina]
    llindar = np.percentile(valors_retina, percentil)

    # Zones mes brillants del canal.
    mascara_disc_optic = canal_suau > llindar
    mascara_disc_optic = mascara_disc_optic & mascara_retina

     # Eliminem regions petites i unim zones properes que poden formar
    # part del mateix disc optic.
    
    mascara_disc_optic=remove_small_objects(mascara_disc_optic,min_size=area_minima)
    mascara_disc_optic=closing(mascara_disc_optic,disk(radi_tancament))
    mascara_disc_optic=remove_small_holes(mascara_disc_optic,area_threshold=area_minima)

    # Separem les diferents regions connectades per poder escollir
    # quina d'elles és la candidata més probable a disc optic.
    etiquetes=label(mascara_disc_optic)
    regions=regionprops(etiquetes)

    if len(regions)==0:
        mascara_disc=np.zeros_like(mascara_retina,dtype=bool)
        return canal_suau,mascara_disc_optic,mascara_disc

    millor_etiqueta=regions[0].label
    millor_puntuacio=-1

    # El disc optic hauria de ser una regio relativament gran,
    # compacta i poc allargada.
    for regio in regions:
        area=regio.area
        solidesa=regio.solidity
        excentricitat=regio.eccentricity

        #d'aquesta manera mesurem com de probable és que aquesta regió sigui
        #el disc `optic i si tenim més d'una regió comparem quina té millor puntuació
        puntuacio=area*solidesa/(1+excentricitat)

        if puntuacio>millor_puntuacio:
            millor_puntuacio=puntuacio
            millor_etiqueta=regio.label

    mascara_disc=etiquetes==millor_etiqueta

    # Ampliem lleugerament la mascara per eliminar també el contorn
    # del disc quan la restem de les regions candidates.
    mascara_disc=dilation(mascara_disc,disk(radi_dilatacio))
    mascara_disc=mascara_disc&mascara_retina

    return canal_suau,mascara_disc_optic,mascara_disc


def elimina_disc_optic(mascara_candidats,mascara_disc):
    """
    Eliminem de la mascara candidata tots els pixels que formen
    part del disc optic detectat.
    """

    mascara_sense_disc=np.copy(mascara_candidats)
    mascara_sense_disc[mascara_disc]=False

    return mascara_sense_disc

if __name__=="__main__":

    #Prova Bloc 6: utilitzem una imatge del dataset per comprovar
    #visualment com es detecta i s'elimina el disc optic

    from pathlib import Path
    import B2_preprocessament as prep
    from B4_millora_contrast import millora_contrast
    from B4_segmentacio import obtenir_mascares_baseline
    from B5_nick_inrg import invertir_canal,segmentar_nick

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

    #Agafem el canal V amb CLAHE perquè és un dels canals
    #que utilitzem després per segmentar possibles exsudats
    canal_v_clahe=contrast[4]

    #Per detectar el disc optic fem servir una combinacio de R i V,
    #ja que el disc optic sol destacar en canals brillants
    canal_r=canals[0]
    canal_v=canals[6]
    canal_disc=0.5*canal_r+0.5*canal_v
    canal_disc=np.clip(canal_disc,0.0,1.0)

    canal_suau,mascara_disc_optic,mascara_disc=detecta_disc_optic(
        canal_disc,
        mascara_retina,
        percentil=98,
        sigma=2)

    #Apliquem la segmentacio basica del Bloc 4 sobre el canal V CLAHE
    mascares=obtenir_mascares_baseline(
        canal_v_clahe,
        mascara_retina)

    mascara_otsu=mascares["Otsu"]
    mascara_adaptatiu=mascares["Adaptatiu"]

    #Apliquem NICK del Bloc 5 sobre el mateix canal
    #Com NICK busca regions fosques, invertim el canal abans
    canal_invertit=invertir_canal(canal_v_clahe)

    mascara_nick=segmentar_nick(
        canal_invertit,
        mascara_retina,
        mida_finestra=95,
        k=-0.2)

    #Eliminem el disc optic de les mascares candidates
    mascara_otsu_sense_disc=elimina_disc_optic(
        mascara_otsu,
        mascara_disc)

    mascara_adaptatiu_sense_disc=elimina_disc_optic(
        mascara_adaptatiu,
        mascara_disc)

    mascara_nick_sense_disc=elimina_disc_optic(
        mascara_nick,
        mascara_disc)

    print("\n======================================")
    print("PROVA BLOC 6")
    print("Dataset:",dataset)
    print("Imatge:",str(numero).zfill(3))
    print("======================================")

    #Mostrem el disc optic detectat i comparem algunes mascares
    #abans i després d'eliminar aquesta zona
    plt.figure(figsize=(16,8))
    plt.suptitle(
        "PROVA BLOC 6 - "+dataset+" "+str(numero).zfill(3),
        fontsize=16)

    plt.subplot(2,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,4,2)
    plt.imshow(imatge)
    plt.imshow(mascara_disc,cmap="Reds",alpha=0.45)
    plt.title("Disc optic detectat")
    plt.axis("off")

    plt.subplot(2,4,3)
    plt.imshow(mascara_otsu,cmap="gray")
    plt.title("Otsu amb disc")
    plt.axis("off")

    plt.subplot(2,4,4)
    plt.imshow(mascara_otsu_sense_disc,cmap="gray")
    plt.title("Otsu sense disc")
    plt.axis("off")

    plt.subplot(2,4,5)
    plt.imshow(mascara_adaptatiu,cmap="gray")
    plt.title("Adaptatiu amb disc")
    plt.axis("off")

    plt.subplot(2,4,6)
    plt.imshow(mascara_adaptatiu_sense_disc,cmap="gray")
    plt.title("Adaptatiu sense disc")
    plt.axis("off")

    plt.subplot(2,4,7)
    plt.imshow(mascara_nick,cmap="gray")
    plt.title("NICK amb disc")
    plt.axis("off")

    plt.subplot(2,4,8)
    plt.imshow(mascara_nick_sense_disc,cmap="gray")
    plt.title("NICK sense disc")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()


#Comentari IA:
# Hem utilitzat IA com a suport per entendre com aplicar l'algoritme per detectar
# el disc òptic, separar regions entre elles i escollir una regió calculant 
#una puntuacio per cada regio, afavorint les que tenen una area gran i una forma 
#compacta. (línies 63-90)
