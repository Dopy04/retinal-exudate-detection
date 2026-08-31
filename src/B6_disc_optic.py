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


#Comentari IA:
# Hem utilitzat IA com a suport per entendre com aplicar l'algoritme per detectar
# el disc òptic, separar regions entre elles i escollir una regió calculant 
#una puntuacio per cada regio, afavorint les que tenen una area gran i una forma 
#compacta. (línies 63-90)