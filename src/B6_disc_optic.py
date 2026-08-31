# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:35:27 2026

@author: pablo
"""
"""
Bloc 7 - Eliminacio del disc optic.

Detectem el disc optic a partir de canals brillants de la retina.
En aquesta versio provem tres opcions:

    1. Canal V
    2. Canal R
    3. Combinacio R + V

Despres aquesta mascara del disc optic es pot restar d'una mascara candidata
d'exsudats que vingui del bloc 4 o del bloc 6.

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    closing,
    dilation,
    disk,
)
from skimage.measure import label, regionprops


def detecta_disc_optic(canal_od, mascara_retina, percentil=98,
                       area_minima=600, radi_tancament=8,
                       radi_dilatacio=12, sigma=2):
    """
    Detecta una possible mascara del disc optic a partir d'un canal.

    canal_od:
        Canal que usem per buscar el disc optic.

    mascara_retina:
        Mascara de la retina per eliminar el fons negre.

    sigma:
        Suavitzat gaussian. Serveix per reduir l'efecte dels vasos que
        passen per damunt del disc optic. Si sigma=0, no es suavitza.

    Retorna:
        [0] canal_suau
        [1] mascara_disc_optic
        [2] mascara_disc
    """

    canal_od = canal_od.astype(np.float64)
    canal_od = np.clip(canal_od, 0.0, 1.0)

    # Suavitzem una mica el canal perque els vasos no divideixin tant
    # la zona brillant del disc optic.
    if sigma > 0:
        canal_suau = gaussian_filter(canal_od, sigma=sigma)
    else:
        canal_suau = np.copy(canal_od)

    canal_suau[~mascara_retina] = 0

    # Calculem el llindar nomes dins de la retina.
    valors_retina = canal_suau[mascara_retina]
    llindar = np.percentile(valors_retina, percentil)

    # Zones mes brillants del canal.
    mascara_disc_optic = canal_suau > llindar
    mascara_disc_optic = mascara_disc_optic & mascara_retina

    # Neteja morfologica.
    mascara_disc_optic = remove_small_objects(mascara_disc_optic, min_size=area_minima)
    mascara_disc_optic = closing(mascara_disc_optic, disk(radi_tancament))
    mascara_disc_optic = remove_small_holes(mascara_disc_optic, area_threshold=area_minima)

    # Busquem components connectades.
    etiquetes = label(mascara_disc_optic)
    regions = regionprops(etiquetes, intensity_image=canal_suau)

    # Si no detecta cap regio, retornem mascara buida.
    if len(regions) == 0:
        mascara_disc = np.zeros_like(mascara_retina, dtype=bool)
        return canal_suau, mascara_disc_optic, mascara_disc

    millor_etiqueta = regions[0].label
    millor_puntuacio = -1

    for regio in regions:
        area = regio.area
        solidesa = regio.solidity
        excentricitat = regio.eccentricity

        # Volem una regio gran i compacta.
        puntuacio = area * solidesa / (1 + excentricitat)

        if puntuacio > millor_puntuacio:
            millor_puntuacio = puntuacio
            millor_etiqueta = regio.label

    mascara_disc = etiquetes == millor_etiqueta

    # Fem la mascara una mica mes gran per eliminar tambe el contorn.
    mascara_disc = dilation(mascara_disc, disk(radi_dilatacio))
    mascara_disc = mascara_disc & mascara_retina

    return canal_suau, mascara_disc_optic, mascara_disc



def elimina_disc_optic(mascara_candidats, mascara_disc):
    """
    Elimina el disc optic d'una mascara candidata d'exsudats.

    mascara_candidats:
        Mascara que ve del bloc 4 o del bloc 6.
        True = possible exsudat.

    mascara_disc:
        Mascara del disc optic detectat.
    """

    mascara_sense_disc = np.copy(mascara_candidats)
    mascara_sense_disc[mascara_disc] = False

    return mascara_sense_disc

