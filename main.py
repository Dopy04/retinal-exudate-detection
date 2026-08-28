# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:56:30 2026

@author: gurpi
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.metriques import calcular_metriques,imprimir_metriques,comparar_metodes
from src.preprocessament_3 import preprocessa_imatge,visualitza_preprocessament
from src.segmentacio import obtenir_mascares_baseline
from src.mejora_contraste_5 import millora_contrast,visualitza_contrast
from src.segmentacio import segmentar_otsu
from src.nick_inrg import invertir_canal,segmentar_nick,segmentar_inrg
from src.disc_optic import detecta_disc_optic,elimina_disc_optic
from src.postprocessament import netejar_mascara
from src.watershed import transformar_distancia,trobar_marcadors,aplicar_watershed

# BLOC 2 - METRIQUES

mascara_gt_prova=np.array([
    [0,0,1,1],
    [0,0,1,1],
    [0,0,0,0],
    [1,1,0,0]
])

mascara_predita_prova=np.array([
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [1,0,0,0]
])

metriques=calcular_metriques(mascara_predita_prova,mascara_gt_prova)

imprimir_metriques("Prova bloc 2",metriques)


# BLOC 3 - PREPROCESSAMENT


# Ruta de la carpeta principal del projecte
ruta_projecte=Path(__file__).resolve().parent

# Ruta de la carpeta on tenim les imatges
ruta_dades=ruta_projecte/"data_samples"


# Posem les 3 imatges i els seus Ground Truth
imatges=[
    ("Exudate_JSIEC_002.jpg","Exudate_JSIEC_002_Groundtruth.jpg"),
    ("Exudate_DR2019_001.jpg","Exudate_DR2019_001_Groundtruth.jpg"),
    ("Exudate_DR2019_003.jpg","Exudate_DR2019_003_Groundtruth.jpg")
]


# Provem les 3 imatges una per una
for nom_imatge,nom_gt in imatges:

    print("\n------------------------------")
    print("Imatge:",nom_imatge)
    print("------------------------------")

    # Creem les rutes completes
    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # Fem tot el preprocessament
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,ruta_gt
    )

    # Mostrem les dimensions per comprovar que tot coincideix
    print("Mida imatge:",imatge.shape)
    print("Mida Ground Truth:",mascara_gt.shape)
    print("Mida mascara retina:",mascara_retina.shape)
    print("Nombre de canals:",len(canals))

    # Comprovem que podem comparar pixel a pixel
    if imatge.shape[:2]==mascara_gt.shape:
        print("OK: imatge i Ground Truth tenen la mateixa mida")
    else:
        print("ERROR: les mides no coincideixen")

    # Mostrem els resultats del preprocessament
    visualitza_preprocessament(
        imatge,
        mascara_gt,
        mascara_retina,
        canals
    )
    
# ============================================================
# BLOC 4 - SEGMENTACIO BASELINE
# ============================================================

print("\n==============================")
print("BLOC 4 - SEGMENTACIO BASELINE")
print("==============================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 4:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # Tornem a fer el preprocessament per obtenir
    # els canals i la mascara de retina
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,ruta_gt
    )

    # Utilitzem el canal G
    canal_g=canals[1]

    # Executem tots els metodes de segmentacio baseline
    mascares=obtenir_mascares_baseline(
        canal_g,
        mascara_retina
    )

    # Comparem totes les mascares amb el Ground Truth
    resultats=comparar_metodes(
        mascares,
        mascara_gt
    )

    # Mostrem les metriques de cada metode
    for nom_metode in resultats:

        imprimir_metriques(
            nom_metode,
            resultats[nom_metode]
        )


    # --------------------------------------------------------
    # VISUALITZACIO
    # --------------------------------------------------------

    plt.figure(figsize=(14,8))

    plt.suptitle(
        "PROVA BLOC 4 - "+nom_imatge+" - Canal G",
        fontsize=16
    )

    # Imatge original
    plt.subplot(2,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    # Canal G
    plt.subplot(2,4,2)
    plt.imshow(canal_g,cmap="gray")
    plt.title("Canal G")
    plt.axis("off")

    # Ground Truth
    plt.subplot(2,4,3)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    # Otsu
    plt.subplot(2,4,4)
    plt.imshow(mascares["Otsu"],cmap="gray")
    plt.title("Otsu")
    plt.axis("off")

    # Llindar fix
    plt.subplot(2,4,5)
    plt.imshow(mascares["Llindar_fix"],cmap="gray")
    plt.title("Llindar fix")
    plt.axis("off")

    # Adaptatiu
    plt.subplot(2,4,6)
    plt.imshow(mascares["Adaptatiu"],cmap="gray")
    plt.title("Adaptatiu")
    plt.axis("off")

    # Sauvola
    plt.subplot(2,4,7)
    plt.imshow(mascares["Sauvola"],cmap="gray")
    plt.title("Sauvola")
    plt.axis("off")

    # K-means
    plt.subplot(2,4,8)
    plt.imshow(mascares["Kmeans"],cmap="gray")
    plt.title("K-means")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
    
# ============================================================
# BLOC 5 - MILLORA DE CONTRAST
# ============================================================

print("\n============================")
print("BLOC 5 - MILLORA DE CONTRAST")
print("============================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 5:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # Fem primer el preprocessament del bloc 3
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,
        ruta_gt
    )

    # Apliquem els metodes de millora de contrast
    contrast=millora_contrast(
        imatge_retina,
        mascara_retina,
        canals
    )

    # Guardem alguns resultats per comprovar-los
    luma_eq=contrast[0]
    luma_clahe=contrast[1]
    g_clahe=contrast[2]
    b_clahe=contrast[3]
    v_clahe=contrast[4]
    v_kmmrc=contrast[5]
    v_kmmrc_clahe=contrast[6]
    v_kmmrc_clahe_filtrat=contrast[7]
    imatge_kmmrc=contrast[8]

    # Comprovem les mides
    print("Mida G CLAHE:",g_clahe.shape)
    print("Mida V CLAHE:",v_clahe.shape)
    print("Mida V KMMRC:",v_kmmrc.shape)
    print("Mida KMMRC + CLAHE:",v_kmmrc_clahe.shape)
    print("Mida resultat final:",v_kmmrc_clahe_filtrat.shape)

    # Comprovem que tots els resultats tenen la mateixa mida que el GT
    if v_kmmrc_clahe_filtrat.shape==mascara_gt.shape:
        print("OK: el resultat del bloc 5 te la mateixa mida que el Ground Truth")
    else:
        print("ERROR: les mides no coincideixen")

    # Visualitzem els resultats
    visualitza_contrast(
        imatge_retina,
        canals,
        contrast,
        "PROVA BLOC 5 - "+nom_imatge
    )
    
# ============================================================
# PROVA BLOC 5B - CONTRAST + OTSU
# ============================================================

print("\n================================")
print("PROVA BLOC 5B - CONTRAST + OTSU")
print("================================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 5B:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # Primer fem el preprocessament
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,
        ruta_gt
    )

    # Bloc 5: millora de contrast
    contrast=millora_contrast(
        imatge_retina,
        mascara_retina,
        canals
    )

    # Canals que volem comparar
    g_original=canals[1]

    g_clahe=contrast[2]
    luma_clahe=contrast[1]
    v_clahe=contrast[4]

    # --------------------------------------------------------
    # OTSU SOBRE CADA CANAL
    # --------------------------------------------------------

    mascara_g_original=segmentar_otsu(
        g_original,
        mascara_retina
    )

    mascara_g_clahe=segmentar_otsu(
        g_clahe,
        mascara_retina
    )

    mascara_luma_clahe=segmentar_otsu(
        luma_clahe,
        mascara_retina
    )

    mascara_v_clahe=segmentar_otsu(
        v_clahe,
        mascara_retina
    )

    # Guardem totes les mascares en un diccionari
    mascares_5b={
        "G original + Otsu":mascara_g_original,
        "G CLAHE + Otsu":mascara_g_clahe,
        "Luma CLAHE + Otsu":mascara_luma_clahe,
        "V CLAHE + Otsu":mascara_v_clahe
    }

    # --------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------

    resultats_5b=comparar_metodes(
        mascares_5b,
        mascara_gt
    )

    for nom_metode in resultats_5b:

        imprimir_metriques(
            nom_metode,
            resultats_5b[nom_metode]
        )

    # --------------------------------------------------------
    # VISUALITZACIO
    # --------------------------------------------------------

    plt.figure(figsize=(14,8))

    plt.suptitle(
        "PROVA BLOC 5B - "+nom_imatge,
        fontsize=16
    )

    # Ground Truth
    plt.subplot(2,4,1)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    # G original
    plt.subplot(2,4,2)
    plt.imshow(g_original,cmap="gray")
    plt.title("G original")
    plt.axis("off")

    # G CLAHE
    plt.subplot(2,4,3)
    plt.imshow(g_clahe,cmap="gray")
    plt.title("G CLAHE")
    plt.axis("off")

    # Luma CLAHE
    plt.subplot(2,4,4)
    plt.imshow(luma_clahe,cmap="gray")
    plt.title("Luma CLAHE")
    plt.axis("off")

    # G original + Otsu
    plt.subplot(2,4,5)
    plt.imshow(mascara_g_original,cmap="gray")
    plt.title("G original + Otsu")
    plt.axis("off")

    # G CLAHE + Otsu
    plt.subplot(2,4,6)
    plt.imshow(mascara_g_clahe,cmap="gray")
    plt.title("G CLAHE + Otsu")
    plt.axis("off")

    # Luma CLAHE + Otsu
    plt.subplot(2,4,7)
    plt.imshow(mascara_luma_clahe,cmap="gray")
    plt.title("Luma CLAHE + Otsu")
    plt.axis("off")

    # V CLAHE + Otsu
    plt.subplot(2,4,8)
    plt.imshow(mascara_v_clahe,cmap="gray")
    plt.title("V CLAHE + Otsu")
    plt.axis("off")

    plt.tight_layout(
        rect=[0,0,1,0.95]
    )

    plt.show()

# ============================================================
# BLOC 6 - NICK / INRG
# ============================================================

print("\n============================")
print("BLOC 6 - NICK / INRG")
print("============================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 6:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # --------------------------------------------------------
    # BLOC 3 - PREPROCESSAMENT
    # --------------------------------------------------------

    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,
        ruta_gt
    )

    # Canal G original
    g_original=canals[1]

    # --------------------------------------------------------
    # BLOC 5 - CLAHE
    # --------------------------------------------------------

    contrast=millora_contrast(
        imatge_retina,
        mascara_retina,
        canals
    )

    g_clahe=contrast[2]

    # --------------------------------------------------------
    # NICK
    # --------------------------------------------------------

    # NICK esta pensat per detectar regions fosques.
    # Com els exsudats son brillants, invertim els canals.
    g_invertit=invertir_canal(g_original)
    g_clahe_invertit=invertir_canal(g_clahe)

    mascara_nick_g=segmentar_nick(
        g_invertit,
        mascara_retina,
        mida_finestra=95,
        k=-0.2
    )

    mascara_nick_clahe=segmentar_nick(
        g_clahe_invertit,
        mascara_retina,
        mida_finestra=95,
        k=-0.2
    )

    # --------------------------------------------------------
    # INRG
    # --------------------------------------------------------

    # ATENCIO:
    # INRG ja inverteix el canal dins de la seva funcio.
    # Per tant aqui passem el canal ORIGINAL, no invertit.

    mascara_inrg_g=segmentar_inrg(
        g_original,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
        iteracions=3,
        delta=0.1
    )

    mascara_inrg_clahe=segmentar_inrg(
        g_clahe,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
        iteracions=3,
        delta=0.1
    )

    # --------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------

    mascares_6={
        "G original + NICK":mascara_nick_g,
        "G CLAHE + NICK":mascara_nick_clahe,
        "G original + INRG":mascara_inrg_g,
        "G CLAHE + INRG":mascara_inrg_clahe
    }

    resultats_6=comparar_metodes(
        mascares_6,
        mascara_gt
    )

    for nom_metode in resultats_6:

        imprimir_metriques(
            nom_metode,
            resultats_6[nom_metode]
        )

    # --------------------------------------------------------
    # VISUALITZACIO
    # --------------------------------------------------------

    plt.figure(figsize=(14,8))

    plt.suptitle(
        "PROVA BLOC 6 - "+nom_imatge,
        fontsize=16
    )

    # Ground Truth
    plt.subplot(2,4,1)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    # Canal G original
    plt.subplot(2,4,2)
    plt.imshow(g_original,cmap="gray")
    plt.title("G original")
    plt.axis("off")

    # Canal G invertit
    plt.subplot(2,4,3)
    plt.imshow(g_invertit,cmap="gray")
    plt.title("G invertit")
    plt.axis("off")

    # Canal G CLAHE
    plt.subplot(2,4,4)
    plt.imshow(g_clahe,cmap="gray")
    plt.title("G CLAHE")
    plt.axis("off")

    # NICK sobre G original
    plt.subplot(2,4,5)
    plt.imshow(mascara_nick_g,cmap="gray")
    plt.title("G + NICK")
    plt.axis("off")

    # NICK sobre G CLAHE
    plt.subplot(2,4,6)
    plt.imshow(mascara_nick_clahe,cmap="gray")
    plt.title("G CLAHE + NICK")
    plt.axis("off")

    # INRG sobre G original
    plt.subplot(2,4,7)
    plt.imshow(mascara_inrg_g,cmap="gray")
    plt.title("G + INRG")
    plt.axis("off")

    # INRG sobre G CLAHE
    plt.subplot(2,4,8)
    plt.imshow(mascara_inrg_clahe,cmap="gray")
    plt.title("G CLAHE + INRG")
    plt.axis("off")

    plt.tight_layout(
        rect=[0,0,1,0.95],
        h_pad=2
    )

    plt.show()
    
    
# ============================================================
# BLOC 7 - ELIMINACIO DEL DISC OPTIC
# ============================================================

print("\n================================")
print("BLOC 7 - ELIMINACIO DISC OPTIC")
print("================================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 7:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt

    # --------------------------------------------------------
    # BLOC 3 - PREPROCESSAMENT
    # --------------------------------------------------------

    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,
        ruta_gt
    )

    # Canals que utilitzarem per detectar el disc optic
    canal_r=canals[0]
    canal_g=canals[1]
    canal_v=canals[6]

    # Combinacio R + V.
    # Dividim entre 2 per mantenir els valors entre 0 i 1.
    canal_rv=(canal_r+canal_v)/2


    # --------------------------------------------------------
    # DETECCIO DEL DISC OPTIC
    # --------------------------------------------------------

    # Prova utilitzant canal R
    r_suau,r_candidats,disc_r=detecta_disc_optic(
        canal_r,
        mascara_retina
    )

    # Prova utilitzant canal V
    v_suau,v_candidats,disc_v=detecta_disc_optic(
        canal_v,
        mascara_retina
    )

    # Prova utilitzant combinacio R + V
    rv_suau,rv_candidats,disc_rv=detecta_disc_optic(
        canal_rv,
        mascara_retina
    )


    # --------------------------------------------------------
    # SEGMENTACIO BASELINE
    # --------------------------------------------------------

    # Utilitzem Otsu sobre G com a mascara candidata
    # per veure si eliminar el disc optic millora el resultat.
    mascara_otsu=segmentar_otsu(
        canal_g,
        mascara_retina
    )


    # --------------------------------------------------------
    # ELIMINEM EL DISC OPTIC
    # --------------------------------------------------------

    mascara_otsu_sense_r=elimina_disc_optic(
        mascara_otsu,
        disc_r
    )

    mascara_otsu_sense_v=elimina_disc_optic(
        mascara_otsu,
        disc_v
    )

    mascara_otsu_sense_rv=elimina_disc_optic(
        mascara_otsu,
        disc_rv
    )


    # --------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------

    mascares_7={
        "Otsu original":mascara_otsu,
        "Otsu - disc R":mascara_otsu_sense_r,
        "Otsu - disc V":mascara_otsu_sense_v,
        "Otsu - disc R+V":mascara_otsu_sense_rv
    }

    resultats_7=comparar_metodes(
        mascares_7,
        mascara_gt
    )

    for nom_metode in resultats_7:

        imprimir_metriques(
            nom_metode,
            resultats_7[nom_metode]
        )


    # Area del disc detectat
    print("\nPixels disc R:",disc_r.sum())
    print("Pixels disc V:",disc_v.sum())
    print("Pixels disc R+V:",disc_rv.sum())


    # --------------------------------------------------------
    # VISUALITZACIO
    # --------------------------------------------------------

    plt.figure(figsize=(14,11))

    plt.suptitle(
        "PROVA BLOC 7 - "+nom_imatge,
        fontsize=16
    )


    # FILA 1 -------------------------------------------------

    plt.subplot(3,4,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(3,4,2)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(3,4,3)
    plt.imshow(canal_r,cmap="gray")
    plt.title("Canal R")
    plt.axis("off")

    plt.subplot(3,4,4)
    plt.imshow(canal_v,cmap="gray")
    plt.title("Canal V")
    plt.axis("off")


    # FILA 2 -------------------------------------------------

    plt.subplot(3,4,5)
    plt.imshow(r_candidats,cmap="gray")
    plt.title("Candidats disc R")
    plt.axis("off")

    plt.subplot(3,4,6)
    plt.imshow(disc_r,cmap="gray")
    plt.title("Disc detectat R")
    plt.axis("off")

    plt.subplot(3,4,7)
    plt.imshow(disc_v,cmap="gray")
    plt.title("Disc detectat V")
    plt.axis("off")

    plt.subplot(3,4,8)
    plt.imshow(disc_rv,cmap="gray")
    plt.title("Disc detectat R+V")
    plt.axis("off")


    # FILA 3 -------------------------------------------------

    plt.subplot(3,4,9)
    plt.imshow(mascara_otsu,cmap="gray")
    plt.title("Otsu original")
    plt.axis("off")

    plt.subplot(3,4,10)
    plt.imshow(mascara_otsu_sense_r,cmap="gray")
    plt.title("Otsu - disc R")
    plt.axis("off")

    plt.subplot(3,4,11)
    plt.imshow(mascara_otsu_sense_v,cmap="gray")
    plt.title("Otsu - disc V")
    plt.axis("off")

    plt.subplot(3,4,12)
    plt.imshow(mascara_otsu_sense_rv,cmap="gray")
    plt.title("Otsu - disc R+V")
    plt.axis("off")


    plt.tight_layout(
        rect=[0,0,1,0.95],
        h_pad=2
    )

    plt.show()


# ============================================================
# BLOC 8 - MORFOLOGIA MATEMATICA
# ============================================================

print("\n==============================")
print("BLOC 8 - MORFOLOGIA MATEMATICA")
print("==============================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 8:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt


    # --------------------------------------------------------
    # PREPROCESSAMENT
    # --------------------------------------------------------

    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,
        ruta_gt
    )

    canal_r=canals[0]
    canal_g=canals[1]


    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    mascara_otsu=segmentar_otsu(
        canal_g,
        mascara_retina
    )


    # --------------------------------------------------------
    # ELIMINACIO DEL DISC OPTIC
    # --------------------------------------------------------

    # Fem servir el canal R.
    # Utilitzem una dilatacio mes petita que al Bloc 7
    # per no eliminar una zona circular massa gran.

    r_suau,r_candidats,disc_r=detecta_disc_optic(
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3
    )

    mascara_sense_disc=elimina_disc_optic(
        mascara_otsu,
        disc_r
    )


    # --------------------------------------------------------
    # MORFOLOGIA
    # --------------------------------------------------------

    # Eliminem objectes petits, fem closing
    # i omplim petits forats.

    mascara_morfologia=netejar_mascara(
        mascara_sense_disc,
        mida_minima=50,
        mida_forat=50
    )


    # --------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------

    mascares_8={
        "Otsu original":mascara_otsu,
        "Otsu - disc":mascara_sense_disc,
        "Otsu - disc + morfologia":mascara_morfologia
    }

    resultats_8=comparar_metodes(
        mascares_8,
        mascara_gt
    )

    for nom_metode in resultats_8:

        imprimir_metriques(
            nom_metode,
            resultats_8[nom_metode]
        )


    # --------------------------------------------------------
    # VISUALITZACIO
    # --------------------------------------------------------

    plt.figure(figsize=(14,8))

    plt.suptitle(
        "PROVA BLOC 8 - "+nom_imatge,
        fontsize=16
    )

    plt.subplot(2,3,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.imshow(mascara_otsu,cmap="gray")
    plt.title("Otsu")
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.imshow(disc_r,cmap="gray")
    plt.title("Disc optic")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.imshow(mascara_sense_disc,cmap="gray")
    plt.title("Otsu - disc")
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.imshow(mascara_morfologia,cmap="gray")
    plt.title("Otsu - disc + morfologia")
    plt.axis("off")

    plt.tight_layout(
        rect=[0,0,1,0.95],
        h_pad=2
    )

    plt.show()
    
    
# ============================================================
# BLOC 9 - WATERSHED
# ============================================================

print("\n====================")
print("BLOC 9 - WATERSHED")
print("====================")


for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("PROVA BLOC 9:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt


    # --------------------------------------------------------
    # PREPROCESSAMENT
    # --------------------------------------------------------

    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
        ruta_imatge,
        ruta_gt
    )

    canal_r=canals[0]
    canal_g=canals[1]


    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    mascara_otsu=segmentar_otsu(
        canal_g,
        mascara_retina
    )


    # --------------------------------------------------------
    # DISC OPTIC
    # --------------------------------------------------------

    r_suau,r_candidats,disc_r=detecta_disc_optic(
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3
    )

    mascara_sense_disc=elimina_disc_optic(
        mascara_otsu,
        disc_r
    )


    # --------------------------------------------------------
    # MORFOLOGIA
    # --------------------------------------------------------

    mascara_morfologia=netejar_mascara(
        mascara_sense_disc,
        mida_minima=50,
        mida_forat=50
    )


    # --------------------------------------------------------
    # WATERSHED
    # --------------------------------------------------------

    # Transformada de distancia
    distancia=transformar_distancia(
        mascara_morfologia
    )

    # Busquem els centres dels possibles blobs
    marcadors=trobar_marcadors(
        distancia,
        mascara_morfologia,
        distancia_minima=5
    )

    # Separem els blobs
    etiquetes=aplicar_watershed(
        mascara_morfologia,
        distancia_minima=5
    )


    # Watershed retorna etiquetes:
    # 0=fons
    # 1=blob 1
    # 2=blob 2
    # ...
    #
    # Si volem tornar a una mascara binaria:
    mascara_watershed=etiquetes>0


    # Nombre de blobs detectats
    nombre_blobs=np.max(etiquetes)

    print("Nombre de blobs després de Watershed:",nombre_blobs)


    # --------------------------------------------------------
    # METRIQUES
    # --------------------------------------------------------

    mascares_9={
        "Otsu - disc + morfologia":mascara_morfologia,
        "Watershed binari":mascara_watershed
    }

    resultats_9=comparar_metodes(
        mascares_9,
        mascara_gt
    )

    for nom_metode in resultats_9:

        imprimir_metriques(
            nom_metode,
            resultats_9[nom_metode]
        )


    # --------------------------------------------------------
    # VISUALITZACIO
    # --------------------------------------------------------

    plt.figure(figsize=(14,8))

    plt.suptitle(
        "PROVA BLOC 9 - "+nom_imatge,
        fontsize=16
    )


    # Ground Truth
    plt.subplot(2,3,1)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")


    # Mascara abans de watershed
    plt.subplot(2,3,2)
    plt.imshow(mascara_morfologia,cmap="gray")
    plt.title("Mascara morfologia")
    plt.axis("off")


    # Transformada de distancia
    plt.subplot(2,3,3)
    plt.imshow(distancia,cmap="gray")
    plt.title("Transformada distancia")
    plt.axis("off")


    # Marcadors
    plt.subplot(2,3,4)
    plt.imshow(marcadors,cmap="nipy_spectral")
    plt.title("Marcadors")
    plt.axis("off")


    # Watershed etiquetat
    plt.subplot(2,3,5)
    plt.imshow(etiquetes,cmap="nipy_spectral")
    plt.title("Watershed - blobs")
    plt.axis("off")


    # Watershed binari
    plt.subplot(2,3,6)
    plt.imshow(mascara_watershed,cmap="gray")
    plt.title("Watershed binari")
    plt.axis("off")


    plt.tight_layout(
        rect=[0,0,1,0.95],
        h_pad=2
    )

    plt.show()
    
