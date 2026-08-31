# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:56:30 2026

@author: gurpi
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.metriques import calcular_metriques, imprimir_metriques, comparar_metodes
from src.preprocessament_3 import preprocessa_imatge, visualitza_preprocessament
from src.segmentacio import obtenir_mascares_baseline
from src.mejora_contraste_5 import millora_contrast, visualitza_contrast
from src.segmentacio import segmentar_otsu
from src.nick_inrg import invertir_canal, segmentar_nick, segmentar_inrg
from src.disc_optic import detecta_disc_optic, elimina_disc_optic
from src.postprocessament import netejar_mascara
from src.watershed import transformar_distancia, trobar_marcadors, aplicar_watershed

from src.deteccio_blobs import (
    crea_taula_features,
    imprimeix_resum_features,
    ajunta_taules
)

from src.classificador_SVM import (
    separa_train_test_per_imatge,
    entrena_svm,
    imprimeix_features_seleccionades,
    prediu_taula,
    avalua_prediccio_blobs,
    reconstrueix_mascara_svm
)
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
    

# ============================================================
# BLOC 10 - FEATURES DELS BLOBS
# PROVA DELS DOS PIPELINES
# ============================================================

print("\n==========================================")
print("BLOC 10 - FEATURES DELS DOS PIPELINES")
print("==========================================")


# Guardarem les taules de cada imatge.
taules_otsu = []
taules_inrg = []


# Guardarem també les etiquetes de Watershed,
# la GT i la imatge perquè les necessitarem al Bloc 11.
dades_otsu = {}
dades_inrg = {}


for nom_imatge,nom_gt in imatges:

    print("\n==========================================")
    print("PROVA BLOC 10:",nom_imatge)
    print("==========================================")


    ruta_imatge = ruta_dades/nom_imatge
    ruta_gt = ruta_dades/nom_gt

    
    
# ============================================================
# BLOC 10 + 11 - FEATURES DE BLOBS + SVM
# ============================================================

print("\n====================================")
print("BLOC 10 + 11 - FEATURES + SVM")
print("====================================")


# Guardarem les taules de features de totes les imatges.
taules_features=[]

# Guardem tambe les mascares i GT per reconstruir despres.
mascares_candidats={}
mascares_gt={}
imatges_originals={}


# ============================================================
# PRIMER: CREEM FEATURES DE CADA IMATGE
# ============================================================

for nom_imatge,nom_gt in imatges:

    print("\n----------------------------------")
    print("BLOC 10 - FEATURES:",nom_imatge)
    print("----------------------------------")

    ruta_imatge=ruta_dades/nom_imatge
    ruta_gt=ruta_dades/nom_gt
>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc


    # --------------------------------------------------------
    # PREPROCESSAMENT
    # --------------------------------------------------------

<<<<<<< HEAD
    imatge,mascara_gt,mascara_retina,imatge_retina,canals = preprocessa_imatge(
=======
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(
>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc
        ruta_imatge,
        ruta_gt
    )

<<<<<<< HEAD
    canal_r = canals[0]
    canal_g = canals[1]


    # --------------------------------------------------------
    # DETECCIO DEL DISC OPTIC
    # --------------------------------------------------------

    # De moment mantenim els parametres que hem utilitzat
    # en les ultimes proves.
    _,_,disc_r = detecta_disc_optic(
=======
    canal_r=canals[0]
    canal_g=canals[1]


    # --------------------------------------------------------
    # MASCARA CANDIDATA
    # --------------------------------------------------------
    # Per a la primera prova del SVM usem la branca que ja teniem:
    #
    # G original -> Otsu -> disc optic -> morfologia -> Watershed
    #
    # Aquesta mascara NO es encara el resultat final.
    # Es una mascara de candidats que el SVM intentara filtrar.

    mascara_otsu=segmentar_otsu(
        canal_g,
        mascara_retina
    )


    # --------------------------------------------------------
    # DISC OPTIC
    # --------------------------------------------------------

    r_suau,r_candidats,disc_r=detecta_disc_optic(
>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc
        canal_r,
        mascara_retina,
        radi_tancament=4,
        radi_dilatacio=3
    )

<<<<<<< HEAD

    # ========================================================
    # PIPELINE A
    # OTSU
    # ========================================================

    mascara_otsu = segmentar_otsu(
        canal_g,
        mascara_retina
    )


    mascara_otsu_disc = elimina_disc_optic(
=======
    mascara_sense_disc=elimina_disc_optic(
>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc
        mascara_otsu,
        disc_r
    )


<<<<<<< HEAD
    mascara_otsu_morf = netejar_mascara(
        mascara_otsu_disc,
=======
    # --------------------------------------------------------
    # MORFOLOGIA
    # --------------------------------------------------------
    # Ara mantenim la mateixa configuracio que teniem al Bloc 9.
    # Mes endavant podem provar mida_minima=10 i mida_forat=20.

    mascara_morfologia=netejar_mascara(
        mascara_sense_disc,
>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc
        mida_minima=50,
        mida_forat=50
    )


<<<<<<< HEAD
    etiquetes_otsu = aplicar_watershed(
        mascara_otsu_morf,
        distancia_minima=5
    )


    # --------------------------------------------------------
    # FEATURES PIPELINE A
    # --------------------------------------------------------

    taula_otsu = crea_taula_features(
        imatge,
        canals,
        etiquetes_otsu,
        mascara_gt,
        nom_imatge=nom_imatge,
        nom_canal="G",
        nom_metode="Otsu"
    )


    print("\n---------- PIPELINE A: OTSU ----------")

    print(
        "Blobs Watershed:",
        np.max(etiquetes_otsu)
    )

    imprimeix_resum_features(
        taula_otsu,
        nom_imatge+" - Otsu"
    )


    # Metriques abans del SVM.
    mascara_candidats_otsu = etiquetes_otsu > 0

    imprimir_metriques(
        "PIPELINE A - Otsu abans SVM",
        calcular_metriques(
            mascara_candidats_otsu,
            mascara_gt
        )
    )


    taules_otsu.append(
        taula_otsu
    )


    dades_otsu[nom_imatge] = {

        "imatge":imatge,

        "gt":mascara_gt,

        "etiquetes":etiquetes_otsu,

        "candidats":mascara_candidats_otsu
    }


    # ========================================================
    # PIPELINE B
    # INRG
    # ========================================================

    # INRG inverteix internament el canal.
    # Per tant li passem directament el canal G original.

    mascara_inrg = segmentar_inrg(
        canal_g,
        mascara_retina,
        mida_finestra=95,
        k=-0.2,
        iteracions=3,
        delta=0.1
    )


    mascara_inrg_disc = elimina_disc_optic(
        mascara_inrg,
        disc_r
    )


    mascara_inrg_morf = netejar_mascara(
        mascara_inrg_disc,
        mida_minima=50,
        mida_forat=50
    )


    etiquetes_inrg = aplicar_watershed(
        mascara_inrg_morf,
        distancia_minima=5
    )


    # --------------------------------------------------------
    # FEATURES PIPELINE B
    # --------------------------------------------------------

    taula_inrg = crea_taula_features(
        imatge,
        canals,
        etiquetes_inrg,
        mascara_gt,
        nom_imatge=nom_imatge,
        nom_canal="G",
        nom_metode="INRG"
    )


    print("\n---------- PIPELINE B: INRG ----------")

    print(
        "Blobs Watershed:",
        np.max(etiquetes_inrg)
    )

    imprimeix_resum_features(
        taula_inrg,
        nom_imatge+" - INRG"
    )


    mascara_candidats_inrg = etiquetes_inrg > 0


    imprimir_metriques(
        "PIPELINE B - INRG abans SVM",
        calcular_metriques(
            mascara_candidats_inrg,
            mascara_gt
        )
    )


    taules_inrg.append(
        taula_inrg
    )


    dades_inrg[nom_imatge] = {

        "imatge":imatge,

        "gt":mascara_gt,

        "etiquetes":etiquetes_inrg,

        "candidats":mascara_candidats_inrg
    }


# ============================================================
# AJUNTEM LES TAULES
# ============================================================

taula_total_otsu = ajunta_taules(
    taules_otsu
)

taula_total_inrg = ajunta_taules(
    taules_inrg
)


print("\n==========================================")
print("RESUM BLOC 10")
print("==========================================")


print("\nPIPELINE A - OTSU")

print(
    "Blobs totals:",
    len(taula_total_otsu)
)

print(
    "Positius:",
    np.sum(taula_total_otsu["etiqueta"]==1)
)

print(
    "Negatius:",
    np.sum(taula_total_otsu["etiqueta"]==0)
)

print(
    "Ambigus:",
    np.sum(taula_total_otsu["etiqueta"]==-1)
)


print("\nPIPELINE B - INRG")

print(
    "Blobs totals:",
    len(taula_total_inrg)
)

print(
    "Positius:",
    np.sum(taula_total_inrg["etiqueta"]==1)
)

print(
    "Negatius:",
    np.sum(taula_total_inrg["etiqueta"]==0)
)

print(
    "Ambigus:",
    np.sum(taula_total_inrg["etiqueta"]==-1)
)

# ============================================================
# BLOC 11 - SVM
# COMPARACIO DELS DOS PIPELINES
# ============================================================

print("\n==========================================")
print("BLOC 11 - SVM DELS DOS PIPELINES")
print("==========================================")


# Guardarem tots els resultats per calcular
# una mitjana final.
resum_final = []


# ============================================================
# FUNCIO DE PROVA D'UN PIPELINE
# ============================================================

def prova_svm_pipeline(
    taula_total,
    dades,
    nom_test,
    nom_pipeline
):

    print("\n------------------------------------------")
    print(nom_pipeline)
    print("Imatge test:",nom_test)
    print("------------------------------------------")


    # --------------------------------------------------------
    # TRAIN / TEST PER IMATGE
    # --------------------------------------------------------

    taula_train,taula_test = separa_train_test_per_imatge(
        taula_total,
        [nom_test]
    )


    print(
        "Blobs train:",
        len(taula_train)
    )

    print(
        "Blobs test:",
        len(taula_test)
    )


    print(
        "Positius train:",
        np.sum(taula_train["etiqueta"]==1)
    )

    print(
        "Negatius train:",
        np.sum(taula_train["etiqueta"]==0)
    )

    print(
        "Ambigus train:",
        np.sum(taula_train["etiqueta"]==-1)
=======
    # --------------------------------------------------------
    # WATERSHED
    # --------------------------------------------------------

    etiquetes_ws=aplicar_watershed(
        mascara_morfologia,
        distancia_minima=5
    )

    mascara_watershed=etiquetes_ws>0


    # --------------------------------------------------------
    # BLOC 10 - FEATURES
    # --------------------------------------------------------

    taula=crea_taula_features(
        imatge,
        canals,
        mascara_watershed,
        mascara_gt,
        nom_imatge=nom_imatge,
        nom_canal="G original",
        nom_metode="Otsu + OD + morfologia + WS",
        llindar_positiu=0.30,
        llindar_negatiu=0.05
    )

    imprimeix_resum_features(
        taula,
        nom_imatge
    )

    taules_features.append(taula)

    # Guardem coses per reconstruir la mascara despres del SVM.
    mascares_candidats[nom_imatge]=mascara_watershed
    mascares_gt[nom_imatge]=mascara_gt
    imatges_originals[nom_imatge]=imatge


# ============================================================
# SEGON: AJUNTEM TOTES LES TAULES
# ============================================================

taula_total=ajunta_taules(
    taules_features
)

print("\n====================================")
print("TAULA TOTAL FEATURES")
print("====================================")

print("Nombre total de blobs:",len(taula_total))

print("Positius totals:",np.sum(taula_total["etiqueta"]==1))
print("Negatius totals:",np.sum(taula_total["etiqueta"]==0))
print("Ambigus totals:",np.sum(taula_total["etiqueta"]==-1))


# ============================================================
# TERCER: SEPAREM TRAIN I TEST PER IMATGE
# ============================================================

# Per fer una primera prova, deixem una imatge per test.
# Pots canviar aquesta llista per provar una altra.
imatges_test=[
    "Exudate_DR2019_003.jpg"
]

taula_train,taula_test=separa_train_test_per_imatge(
    taula_total,
    imatges_test
)

print("\n====================================")
print("SEPARACIO TRAIN / TEST")
print("====================================")

print("Imatges test:",imatges_test)
print("Blobs train:",len(taula_train))
print("Blobs test:",len(taula_test))

print("Train positius:",np.sum(taula_train["etiqueta"]==1))
print("Train negatius:",np.sum(taula_train["etiqueta"]==0))
print("Train ambigus:",np.sum(taula_train["etiqueta"]==-1))

print("Test positius:",np.sum(taula_test["etiqueta"]==1))
print("Test negatius:",np.sum(taula_test["etiqueta"]==0))
print("Test ambigus:",np.sum(taula_test["etiqueta"]==-1))


# ============================================================
# QUART: ENTRENEM EL SVM
# ============================================================

# k_chi2=3 segueix la idea inicial de la revisio avançada:
# començar amb poques features seleccionades.
escalador,selector,model=entrena_svm(
    taula_train,
    k_chi2=3
)

imprimeix_features_seleccionades(
    selector
)


# ============================================================
# CINQUE: AVALUEM EL SVM A NIVELL DE BLOBS
# ============================================================

taula_test_predita=prediu_taula(
    taula_test,
    escalador,
    selector,
    model
)

avalua_prediccio_blobs(
    taula_test_predita
)


# ============================================================
# SISE: RECONSTRUIM MASCARES I COMPAREM AMB GT
# ============================================================

for nom_test in imatges_test:

    print("\n----------------------------------")
    print("RESULTAT SVM:",nom_test)
    print("----------------------------------")

    mascara_candidata=mascares_candidats[nom_test]
    mascara_gt=mascares_gt[nom_test]
    imatge=imatges_originals[nom_test]

    # Agafem nomes els blobs de la imatge de test actual.
    taula_imatge=taula_test_predita[
        taula_test_predita["nom_imatge"]==nom_test
    ]

    # Reconstruim la mascara final deixant nomes els blobs
    # que el SVM ha classificat com exsudat.
    mascara_svm=reconstrueix_mascara_svm(
        mascara_candidata,
        taula_imatge
>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc
    )


    # --------------------------------------------------------
<<<<<<< HEAD
    # ENTRENAR SVM
    # --------------------------------------------------------

    try:

        escalador,selector,model = entrena_svm(
            taula_train,
            k_chi2=3
        )

    except ValueError as error:

        print(
            "ERROR ENTRENANT:",
            error
        )

        return None


    # Features seleccionades per Chi2.
    imprimeix_features_seleccionades(
        selector
    )


    # --------------------------------------------------------
    # PREDICCIO DELS BLOBS
    # --------------------------------------------------------

    taula_predita = prediu_taula(
        taula_test,
        escalador,
        selector,
        model
    )


    avalua_prediccio_blobs(
        taula_predita
    )


    # --------------------------------------------------------
    # RECONSTRUIR MASCARA
    # --------------------------------------------------------

    etiquetes = dades[
        nom_test
    ]["etiquetes"]


    mascara_gt = dades[
        nom_test
    ]["gt"]


    mascara_abans = dades[
        nom_test
    ]["candidats"]


    mascara_svm = reconstrueix_mascara_svm(
        etiquetes,
        taula_predita
    )


    # --------------------------------------------------------
    # METRIQUES PIXEL A PIXEL
    # --------------------------------------------------------

    metriques_abans = calcular_metriques(
        mascara_abans,
        mascara_gt
    )


    metriques_svm = calcular_metriques(
=======
    # METRIQUES ABANS I DESPRES DEL SVM
    # --------------------------------------------------------

    metriques_abans=calcular_metriques(
        mascara_candidata,
        mascara_gt
    )

    metriques_despres=calcular_metriques(

        mascara_svm,
        mascara_gt
    )



    print("\nMETRIQUES ABANS DEL SVM")

    imprimir_metriques(
        nom_pipeline+" abans SVM",
        metriques_abans
    )


    print("\nMETRIQUES DESPRES DEL SVM")

    imprimir_metriques(
        nom_pipeline+" + SVM",
        metriques_svm
    )


    return {

        "mascara_abans":mascara_abans,

        "mascara_svm":mascara_svm,

        "metriques_abans":metriques_abans,

        "metriques_svm":metriques_svm
    }


# ============================================================
# LEAVE-ONE-IMAGE-OUT
# ============================================================

for nom_test,nom_gt in imatges:

    print("\n\n##########################################")
    print("IMATGE DE TEST:",nom_test)
    print("##########################################")


    # --------------------------------------------------------
    # PIPELINE A
    # --------------------------------------------------------

    resultat_otsu = prova_svm_pipeline(

        taula_total_otsu,

        dades_otsu,

        nom_test,

        "PIPELINE A - OTSU"

    imprimir_metriques(
        "Abans SVM",
        metriques_abans
    )

    imprimir_metriques(
        "Despres SVM",
        metriques_despres>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc
    )


    # --------------------------------------------------------

    # PIPELINE B
    # --------------------------------------------------------

    resultat_inrg = prova_svm_pipeline(

        taula_total_inrg,

        dades_inrg,

        nom_test,

        "PIPELINE B - INRG"
    )


    # Si algun SVM no s'ha pogut entrenar,
    # passem a la següent imatge.
    if resultat_otsu is None or resultat_inrg is None:

        continue


    # --------------------------------------------------------
    # GUARDAR RESULTATS
    # --------------------------------------------------------

    resum_final.append({

        "imatge":nom_test,

        "otsu_abans":resultat_otsu["metriques_abans"],

        "otsu_svm":resultat_otsu["metriques_svm"],

        "inrg_abans":resultat_inrg["metriques_abans"],

        "inrg_svm":resultat_inrg["metriques_svm"]
    })


    # --------------------------------------------------------
    # VISUALITZACIO FINAL
    # --------------------------------------------------------

    imatge_original = dades_otsu[
        nom_test
    ]["imatge"]


    mascara_gt = dades_otsu[
        nom_test
    ]["gt"]


    plt.figure(
        figsize=(16,9)
    )


    plt.suptitle(
        "COMPARACIO PIPELINES - "+nom_test,
        fontsize=16
    )


    # ORIGINAL
    plt.subplot(2,3,1)

    plt.imshow(
        imatge_original
    )

    plt.title(
        "Original"
    )

    plt.axis(
        "off"
    )


    # GROUND TRUTH
    plt.subplot(2,3,2)

    plt.imshow(
        mascara_gt,
        cmap="gray"
    )

    plt.title(
        "Ground Truth"
    )

    plt.axis(
        "off"
    )


    # OTSU ABANS DEL SVM
    plt.subplot(2,3,3)

    plt.imshow(
        resultat_otsu["mascara_abans"],
        cmap="gray"
    )

    plt.title(
        "Otsu abans SVM"
    )

    plt.axis(
        "off"
    )


    # OTSU DESPRES SVM
    plt.subplot(2,3,4)

    plt.imshow(
        resultat_otsu["mascara_svm"],
        cmap="gray"
    )

    plt.title(
        "Otsu + SVM"
    )

    plt.axis(
        "off"
    )


    # INRG ABANS DEL SVM
    plt.subplot(2,3,5)

    plt.imshow(
        resultat_inrg["mascara_abans"],
        cmap="gray"
    )

    plt.title(
        "INRG abans SVM"
    )

    plt.axis(
        "off"
    )


    # INRG DESPRES DEL SVM
    plt.subplot(2,3,6)

    plt.imshow(
        resultat_inrg["mascara_svm"],
        cmap="gray"
    )

    plt.title(
        "INRG + SVM"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout(
        rect=[0,0,1,0.94]
    )

    plt.show()
    

    # VISUALITZACIO
    # --------------------------------------------------------
    # Mantenim la representacio de les proves:
    # Ground Truth en gris i mascara obtinguda en verd.

    plt.figure(figsize=(14,5))

    plt.suptitle(
        "BLOC 11 - SVM - "+nom_test,
        fontsize=16
    )

    plt.subplot(1,3,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mascara_gt,cmap="gray",alpha=1)
    plt.imshow(mascara_candidata,cmap="Greens",alpha=0.5)
    plt.title("Abans SVM")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(mascara_gt,cmap="gray",alpha=1)
    plt.imshow(mascara_svm,cmap="Greens",alpha=0.5)
    plt.title("Despres SVM")
    plt.axis("off")

    plt.tight_layout(
        rect=[0,0,1,0.90]
    )

    plt.show()

