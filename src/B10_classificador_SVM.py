
# -- coding: utf-8 --
"""
Bloc 10: Classificador SVM de blobs.

En aquest bloc utilitzem les features calculades al bloc anterior
per entrenar un SVM que decideixi si cada blob candidat correspon
a un exsudat o a un fals positiu.

Abans d'entrenar eliminem els blobs ambigus, normalitzem les features
i utilitzem Chi2 per seleccionar les que aporten més informació.

Finalment reconstruim una mascara conservant només els blobs que
el SVM ha classificat com exsudats.
"""

import numpy as np

from skimage.measure import label

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.svm import SVC
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

from src.B9_deteccio_blobs import columnes_svm


def elimina_ambigus(taula):
    """
    Eliminem els blobs que tenen etiqueta -1.

    Aquests blobs tenen un solapament intermedi amb el Ground Truth
    i no els utilitzem per entrenar ni avaluar el classificador.
    """

    taula_bona=taula[taula["etiqueta"]!=-1].copy()

    return taula_bona


def separa_train_test_per_imatge(taula,imatges_test):
    """
    Separem les dades de train i test utilitzant el nom de la imatge.

    D'aquesta manera evitem que blobs d'una mateixa imatge apareguin
    tant durant l'entrenament com durant el test.
    """

    taula_train=taula[~taula["nom_imatge"].isin(imatges_test)].copy()
    taula_test=taula[taula["nom_imatge"].isin(imatges_test)].copy()

    return taula_train,taula_test


def prepara_x_y(taula):
    """
    Separem les features que utilitzara el SVM de les etiquetes reals.

    X conté les caracteristiques dels blobs i y indica si són
    exsudats o falsos positius.
    """

    columnes=columnes_svm()

    x=taula[columnes].values
    y=taula["etiqueta"].values.astype(int)

    return x,y,columnes


def entrena_svm(taula_train,k_chi2=3):
    """
    Entrenem el classificador SVM.

    Primer eliminem els casos ambigus i normalitzem totes les features
    entre 0 i 1. Després utilitzem Chi2 per quedar-nos amb les features
    més útils i finalment entrenem un SVM amb kernel RBF.
    """

    taula_train=elimina_ambigus(taula_train)

    x_train,y_train,columnes=prepara_x_y(taula_train)

    # Necessitem exemples positius i negatius perquè el SVM
    # pugui aprendre a separar les dues classes.
    classes=np.unique(y_train)

    if len(classes)<2:
        raise ValueError("El train nomes te una classe. Calen positius i negatius.")

    # Normalitzem les features entre 0 i 1. A més de posar-les
    # en una escala semblant, Chi2 necessita treballar amb valors positius.
    escalador=MinMaxScaler()
    x_train_norm=escalador.fit_transform(x_train)

    # Chi2 dona una puntuacio a cada feature segons la seva relacio
    # amb la classe i seleccionem les k que obtenen millor resultat.
    selector=SelectKBest(score_func=chi2,k=k_chi2)
    x_train_sel=selector.fit_transform(x_train_norm,y_train)

    # Utilitzem un kernel RBF perquè permet separar les classes
    # encara que la frontera entre elles no sigui lineal.
    # Donem més pes als positius perquè normalment tenim molts
    # més falsos positius que exsudats.
    model=SVC(kernel="rbf",C=1.0,gamma="scale",class_weight={0:1,1:2})

    model.fit(x_train_sel,y_train)

    return escalador,selector,model


def imprimeix_features_seleccionades(selector):
    """
    Mostrem les features seleccionades per Chi2 i la puntuacio
    que ha obtingut cadascuna.
    """

    columnes=columnes_svm()
    mascara=selector.get_support()
    puntuacions=selector.scores_

   
    print("FEATURES SELECCIONADES AMB CHI2")

    for i in range(len(columnes)):

        if mascara[i]:
            print(columnes[i],"score:",round(puntuacions[i],4))


def prediu_taula(taula,escalador,selector,model):
    """
    Apliquem el SVM sobre els blobs d'una taula.

    Afegim una nova columna anomenada prediccio_svm:
        1 = exsudat
        0 = fals positiu
    """

    x,y,columnes=prepara_x_y(taula)

    # Hem d'aplicar exactament la mateixa normalitzacio i seleccio
    # de features que s'ha utilitzat durant l'entrenament.
    x_norm=escalador.transform(x)
    x_sel=selector.transform(x_norm)

    prediccio=model.predict(x_sel)

    taula_predita=taula.copy()
    taula_predita["prediccio_svm"]=prediccio

    return taula_predita


def avalua_prediccio_blobs(taula_predita):
    """
    Avaluem el resultat del SVM a nivell de blobs.

    Aquestes metriques indiquen si el classificador reconeix correctament
    els blobs, però encara no són les metriques finals de segmentacio
    calculades pixel a pixel.
    """

    taula_eval=elimina_ambigus(taula_predita)

    y_real=taula_eval["etiqueta"].values.astype(int)
    y_pred=taula_eval["prediccio_svm"].values.astype(int)

    precisio=precision_score(y_real,y_pred,zero_division=0)
    recall=recall_score(y_real,y_pred,zero_division=0)
    f1=f1_score(y_real,y_pred,zero_division=0)
    accuracy=accuracy_score(y_real,y_pred)

    print("AVALUACIO SVM A NIVELL DE BLOBS")


    print("Precision:",round(precisio,4))
    print("Recall:",round(recall,4))
    print("F1:",round(f1,4))
    print("Accuracy:",round(accuracy,4))


def reconstrueix_mascara_svm(mascara_candidats,taula_predita):
    """
    Reconstruim la mascara final a partir de les prediccions del SVM.

    Conservem els blobs classificats com exsudats i eliminem
    els que el classificador considera falsos positius.
    """

    mascara_candidats=mascara_candidats.astype(bool)

    # Tornem a separar la mascara en components connectades utilitzant
    # el mateix connectivity=2 que al Bloc 9. Això permet que els blob_id
    # coincideixin amb els que tenim guardats a la taula.
    etiquetes=label(mascara_candidats,connectivity=2)

    mascara_final=np.zeros_like(mascara_candidats,dtype=bool)

    # Recorrem les prediccions i recuperem només les regions
    # que el SVM ha classificat com exsudats.
    for i in range(len(taula_predita)):

        fila=taula_predita.iloc[i]

        blob_id=int(fila["blob_id"])
        prediccio=int(fila["prediccio_svm"])

        if prediccio==1:
            mascara_blob=etiquetes==blob_id
            mascara_final=mascara_final|mascara_blob

    return mascara_final


def classifica_mascara(mascara_candidats,taula_features,escalador,selector,model):
    """
    Funcio que permet aplicar directament el classificador
    sobre una mascara candidata.

    Primer classifiquem els seus blobs i després reconstruim
    la mascara conservant només els que han estat acceptats.
    """

    taula_predita=prediu_taula(taula_features,escalador,selector,model)

    mascara_svm=reconstrueix_mascara_svm(mascara_candidats,taula_predita)

    return mascara_svm,taula_predita




"""
No  farem cap prova de funcionament a aquest bloc perquè el correcte funcionament
significa entrenar un SVM amb les imatges, d'una manera idèntica a com fem al main.py 
per exemple, de manera que repetirho aquí seria d'alguna manera repetirse. 
per veure el correcte funcionament el millor és anar al main i veure ahi el resultat
"""

# Comentari IA:
# Hem utilitzat IA com a suport sobretot en la part d'entrenament del SVM,
# ja que al principi no teniem clar com preparar les dades ni quin ordre
# seguir entre la normalitzacio, la seleccio de features i l'entrenament.
#
# Ens ha servit per entendre com separar correctament X i y abans d'aplicar Chi2 i com 
# classificar segpns les features seleccionades. Per com reconstruir finalment
# la mascara a partir dels blobs classificats pel SVM també ha sigut necessaria
