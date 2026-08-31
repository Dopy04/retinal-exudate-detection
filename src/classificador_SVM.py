"""
Bloc 11 - Classificador SVM de blobs.

Aquest bloc agafa la taula de features del Bloc 10
i entrena un SVM per decidir quins blobs son exsudats
i quins son falsos positius.

Pipeline:

    taula features
        ↓
    eliminar blobs ambigus
        ↓
    normalitzar features
        ↓
    Chi2
        ↓
    SVM
        ↓
    prediccio dels blobs
        ↓
    reconstruccio de la mascara
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC

<<<<<<< HEAD
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

=======
>>>>>>> c4eff620fe55a851d55432915c358e766d1869cc
from src.deteccio_blobs import columnes_svm


# -------------------------------------------------------------------------
# 1. ELIMINAR BLOBS AMBIGUS
# -------------------------------------------------------------------------

def elimina_ambigus(
    taula
):
    """
    Elimina els blobs amb etiqueta -1.

    etiqueta = 1   exsudat
    etiqueta = 0   fals positiu
    etiqueta = -1  ambigu
    """

    taula_bona = taula[
        taula["etiqueta"] != -1
    ].copy()

    return taula_bona


# -------------------------------------------------------------------------
# 2. SEPARAR TRAIN I TEST
# -------------------------------------------------------------------------

def separa_train_test_per_imatge(
    taula,
    imatges_test
):
    """
    Separa train i test segons el nom de la imatge.

    D'aquesta manera no entrenem i testegem
    amb blobs de la mateixa imatge.
    """

    taula_train = taula[
        ~taula["nom_imatge"].isin(imatges_test)
    ].copy()


    taula_test = taula[
        taula["nom_imatge"].isin(imatges_test)
    ].copy()


    return (
        taula_train,
        taula_test
    )


# -------------------------------------------------------------------------
# 3. PREPARAR X
# -------------------------------------------------------------------------

def prepara_x(
    taula
):
    """
    Prepara les features X.

    Aquesta funcio serveix tant per entrenar
    com per fer prediccions.
    """

    columnes = columnes_svm()

    x = taula[
        columnes
    ].values.astype(float)


    # Evitem problemes si apareix algun
    # NaN o infinit en les features.
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        x,
        columnes
    )


def prepara_x_y(
    taula
):
    """
    Prepara X i y per entrenar el SVM.
    """

    x,columnes = prepara_x(
        taula
    )

    y = taula[
        "etiqueta"
    ].values.astype(int)


    return (
        x,
        y,
        columnes
    )


# -------------------------------------------------------------------------
# 4. ENTRENAR SVM
# -------------------------------------------------------------------------

def entrena_svm(
    taula_train,
    k_chi2=3
):
    """
    Entrena el classificador SVM.

    Passos:

        1. eliminar blobs ambigus
        2. normalitzar a [0,1]
        3. seleccionar features amb Chi2
        4. entrenar SVM
    """

    taula_train = elimina_ambigus(
        taula_train
    )


    # Comprovem que tenim blobs
    if len(taula_train) == 0:

        raise ValueError(
            "No hi ha blobs valids per entrenar."
        )


    x_train,y_train,columnes = prepara_x_y(
        taula_train
    )


    # ---------------------------------------------------------
    # COMPROVAR CLASSES
    # ---------------------------------------------------------

    classes = np.unique(
        y_train
    )


    if len(classes) < 2:

        raise ValueError(
            "El train nomes te una classe. "
            "Calen positius i negatius."
        )


    # ---------------------------------------------------------
    # NORMALITZACIO
    # ---------------------------------------------------------

    escalador = MinMaxScaler()


    x_train_norm = escalador.fit_transform(
        x_train
    )


    # ---------------------------------------------------------
    # SELECCIO DE FEATURES AMB CHI2
    # ---------------------------------------------------------

    selector = SelectKBest(
        score_func=chi2,
        k=k_chi2
    )


    x_train_sel = selector.fit_transform(
        x_train_norm,
        y_train
    )


    # ---------------------------------------------------------
    # SVM
    # ---------------------------------------------------------

    model = SVC(

        kernel="rbf",

        C=1.0,

        gamma="scale",

        class_weight="balanced"
    )


    model.fit(
        x_train_sel,
        y_train
    )


    return (
        escalador,
        selector,
        model
    )


# -------------------------------------------------------------------------
# 5. MOSTRAR FEATURES SELECCIONADES
# -------------------------------------------------------------------------

def imprimeix_features_seleccionades(
    selector
):
    """
    Mostra quines features ha seleccionat Chi2.
    """

    columnes = columnes_svm()

    mascara = selector.get_support()

    puntuacions = selector.scores_


    print("\n====================================")

    print(
        "FEATURES SELECCIONADES AMB CHI2"
    )

    print("====================================")


    for i in range(
        len(columnes)
    ):

        if mascara[i]:

            print(
                columnes[i],
                "score:",
                round(
                    puntuacions[i],
                    4
                )
            )


# -------------------------------------------------------------------------
# 6. PREDIR BLOBS
# -------------------------------------------------------------------------

def prediu_taula(
    taula,
    escalador,
    selector,
    model
):
    """
    Aplica el SVM sobre una taula de blobs.

    Afegeix una columna nova:

        prediccio_svm
    """

    if len(taula) == 0:

        taula_predita = taula.copy()

        taula_predita[
            "prediccio_svm"
        ] = []

        return taula_predita


    x,columnes = prepara_x(
        taula
    )


    # Mateixa normalitzacio que hem utilitzat
    # durant l'entrenament.
    x_norm = escalador.transform(
        x
    )


    # Mateixes features seleccionades
    x_sel = selector.transform(
        x_norm
    )


    # Prediccio del SVM
    prediccio = model.predict(
        x_sel
    )


    taula_predita = taula.copy()


    taula_predita[
        "prediccio_svm"
    ] = prediccio


    return taula_predita


# -------------------------------------------------------------------------
# 7. AVALUACIO A NIVELL DE BLOBS
# -------------------------------------------------------------------------

def avalua_prediccio_blobs(
    taula_predita
):
    """
    Avalua el SVM a nivell de blobs.

    Aquestes metriques no son encara
    les metriques finals de segmentacio.
    """

    taula_eval = elimina_ambigus(
        taula_predita
    )


    if len(taula_eval) == 0:

        print(
            "No hi ha blobs valids per avaluar."
        )

        return


    y_real = taula_eval[
        "etiqueta"
    ].values.astype(int)


    y_pred = taula_eval[
        "prediccio_svm"
    ].values.astype(int)


    precisio = precision_score(
        y_real,
        y_pred,
        zero_division=0
    )


    recall = recall_score(
        y_real,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_real,
        y_pred,
        zero_division=0
    )


    accuracy = accuracy_score(
        y_real,
        y_pred
    )


    print("\n====================================")

    print(
        "AVALUACIO SVM A NIVELL DE BLOBS"
    )

    print("====================================")


    print(
        "Precision:",
        round(precisio,4)
    )

    print(
        "Recall:",
        round(recall,4)
    )

    print(
        "F1:",
        round(f1,4)
    )

    print(
        "Accuracy:",
        round(accuracy,4)
    )


# -------------------------------------------------------------------------
# 8. RECONSTRUIR MASCARA
# -------------------------------------------------------------------------

def reconstrueix_mascara_svm(
    etiquetes,
    taula_predita
):
    """
    Reconstrueix la mascara final utilitzant
    directament les etiquetes de Watershed.

    Si:

        prediccio_svm = 1

    el blob es conserva.

    Si:

        prediccio_svm = 0

    el blob s'elimina.
    """

    etiquetes = np.asarray(
        etiquetes,
        dtype=int
    )


    mascara_final = np.zeros(
        etiquetes.shape,
        dtype=bool
    )


    for i in range(
        len(taula_predita)
    ):

        fila = taula_predita.iloc[i]


        blob_id = int(
            fila["blob_id"]
        )


        prediccio = int(
            fila["prediccio_svm"]
        )


        # Nomes conservem els blobs
        # classificats com exsudats.
        if prediccio == 1:

            mascara_blob = (
                etiquetes == blob_id
            )


            mascara_final = (
                mascara_final |
                mascara_blob
            )


    return mascara_final


# -------------------------------------------------------------------------
# 9. FUNCIO COMPLETA DE CLASSIFICACIO
# -------------------------------------------------------------------------

def classifica_mascara(
    etiquetes,
    taula_features,
    escalador,
    selector,
    model
):
    """
    Aplica directament el SVM
    als blobs obtinguts amb Watershed.

    Retorna:

        mascara_svm
        taula_predita
    """

    taula_predita = prediu_taula(

        taula_features,

        escalador,

        selector,

        model
    )


    mascara_svm = reconstrueix_mascara_svm(

        etiquetes,

        taula_predita
    )


    return (
        mascara_svm,
        taula_predita
    )