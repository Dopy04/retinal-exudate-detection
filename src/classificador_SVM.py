"""
Bloc 11 - Classificador SVM de blobs.

Aquest bloc agafa la taula de features del Bloc 10 i entrena un SVM
per decidir quins blobs son exsudats i quins son falsos positius.

La idea es:

    taula features
        ↓
    eliminar blobs ambigus
        ↓
    seleccionar columnes valides
        ↓
    Chi2 per triar les millors features
        ↓
    SVM
        ↓
    reconstruir mascara final nomes amb els blobs acceptats
"""

import numpy as np

from skimage.measure import label

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.deteccio_blobs import columnes_svm


# -------------------------------------------------------------------------
# 1. Preparar dades per al SVM
# -------------------------------------------------------------------------

def  elimina_ambigus(taula):
    """
    Elimina els blobs amb etiqueta -1.

    etiqueta = 1   exsudat
    etiqueta = 0   fals positiu
    etiqueta = -1  ambigu

    Els ambigus no els usem per entrenar al principi.
    """

    taula_bona = taula[taula["etiqueta"] != -1].copy()

    return taula_bona


def separa_train_test_per_imatge(taula, imatges_test):
    """
    Separa train i test segons el nom de la imatge.

    Aixo es important perquè no volem entrenar i testejar amb blobs
    de la mateixa imatge.
    """

    taula_train = taula[~taula["nom_imatge"].isin(imatges_test)].copy()
    taula_test = taula[taula["nom_imatge"].isin(imatges_test)].copy()

    return taula_train, taula_test


def prepara_x_y(taula):
    """
    Prepara X i y per al SVM.

    X conte nomes les features.
    y conte l'etiqueta.

    No usem solapament_gt com a feature perquè en una imatge nova
    no tindrem Ground Truth.
    """

    columnes = columnes_svm()

    x = taula[columnes].values
    y = taula["etiqueta"].values.astype(int)

    return x, y, columnes


# -------------------------------------------------------------------------
# 2. Entrenar SVM amb Chi2
# -------------------------------------------------------------------------

def entrena_svm(taula_train, k_chi2=3):
    """
    Entrena el classificador.

    Primer normalitzem les features a [0, 1], perquè Chi2 necessita
    valors no negatius.

    Despres seleccionem les k millors features amb Chi2.

    Finalment entrenem un SVM amb kernel RBF.
    """

    taula_train =  elimina_ambigus(taula_train)

    x_train, y_train, columnes = prepara_x_y(taula_train)

    # El SVM necessita exemples de les dues classes.
    # Si nomes tenim positius o nomes negatius, no pot aprendre.
    classes = np.unique(y_train)

    if len(classes) < 2:
        raise ValueError("El train nomes te una classe. Calen positius i negatius.")

    # Normalitzem totes les features entre 0 i 1.
    escalador = MinMaxScaler()
    x_train_norm = escalador.fit_transform(x_train)

    # Seleccionem les millors features.
    selector = SelectKBest(
        score_func=chi2,
        k=k_chi2,
    )

    x_train_sel = selector.fit_transform(
        x_train_norm,
        y_train,
    )

    # class_weight balanced ajuda quan tenim molts mes negatius que positius.
    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
    )

    model.fit(
        x_train_sel,
        y_train,
    )

    return escalador, selector, model


def imprimeix_features_seleccionades(selector):
    """
    Mostra quines features ha triat Chi2.
    """

    columnes = columnes_svm()
    mascara = selector.get_support()
    puntuacions = selector.scores_

    print("\n====================================")
    print("FEATURES SELECCIONADES AMB CHI2")
    print("====================================")

    for i in range(len(columnes)):

        if mascara[i]:
            print(columnes[i], "score:", round(puntuacions[i], 4))


# -------------------------------------------------------------------------
# 3. Predir blobs nous
# -------------------------------------------------------------------------

def prediu_taula(taula, escalador, selector, model):
    """
    Aplica el SVM sobre una taula de blobs.

    Retorna la mateixa taula, pero amb una columna nova:
        prediccio_svm
    """

    x, y, columnes = prepara_x_y(taula)

    x_norm = escalador.transform(x)
    x_sel = selector.transform(x_norm)

    prediccio = model.predict(x_sel)

    taula_predita = taula.copy()
    taula_predita["prediccio_svm"] = prediccio

    return taula_predita


def avalua_prediccio_blobs(taula_predita):
    """
    Avalua el SVM a nivell de blobs.

    Aquesta metrica no es encara la metrica final de segmentacio.
    Nomes ens diu si el classificador encerta blobs positius/negatius.
    """

    taula_eval =  elimina_ambigus(taula_predita)

    y_real = taula_eval["etiqueta"].values.astype(int)
    y_pred = taula_eval["prediccio_svm"].values.astype(int)

    precisio = precision_score(y_real, y_pred, zero_division=0)
    recall = recall_score(y_real, y_pred, zero_division=0)
    f1 = f1_score(y_real, y_pred, zero_division=0)
    accuracy = accuracy_score(y_real, y_pred)

    print("\n====================================")
    print("AVALUACIO SVM A NIVELL DE BLOBS")
    print("====================================")

    print("Precision:", round(precisio, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("Accuracy:", round(accuracy, 4))


# -------------------------------------------------------------------------
# 4. Reconstruir mascara final
# -------------------------------------------------------------------------

def reconstrueix_mascara_svm(mascara_candidats, taula_predita):
    """
    Reconstrueix una mascara final amb els blobs acceptats pel SVM.

    Si prediccio_svm = 1, el blob es conserva.
    Si prediccio_svm = 0, el blob s'elimina.
    """

    mascara_candidats = mascara_candidats.astype(bool)

    # Tornem a etiquetar la mascara candidata.
    # Ha de coincidir amb el label utilitzat en el Bloc 10.
    etiquetes = label(
        mascara_candidats,
        connectivity=2,
    )

    mascara_final = np.zeros_like(
        mascara_candidats,
        dtype=bool,
    )

    for i in range(len(taula_predita)):

        fila = taula_predita.iloc[i]

        blob_id = int(fila["blob_id"])
        prediccio = int(fila["prediccio_svm"])

        if prediccio == 1:
            mascara_blob = etiquetes == blob_id
            mascara_final = mascara_final | mascara_blob

    return mascara_final


def classifica_mascara(
    mascara_candidats,
    taula_features,
    escalador,
    selector,
    model,
):
    """
    Funcio directa per aplicar el SVM a una mascara candidata.

    Retorna:
        mascara_svm
        taula_predita
    """

    taula_predita = prediu_taula(
        taula_features,
        escalador,
        selector,
        model,
    )

    mascara_svm = reconstrueix_mascara_svm(
        mascara_candidats,
        taula_predita,
    )

    return mascara_svm, taula_predita

