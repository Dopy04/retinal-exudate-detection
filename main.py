# -*- coding: utf-8 -*-
"""
Main final del projecte.

En aquest codi es proposa comparar el millor resultat del pipeline del
paper 2025 amb el pipeline que nosaltres proposem per la detecció d'exsudats:

    1. Pipeline adaptat del paper:
       V KMMRC -> INRG -> eliminacio del disc optic ->
       morfologia -> Watershed -> features -> SVM.

    2. Pipeline final proposat:
       V CLAHE -> segmentacio adaptativa -> eliminacio del disc optic ->
       morfologia -> Watershed -> features -> SVM.

La primera aproximacio segueix més de prop la idea del treball de referencia,
però aplicant-la al nostre cas concret d'exudats i no hemorragie"s

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import KFold

from src.B1_metriques import calcular_metriques,imprimir_metriques
from src.B2_preprocessament import preprocessa_imatge
from src.B4_segmentacio import segmentar_adaptatiu
from src.B4_millora_contrast import millora_contrast
from src.B5_nick_inrg import segmentar_inrg
from src.B6_disc_optic import detecta_disc_optic,elimina_disc_optic
from src.B7_postprocessament import netejar_mascara
from src.B8_watershed import aplicar_watershed
from src.B9_deteccio_blobs import crea_taula_features
from src.B10_classificador_SVM import entrena_svm,prediu_taula,reconstrueix_mascara_svm


#Configuracio general

ruta_projecte=Path(__file__).resolve().parent
ruta_dades=ruta_projecte/"data_samples"

#si es vol provar es pot canviar el numero per provar amb altres imatges
imatges_exemple=["DR2019_047","DR2019_008"]

k_chi2=8
llindar_positiu=0.25
llindar_negatiu=0.05
n_folds=5


#Dataset

def llista_imatges_dataset():
    """
    Busca totes les parelles imatge-Ground Truth disponibles
    als datasets DR2019 i JSIEC.
    """

    #Guardem el nom de cada dataset i les seves carpetes
    carpetes=[("DR2019","Exudate-DR2019","Exudate-DR2019-Groundtruth"),
              ("JSIEC","Exudate-JSIEC","Exudate-JSIEC-Groundtruth")]

    parelles=[]

    #Recorrem totes les imatges i suposem que el Ground Truth existeix
    for dataset,nom_carpeta,nom_carpeta_gt in carpetes:
        carpeta=ruta_dades/nom_carpeta
        carpeta_gt=ruta_dades/nom_carpeta_gt

        for ruta_imatge in sorted(carpeta.glob("*.jpg")):
            ruta_gt=carpeta_gt/ruta_imatge.name
            id_imatge=dataset+"_"+ruta_imatge.stem
            parelles.append({"id":id_imatge,"dataset":dataset,"ruta_imatge":ruta_imatge,"ruta_gt":ruta_gt})

    return parelles


def carrega_imatge(parella):
    """Carrega una imatge i aplica el preprocessament del bloc 2."""

    #Aquesta funcio només fa més curta la crida al preprocessament
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=preprocessa_imatge(parella["ruta_imatge"],parella["ruta_gt"])

    return imatge,mascara_gt,mascara_retina,imatge_retina,canals


#Funcions comunes dels pipelines

def elimina_disc(mascara,canals,mascara_retina):
    """Detecta el disc optic amb el canal R i l'elimina de la mascara."""

    #El canal R ajuda a trobar el disc optic perquè sol ser una zona brillant
    canal_r=canals[0]
    _,_,mascara_disc=detecta_disc_optic(canal_r,mascara_retina,radi_tancament=4,radi_dilatacio=3)

    #Traiem aquesta zona de la mascara candidata
    mascara_sense_disc=elimina_disc_optic(mascara,mascara_disc)

    return mascara_sense_disc,mascara_disc


def finalitza_pipeline(mascara_segmentacio,canals,mascara_retina):
    """
    Etapes comunes dels dos pipelines:
    eliminacio del disc optic, morfologia i Watershed.
    """

    #Primer eliminem el disc optic perquè podria comptar com a fals positiu
    mascara_sense_disc,mascara_disc=elimina_disc(mascara_segmentacio,canals,mascara_retina)

    #Després netegem la mascara i eliminem regions petites
    mascara_morfologia=netejar_mascara(mascara_sense_disc,mida_minima=50,mida_forat=50)

    #Amb Watershed separem regions 
    etiquetes=aplicar_watershed(mascara_morfologia,distancia_minima=5)


    mascara_candidats=etiquetes>0

    return mascara_disc,mascara_sense_disc,mascara_morfologia,etiquetes,mascara_candidats


#Pipeline del paper

def pipeline_paper(imatge_retina,canals,mascara_retina):
    """
    Pipeline adaptat del treball de referencia:
    V KMMRC, INRG, disc optic, morfologia, Watershed.
    """

    #Calculem els canals millorats i agafem el V KMMRC
    contrast=millora_contrast(imatge_retina,mascara_retina,canals)
    v_kmmrc=contrast[5]

    #Segmentem amb INRG sobre el canal millorat
    mascara_segmentacio=segmentar_inrg(v_kmmrc,mascara_retina,mida_finestra=95,
                                       k=-0.2,iteracions=3,delta=0.1)

    #identifiquem tot el que ens retorna la funció anterior per poder ferho servir
    mascara_disc,mascara_sense_disc,mascara_morfologia,etiquetes,mascara_candidats = finalitza_pipeline(mascara_segmentacio,canals,mascara_retina)

    return {"canal":v_kmmrc,"segmentacio":mascara_segmentacio,"disc":mascara_disc,
            "sense_disc":mascara_sense_disc,"morfologia":mascara_morfologia,"etiquetes":etiquetes,"candidats":mascara_candidats}


#Pipeline final

def pipeline_millor(imatge_retina,canals,mascara_retina):
    """
    Pipeline que ha donat millors resultats:
    V CLAHE, Adaptatiu, disc optic,  morfologia, Watershed.
    """

    #Calculem els canals millorats i agafem el V CLAHE
    contrast=millora_contrast(imatge_retina,mascara_retina,canals)
    v_clahe=contrast[4]

    #Fem la segmentacio adaptativa sobre V CLAHE
    mascara_segmentacio=segmentar_adaptatiu(v_clahe,mascara_retina)

   
    mascara_disc,mascara_sense_disc,mascara_morfologia,etiquetes,mascara_candidats = finalitza_pipeline(mascara_segmentacio,canals,mascara_retina)

    return {"canal":v_clahe,"segmentacio":mascara_segmentacio,"disc":mascara_disc,"sense_disc":mascara_sense_disc,
            "morfologia":mascara_morfologia,"etiquetes":etiquetes,"candidats":mascara_candidats}


#Features

def crea_features_pipeline(imatge,canals,mascara_gt,resultat,id_imatge,nom_canal,nom_metode):
    """Calcula les features dels blobs candidats abans de passar-los al SVM."""

    #Cada blob candidat es converteix en una fila de la taula
    taula=crea_taula_features(imatge,canals,resultat["candidats"],mascara_gt,nom_imatge=id_imatge,
                              nom_canal=nom_canal,nom_metode=nom_metode,llindar_positiu=llindar_positiu,
                              llindar_negatiu=llindar_negatiu)

    return taula


#Processament del dataset

def processa_dataset(parelles):
    """
    Processa totes les imatges pels dos pipelines fins arribar
    a les features utilitzades pel SVM.
    """

    dades={}

    print("\nProcessant dataset complet")

    #Processem totes les imatges una vegada i guardem els resultats
    for i,parella in enumerate(parelles):
        id_imatge=parella["id"]

        print(str(i+1)+"/"+str(len(parelles)),id_imatge)

        #Preprocessament comu
        imatge,gt,mascara_retina,imatge_retina,canals=carrega_imatge(parella)

        #Resultats dels dos pipelines
        resultat_paper=pipeline_paper(imatge_retina,canals,mascara_retina)
        resultat_millor=pipeline_millor(imatge_retina,canals,mascara_retina)

        #Taules de features de cada pipeline
        taula_paper=crea_features_pipeline(imatge,canals,gt,resultat_paper,id_imatge,"V KMMRC","INRG")
        taula_millor=crea_features_pipeline(imatge,canals,gt,resultat_millor,id_imatge,"V CLAHE","Adaptatiu")

        #Guardem tot per poder entrenar i avaluar sense repetir el proces
        dades[id_imatge]={"parella":parella,"imatge":imatge,"gt":gt,"paper":resultat_paper,
                          "millor":resultat_millor,"taula_paper":taula_paper,"taula_millor":taula_millor}

    return dades


#SVM

def ajunta_taules_ids(dades,ids,nom_taula):
    """Junta les features de les imatges que utilitzarem per entrenar."""

    #Agafem les taules de les imatges de train i les unim en una sola
    taules=[dades[id_imatge][nom_taula] for id_imatge in ids]

    return pd.concat(taules,ignore_index=True)


def entrena_model_pipeline(dades,ids_train,nom_taula):
    """Entrena el SVM només amb les imatges que formen part del train."""

    #Primer unim les features i després entrenem el SVM
    taula_train=ajunta_taules_ids(dades,ids_train,nom_taula)
    escalador,selector,model=entrena_svm(taula_train,k_chi2=k_chi2)

    return escalador,selector,model


def aplica_svm_imatge(dada,nom_pipeline,nom_taula,model_svm):
    """Aplica el SVM als blobs candidats d'una imatge de test."""

    escalador,selector,model=model_svm

    #Classifiquem els blobs de la imatge i reconstruim la mascara final
    taula_predita=prediu_taula(dada[nom_taula],escalador,selector,model)
    mascara_final=reconstrueix_mascara_svm(dada[nom_pipeline]["candidats"],taula_predita)

    return mascara_final


#Visualitzacio

def mostra_pipeline(dada,mascara_final,nom_pipeline):
    """Mostra tots els passos d'un pipeline."""

    if nom_pipeline=="paper":
        titol="PIPELINE PAPER - KMMRC + INRG"
        titol_canal="V KMMRC"
        titol_segmentacio="INRG"
    else:
        titol="PIPELINE FINAL - V CLAHE + ADAPTATIU"
        titol_canal="V CLAHE"
        titol_segmentacio="Adaptatiu"

    resultat=dada[nom_pipeline]

    #Calculem Dice i IoU per posar el resultat principal al titol
    metriques=calcular_metriques(mascara_final,dada["gt"])

    plt.figure(figsize=(18,9))
    plt.suptitle(titol+" - "+dada["parella"]["id"]+" | Dice="+str(round(metriques["dice"],3))+" | IoU="+str(round(metriques["iou"],3)),fontsize=15)

    plt.subplot(2,4,1)
    plt.imshow(dada["imatge"])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2,4,2)
    plt.imshow(dada["gt"],cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2,4,3)
    plt.imshow(resultat["canal"],cmap="gray")
    plt.title(titol_canal)
    plt.axis("off")

    plt.subplot(2,4,4)
    plt.imshow(resultat["segmentacio"],cmap="gray")
    plt.title(titol_segmentacio)
    plt.axis("off")

    plt.subplot(2,4,5)
    plt.imshow(resultat["sense_disc"],cmap="gray")
    plt.title("Sense disc optic")
    plt.axis("off")

    plt.subplot(2,4,6)
    plt.imshow(resultat["morfologia"],cmap="gray")
    plt.title("Morfologia")
    plt.axis("off")

    plt.subplot(2,4,7)
    plt.imshow(resultat["etiquetes"],cmap="nipy_spectral")
    plt.title("Watershed")
    plt.axis("off")

    plt.subplot(2,4,8)
    plt.imshow(mascara_final,cmap="gray")
    plt.title("Resultat final SVM")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.94])
    plt.show()


#Prova de les dues imatges

def prova_imatges_exemple(dades):
    """
    Entrena els dos SVM sense utilitzar les dues imatges d'exemple
    i després mostra els dos pipelines sobre cadascuna.
    """

    ids_totals=list(dades.keys())

    #Les imatges que mostrem no entren al train d'aquesta prova visual
    ids_train=[id_imatge for id_imatge in ids_totals if id_imatge not in imatges_exemple]

    #Entrenem un SVM per cada pipeline
    model_paper=entrena_model_pipeline(dades,ids_train,"taula_paper")
    model_millor=entrena_model_pipeline(dades,ids_train,"taula_millor")

    print("\nProva de les dues imatges")

    for id_imatge in imatges_exemple:
        dada=dades[id_imatge]

        #Apliquem cada model a la imatge d'exemple
        mascara_paper=aplica_svm_imatge(dada,"paper","taula_paper",model_paper)
        mascara_millor=aplica_svm_imatge(dada,"millor","taula_millor",model_millor)

        #Calculem les metriques finals de cada pipeline
        metriques_paper=calcular_metriques(mascara_paper,dada["gt"])
        metriques_millor=calcular_metriques(mascara_millor,dada["gt"])

        print("\nImatge:",id_imatge)

        imprimir_metriques("Pipeline paper",metriques_paper)
        imprimir_metriques("Pipeline final",metriques_millor)

        mostra_pipeline(dada,mascara_paper,"paper")
        mostra_pipeline(dada,mascara_millor,"millor")


#Metriques globals

def resum_metriques(llista):
    """Calcula la mitjana i la desviacio de cada metrica."""

    claus=["precisio","recall","f1","dice","iou","accuracy"]

    mitjana={}
    desviacio={}

    #Calculem el valor mitja i la desviacio per cada metrica
    for clau in claus:
        valors=[metriques[clau] for metriques in llista]
        mitjana[clau]=np.mean(valors)
        desviacio[clau]=np.std(valors)

    return mitjana,desviacio


def imprimeix_resum_global(nom,llista):
    """Mostra les metriques globals com mitjana +/- desviacio."""

    mitjana,desviacio=resum_metriques(llista)

    print("\n"+nom)

    for clau in ["precisio","recall","f1","dice","iou","accuracy"]:
        print(clau.capitalize()+":",round(mitjana[clau],4),"+/-",round(desviacio[clau],4))



def avalua_dataset_complet(dades):
    """
    Avalua els dos pipelines sobre tot el dataset amb 5-fold
    cross-validation separant sempre train i test per imatge.
    """

    ids=np.array(sorted(dades.keys()))

    #KFold separa les imatges en train i test cinc vegades diferents
    kfold=KFold(n_splits=n_folds,shuffle=True,random_state=0)

    metriques_paper=[]
    metriques_millor=[]

    print("\nAvaluacio global - "+str(n_folds)+" folds")

    for numero_fold,(index_train,index_test) in enumerate(kfold.split(ids)):
        ids_train=ids[index_train].tolist()
        ids_test=ids[index_test].tolist()

        print("Fold",numero_fold+1,"- train:",len(ids_train),"test:",len(ids_test))

        #Cada pipeline entrena amb les seves propies features
        model_paper=entrena_model_pipeline(dades,ids_train,"taula_paper")
        model_millor=entrena_model_pipeline(dades,ids_train,"taula_millor")

        #Apliquem els models sobre totes les imatges del test
        for id_imatge in ids_test:
            dada=dades[id_imatge]

            mascara_paper=aplica_svm_imatge(dada,"paper","taula_paper",model_paper)
            mascara_millor=aplica_svm_imatge(dada,"millor","taula_millor",model_millor)

            metriques_paper.append(calcular_metriques(mascara_paper,dada["gt"]))
            metriques_millor.append(calcular_metriques(mascara_millor,dada["gt"]))

    imprimeix_resum_global("Pipeline paper - mitjana global",metriques_paper)
    imprimeix_resum_global("Pipeline final - mitjana global",metriques_millor)


#Main

if __name__=="__main__":

    #Busquem les imatges i Ground Truth disponibles
    parelles=llista_imatges_dataset()

    print("Parelles trobades:",len(parelles))

    #Processem totes les imatges i guardem els resultats
    dades=processa_dataset(parelles)

    print("Imatges processades:",len(dades))

    #Mostrem dos exemples visuals
    prova_imatges_exemple(dades)

    #Calculem la mitjana final amb validacio creuada
    avalua_dataset_complet(dades)

    print("\nAcabat.")