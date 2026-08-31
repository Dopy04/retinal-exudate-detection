# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 23:42:52 2026

@author: gurpi
"""

"""
Bloc 5: segmentacio amb NICK i INRG.

En aquest bloc utilitzem NICK per calcular un llindar local per cada
zona de la imatge i posteriorment INRG aplica NICK diverses iteracions
fins que el canvi entre una iteracio i la següent sigui petit.

NICK i INRG s'utilitzen al paper de referencia per detectar hemorragies.
Com en el nostre cas volem detectar exsudats, que són regions brillants,
invertim el canal perquè passin a ser regions fosques.
"""

import numpy as np
from scipy.ndimage import uniform_filter

def invertir_canal(canal):
    """
    Invertim la intensitat del canal suposant que esta normalitzat
    entre 0 i 1.

    D'aquesta forma els exsudats passen de ser regions brillants
    a regions fosques.
    """

    canal_invertit=1-canal

    return canal_invertit


def segmentar_nick(canal,mascara_retina=None,mida_finestra=95,k=-0.10):
    """
    Fem una segmentacio amb NICK thresholding.

    NICK calcula un llindar diferent per cada zona de la imatge
    utilitzant la mitjana i la variacio local dels pixels.

     La mida_finestra defineix la zona local que utilitzem per calcular
    el llindar i k és el parametre del metode NICK.
    """

    #Calculem la mitjana local de cada zona
    mitjana=uniform_filter(canal,size=mida_finestra)

    #També necessitem la mitjana de I^2 per poder calcular la variancia
    mitjana_quadrat=uniform_filter(canal**2,size=mida_finestra)

    #Calculem la variancia local amb E[I^2]-E[I]^2
    variancia=mitjana_quadrat-mitjana**2

    #Poden aparèixer valors negatius molt petits per errors numerics
    variancia=np.maximum(variancia,0)

    #A partir de la variancia calculem la desviacio local
    desviacio=np.sqrt(variancia)

    #Calculem el llindar local de NICK
    llindar=mitjana+k*desviacio

    #Com treballem amb la imatge invertida els exsudats són foscos
    #i seleccionem els pixels que estan per sota del llindar
    mascara=canal<llindar

    #Si tenim mascara de retina eliminem les deteccions del fons
    if mascara_retina is not None:
        mascara=mascara&mascara_retina

    return mascara


def segmentar_inrg(canal,mascara_retina=None,mida_finestra=95,k=-0.2,
                   iteracions=3,delta=0.1):
    """
    Apliquem el metode INRG.

    Primer fem una segmentacio inicial amb NICK i després modifiquem
    la intensitat de les regions detectades per tornar a aplicar NICK.

    Les noves regions es van acumulant fins arribar al nombre
    d'iteracions o fins que el creixement sigui molt petit.
    """

    #Invertim el canal perquè els exsudats passin a ser zones fosques
    imatge=invertir_canal(canal)

    #Fem la primera segmentacio amb NICK
    mascara=segmentar_nick(imatge,mascara_retina,mida_finestra,k)

    #Repetim el procés a partir de la segona iteracio
    for i in range(1,iteracions):

        mascara_anterior=mascara.copy()

        #Guardem l'area detectada abans de buscar noves regions
        area_anterior=np.sum(mascara_anterior)

        #Si no tenim cap regio detectada no podem continuar
        if area_anterior==0:
            break

        #Calculem la intensitat mitjana de la regio detectada
        valor_mitja=np.mean(imatge[mascara_anterior])

        #Substituim aquesta regio per la seva intensitat mitjana
        #abans de tornar a aplicar NICK
        imatge[mascara_anterior]=valor_mitja

        #Tornem a aplicar NICK sobre la imatge modificada
        mascara_nova=segmentar_nick(imatge,mascara_retina,mida_finestra,k)

        #Acumulem les regions que ja teniem amb les noves
        mascara=mascara_anterior|mascara_nova

        area_actual=np.sum(mascara)

        #Mirem quant ha crescut la mascara respecte la iteracio anterior
        creixement=(area_actual-area_anterior)/area_anterior

        #Si el creixement és menor que delta parem el procés
        if creixement<delta:
            break

    return mascara
if __name__=="__main__":

    #Prova Bloc 5: apliquem NICK i INRG sobre una imatge real
    #per comparar visualment les regions que detecta cada metode

    from pathlib import Path
    import matplotlib.pyplot as plt
    import B2_preprocessament as prep
    from B1_metriques import calcular_metriques,imprimir_metriques

    #Escollim el dataset i la imatge que volem provar
    #Per provar una altra només cal canviar aquests dos valors
    dataset="DR2019"
    numero=3

    ruta_projecte=Path(__file__).resolve().parent.parent
    ruta_dades=ruta_projecte/"data_samples"

    #Obtenim automaticament la imatge i el seu Ground Truth
    ruta_imatge,ruta_gt=prep.obtenir_rutes_dataset(
        ruta_dades,dataset,numero)

    #Fem el preprocessament i obtenim els diferents canals
    imatge,mascara_gt,mascara_retina,imatge_retina,canals=prep.preprocessa_imatge(
        ruta_imatge,ruta_gt)

    #Utilitzem el canal G per aplicar NICK i INRG
    canal_g=canals[1]

    #NICK treballa amb el canal invertit perquè els exsudats
    #passin de ser regions brillants a regions fosques
    canal_invertit=invertir_canal(canal_g)

    mascara_nick=segmentar_nick(
        canal_invertit,
        mascara_retina)

    #INRG rep el canal original perquè ja fa la inversio internament
    mascara_inrg=segmentar_inrg(
        canal_g,
        mascara_retina)

    #Calculem les metriques dels dos metodes
    metriques_nick=calcular_metriques(mascara_nick,mascara_gt)
    metriques_inrg=calcular_metriques(mascara_inrg,mascara_gt)

    print("\n======================================")
    print("PROVA BLOC 5")
    print("Dataset:",dataset)
    print("Imatge:",str(numero).zfill(3))
    print("======================================")

    imprimir_metriques("NICK",metriques_nick)
    imprimir_metriques("INRG",metriques_inrg)

    #Mostrem la imatge original,el canal invertit i les dues segmentacions
    plt.figure(figsize=(16,7))
    plt.suptitle(
        "PROVA BLOC 5 - "+dataset+" "+str(numero).zfill(3),
        fontsize=16)

    plt.subplot(1,5,1)
    plt.imshow(imatge)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,5,2)
    plt.imshow(mascara_gt,cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1,5,3)
    plt.imshow(canal_invertit,cmap="gray")
    plt.title("Canal G invertit")
    plt.axis("off")

    plt.subplot(1,5,4)
    plt.imshow(mascara_nick,cmap="gray")
    plt.title("NICK")
    plt.axis("off")

    plt.subplot(1,5,5)
    plt.imshow(mascara_inrg,cmap="gray")
    plt.title("INRG")
    plt.axis("off")

    plt.tight_layout(rect=[0,0,1,0.92])
    plt.show()
    

# Ús d'IA:
# En aquest bloc el metode s'ha implementat a partir del paper del 2022.
# Nosaltres l'hem simplificat i adaptat al nostre cas, ja que treballem
# només amb exsudats.
# La IA s'ha utilitzat principalment per comprovar que aquesta simplificació
# mantingués el funcionament general del metode original.