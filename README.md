# Segmentació d'exsudats en imatges de retina

Aquest projecte busca detectar i segmentar exsudats en imatges de retina utilitzant diferents tècniques de processament d'imatge.

Hem separat el projecte en diferents blocs perquè sigui més fàcil entendre que fa cada part i poder provar-les per separat. Després, al `main.py`, es junten tots els blocs per executar el pipeline complet.

## Estructura del projecte

```text
retina-exudates-piva/
│
├── main.py
├── data_samples/
├── src/
│   ├── B1_metriques.py
│   ├── B2_preprocessament.py
│   ├── B3_millora_contrast.py
│   ├── B4_segmentacio.py
│   ├── B5_nick_inrg.py
│   ├── B6_disc_optic.py
│   ├── B7_postprocessament.py
│   ├── B8_watershed.py
│   ├── B9_deteccio_blobs.py
│   └── B10_classificador_SVM.py
└── README.md
```

Els fitxers de Git no fan falta per executar el projecte, per tant no s'han de tocar.

## Blocs
El projecte està separat en 10 blocs. Cada un fa una part concreta del procés.

### Bloc 1 - Mètriques
Serveix per comparar una màscara obtinguda amb el Ground Truth. Calcula TP,FP, FN y TN i a partir d'aixo obtenim Precision, Recall, F1, Dice, IoU i Accuracy.

### Bloc 2 - Preprocessament
Carrega la imatge i el Ground Truth,normalitza la imatge i genera la màscara de retina per eliminar el fons negre exterior. També extreu els diferents canals que utilitzem després

### Bloc 3 - Millora de contrast
En aquest bloc provem diferents mètodes de contrast com equalització d'histograma,CLAHE i KMMRC. La idea és veure si millorar el contrast ajuda realment a detectar millor els exsudats.

### Bloc 4 - Segmentació
Aquí tenim els primers mètodes de segmentació que utilitzem per comparar resultats. Tenim Otsu, llindar fix, llindar adaptatiu, Sauvola i K-means.

### Bloc 5 - NICK i INRG
Implementa NICK i INRG,que venen del treball de referència.Com els exsudats són zones brillants, invertim el canal abans d'aplicar el mètode.

### Bloc 6 - Disc òptic
Busca aproximadament la zona del disc òptic i l'elimina de la màscara. Això es fa perquè el disc òptic també és molt brillant i es pot confondre amb un exsudat.

### Bloc 7 - Postprocessament
Aplica morfologia matemàtica per netejar una mica les màscares.S'eliminen objectes petits, es tanquen discontinuïtats i també s'eliminen forats petits dins dels blobs.

### Bloc 8 - Watershed
S'utilitza per separar blobs que han quedat connectats.No detecta exsudats nous,simplement separa regions perquè després es puguin estudiar una per una.

### Bloc 9 - Detecció de blobs i característiques
Agafa cada blob i calcula diferents característiques de forma, mida, intensitat i color.Després aquestes dades s'utilitzen en el classificador.

### Bloc 10 - Classificador SVM
Utilitza les característiques del bloc anterior per entrenar una SVM.L'objectiu és decidir quins blobs són exsudats reals i quins són falsos positius.

## Proves de cada bloc

Al final de cada bloc hi ha una prova per poder executar aquell fitxer sol i veure que fa.

Per exemple:

```bash
python src/B4_segmentacio.py
```

Això executa només la prova del Bloc 4.

En els blocs que necessiten una imatge es pot escollir molt fàcilment quina volem provar:

```python
dataset="DR2019"
numero=3
```

Si volem provar una altra només canviem aquests dos valors:

```python
dataset="JSIEC"
numero=12
```

El programa busca directament la imatge i el Ground Truth que li correspon, així no hem d'escriure les dues rutes a ma cada vegada.

## Data samples

Les imatges que utilitzem estan dins de `data_samples`.

```text
data_samples/
│
├── Exudate-DR2019/
├── Exudate-DR2019-Groundtruth/
├── Exudate-JSIEC/
└── Exudate-JSIEC-Groundtruth/
```

Les imatges i els Ground Truth es relacionen amb el mateix número.

Per exemple:

```text
Exudate-DR2019/003.jpg
```

correspon amb:

```text
Exudate-DR2019-Groundtruth/003.jpg
```

## Main
El fitxer principal és `main.py`.

Aquí es connecten els diferents blocs i s'executa el projecte complet.

La idea és fer dues coses principals.Primer agafarem dues imatges del dataset i passarem el pipeline del mètode de referència i el pipeline que a nosaltres ens ha funcionat millor. Això ens permet comparar visualment els dos mètodes i veure les diferències.
Després el main recorrerà totes les imatges disponibles, calcularà les mètriques de cada resultat i al final farà una mitjana general.

Per executar-ho:

```bash
python main.py
```

## Llibreries
Les llibreries principals que utilitzem són:

```text
NumPy
SciPy
Matplotlib
scikit-image
scikit-learn
panda
```
```bash
```

## Ús d'IA
En els blocs on s'ha utilitzat IA com a suport hi ha una indicació al final del fitxer explicant per a que s'ha utilitzat.
Principalment s'ha fet servir per ajudar a entendre alguns mètodes, revisar errors, comprovar parts del codi i ajudar amb la documentació. Alguns elements de markdown s'han fet amb ajuda de la IA, per a que la representació 
visual sigués la adecuada.
