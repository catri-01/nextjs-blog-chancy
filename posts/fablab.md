---
title: "Fablab"
subtitle: "Détail du projet Fablab"
date: "2023-06-09"
---

![images/fab](/images/1.png)

Ce code utilise TensorFlow et TensorFlow Hub pour effectuer certaines opérations. Voici une explication ligne par ligne :

1. **import tensorflow_hub as hub**: Cette ligne importe la bibliothèque TensorFlow Hub, qui est une bibliothèque Python permettant de charger et d'utiliser des modèles pré-entraînés provenant de TensorFlow Hub. Ces modèles peuvent être utilisés pour effectuer diverses tâches telles que la classification d'images, la détection d'objets, la génération de texte, etc.

2. **import tensorflow as tf**: Cette ligne importe la bibliothèque TensorFlow, qui est une bibliothèque open source utilisée pour développer et former des modèles d'apprentissage automatique. TensorFlow est très populaire dans le domaine de l'apprentissage automatique et est souvent utilisé pour créer et entraîner des réseaux de neurones.

3. **print("TF hub version :",hub.__version__)**: Cette ligne affiche la version de TensorFlow Hub utilisée dans le code.

4. **print("TF version:",tf.__version__)**: Cette ligne affiche la version de TensorFlow utilisée dans le code.

5. **print("GPU","available" if tf.config.list_physical_devices("GPU")else "not available")**: Cette ligne vérifie la disponibilité d'un processeur graphique (GPU) pour l'exécution du code. Elle utilise la fonction `tf.config.list_physical_devices("GPU")` pour obtenir la liste des dispositifs physiques disponibles, puis vérifie si la liste est vide ou non. Si un GPU est disponible, il affiche "available" (disponible), sinon il affiche "not available" (non disponible).

En résumé, ce code importe les bibliothèques TensorFlow et TensorFlow Hub, affiche les versions de ces bibliothèques, puis vérifie la disponibilité d'un GPU pour l'exécution du code.


![images/fab](/images/2.png)

Ce code utilise la bibliothèque pandas pour charger et manipuler des données tabulaires. Il utilise également la fonction `files.upload()` de la bibliothèque Google Colab pour télécharger un fichier kaggle.json.

Voici une explication ligne par ligne :

1. **import pandas as pd**: Cette ligne importe la bibliothèque pandas sous l'alias **pd**, qui est couramment utilisé pour travailler avec des données tabulaires en Python. Pandas fournit des structures de données et des fonctions pour manipuler et analyser facilement les données.

2. **from google.colab import files**: Cette ligne importe la fonction **files.upload()** de la bibliothèque Google Colab. Cette fonction permet de télécharger des fichiers depuis l'environnement Google Colab.

3. **files.upload()**: Cette ligne exécute la fonction **files.upload()** pour télécharger un fichier kaggle.json. Lorsque cette ligne est exécutée, une boîte de dialogue apparaîtra vous demandant de sélectionner le fichier kaggle.json à partir de votre système de fichiers local. Une fois le fichier sélectionné, il sera téléchargé dans l'environnement Google Colab.

Ce code est utilisé dans un environnement spécifique appelé Google Colab, qui est une plateforme d'apprentissage automatique basée sur Jupyter Notebook. La fonction **files.upload()** est spécifique à Google Colab et permet aux utilisateurs de télécharger des fichiers directement dans leur environnement de travail.

![images/fab](/images/3.png)

La ligne de code **!kaggle datasets list -s dogbreedidfromcomp** est une commande utilisée dans l'environnement Google Colab pour rechercher des jeux de données sur Kaggle.

Voici une explication de cette ligne de code :

1. **!** : Dans Google Colab, le symbole **!** est utilisé pour exécuter des commandes système.

2. **kaggle datasets list** : C'est la commande pour lister tous les jeux de données disponibles sur Kaggle.

3. **-s dogbreedidfromcomp** : L'option **-s** est utilisée pour spécifier un critère de recherche. Dans ce cas, le critère de recherche est "dogbreedidfromcomp", ce qui signifie que la commande listera les jeux de données qui contiennent ce terme dans leur nom ou leur description.

En exécutant cette commande, vous obtiendrez une liste de jeux de données correspondant au critère de recherche "dogbreedidfromcomp" sur Kaggle.


![images/fab](/images/4.png)

Ce code utilise la bibliothèque Google Colab pour monter Google Drive dans l'environnement de travail.

Voici une explication ligne par ligne :

1. **from google.colab import drive**: Cette ligne importe la fonction **drive** du module **google.colab**, qui permet d'interagir avec Google Drive.

2. **drive.mount('/content/drive')**: Cette ligne exécute la fonction **mount()** de **drive** pour monter Google Drive dans l'environnement de travail de Google Colab. L'argument **'/content/drive'** spécifie le chemin d'accès où Google Drive sera monté.

Lorsque vous exécutez cette ligne de code, une boîte de dialogue apparaîtra vous demandant de vous authentifier avec votre compte Google. Une fois que vous vous êtes authentifié, Google Colab aura accès à votre Google Drive et vous pourrez accéder aux fichiers et répertoires qui s'y trouvent.

Après avoir monté Google Drive, vous pouvez naviguer dans votre arborescence de fichiers, lire et écrire des fichiers, et utiliser les données qui se trouvent dans Google Drive dans votre environnement de travail Google Colab.


![images/fab](/images/5.png)

Ce code vérifie si le nombre de noms de fichiers dans un répertoire correspond au nombre réel de fichiers image présents dans ce répertoire.

Voici une explication ligne par ligne :

1. **import os**: Cette ligne importe le module **os** qui fournit des fonctionnalités pour interagir avec le système d'exploitation, telles que la manipulation des fichiers et des répertoires.

2. **if len(os.listdir("dog_dataset/train/")) == len(filenames):**: Cette ligne vérifie si le nombre de fichiers dans le répertoire **"dog_dataset/train/"** correspond au nombre de fichiers stockés dans la variable **filenames**. **os.listdir("dog_dataset/train/")** renvoie la liste des noms de fichiers présents dans le répertoire spécifié, et **len()** renvoie la longueur de cette liste. **len(filenames)** donne le nombre de fichiers dans la variable **filenames**. Si ces deux nombres sont égaux, cela signifie que le nombre de noms de fichiers correspond au nombre réel de fichiers image.

3. **print("proceed!!!")**: Si le nombre de fichiers correspond, cette ligne affiche "proceed!!!" pour indiquer que vous pouvez continuer avec les opérations prévues.

4. **print("error occured!!!")**: Si le nombre de fichiers ne correspond pas, cette ligne affiche "error occured!!!" pour indiquer qu'une erreur s'est produite et que les nombres ne correspondent pas.

En résumé, ce code permet de vérifier si le nombre de noms de fichiers dans un répertoire correspond au nombre réel de fichiers image présents dans ce répertoire. Si les nombres correspondent, le code affiche "proceed!!!", sinon il affiche "error occured!!!".


![images/fab](/images/6.png)

Ce code enregistre une liste de races de chiens uniques dans un fichier texte appelé "breeds.txt". Voici une explication ligne par ligne :

1. **unique_breeds**: Il est supposé que **unique_breeds** est une liste contenant les races de chiens uniques.

2. **textfile = open("breeds.txt", "w")**: Cette ligne ouvre le fichier "breeds.txt" en mode écriture ("w"). Si le fichier n'existe pas, il sera créé. La variable **textfile** est utilisée pour représenter le fichier ouvert.

3. **for element in unique_breeds:** : Cette ligne commence une boucle `for` qui itère sur chaque élément de la liste **unique_breeds**.

4. **textfile.write(element + "\n")**: Cette ligne écrit chaque élément de **unique_breeds** dans le fichier texte. L'élément est suivi du caractère de saut de ligne (**"\n"**) pour séparer chaque race de chien sur une nouvelle ligne.

5. **textfile.close()**: Cette ligne ferme le fichier "breeds.txt" après avoir terminé l'écriture. Cela garantit que toutes les opérations d'écriture sont terminées et que les ressources associées au fichier sont libérées.

En résumé, ce code parcourt une liste de races de chiens uniques et écrit chaque race dans un fichier texte, chaque race étant sur une nouvelle ligne. Cela permet de sauvegarder la liste des races de chiens dans un format lisible par les humains et qui peut être utilisé ultérieurement.


![images/fab](/images/7.png)

La première ligne de code **print(labels[0])** affiche la valeur de la première étiquette dans la variable `labels`. Cette ligne de code permet de visualiser l'étiquette avant la transformation.

La deuxième ligne de code **labels[0] == unique_breeds** compare la première étiquette avec la liste **unique_breeds**. Elle renvoie un tableau de booléens indiquant si chaque élément de **unique_breeds** est égal à la première étiquette. Le résultat sera un tableau de la même longueur que **unique_breeds**, où chaque élément sera **True** s'il correspond à la première étiquette, et **False** sinon.

Cette ligne de code est utilisée pour effectuer une comparaison entre l'étiquette d'un élément spécifique et une liste de valeurs possibles. Cela peut être utile pour vérifier si une étiquette donnée correspond à une valeur spécifique ou pour effectuer des opérations de filtrage sur les étiquettes.


![images/fab](/images/8.png)

Le code **boolean_labels = [label == unique_breeds for label in labels]** transforme toutes les étiquettes de la liste **labels** en un tableau de booléens, en les comparant avec la liste **unique_breeds**. Chaque élément de **boolean_labels** correspondra à une étiquette de **labels** et sera **True** si l'étiquette est présente dans **unique_breeds**, et **False** sinon.

Voici une explication ligne par ligne :

1. **boolean_labels = [label == unique_breeds for label in labels]** : Cette ligne utilise une liste en compréhension pour itérer sur chaque étiquette **label** de la liste **labels**. Pour chaque **label**, elle vérifie si **label** est égal à l'un des éléments de **unique_breeds**. Si c'est le cas, elle ajoute **True** à **boolean_labels**, sinon elle ajoute **False**. Ainsi, **boolean_labels** sera une liste de booléens correspondant à la présence ou à l'absence de chaque étiquette de **labels** dans **unique_breeds**.

2. **boolean_labels[:2]** : Cette ligne affiche les deux premiers éléments de **boolean_labels**. Cela permet de vérifier les résultats de la transformation pour les premières étiquettes.

En résumé, ce code transforme toutes les étiquettes de la liste **labels** en un tableau de booléens indiquant si chaque étiquette est présente dans **unique_breeds** ou non.


![images/fab](/images/9.png)

Le code fourni illustre comment transformer un tableau de booléens en entiers en utilisant la bibliothèque NumPy. Voici une explication ligne par ligne :

1. `print(labels[0])`: Cette ligne affiche la valeur de la première étiquette dans la variable `labels`. Cela permet de visualiser l'étiquette originale avant la transformation.

2. `print(np.where(unique_breeds == labels[0]))`: Cette ligne utilise la fonction `np.where()` de la bibliothèque NumPy pour trouver les indices où l'étiquette originale (`labels[0]`) apparaît dans le tableau `unique_breeds`. La fonction `np.where()` renvoie les indices correspondants. Cela peut être utile pour déterminer la position de l'étiquette dans un tableau.

3. `print(boolean_labels[0].argmax())`: Cette ligne utilise la méthode `argmax()` du tableau booléen `boolean_labels[0]` pour trouver l'indice où l'étiquette apparaît pour la première fois. L'indice renvoyé correspondra à la première occurrence de `True` dans le tableau booléen. Cela peut être utile si vous voulez obtenir l'indice de l'étiquette dans le tableau booléen.

4. `print(boolean_labels[0].astype(int))`: Cette ligne utilise la méthode `astype()` du tableau booléen `boolean_labels[0]` pour convertir les valeurs booléennes en entiers. Les valeurs `True` sont converties en 1 et les valeurs `False` sont converties en 0. Cela permet de représenter le tableau booléen sous forme d'entiers.

En résumé, ce code illustre différentes méthodes pour manipuler un tableau booléen, y compris la recherche d'indices où une étiquette apparaît, la recherche de l'indice de la première occurrence d'une valeur `True`, et la conversion du tableau booléen en un tableau d'entiers.


![images/fab](/images/10.png)

Le code fourni permet d'afficher et de manipuler la troisième étiquette et son tableau booléen correspondant. Voici une explication ligne par ligne :

1. `print(labels[2])`: Cette ligne affiche la valeur de la troisième étiquette dans la variable `labels`. Cela permet de visualiser l'étiquette originale avant la transformation.

2. `print(boolean_labels[2].argmax())`: Cette ligne utilise la méthode `argmax()` du tableau booléen `boolean_labels[2]` pour trouver l'indice où l'étiquette apparaît pour la première fois. L'indice renvoyé correspondra à la première occurrence de `True` dans le tableau booléen. Cela peut être utile si vous voulez obtenir l'indice de l'étiquette dans le tableau booléen.

3. `print(boolean_labels[2].astype(int))`: Cette ligne utilise la méthode `astype()` du tableau booléen `boolean_labels[2]` pour convertir les valeurs booléennes en entiers. Les valeurs `True` sont converties en 1 et les valeurs `False` sont converties en 0. Cela permet de représenter le tableau booléen sous forme d'entiers.

En résumé, ces lignes de code affichent l'étiquette originale, trouvent l'indice de la première occurrence de cette étiquette dans le tableau booléen et convertissent le tableau booléen en un tableau d'entiers pour la troisième étiquette. Cela permet de manipuler et de représenter les informations sous différentes formes en fonction des besoins spécifiques de l'analyse des données.


![images/fab](/images/11.png)

Les lignes de code fournies permettent de paramétrer les variables `x` et `y` avec les valeurs des variables `filenames` et `boolean_labels` respectivement.

- `x = filenames`: Cette ligne assigne à la variable `x` la valeur de la variable `filenames`. Cela signifie que `x` contient la liste des noms de fichiers.
- `y = boolean_labels`: Cette ligne assigne à la variable `y` la valeur de la variable `boolean_labels`. Cela signifie que `y` contient la liste des tableaux de booléens qui représentent les étiquettes transformées en valeurs booléennes.

Ces paramétrages sont couramment utilisés dans des tâches de machine learning où `x` représente les caractéristiques (dans ce cas les noms de fichiers) et `y` représente les étiquettes (dans ce cas les tableaux de booléens). Ces variables peuvent ensuite être utilisées pour entraîner des modèles d'apprentissage automatique, effectuer des prédictions, etc.


![images/fab](/images/12.png)


![images/fab](/images/13.png)

Dans le code fourni, le nombre d'images utilisé pour l'expérimentation est défini par la variable `Num_images`. La valeur de cette variable est fixée à 10000, mais elle peut être ajustée en utilisant un curseur (slider) dans la plage de 1000 à 10000, avec un pas de 1000.

Ensuite, les données sont divisées en ensembles d'apprentissage et de validation à l'aide de la fonction `train_test_split` de la bibliothèque scikit-learn. Les ensembles d'apprentissage et de validation sont créés à partir des `Num_images` premières images dans les tableaux `x` et `y`. L'ensemble de validation est défini pour avoir une taille de 20% des données totales.

Enfin, les longueurs des ensembles d'apprentissage (`x_train`, `y_train`) et de validation (`x_val`, `y_val`) sont affichées pour vérification.

![images/fab](/images/14.png)

Les données d'entraînement (`x_train`) et les étiquettes correspondantes (`y_train`) sont affichées ci-dessous :

```
(array([image_1, image_2]), array([label_1, label_2]))
```

Cela représente les deux premières images et leurs étiquettes respectives dans l'ensemble d'entraînement. Les valeurs réelles des images et des étiquettes sont remplacées par les termes génériques "image_1", "image_2", "label_1" et "label_2" dans cet exemple.

![images/fab](/images/15.png)

Pour convertir une image en un tableau NumPy, vous pouvez utiliser la fonction `imread` du module `matplotlib.pyplot`. Dans l'exemple donné, l'image est lue à partir du fichier correspondant à l'indice 42 dans le tableau `filenames` et est stockée dans la variable `image`.

Pour connaître la forme du tableau `image`, c'est-à-dire ses dimensions, vous pouvez utiliser l'attribut `shape` du tableau NumPy :

``` python
image.shape
```

Cela renverra un tuple indiquant les dimensions de l'image, par exemple `(hauteur, largeur, nombre de canaux)`. Si l'image est en niveaux de gris, elle n'aura qu'un seul canal, tandis qu'une image en couleur aura trois canaux correspondant aux composantes rouge, verte et bleue (RVB).

Enfin, pour afficher les trois premières lignes de l'image, vous pouvez utiliser la notation d'indexation de tableau NumPy :

``` python
image[:3]
```

Cela affichera les trois premières lignes de l'image, où chaque ligne représente une série de valeurs de pixels.

Pour déterminer la valeur maximale et minimale des pixels dans une image, vous pouvez utiliser les fonctions `max()` et `min()` du tableau NumPy `image`. 

Voici comment vous pouvez obtenir ces valeurs :

``` python
image.max()  # Valeur maximale des pixels
image.min()  # Valeur minimale des pixels
```

Dans le contexte de l'image RGB, la valeur des canaux rouge, vert et bleu peut varier de 0 à 255. En utilisant ces fonctions, vous pouvez vérifier la plage de valeurs présente dans votre image.

![images/fab](/images/16.png)

Ce code traite les images en les convertissant en tenseurs à l'aide de TensorFlow. Voici une explication détaillée du code :

1. `tf.constant(image)[:2]`: Cette ligne crée un tenseur constant à partir du tableau NumPy `image` en utilisant `tf.constant`. Ensuite, l'opération de découpage `[:2]` est appliquée pour afficher les deux premières lignes du tenseur.

2. `IMG_SIZE = 224`: Cette ligne définit la taille souhaitée pour les images. Les images seront redimensionnées à cette taille.

3. `process_img(image_path)`: C'est une fonction qui prend en entrée le chemin d'un fichier image et effectue le prétraitement de l'image. Voici les étapes du prétraitement :

   - `tf.io.read_file(image_path)`: Cette ligne lit le contenu du fichier image spécifié par `image_path` et renvoie les données binaires de l'image.
   - `tf.image.decode_jpeg(image, channels=3)`: Cette ligne décode les données binaires de l'image JPEG en un tenseur numérique avec 3 canaux de couleur (rouge, vert, bleu). Cela convertit l'image en un tenseur 3D.
   - `tf.image.convert_image_dtype(image, tf.float32)`: Cette ligne convertit les valeurs du tenseur d'image de la plage 0-255 à la plage 0-1 en les divisant par 255. Cela permet de normaliser les valeurs des pixels entre 0 et 1, ce qui est souvent utilisé lors de l'entraînement de modèles de deep learning.
   - `tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])`: Cette ligne redimensionne l'image à la taille souhaitée `[IMG_SIZE, IMG_SIZE]`. Cela garantit que toutes les images ont la même taille pour être utilisées dans un modèle d'apprentissage automatique.

4. `get_image_label(image_path, label)`: Cette fonction prend en entrée le chemin d'un fichier image et son étiquette associée, puis applique la fonction `process_img` pour prétraiter l'image. Ensuite, elle renvoie un tuple contenant l'image prétraitée et l'étiquette correspondante.

En utilisant ces fonctions, vous pouvez convertir les images en tenseurs et les prétraiter avant de les utiliser pour l'apprentissage automatique ou d'autres tâches.


![images/fab](/images/17.png)

Pour implémenter le code ci-dessus et prétraiter une image spécifique, vous pouvez utiliser la fonction `process_img` et `tf.constant` de la manière suivante :

```python
(process_img(x[42]), tf.constant(y[42]))
```

Cela appliquera la fonction `process_img` à l'image correspondant à l'indice 42 dans le tableau `x`. Ensuite, le résultat sera combiné avec l'étiquette correspondante à l'indice 42 dans le tableau `y` en utilisant `tf.constant` pour créer un tenseur constant. Le tuple résultant contiendra l'image prétraitée et l'étiquette correspondante.

Assurez-vous que les tableaux `x` et `y` contiennent les images et les étiquettes respectives, et que l'indice 42 correspond à une image valide dans ces tableaux.

![images/fab](/images/18.png)

Le code ci-dessus définit la taille du lot (`BATCH_SIZE`) comme 32 et implémente la fonction `create_data_batch` pour transformer les données en lots.

La fonction `create_data_batch` prend en entrée un tableau d'images `X` et éventuellement un tableau d'étiquettes `Y`. Voici comment elle fonctionne :

- Si `test_data` est `True`, la fonction crée des lots de données de test en utilisant uniquement les chemins de fichiers des images. Elle crée un ensemble de données à partir des chemins de fichiers en utilisant `tf.data.Dataset.from_tensor_slices`, puis applique la fonction `process_img` pour prétraiter chaque image. Ensuite, elle crée des lots de taille `BATCH_SIZE` en utilisant `.batch(BATCH_SIZE)`. Les étiquettes ne sont pas nécessaires pour les données de test, car elles ne sont pas utilisées.
- Si `valid_data` est `True`, la fonction crée des lots de données de validation en utilisant à la fois les chemins de fichiers des images et les étiquettes correspondantes. Elle crée un ensemble de données à partir des paires `(chemin de fichier, étiquette)` en utilisant `tf.data.Dataset.from_tensor_slices`, puis applique la fonction `get_image_label` pour prétraiter chaque image et étiquette. Ensuite, elle crée des lots de taille `BATCH_SIZE`.
- Sinon, la fonction crée des lots de données d'entraînement. Elle mélange d'abord les chemins de fichiers et les étiquettes en utilisant `data.shuffle` pour éviter tout biais dans l'ordre des données. Ensuite, elle crée des tuples `(image, étiquette)` en utilisant `get_image_label` pour prétraiter chaque image et étiquette. Enfin, elle crée des lots de taille `BATCH_SIZE` à partir des tuples.

La fonction renvoie l'ensemble de données sous forme de lot, prêt à être utilisé pour l'entraînement, la validation ou les tests.

Pour utiliser cette fonction, vous pouvez l'appeler avec les données d'entraînement, de validation ou de test, en spécifiant les arguments appropriés (`X`, `Y`, `valid_data=True`, `test_data=True`, etc.). Par exemple, pour créer des lots de données d'entraînement à partir de `x_train` et `y_train`, vous pouvez utiliser :

```python
train_data_batch = create_data_batch(x_train, y_train)
```


![images/fab](/images/19.png)


Le code ci-dessus crée des lots de données de formation et de validation à partir des ensembles d'entraînement et de validation. Il utilise la fonction `create_data_batch` pour transformer les données en lots.

Voici ce que fait le code :

- Il crée le lot de données d'entraînement en utilisant la fonction `create_data_batch` avec les tableaux `x_train` et `y_train`.
- Il crée le lot de données de validation en utilisant la fonction `create_data_batch` avec les tableaux `x_val` et `y_val`, et en spécifiant `valid_data=True`.
- Ensuite, il affiche la spécification de type d'un élément de chaque lot de données en utilisant `train_data.element_spec` et `val_data.element_spec`. Cela donne des informations sur la structure des éléments dans chaque lot de données, c'est-à-dire les types de données et les formes.
- Ensuite, il définit une fonction `show_25_img` pour afficher un tracé de 25 images et leurs étiquettes à partir d'un lot de données. Cette fonction prend en entrée un tableau d'images `images` et un tableau d'étiquettes `labels`.
- Enfin, il utilise la fonction `next(train_data.as_numpy_iterator())` pour obtenir le prochain lot de données d'entraînement sous forme de tableau numpy. Il assigne les images au tableau `train_img` et les étiquettes au tableau `train_labels`. Il affiche également la taille des tableaux `train_img` et `train_labels`.

Pour visualiser les images du lot de données d'entraînement, vous pouvez utiliser la fonction `show_25_img` en passant les tableaux `train_img` et `train_labels` :

```python
show_25_img(train_img, train_labels)
```

Cela affichera un tracé de 25 images avec leurs étiquettes correspondantes.


![images/fab](/images/20.png)

![images/fab](/images/21.png)

Le code ci-dessus crée des lots de données de formation et de validation à partir des ensembles d'entraînement et de validation. Il utilise la fonction `create_data_batch` pour transformer les données en lots.

Voici ce que fait le code :

- Il crée le lot de données d'entraînement en utilisant la fonction `create_data_batch` avec les tableaux `x_train` et `y_train`.
- Il crée le lot de données de validation en utilisant la fonction `create_data_batch` avec les tableaux `x_val` et `y_val`, et en spécifiant `valid_data=True`.
- Ensuite, il affiche la spécification de type d'un élément de chaque lot de données en utilisant `train_data.element_spec` et `val_data.element_spec`. Cela donne des informations sur la structure des éléments dans chaque lot de données, c'est-à-dire les types de données et les formes.
- Ensuite, il définit une fonction `show_25_img` pour afficher un tracé de 25 images et leurs étiquettes à partir d'un lot de données. Cette fonction prend en entrée un tableau d'images `images` et un tableau d'étiquettes `labels`.
- Enfin, il utilise la fonction `next(train_data.as_numpy_iterator())` pour obtenir le prochain lot de données d'entraînement sous forme de tableau numpy. Il assigne les images au tableau `train_img` et les étiquettes au tableau `train_labels`. Il affiche également la taille des tableaux `train_img` et `train_labels`.

Pour visualiser les images du lot de données d'entraînement, vous pouvez utiliser la fonction `show_25_img` en passant les tableaux `train_img` et `train_labels` :

```python
show_25_img(train_img, train_labels)
```

Cela affichera un tracé de 25 images avec leurs étiquettes correspondantes.

![images/fab](/images/22.png)

Le code ci-dessus configure la forme d'entrée et de sortie du modèle, et crée ensuite un modèle Keras en utilisant TensorFlow Hub.

- `INPUT_SHAPE` est défini comme `[None, IMG_SIZE, IMG_SIZE, 3]`, ce qui représente la forme d'entrée du modèle. Il s'agit d'un tenseur de 4 dimensions avec une dimension de lot variable, une hauteur et une largeur de `IMG_SIZE`, et 3 canaux de couleur (RGB).

- `OUTPUT_SHAPE` est défini comme la longueur de la liste `unique_breeds`, ce qui représente le nombre de classes de sortie du modèle. Cela détermine la forme de sortie du modèle.

- `MODEL_URL` est l'URL du modèle pré-entraîné à utiliser à partir de TensorFlow Hub. Dans ce cas, le modèle utilisé est "MobileNet V2" pré-entraîné sur ImageNet.

- La fonction `create_model` prend en compte les paramètres d'entrée suivants : `input_shape`, `output_shape` et `url`. Elle construit un modèle Keras en utilisant la séquence de couches suivante :
  - La première couche est un `hub.KerasLayer` qui charge le modèle pré-entraîné à partir de l'URL spécifié. Cela permet d'utiliser le modèle pré-entraîné comme une couche dans notre modèle.
  - La deuxième couche est une couche dense avec `units = OUTPUT_SHAPE` et une fonction d'activation softmax. Cela sert de couche de sortie du modèle.
  
- Le modèle est compilé en utilisant la fonction de perte `CategoricalCrossentropy`, l'optimiseur Adam et les métriques d'exactitude (`accuracy`).

- Enfin, le modèle est construit en utilisant la forme d'entrée `INPUT_SHAPE`.

- Une fois le modèle créé, `model.summary()` est utilisé pour afficher un résumé du modèle, y compris les différentes couches, le nombre de paramètres et la forme de sortie de chaque couche.

Le modèle ainsi créé est un modèle de classification d'images basé sur MobileNet V2, prêt à être entraîné sur les données fournies.


![images/fab](/images/23.png)

Le code ci-dessus comprend les étapes suivantes :

- `%load_ext tensorboard` est utilisé pour charger l'extension TensorBoard pour l'utilisation de TensorBoard dans le notebook.

- La fonction `create_tensorboard_callback` est créée pour construire un objet TensorBoard Callback. Cela permettra d'enregistrer les journaux TensorBoard pendant l'entraînement du modèle. Le répertoire de journaux est créé en utilisant la date et l'heure actuelles.

- `early_stop` est un rappel `EarlyStopping` qui arrête l'entraînement si la performance sur la validation ne s'améliore pas pendant un certain nombre d'époques défini par le paramètre `patience`.

- `NUM_EPOCHS` est défini comme le nombre d'époques à effectuer lors de l'entraînement du modèle. La valeur est définie par un curseur qui peut être ajusté.

- La fonction `train_model` est créée pour entraîner le modèle. Elle suit les étapes suivantes :
  - Créer un modèle en utilisant la fonction `create_model`.
  - Créer un objet TensorBoard Callback en utilisant la fonction `create_tensorboard_callback`.
  - Ajuster le modèle aux données d'entraînement en utilisant la méthode `fit`. Les rappels de TensorBoard et d'EarlyStopping sont transmis à la méthode `fit` pour enregistrer les journaux TensorBoard et arrêter l'entraînement si nécessaire.

- En fin de compte, la fonction `train_model` retourne le modèle entraîné.

La fonction `train_model` peut être utilisée pour entraîner le modèle en appelant simplement cette fonction. Les journaux TensorBoard seront enregistrés et peuvent être visualisés à l'aide de TensorBoard.

![images/fab](/images/24.png)

Le modèle a été ajusté aux données en appelant la fonction `train_model()`. Le modèle a été entraîné pendant un nombre d'époques défini par la variable `NUM_EPOCHS`. Pendant l'entraînement, les journaux TensorBoard ont été enregistrés à l'emplacement spécifié dans la fonction `create_tensorboard_callback()`.

Après l'entraînement, vous pouvez utiliser la commande `%tensorboard --logdir "/content/drive/MyDrive/Dog breed prediction/logs"` pour visualiser les journaux TensorBoard dans le notebook. Cela affichera TensorBoard avec les journaux enregistrés dans le répertoire spécifié. Vous pouvez utiliser l'interface TensorBoard pour explorer les métriques d'entraînement et de validation, les graphiques, les histogrammes, etc., afin de mieux comprendre les performances du modèle pendant l'entraînement.

Notez que vous devrez peut-être ajuster le chemin `/content/drive/MyDrive/Dog breed prediction/logs` pour correspondre à l'emplacement réel des journaux sur votre système.

![images/fab](/images/25.png)

La variable `preds` contient les prédictions du modèle sur les données de validation. Chaque élément de `preds` est un tableau de valeurs de probabilité correspondant aux différentes classes de races de chiens. 

En examinant `preds[0]`, vous pouvez voir les valeurs de probabilité associées à chaque classe de race de chien pour la première image dans les données de validation. La valeur la plus élevée parmi ces probabilités correspond à la prédiction du modèle pour la race de chien de cette image. 

La somme des valeurs de probabilité dans `preds[0]` est égale à 1, car les valeurs de probabilité représentent la distribution de probabilité sur les classes de races de chiens.


![images/fab](/images/26.png)


Dans cette partie du code, nous examinons la première prédiction à l'index 42 dans les données de validation.

- `preds[index]` renvoie les probabilités de prédiction pour l'image à l'index 42.
- `np.max(preds[index])` renvoie la valeur maximale de probabilité de prédiction pour cette image.
- `np.sum(preds[index])` renvoie la somme de toutes les probabilités de prédiction pour cette image.
- `np.argmax(preds[index])` renvoie l'index de la classe prédite avec la probabilité la plus élevée.
- `unique_breeds[np.argmax(preds[index])]` renvoie le label correspondant à la classe prédite.

Ainsi, nous affichons la prédiction, la valeur maximale de probabilité, la somme des probabilités, l'index de la classe prédite et le label prédit pour l'image à l'index 42.

La fonction `get_preds_label` prend en entrée les probabilités de prédiction `preds_prob` et renvoie le label correspondant à la prédiction avec la probabilité la plus élevée. Dans notre exemple, nous avons utilisé `preds[111]` pour obtenir la prédiction pour la 112e image dans les données de validation. Le résultat renvoyé par `get_preds_label` est le label prédit pour cette image.


![images/fab](/images/27.png)


La fonction `unbatchify` est utilisée pour dégrouper les données d'un lot (batch) et obtenir les images et les étiquettes correspondantes.

- `data.unbatch()` est utilisé pour dégrouper les données.
- `as_numpy_iterator()` est utilisé pour itérer sur les données dégroupées et les convertir en itérateur numpy.
- En utilisant une boucle, chaque image et étiquette sont extraites de l'itérateur numpy, et l'image est ajoutée à la liste `images_unbatch` et l'étiquette correspondante est ajoutée à la liste `labels_unbatch`.
- Enfin, les listes `images_unbatch` et `labels_unbatch` sont renvoyées.

Ainsi, `val_img` contiendra les images dégroupées et `val_labels` contiendra les étiquettes correspondantes. L'exemple affiche la première image et son étiquette dans les données dégroupées.


![images/fab](/images/28.png)

La fonction `get_preds_label` est utilisée pour obtenir le label prédit à partir des probabilités de prédiction. Elle prend en entrée les probabilités de prédiction (`pred_prob`) et renvoie le label prédit correspondant en utilisant la fonction `get_preds_label`.

L'exemple `get_preds_label(val_labels[0])` utilise la fonction pour obtenir le label prédit de la première image dans les données de validation dégroupées.

La fonction `plot_pred` est utilisée pour tracer une image avec son label prédit. Elle prend en entrée les probabilités de prédiction (`pred_prob`), les étiquettes réelles (`labels`), les images (`images`) et un indice (`n`) pour sélectionner l'image à tracer.

La fonction extrait les probabilités de prédiction, l'étiquette réelle et l'image correspondante en utilisant l'indice donné. Ensuite, elle utilise la fonction `get_preds_label` pour obtenir le label prédit à partir des probabilités de prédiction. Elle trace ensuite l'image, supprime les ticks sur les axes, change la couleur du titre en fonction de la nature de la prédiction (verte si correcte, rouge sinon) et affiche le titre du plot avec le label prédit, le pourcentage de la plus haute probabilité de prédiction et l'étiquette réelle.

L'exemple `plot_pred(pred_prob=preds, labels=val_labels, images=val_img, n=26)` utilise la fonction pour tracer la 26e image avec son label prédit à partir des prédictions `preds`, des étiquettes réelles `val_labels` et des images `val_img`.

![images/fab](/images/29.png)

La fonction `plot_pred_confidences` est utilisée pour tracer un graphique montrant les 10 meilleures prédictions de confiance pour une image donnée. Elle prend en entrée les probabilités de prédiction (`prediction_prob`), les étiquettes réelles (`labels`) et l'indice de l'image à tracer (`n`).

La fonction extrait les probabilités de prédiction et l'étiquette réelle correspondante en utilisant l'indice donné. Ensuite, elle trouve les 10 meilleurs indices de confiance des prédictions en utilisant la fonction `argsort` pour trier les probabilités dans l'ordre décroissant et en sélectionnant les 10 premiers. Elle trouve également les 10 premières valeurs de prédictions et les 10 premiers labels correspondants.

Ensuite, la fonction configure le tracé en utilisant la fonction `bar` pour créer un graphique à barres. Les barres représentent les valeurs de prédictions, l'axe des x est étiqueté avec les 10 meilleurs labels, et les étiquettes sont tournées verticalement pour une meilleure lisibilité.

La couleur de la barre correspondant à l'étiquette réelle est changée en vert si elle fait partie des 10 meilleurs labels prédits. Si l'étiquette réelle ne fait pas partie des 10 meilleurs labels, la couleur reste la même.

L'exemple `plot_pred_confidences(prediction_prob=preds, labels=val_labels, n=26)` utilise la fonction pour tracer le graphique des 10 meilleures prédictions de confiance pour la 26e image, en utilisant les probabilités de prédiction `preds`, les étiquettes réelles `val_labels` et l'indice de l'image.


![images/fab](/images/30.png)

La fonction `plot_pred_confidences` a été utilisée pour tracer le graphique des 10 meilleures prédictions de confiance pour la 59e image. Les probabilités de prédiction sont fournies par `preds`, et les étiquettes réelles sont fournies par `val_labels`.

Les barres du graphique représentent les valeurs de prédictions, et l'axe des x est étiqueté avec les 10 meilleurs labels prédits. La barre correspondant à l'étiquette réelle est colorée en vert si elle fait partie des 10 meilleurs labels prédits.

Cela permet de visualiser les prédictions de confiance pour cette image spécifique, en mettant en évidence la prédiction correcte si elle fait partie des prédictions les plus probables.



![images/fab](/images/31.png)

Ce code permet de visualiser les prédictions du modèle pour plusieurs images de validation. Les prédictions sont affichées sous forme de paires de graphiques pour chaque image.

Dans chaque paire de graphiques, le graphique de gauche utilise la fonction `plot_pred` pour afficher l'image, la prédiction la plus probable avec son pourcentage de confiance, et l'étiquette réelle. Le titre de l'image est coloré en vert si la prédiction est correcte.

Le graphique de droite utilise la fonction `plot_pred_confidences` pour afficher un graphique à barres des 10 meilleures prédictions de confiance, avec les étiquettes correspondantes. La barre correspondant à l'étiquette réelle est colorée en vert si elle fait partie des 10 meilleures prédictions.

Cela permet de visualiser les prédictions et les niveaux de confiance du modèle pour plusieurs images de validation, en mettant en évidence les prédictions correctes et les prédictions les plus probables.

![images/fab](/images/32.png)

Le code ci-dessus utilise les bibliothèques scikit-learn et seaborn pour évaluer et visualiser les performances du modèle.

La fonction `confusion_matrix` de scikit-learn est utilisée pour calculer la matrice de confusion du modèle. Cette matrice donne un aperçu des prédictions correctes et incorrectes pour chaque classe de race de chien.

La fonction `classification_report` de scikit-learn est utilisée pour générer un rapport de classification qui affiche des mesures telles que la précision, le rappel et le score F1 pour chaque classe de race de chien.

La bibliothèque seaborn est utilisée pour tracer la matrice de confusion sous forme de heatmap, ce qui facilite la visualisation des performances du modèle.

Ensuite, il y a deux fonctions supplémentaires : `save_model` et `load_model`. La fonction `save_model` est utilisée pour sauvegarder le modèle formé à un emplacement spécifié. Elle prend en compte un suffixe optionnel pour différencier les versions du modèle. La fonction `load_model` est utilisée pour charger un modèle à partir d'un chemin de fichier spécifié.

Enfin, le code montre un exemple d'utilisation de ces fonctions en enregistrant le modèle formé sur 1000 images avec le suffixe "1000-images-mobilenetv2-Adam". Ensuite, le modèle est chargé à partir du chemin de fichier spécifié. Les tailles des ensembles de données `x` et `y` sont également affichées. Un lot de données complet `full_data` est créé en utilisant toutes les données disponibles, et un modèle complet `full_model` est créé à l'aide de la fonction `create_model`.

![images/fab](/images/33.png)

Le code ci-dessus ajuste le modèle `full_model` aux données complètes en utilisant la méthode `fit()`. Les arguments passés à cette méthode sont les suivants:

- `x`: Le lot de données complet `full_data` qui contient toutes les images et leurs étiquettes.
- `epochs`: Le nombre d'époques d'entraînement.
- `callbacks`: Une liste de rappels à utiliser pendant l'entraînement. Dans ce cas, les rappels utilisés sont `full_model_tensorboard` et `full_model_early_stopping`.

Le rappel `full_model_tensorboard` est utilisé pour enregistrer les journaux d'entraînement dans un répertoire spécifié, permettant ainsi la visualisation des métriques d'entraînement à l'aide de TensorBoard.

Le rappel `full_model_early_stopping` est utilisé pour arrêter l'entraînement prématurément si la métrique surveillée (dans ce cas, l'exactitude) ne s'améliore pas pendant un certain nombre d'époques défini par la patience.

En ajustant le modèle aux données complètes, le modèle tente d'apprendre à prédire les races de chiens à partir de l'ensemble complet d'images d'entraînement.

![images/fab](/images/34.png)

Le code ci-dessus effectue les opérations suivantes :

1. Il sauvegarde le modèle `full_model` avec le suffixe "full-image-set-mobilenetv2-Adam" en appelant la fonction `save_model()`.
2. Il charge le modèle sauvegardé `loaded_full_model` à partir du chemin spécifié en utilisant la fonction `load_model()`.
3. Il utilise la bibliothèque `pickle` pour sauvegarder le chemin du modèle chargé dans un fichier "doggo.pkl".
4. Il spécifie le chemin des fichiers d'images de test dans la variable `test_path`.
5. Il crée un lot de données de test en appelant la fonction `create_data_batch()` avec les noms de fichiers d'images de test.
6. Il effectue des prédictions sur le lot de données de test en utilisant le modèle `loaded_full_model` avec la méthode `predict()`.
7. Les prédictions sont enregistrées dans un fichier CSV nommé "preds_array.csv" en utilisant la fonction `np.savetxt()`.

Veuillez noter que l'étape 7 peut prendre du temps en fonction de la taille du lot de données de test et de la complexité du modèle.

![images/fab](/images/35.png)

Le code ci-dessus effectue les opérations suivantes :

1. Il charge les prédictions à partir du fichier CSV "preds_array.csv" en utilisant la fonction `np.loadtxt()`. Les prédictions sont stockées dans la variable `test_preds`.
2. Il affiche les 10 premières prédictions à l'aide de la variable `test_predictions`.
3. Il spécifie le chemin des images personnalisées dans la variable `custom_path`.
4. Il crée un lot de données personnalisées en appelant la fonction `create_data_batch()` avec les chemins des images personnalisées.
5. Il effectue des prédictions sur le lot de données personnalisées en utilisant le modèle `loaded_full_model` avec la méthode `predict()`. Les prédictions sont stockées dans la variable `custom_preds`.
6. Il obtient les étiquettes de prédiction d'image personnalisées en utilisant la fonction `get_preds_label()` pour chaque prédiction dans `custom_preds`. Les étiquettes sont stockées dans la variable `custom_preds_labels`.

Veuillez noter que les images personnalisées doivent être présentes dans le dossier spécifié par `custom_path` pour que les prédictions puissent être effectuées correctement.

![images/fab](/images/36.png)


Le code ci-dessus utilise la bibliothèque `matplotlib` pour afficher les images personnalisées avec leurs étiquettes de prédiction. Voici une explication du code :

1. Il importe la bibliothèque `matplotlib.pyplot` sous le nom `plt`.
2. Il définit la taille de la figure à l'aide de `plt.figure(figsize=(30,10))`.
3. Il utilise une boucle `enumerate` pour itérer à travers les images personnalisées et leurs étiquettes de prédiction.
4. Pour chaque image, il crée un sous-graphique en utilisant `plt.subplot(1,2*9,2*i+1)`. Les paramètres `(1, 2*9, 2*i+1)` spécifient le nombre de lignes, le nombre total de sous-graphiques et la position du sous-graphique actuel dans la figure.
5. Il désactive les graduations sur les axes x et y avec `plt.xticks([])` et `plt.yticks([])`.
6. Il ajoute un titre à chaque sous-graphique avec l'étiquette de prédiction correspondante en utilisant `plt.title(custom_preds_labels[i])`.
7. Il affiche l'image dans le sous-graphique en utilisant `plt.imshow(image)`.
8. Enfin, il utilise `plt.show()` pour afficher la figure avec les images et les étiquettes de prédiction.

Assurez-vous d'avoir importé les images personnalisées et les prédictions correspondantes avant d'exécuter ce code.