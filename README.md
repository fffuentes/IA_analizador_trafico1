# IA_analizador_trafico1
___
##### Universidad mariano Gálvez de Guatemala
##### Centro universitario Retalhuleu
##### Curso de INTELIGENCIA ARTIFICIAL
##### Catedrático ing. Jorge salvador santos
___

# Introducción
- Justificación: Este proyecto esta enfocado al aprendizaje del funcionamiento de la detección de objetos mediante las tecnologías OpenCV y Tensorflow ya ambas están orientadas a esta parte de la inteligencia artificial
Objetivos: Poder implementar de manera funcional dicho proyecto para poder realizar la detección de vehículos, poder calcular su velocidad y calcificación de color y tamaño

- Propósito: entender y comprender por completo el funcionamiento de la Api de clasificación de tensorflow y su adaptación a los ambientes de deteccion
___

# DETECCIÓN DE VEHÍCULOS, SEGUIMIENTO Y CONTEO
- **TensorFlow ™** es una biblioteca de software de código abierto para cálculos numéricos que utilizan gráficos de flujo de datos. Los nodos en el gráfico representan operaciones matemáticas, mientras que los bordes del gráfico representan las matrices de datos multidimensionales (tensores) comunicadas entre ellos.
- **OpenCV (Open Source Computer Vision Library)** es una biblioteca de software de visión de computadora y de aprendizaje automático de código abierto. OpenCV fue construido para proporcionar una infraestructura común para aplicaciones de visión artificial y para acelerar el uso de la percepción de la máquina en los productos comerciales.
- ***La API de conteo de objetos TensorFlow se utiliza como base para el conteo de objetos en este proyecto.***
___

# Capacidades generales de este proyecto
- Detección de la dirección de marcha del vehículo. 
- Predicción de la velocidad del vehículo. 
- Predicción del tamaño aproximado del vehículo. 
- Las imágenes de los vehículos detectados se recortan del cuadro de video y se guardan como nuevas imágenes en la ruta de la carpeta " detected_vehicles " 
- El programa proporciona un archivo .csv como salida ( traffic_measurement.csv ) que incluye las filas "Tipo / Tamaño del vehículo", "Color del vehículo", "Dirección del movimiento del vehículo", "Velocidad del vehículo (km / h)", después del final de El proceso para el archivo de vídeo de origen. 
- Reconocimiento del color aproximado del vehículo. 
___
# Arquitectura del sistema
<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/35445395-8dba4406-02c2-11e8-84bf-b480edbe9472.jpg">
</p>

- La detección y clasificación de vehículos se han desarrollado utilizando la API de detección de objetos TensorFlow
Ruta :  vehicle_counting_tensorflow / vehicle_detection_main.py 

- La predicción de la velocidad del vehículo se ha desarrollado utilizando OpenCV mediante la manipulación y el cálculo de píxeles de imagen
 Ruta : vehicle_counting_tensorflow / utils / speed_and_direction_prediction_module / 

- La predicción del color del vehículo se ha desarrollado utilizando OpenCV a través del algoritmo de clasificación del Aprendizaje Automático de Vecinos Más Cercanos K es Características del histograma de color entrenado
 Ruta : vehicle_counting_tensorflow / utils / color_recognition_module / 

- La fuente de vídeo se lee fotograma a fotograma con OpenCV. Cada cuadro se procesa mediante el modelo "SSD with Mobilenet" desarrollado en TensorFlow. Este es un bucle que continúa trabajando hasta llegar al final del video.

#### Modelo
<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/48481757-b1d5a900-e81f-11e8-824b-4317115fe5b4.png">
</p>

# Requerimientos

- pc con Procesadores Intel® Core™ i5
- 8 Gb de memoria RAM
- 80 Gb de espacio en el disco 
___

# Instalación

1. Para el correcto funcinamiento sera necesario instalar lo siguiente
 
  - Python 3.6.4
  - OpenCV


2. modulos a instalar via pip
  - Package             Version
------------------- ----------
absl-py             0.6.1
aiohttp             3.4.4
aiohttp-cors        0.7.0
astor               0.7.1
async-timeout       3.0.1
attrs               18.2.0
backcall            0.1.0
bleach              3.0.2
certifi             2018.10.15
chardet             3.0.4
Click               7.0
colorama            0.4.0
contextlib2         0.5.5
cycler              0.10.0
Cython              0.29
decorator           4.3.0
defusedxml          0.5.0
entrypoints         0.2.3
gast                0.2.0
grpcio              1.16.0
h5py                2.8.0
html5lib            1.0.1
idna                2.7
idna-ssl            1.1.0
ipykernel           5.1.0
ipython             7.0.1
ipython-genutils    0.2.0
ipywidgets          7.4.2
jedi                0.13.1
Jinja2              2.10
jsonschema          2.6.0
jupyter             1.0.0
jupyter-client      5.2.3
jupyter-console     6.0.0
jupyter-core        4.4.0
Keras-Applications  1.0.6
Keras-Preprocessing 1.0.5
kiwisolver          1.0.1
lxml                4.2.5
Markdown            3.0.1
MarkupSafe          1.0
matplotlib          3.0.1
mistune             0.8.4
multidict           4.4.2
nbconvert           5.4.0
nbformat            4.4.0
notebook            5.7.0
numpy               1.15.3
opencv-python       3.4.3.18
packaging           18.0
pandocfilters       1.4.2
parso               0.3.1
pickleshare         0.7.5
Pillow              5.3.0
pip                 18.1
pip-review          1.0
prometheus-client   0.4.2
prompt-toolkit      2.0.6
protobuf            3.6.1
psutil              5.4.7
Pygments            2.2.0
pyparsing           2.2.2
pyreadline          2.1
python-dateutil     2.7.4
pytz                2018.6
pywinpty            0.5.4
pywrap              0.1.0
pyzmq               17.1.2
qtconsole           4.4.2
requests            2.20.0
scikit-learn        0.20.0
scipy               1.1.0
Send2Trash          1.5.0
setuptools          40.5.0
simplegeneric       0.8.1
six                 1.11.0
slim                0.3.13
tensorboard         1.12.0
tensorflow          1.11.0
termcolor           1.1.0
terminado           0.8.1
testpath            0.4.2
tf                  1.0.0
tornado             5.1.1
traitlets           4.3.2
urllib3             1.24
wcwidth             0.1.7
webencodings        0.5.1
Werkzeug            0.14.1
wheel               0.32.2
widgetsnbextension  3.4.2
yarl                1.2.6

___
# ejecucion

- al tener instalado todos los programas y modulos de python 
ir a la ubicacion de la carpeta del proyecto y ejecutar el archivo ***vehicle_detection_main.py***


# cita
- author = "Ahmet Özlü",
- title  = "Vehicle Detection, Tracking and Counting by TensorFlow",
- year   = "2018",
- url    = [https://github.com/ahmetozlu/vehicle_counting_tensorflow](https://github.com/ahmetozlu/vehicle_counting_tensorflow)

