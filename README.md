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


