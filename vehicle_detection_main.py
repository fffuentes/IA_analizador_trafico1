
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Importación de detección de objetos
from utils import label_map_util
from utils import visualization_utils as vis_util

# inicializar .csv
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'tipo de vehiculo/tamaño, Color, direccion del vehiculo, Velocidad (km/h)'
    writer.writerows([csv_line.split(',')])

# video de entrada
cap = cv2.VideoCapture('video car prueba/car1.mp4')

# Variables
total_passed_vehicle = 0  # usándolo para contar vehículos

# Por defecto, uso un modelo de "SSD con Mobilenet" aquí. Consulte el zoológico del modelo de detección 
# (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
#  para ver una lista de otros modelos que se pueden ejecutar fuera de la caja con Variaciones de velocidades y precisiones.
# modelo desgarcado
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'


# Ruta al gráfico de detección de congelados. Este es el modelo real que se utiliza para la detección de objetos.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Lista de las cadenas que se utilizan para agregar la etiqueta correcta para cada cuadro.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Cargue un modelo de Tensorflow (congelado) en la memoria.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Cargando mapa de etiquetas
# La etiqueta asigna los índices a los nombres de las categorías, de modo que cuando nuestra red de convolución predice 5, 
# sabemos que esto corresponde al avión. Aquí utilizo funciones de utilidad internas, pero cualquier cosa que devuelva los enteros 
# de asignación de un diccionario a las etiquetas de cadena apropiadas estaría bien
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# codigo de ayuda
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)


# Deteccion
def object_detection_function():
    total_passed_vehicle = 0
    speed = 'esperando...'
    direction = 'esperando...'
    size = 'esperando...'
    color = 'esperando...'
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:

            # Tensores de entrada y salida definidos para detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Cada cuadro representa una parte de la imagen donde se detectó un objeto en particular.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Cada puntaje representa cómo nivel de confianza para cada uno de los objetos.
            # La puntuación se muestra en la imagen del resultado, junto con la etiqueta de la clase.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Para todos los cuadros que se extraen del video de entrada.
            while cap.isOpened():
                (ret, frame) = cap.read()

                if not ret:
                    print ('fin del video')
                    break

                input_frame = frame

                # Ampliar las dimensiones ya que el modelo espera que las imágenes tengan forma: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # deteccion actual
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                # Visualización de los resultados de una detección.
                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    cap.get(1),
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    )

                total_passed_vehicle = total_passed_vehicle + counter

                #insertar texto de información en el cuadro de video
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Vehiculos Detectados: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # cuando el vehículo haya pasado la línea y se haya contado, haga que el color de la línea ROI sea verde
                if counter == 1:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)

                # insertar texto de información en el cuadro de video
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, 190),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )
                cv2.putText(
                    input_frame,
                    'Informacion del ultimo vehiculo',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )
                cv2.putText(
                    input_frame,
                    '-Direccion: ' + direction,
                    (14, 302),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Velocidad(km/h): ' + speed,
                    (14, 312),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Color: ' + color,
                    (14, 322),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )
                cv2.putText(
                    input_frame,
                    '-Tamaño vehiculo/Tipo: ' + size,
                    (14, 332),
                    font,
                    0.4,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    )

                cv2.imshow('Deteccion de vehiculos', input_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if csv_line != 'not_available':
                    with open('traffic_measurement.csv', 'a') as f:
                        writer = csv.writer(f)
                        (size, color, direction, speed) = \
                            csv_line.split(',')
                        writer.writerows([csv_line.split(',')])
            cap.release()
            cv2.destroyAllWindows()


object_detection_function() 