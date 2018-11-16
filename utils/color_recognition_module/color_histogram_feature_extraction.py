

from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
from utils.color_recognition_module import knn_classifier as knn_classifier
current_path = os.getcwd()


def color_histogram_of_test_image(test_src_image):

    # carga de la imagen
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # encuentra los valores de píxel máximos para R, G y B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
    with open(current_path + '/utils/color_recognition_module/'
              + 'test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):

    # detecte el color de la imagen utilizando el nombre del archivo de imagen para etiquetar los datos de entrenamiento
    if 'red' in img_name:
        data_source = 'rojo'
    elif 'yellow' in img_name:
        data_source = 'amarillo'
    elif 'green' in img_name:
        data_source = 'verde'
    elif 'orange' in img_name:
        data_source = 'anaranjado'
    elif 'white' in img_name:
        data_source = 'blanco'
    elif 'black' in img_name:
        data_source = 'negro'
    elif 'blue' in img_name:
        data_source = 'azul'
    elif 'violet' in img_name:
        data_source = 'violeta'

    # carga de la imagen
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # encuentra los valores de píxel máximos para R, G y B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():

    # imágenes de color rojo para entrenamiento
    for f in os.listdir('./training_dataset/red'):
        color_histogram_of_training_image('./training_dataset/red/' + f)

    #  imágenes de color amarillo para entrenamiento
    for f in os.listdir('./training_dataset/yellow'):
        color_histogram_of_training_image('./training_dataset/yellow/' + f)

    #  imágenes de color verde para entrenamiento
    for f in os.listdir('./training_dataset/green'):
        color_histogram_of_training_image('./training_dataset/green/' + f)

    #  imágenes de color anaranjado para entrenamiento
    for f in os.listdir('./training_dataset/orange'):
        color_histogram_of_training_image('./training_dataset/orange/' + f)

    # imágenes de color blanco para entrenamiento
    for f in os.listdir('./training_dataset/white'):
        color_histogram_of_training_image('./training_dataset/white/' + f)

    #  imágenes de color negro para entrenamiento
    for f in os.listdir('./training_dataset/black'):
        color_histogram_of_training_image('./training_dataset/black/' + f)

    # imágenes de color azul para entrenamiento
    for f in os.listdir('./training_dataset/blue'):
        color_histogram_of_training_image('./training_dataset/blue/' + f)
