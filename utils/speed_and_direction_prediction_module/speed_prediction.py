
from utils.image_utils import image_saver

is_vehicle_detected = [0]
current_frame_number_list = [0]
bottom_position_of_detected_vehicle = [0]


def predict_speed(
    top,
    bottom,
    right,
    left,
    current_frame_number,
    crop_img,
    roi_position,
    ):
    speed = 'n.a.'  # Significa no disponible, es solo inicialización.
    direction = 'n.a.'  # Significa no disponible, es solo inicialización.
    scale_constant = 1  # Escala manual porque no realizamos la calibración de la cámara.
    isInROI = True  #Es el objeto que se encuentra dentro de Región de Interés.
    update_csv = False

    if bottom < 250:
        scale_constant = 1  # scale_constant se usa para la escala manual porque no realizamos la calibración de la cámara
    elif bottom > 250 and bottom < 320:
        scale_constant = 2  # scale_constant se usa para la escala manual porque no realizamos la calibración de la cámara
    else:
        isInROI = False

    if len(bottom_position_of_detected_vehicle) != 0 and bottom \
        - bottom_position_of_detected_vehicle[0] > 0 and 205 \
        < bottom_position_of_detected_vehicle[0] \
        and bottom_position_of_detected_vehicle[0] < 210 \
        and roi_position < bottom:
        is_vehicle_detected.insert(0, 1)
        update_csv = True
        image_saver.save_image(crop_img)  # guardar la imagen del vhiculo detectado

    
    if bottom > bottom_position_of_detected_vehicle[0]:
        direction = 'Abajo'
    else:
        direction = 'Arriba'

    if isInROI:
        pixel_length = bottom - bottom_position_of_detected_vehicle[0]
        scale_real_length = pixel_length * 44  # multiplicado por 44 para convertir la longitud del píxel en longitud real en metros (chenge 44 para obtener la longitud en metros para su caso)
        total_time_passed = current_frame_number - current_frame_number_list[0]
        scale_real_time_passed = total_time_passed * 24  # obtener el tiempo total transcurrido para que un vehículo pase por el área de ROI (24 = fps)
        if scale_real_time_passed != 0:
            speed = scale_real_length / scale_real_time_passed / scale_constant  # Realización de escala manual porque no hemos realizado calibración de cámara.
            speed = speed / 6 * 40  # use la constante de referencia para obtener la predicción de la velocidad del vehículo en unidades de kilómetro
            current_frame_number_list.insert(0, current_frame_number)
            bottom_position_of_detected_vehicle.insert(0, bottom)

    return (direction, speed, is_vehicle_detected, update_csv)
