import cv2
import numpy as np

def nothing(x):
    pass

# Ruta al video
video_path = '../data/video_sensor.mp4'

# Abrir el video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

# Inicializar posiciones de las líneas
line_x_pos = 0.5  # Posición vertical de la línea horizontal (como porcentaje del alto)
line_y_pos = 0.5  # Posición horizontal de la línea vertical (como porcentaje del ancho)
radio = 110  # Radio inicial de la circunferencia

# Crear una ventana
cv2.namedWindow('PDI41')

# Crear una barra deslizante para ajustar el radio de la circunferencia
cv2.createTrackbar('Radio', 'PDI41', radio, 200, nothing)  # Radio inicial de 50 píxeles, máximo de 200 píxeles

# Crear barras deslizantes para ajustar los parámetros de segmentación de la aguja
cv2.createTrackbar('Umbral', 'PDI41', 60, 255, nothing)
cv2.createTrackbar('Area Min', 'PDI41', 1000, 10000, nothing)
cv2.createTrackbar('Area Max', 'PDI41', 5000, 20000, nothing)

# Función para actualizar las posiciones de las líneas
def update_lines(key):
    global line_x_pos, line_y_pos
    step = 0.01
    if key == ord('w'):
        line_x_pos = max(0, line_x_pos - step)  # Mover la línea horizontal hacia arriba
    elif key == ord('s'):
        line_x_pos = min(1, line_x_pos + step)  # Mover la línea horizontal hacia abajo
    elif key == ord('a'):
        line_y_pos = max(0, line_y_pos - step)  # Mover la línea vertical hacia la izquierda
    elif key == ord('d'):
        line_y_pos = min(1, line_y_pos + step)  # Mover la línea vertical hacia la derecha

confirm = False
second_confirm = False
third_confirm = False
fourth_confirm = False
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

    # Convertir a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular las posiciones en píxeles
    height, width = gris.shape[:2]
    line_x_pixel = int(line_x_pos * height)
    line_y_pixel = int(line_y_pos * width)
    center_line_x_pixel = width // 2  # Línea vertical central

    # Obtener el valor del radio de la barra deslizante
    if not confirm:
        radio = cv2.getTrackbarPos('Radio', 'PDI41')

    if confirm and not second_confirm:
        mask = np.zeros_like(gris)
        cv2.circle(mask, (line_y_pixel, line_x_pixel), radio, (255), thickness=-1)
        gris[mask == 0] = 255  # Convertir a blanco todo lo que está fuera de la circunferencia

        # Segmentación de la aguja
        umbral_val = cv2.getTrackbarPos('Umbral', 'PDI41')
        area_min = cv2.getTrackbarPos('Area Min', 'PDI41')
        area_max = cv2.getTrackbarPos('Area Max', 'PDI41')

        # Aplicar un umbral para segmentar la aguja
        _, umbral = cv2.threshold(gris, umbral_val, 255, cv2.THRESH_BINARY_INV)

        # Eliminar pequeñas áreas no deseadas (ruido) y mejorar la máscara
        kernel = np.ones((2, 2), np.uint8)  # Ajustar el tamaño del kernel según sea necesario
        mascara = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crear una máscara en blanco
        mascara_aguja = np.zeros_like(mascara)

        # Dibujar solo la aguja en la máscara en blanco
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area_min < area < area_max:
                cv2.drawContours(mascara_aguja, [contorno], -1, (255), thickness=cv2.FILLED)

        # Invertir la máscara para que la aguja sea negra sobre fondo blanco
        mascara_aguja_inv = cv2.bitwise_not(mascara_aguja)

        # Mostrar la máscara resultante
        frame_with_shapes = cv2.cvtColor(mascara_aguja_inv, cv2.COLOR_GRAY2BGR)
    elif second_confirm and not third_confirm:
        # Ajustar una línea a los contornos de la aguja
        mascara_aguja_inv = np.zeros_like(gris)
        for contorno in contornos:
            if area_min < cv2.contourArea(contorno) < area_max:
                [vx, vy, x, y] = cv2.fitLine(contorno, cv2.DIST_L2, 0, 0.01, 0.01)
                cols = gris.shape[1]
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(mascara_aguja_inv, (cols - 1, righty), (0, lefty), (255), 2)

        # Mostrar la línea ajustada
        frame_with_shapes = cv2.cvtColor(mascara_aguja_inv, cv2.COLOR_GRAY2BGR)
    elif third_confirm and not fourth_confirm:
        # Ajustar una línea a los contornos de la aguja y dibujar la línea vertical en el centro
        mascara_aguja_inv = np.zeros_like(gris)
        for contorno in contornos:
            if area_min < cv2.contourArea(contorno) < area_max:
                [vx, vy, x, y] = cv2.fitLine(contorno, cv2.DIST_L2, 0, 0.01, 0.01)
                cols = gris.shape[1]
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(mascara_aguja_inv, (cols - 1, righty), (0, lefty), (255), 2)

        # Dibujar la línea vertical en el centro
        cv2.line(mascara_aguja_inv, (center_line_x_pixel, 0), (center_line_x_pixel, height), (255), 2)

        # Mostrar las líneas ajustadas
        frame_with_shapes = cv2.cvtColor(mascara_aguja_inv, cv2.COLOR_GRAY2BGR)
    elif fourth_confirm:
        # Ajustar una línea a los contornos de la aguja y dibujar la línea vertical en el centro
        mascara_aguja_inv = np.zeros_like(gris)
        for contorno in contornos:
            if area_min < cv2.contourArea(contorno) < area_max:
                [vx, vy, x, y] = cv2.fitLine(contorno, cv2.DIST_L2, 0, 0.01, 0.01)
                cols = gris.shape[1]
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(mascara_aguja_inv, (cols - 1, righty), (0, lefty), (255), 2)

        # Dibujar la línea vertical en el centro
        cv2.line(mascara_aguja_inv, (center_line_x_pixel, 0), (center_line_x_pixel, height), (255), 2)

        # Calcular el ángulo entre la línea de la aguja y la línea vertical
        vector_aguja = np.array([vx, vy])
        vector_vertical = np.array([0, 1])  # Vector vertical en dirección y

        # Calcular el producto escalar y el ángulo
        dot_product = np.dot(vector_aguja, vector_vertical)
        mod_aguja = np.linalg.norm(vector_aguja)
        mod_vertical = np.linalg.norm(vector_vertical)
        cos_angle = dot_product / (mod_aguja * mod_vertical)
        angle = np.arccos(cos_angle) * 180 / np.pi

        # Ajustar el ángulo para que esté en el rango de 0 a 360 grados
        if vy < 0:
            angle = 360 - angle

        # Mostrar el ángulo en la imagen
        frame_with_shapes = cv2.cvtColor(mascara_aguja_inv, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame_with_shapes, f'Angulo: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        # Dibujar las líneas y la circunferencia
        frame_with_shapes = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        cv2.line(frame_with_shapes, (0, line_x_pixel), (width, line_x_pixel), (255, 0, 0), 2)  # Línea horizontal azul
        cv2.line(frame_with_shapes, (line_y_pixel, 0), (line_y_pixel, height), (0, 0, 255), 2)  # Línea vertical roja
        cv2.circle(frame_with_shapes, (line_y_pixel, line_x_pixel), radio, (0, 255, 0), 2)  # Circunferencia verde

    # Mostrar el cuadro con las líneas, la circunferencia y la segmentación
    cv2.imshow('PDI41', frame_with_shapes)

    # Esperar 1 milisegundo y verificar si el usuario presiona una tecla
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    update_lines(key)

    if key == ord('e'):
        if confirm:
            if second_confirm:
                if third_confirm:
                    fourth_confirm = True
                else:
                    third_confirm = True
            else:
                second_confirm = True
        else:
            confirm = True

    if key == ord('p'):  # Pausar/Reanudar el video
        paused = not paused

cap.release()
cv2.destroyAllWindows()

