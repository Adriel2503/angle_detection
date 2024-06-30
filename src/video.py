import cv2
import numpy as np

# Cargar el video
video_path = '../data/video_sensor.mp4'
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    raise Exception("No se pudo abrir el video")

# Crear una ventana para mostrar el video
cv2.namedWindow('Segmentación del Reloj', cv2.WINDOW_NORMAL)

while True:
    # Leer un fotograma del video
    ret, frame = cap.read()
    if not ret:
        break  # Fin del video

    # Convertir el fotograma a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir el rango de color para detectar el reloj (ajustar según sea necesario)
    lower_color = np.array([0, 0, 0])
    upper_color = np.array([180, 255, 50])

    # Crear una máscara para el color del reloj
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una imagen en blanco del mismo tamaño que el fotograma
    white_background = np.ones_like(frame) * 255

    # Dibujar el contorno más grande en la imagen en blanco
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(white_background, [largest_contour], -1, (0, 0, 0), -1)
        segmented_frame = cv2.bitwise_and(frame, frame, mask=cv2.drawContours(np.zeros_like(mask), [largest_contour], -1, 255, -1))

    # Mostrar el fotograma segmentado con fondo blanco
    cv2.imshow('Segmentación del Reloj', segmented_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
