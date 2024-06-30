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

# Crear una ventana
cv2.namedWindow('PDI41')

# Crear barras deslizantes para ajustar los parámetros
cv2.createTrackbar('Umbral', 'PDI41', 60, 255, nothing)
cv2.createTrackbar('Area Min', 'PDI41', 1000, 10000, nothing)
cv2.createTrackbar('Area Max', 'PDI41', 5000, 20000, nothing)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro gaussiano para suavizar la imagen
    suavizado = cv2.GaussianBlur(gris, (15, 15), 0)

    # Aplicar detección de bordes para encontrar el contorno del reloj
    bordes = cv2.Canny(suavizado, 50, 150)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        # Suponer que el contorno más grande es el del reloj
        contorno_reloj = max(contornos, key=cv2.contourArea)

        # Crear una máscara del reloj
        mascara_reloj = np.zeros_like(gris)
        cv2.drawContours(mascara_reloj, [contorno_reloj], -1, (255), thickness=cv2.FILLED)

        # Aplicar la máscara del reloj para obtener solo la región del reloj
        reloj = cv2.bitwise_and(frame, frame, mask=mascara_reloj)

        # Convertir el área del reloj a escala de grises
        gris_reloj = cv2.cvtColor(reloj, cv2.COLOR_BGR2GRAY)

        # Obtener los valores de las barras deslizantes
        umbral_val = cv2.getTrackbarPos('Umbral', 'PDI41')
        area_min = cv2.getTrackbarPos('Area Min', 'PDI41')
        area_max = cv2.getTrackbarPos('Area Max', 'PDI41')

        # Aplicar un umbral para segmentar la aguja
        _, umbral = cv2.threshold(gris_reloj, umbral_val, 255, cv2.THRESH_BINARY_INV)

        # Eliminar pequeñas áreas no deseadas (ruido) y mejorar la máscara
        kernel = np.ones((5, 5), np.uint8)
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
                # Refinar la detección de la aguja eliminando bordes internos no deseados
                epsilon = 0.01 * cv2.arcLength(contorno, True)
                contorno_aprox = cv2.approxPolyDP(contorno, epsilon, True)
                cv2.drawContours(mascara_aguja, [contorno_aprox], -1, (255), thickness=cv2.FILLED)

        # Invertir la máscara para que la aguja sea negra sobre fondo blanco
        mascara_aguja_inv = cv2.bitwise_not(mascara_aguja)

        # Aplicar la máscara de la aguja al área del reloj
        reloj[mascara_aguja_inv == 0] = [255, 255, 255]  # Convertir todo lo que no es la aguja a blanco

        # Mostrar la máscara resultante
        cv2.imshow('mascara', mascara_aguja_inv)

        # Mostrar el cuadro original y el cuadro con el reloj segmentado
        cv2.imshow('PDI41', frame)
        cv2.imshow('Reloj Segmentado', reloj)

    # Esperar 1 milisegundo y verificar si el usuario presiona la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


