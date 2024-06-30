import cv2
import numpy as np

# Ruta del video
video_path = '../data/video_sensor.mp4'
# Ruta para guardar la imagen del frame
output_image_path = '../data/frame.jpg'
# Número del frame que quieres extraer
frame_number = 100

# Abrir el video
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video")
    exit()

# Configurar el frame al que se desea ir
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Leer el frame
ret, frame = cap.read()

# Verificar si se leyó el frame correctamente
if not ret:
    print("Error al leer el frame")
    exit()

# Convertir el frame a escala de grises
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Guardar el frame en escala de grises como una imagen
cv2.imwrite(output_image_path, gray_frame)

# Cerrar el video
cap.release()

print(f"Frame {frame_number} guardado como imagen en escala de grises en {output_image_path}")
