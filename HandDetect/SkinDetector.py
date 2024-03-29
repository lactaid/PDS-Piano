import numpy as np
import cv2

def get_bounding_box(imagen):
    # Convertir la imagen a escala de grises
    gray = imagen
    
    # Umbralizar la imagen para obtener los píxeles del objeto
    _, umbral = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Encontrar los contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si no se encontraron contornos, devolver None
    if len(contornos) == 0:
        return None
    
    # Obtener el cuadro delimitador para el contorno más grande
    x, y, w, h = cv2.boundingRect(max(contornos, key=cv2.contourArea))
    
    return (x, y, w, h)

cap = cv2.VideoCapture(0)  # El argumento 0 indica que se utilizará la primera cámara disponible


def skin_detection_YCRCB(image):
    #Convertir de BGR a YCrCb
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    #Se crea una máscara
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 130, 85), (255,180,135)) 
    #Se aplica la operación de open sobre la imagen binaria
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    return YCrCb_mask

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Bucle para capturar y mostrar el video
while True:
    # Capturar fotograma por fotograma
    ret, frame = cap.read()

    # Verificar si se capturó correctamente el fotograma
    if not ret:
        print("No se pudo capturar el fotograma.")
        break

    #Obtenemos la imagen binaria
    skin = skin_detection_YCRCB(frame)

    #Obtenemos un bounding box
    bounding_box = get_bounding_box(skin)
    if bounding_box is not None:
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Mostrar el fotograma en una ventana
    cv2.imshow('Skin', frame)

    # Esperar 1 milisegundo y verificar si se presionó la tecla 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()