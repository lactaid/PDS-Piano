import mediapipe as mp
import cv2
import numpy as np

limit = 0.0000005
mp_drawing = mp.solutions.drawing_utils
# Modelo de manos
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# Instanciamos el modelo de las manos, detection y tracking son los dos modelos que se usan
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
  # Mientas estamos conectados a la webcam
  while cap.isOpened():
    # Leemos cada cuadro, no usamos ret, frame es el cuadro
    ret, frame = cap.read()

    #DETECCION CON MEDIAPIPE
    # Recoloreamos el feed obtenido de cv2 de BGR a RGB para usarlo en mediapipe
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #Invertimos la imagen de manera horizontal, para detectar de manera correcta cual mano es derecha y cual izquierda
    image = cv2.flip(image,1)
    # Evita que podamos dibujar sobre la imagen
    image.flags.writeable = False
    # Realizamos la deteccion
    results = hands.process(image)
    # Nos permite dibujar sobre la imagen de nuevo
    image.flags.writeable = True

    # Convertimos de RGB a BGR
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # print(results)

    # Renderizar los resultados si detecta akgi (en la variable results)
    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      for landmark in hand_landmarks.landmark:
        if landmark.z < limit:
            print('Tocando Piano')
        else:
            print('No tocando Piano')
        break

      for num, hand in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(image,hand, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(224, 224, 224), thickness=2, circle_radius=2))

    # Mostramos frame
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()