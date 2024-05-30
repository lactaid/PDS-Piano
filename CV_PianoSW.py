import cv2
import numpy as np
import pygame

import tensorflow as tf
from tensorflow.keras.models import load_model

from collections import deque
from statistics import mode

import os

#reduce resolution function
def reduceResolution(img):
  while img.shape[1] > 2500:
    h, w, _ = img.shape
    img = cv2.resize(img, (w//2, h//2))

  return img

#functions to identify piano keys
def getBinaryImage(img):

  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Set a threshold value
  black_threshold_value = 100

  # Threshold the image
  _, b_binary_image = cv2.threshold(gray_image, black_threshold_value, 255, cv2.THRESH_BINARY)

  return b_binary_image

def thickEdges(im_gray):
    #canny edges detection
    edges = cv2.Canny(im_gray, 100, 200)

    #define the kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    #perform dilation
    thick_edges = cv2.dilate(edges, kernel, iterations=12)

    return thick_edges

def calculateDistance(point1, point2): #euclidean distance
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculateCentroid(contour):
    M = cv2.moments(contour)

    if M["m00"] != 0:

      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])

      return (cX, cY)

    else:
      return 0

def cropPiano(contour, im_color):
  '''
  Using a contour should only be applied when we detect the piano. After
  that we should save the bounding box coordinates to just apply them directly
  to afterwards images.
  '''

  #get the bounding box of the contour
  x, y, w, h = cv2.boundingRect(contour)

  #crop the image using the bounding box
  cropped_image = im_color[y:y+h, x:x+w]

  return cropped_image

def getKeys(img_c):

    white_key_order = np.array(['5-c', '5-d', '5-e', '5-f', '5-g', '5-a', '5-b'])
    black_key_order = np.array(['5-cs', '5-ds', '5-fs', '5-gs', '5-as'])

    white_contours = []
    black_contours = []

    white_area = []
    black_area = []

    white_keys = []
    black_keys = []

    #get binary image
    img_b = getBinaryImage(img_c) #es opcional

    #get edges
    edges = thickEdges(img_b)

    #get piano contour for cropping images
    contour_ext, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #in case there is small noise we eliminate it
    if len(contour_ext) > 1:
      temp = 0
      for c in contour_ext:
        area = cv2.contourArea(c)
        if area > temp:
          temp = area
          contour_ext = [c]

    #crop images
    img_c = cropPiano(contour_ext[0], img_c)
    img_b = cropPiano(contour_ext[0], img_b)
    edges = cropPiano(contour_ext[0], edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #masks = []
    for i in range(1, len(contours)): #skip whole piano contour

      #create a mask to evaluate this region within the color image
      mask = np.zeros_like(img_b)
      cv2.drawContours(mask, [contours[i]], -1, (255), thickness=cv2.FILLED)
      #masks.append(mask)

      #compute the average pixel value within the contour region
      mean_value = cv2.mean(img_c, mask=mask)[0]

      #check if the average pixel value is black or white
      if mean_value < 100: #arbitrary number
        #each append contains the value of the max 'x' coordinate and the contour index
        black_contours.append([np.max(contours[i][:, :, 0]), i])

      else:
        white_contours.append([np.max(contours[i][:, :, 0]), i])


    #sort the contours based on the x coordinate, this way we will have the keys
    #in order from left to right and it will be easire to lable
    black_contours = np.array(black_contours) #we need it for the sort
    black_contours = black_contours[black_contours[:, 0].argsort()]

    white_contours = np.array(white_contours) #we need it for the sort
    white_contours = white_contours[white_contours[:, 0].argsort()]

    for bc in black_contours:
      black_keys.append(contours[bc[1]])

    for wc in white_contours:
      white_keys.append(contours[wc[1]])



    '''#draw white keys
    cv2.drawContours(img_c, white_keys, -1, (0, 255, 0), 2)

    #draw black keys
    cv2.drawContours(img_c, black_keys, -1, (255, 0, 0), 2)'''


    # We want ONLY THE KEYS from the white components, so we stablish a
    # threshold based on the median area of the components
    # this is needed to avoid noise

    final_white_keys = []

    #obtain all areas
    for k in white_keys:
      area = cv2.contourArea(k)
      white_area.append(area)


    #find median
    white_median = np.median(np.sort(white_area))

    #set threshold
    w_threshold = 3*(white_median/4)
    upper = white_median + w_threshold
    under = white_median - w_threshold

    #compare individual areas with the median
    for i in range(0, len(white_area)):
      if white_area[i] < upper and white_area[i] > under: #threshold
        final_white_keys.append(white_keys[i])

    #draw white keys
    cv2.drawContours(img_c, final_white_keys, -1, (0, 255, 0), 2)

    #SAME FOR BLACK KEYS

    final_black_keys = []
    #obtain all areas
    for k in black_keys:
      area = cv2.contourArea(k)
      black_area.append(area)

    #find median
    black_median = np.median(np.sort(black_area))

    #set threshold
    b_threshold = 2*(black_median/3)
    upper = black_median + b_threshold
    under = black_median - b_threshold

    #compare individual areas with the median
    for i in range(0, len(black_area)):
      if black_area[i] < upper and black_area[i] > under: #threshold
        final_black_keys.append(black_keys[i])

    #draw black keys
    cv2.drawContours(img_c, final_black_keys, -1, (255, 0, 0), 2)


    white_keys = final_white_keys
    black_keys = final_black_keys


    #HERE WE START THE PROCESS FOR LABELING OUR KEYS

    is_middle_key = []
    white_keys_centroid = []
    black_keys_centroid = []

    #calculate black centroids
    for bk in black_keys:
      centroid2 = calculateCentroid(bk)
      cv2.circle(img_c, centroid2, 5, (0, 0, 255), -1)  # Red color
      black_keys_centroid.append(centroid2)

    #calculate white centroids and measure distance with black centroids
    for wk in white_keys:
      wkc = calculateCentroid(wk)
      white_keys_centroid.append(wkc)
      cv2.circle(img_c, wkc, 5, (0, 255, 0), -1)  # Green color

      distances = []

      #calculate distance between wkc and bkc
      for bkc in black_keys_centroid:
        distance = calculateDistance(wkc, bkc)
        distances.append(distance)

      #obtain the minimum distance calculated
      min = np.min(distances)
      limit = 30 #podríamos calcular de una mejor manera dependiendo del tamaño de la imagen para que no haya falla al etiquetar (esto es lo que hace que falle)
      count = 0

      for d in distances:
        #if a distance is between our min threshold
        if d <= min+limit and d >= min-limit:
          count += 1 #we count it

      #if two distances were counted
      if count == 2:
        is_middle_key.append(1) #it is a middle white key
      else:
        is_middle_key.append(0) #if not, it isn't


    #create and initialize our label arrays
    white_key_label = []
    black_key_label = []

    for wk in white_keys:
      white_key_label.append('nn')

    for bk in black_keys:
      black_key_label.append('nn')

    #we find a 'middle' white key followed by another one, the first would be a G
    for i in range(0, len(is_middle_key)):
      if is_middle_key[i] == 1 and is_middle_key[i+1] == 1:
        white_key_label[i] = '5-g'
        index = i
        break

    #label from found G key backwards
    m = np.where(white_key_order == white_key_label[index])[0][0] - 1
    for j in range(index-1, -1, -1):
      white_key_label[j] = white_key_order[np.abs(m % len(white_key_order))]
      m-=1

    #label from found G key forwards
    m = np.where(white_key_order == white_key_label[index])[0][0] + 1
    for j in range(index+1, len(white_key_label)):
      white_key_label[j] = white_key_order[m % len(white_key_order)]
      m+=1

    #now for labeling black keys

    if white_key_label[0] == '5-c':
      m = 0
    elif white_key_label[0] == '5-f':
      m = 2

    for j in range(0, len(black_key_label)):
      black_key_label[j] = black_key_order[m % len(black_key_order)]
      m+=1

    #draw the labeled notes on the original image
    font = cv2.FONT_HERSHEY_SIMPLEX

    #for white keys
    for i in range(0, len(white_keys_centroid)):
      text = white_key_label[i]
      position = white_keys_centroid[i]  # (x, y) coordinates

      # Draw the text on the image
      cv2.putText(img_c, text, position, font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    #for black keys
    for i in range(0, len(black_keys_centroid)):
      text = black_key_label[i]
      position = black_keys_centroid[i]  # (x, y) coordinates

      # Draw the text on the image
      cv2.putText(img_c, text, position, font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    #show original image
    #cv2.imshow(img_c)

    return contour_ext[0], white_keys, black_keys, white_keys_centroid, black_keys_centroid, white_key_label, black_key_label, img_c


#functions for key pressing detection
def findFingers(image, white_keys_centroid, white_keys_label, black_keys_centroid, black_keys_label):
  #transform image to hsv
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  #create a mask using a color filter based on intensity
  mask = cv2.inRange(hsv, (0, 0, 70), (255,255,150))

  #morphological operations to delete piano edges and contrast fingers
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  mask = cv2.erode(mask, kernel, iterations=4)
  mask = cv2.dilate(mask, kernel, iterations=3)
  
  #cv2.imshow('Fingers mask', mask)

  #find contours
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if len(contours) == 0:
    return 0, 0

  else:

    #we filter the contours to have only the fingers
    #first calculate areas of contours
    areas = []
    for contour in contours:
      areas.append(cv2.contourArea(contour))

    max_area = np.max(areas) #obtain max area (should be a finger)
   
    
    if max_area < 10000: #further reduce noise
      return 0, 0

    new_contours = [] #finger contours
    i = 0
    for area in areas:
      if area >= max_area/3: #threshold areas using max_area
        new_contours.append(contours[i])

      i += 1

    #then we identify which key they are at by proximity same as with shadows
    pressing_keys = []
    indexes = []
    finger_centroids = []
    
    for i in range(len(white_keys_centroid)):
      white_keys_centroid[i] = (white_keys_centroid[i][0], 100)

    for contour in new_contours:
      #cv2.drawContours(image, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
      centroid = calculateCentroid(contour)

      if centroid != 0:
        finger_centroids.append(centroid)


    for sc in finger_centroids:
      i = 0
      min = 10000
      is_white = True

      for wc in white_keys_centroid:
        distance = calculateDistance(sc, wc)

        if distance < min:
          min = distance
          index = i

        i += 1

      i = 0

      for bc in black_keys_centroid:
        distance = calculateDistance(sc, wc)

        if distance < min:
          is_white = False
          min = distance
          index = i

        i += 1

      if is_white:
        '''#we check that it does not have a shadow, which means it is being pressed
        if white_keys_label[index] not in not_pressing_keys:
          cv2.drawContours(image, [white_keys[index]], -1, (0, 255, 0), 2) #VISUAL
          pressing_keys.append(white_keys_label[index])'''

        pressing_keys.append(white_keys_label[index])
        indexes.append(index)

      else:
        #we check that it does not have a shadow, which means it is being pressed
        '''if black_keys_label[index] not in not_pressing_keys:
          cv2.drawContours(image, [black_keys[index]], -1, (0, 255, 0), 2) #VISUAL
          pressing_keys.append(black_keys_label[index])'''

        pressing_keys.append(black_keys_label[index])
        indexes.append(index)


    #print(pressing_keys) #VISUAL
    #cv2_imshow(image) #VISUAL

    return pressing_keys, indexes

'''--------------------------------------------------------------------------'''

def arise(img, pressing_keys, indexes, white_keys_label, black_keys_label):

  #first we transform the image to hsv and grayscale
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #we define kernels to apply 'noisy' edge detection
  kernel_y = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
  kernel_y2 = np.array([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]])

  kernel_x = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
  kernel_x2 = np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]])

  #apply kernels to obtain all edges
  edges_x = cv2.filter2D(image, -1, kernel_x)
  edges_y = cv2.filter2D(image, -1, kernel_y)

  edges_x2 = cv2.filter2D(image, -1, kernel_x2)
  edges_y2 = cv2.filter2D(image, -1, kernel_y2)

  #combine edges found
  edges = cv2.addWeighted(edges_x, 1, edges_y, 1, 0)
  edges2 = cv2.addWeighted(edges_x2, 1, edges_y2, 1, 0)
  edges3 = cv2.addWeighted(edges, 1, edges2, 1, 0)

  #apply dilation + closing
  kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
  edges3 = cv2.erode(edges3, kernel, iterations=1)
  edges3 = cv2.morphologyEx(edges3, cv2.MORPH_CLOSE, kernel, iterations=2)
  

  #convert image to binary
  _, edges3 = cv2.threshold(edges3, 25, 255, cv2.THRESH_BINARY_INV)
  #cv2.imshow('Shadow mask', edges3)
  

  #array for centroids of shadows
  shadow_centroids = []

  the_contours = []

  index = 0
  for wk in white_keys:

    #we only process on white keys that have fingers on them
    if white_keys_label[index] in pressing_keys:
      mask = np.zeros_like(image)
      cv2.drawContours(mask, [wk], -1, 255, thickness=cv2.FILLED)

      mask = cv2.bitwise_not(mask)
      mask = cv2.bitwise_or(mask, edges3)
      cv2.drawContours(mask, [wk], -1, 0, 4) #we draw the outline of the key to connect figures to detect contours

      contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

      for contour in contours:

        #noise reduction + reduced computational cost
        if cv2.contourArea(contour) > 350: #650

          c_mask = np.zeros_like(image)
          cv2.drawContours(c_mask, [contour], -1, 255, thickness=cv2.FILLED)
          #cv2.imshow('c_mask', c_mask)

          #obtain average pixel value of contour using hsv values
          average_hsv_value = cv2.mean(hsv, mask=c_mask)[:3]

          #evaluate intensity to identify shadows
          if average_hsv_value[1] < 115 and average_hsv_value[2] < 70:
            the_contours.append(contour)

    index+=1


  if len(the_contours) == 0:
    really_pressing_keys = []

    for i in range(len(pressing_keys)):
      really_pressing_keys.append(pressing_keys[i])
      cv2.drawContours(img, [white_keys[indexes[i]]], -1, (0, 255, 0), 2) #VISUAL

    return img, really_pressing_keys


  else:
    
    for contour in the_contours:
      '''------ESTO ES LO QUE HACE QUE SE VEAN LAS SOMBRAS!!!-------'''
      #cv2.drawContours(img, [contour], -1, (0, 255, 0), thickness=cv2.FILLED) #VISUAL

      #calculate centroid of shadow element
      centroid = calculateCentroid(contour)
      if centroid != 0:
        shadow_centroids.append(centroid)


    #here starts the process to identify which white keys have shadows
    #we use proximity
    not_pressing_keys = [] #array for keys with shadows

    #iterate through shadow centroids
    for sc in shadow_centroids:
      i = 0 #to know index
      min = 10000 #initializing
      is_white = True #shadow over white key (or bk)

      #iterate through wk centroids
      for wc in white_keys_centroid:
        distance = calculateDistance(sc, wc)

        #find key with minimum distance to shadow
        if distance < min:
          min = distance
          index = i

        i += 1

      '''
      #restart counter to use it with bk
      i = 0

      #iterate through bk centroids
      for bc in black_keys_centroid:
        distance = calculateDistance(sc, bc)

        #find key with minimum distance to shadow
        if distance < min:
          is_white = False #if there is, then it will be black
          min = distance
          index = i

        i += 1
        '''

      #use index obtained before to save the corresponding key where a shadow is
      if is_white:
        #cv2.drawContours(img, [white_keys[index]], -1, (0, 255, 0), 2) #VISUAL
        not_pressing_keys.append(white_keys_label[index])
      else:
        #cv2.drawContours(img, [black_keys[index]], -1, (0, 255, 0), 2) #VISUAL
        not_pressing_keys.append(black_keys_label[index])


    really_pressing_keys = []
    for i in range(len(pressing_keys)):
      if pressing_keys[i] not in not_pressing_keys:
        really_pressing_keys.append(pressing_keys[i])
        cv2.drawContours(img, [white_keys[indexes[i]]], -1, (0, 255, 0), 2) #VISUAL


    #print(really_pressing_keys)
    #cv2_imshow(img) #VISUAL

    #return really_pressing_keys
    return img, really_pressing_keys

# Esta es la clase para tocar notas del piano
'''class Piano:
    def __init__(self, path, pattern='.wav'):
        self.path = path
        self.pattern = pattern
        self.pressed_keys = {}

        try:
            # Inicializar Pygame
            pygame.init()
            
            # Inicializar el mixer de audio para tocar varias notas al mismo tiempo
            pygame.mixer.init()
            
        except pygame.error as e:
            print(f"Error al inicializar Pygame: {e}")

    def playNote(self, note):
        if note in self.pressed_keys:
            # Si la nota ya se está reproduciendo, no hacer nada
            if self.pressed_keys[note]['channel'].get_busy():
                return


            del self.pressed_keys[note]

        complete_path = self.path + note + self.pattern
        
        audio = pygame.mixer.Sound(complete_path)
        channel = pygame.mixer.find_channel()
            
        channel.play(audio)
        self.pressed_keys[note] = {'audio': audio, 'channel': channel}

'''
class Piano:
    def __init__(self, path, pattern='.wav'):
        self.path = path
        self.pattern = pattern
        self.pressed_keys = {}
        # Inicializar Pygame
        pygame.init()

        # Inicializar el mixer de audio para tocar varias notas al mismo tiempo
        pygame.mixer.init()
        
    def PlayNote(self, note):
        if note in self.pressed_keys:
            self.pressed_keys[note].stop()
            del self.pressed_keys[note]
        complete_path = self.path + note + self.pattern
        
        audio = pygame.mixer.Sound(complete_path)
        audio.play()
        
        self.pressed_keys[note] = audio

#Funciones para el modelo de NN
def load_and_preprocess_image(image_array, input_width=256, input_height=256):
    #print('call it')
    # Resize the image to match the input dimensions of the model using OpenCV
    image_array_resized = cv2.resize(image_array, (input_width, input_height))
    
    # Flip the image vertically
    image_array_resized = cv2.flip(image_array_resized, 0)

    # Flip the image horizontally
    image_array_resized = cv2.flip(image_array_resized, 1)
    
    # Normalize pixel values
    image_array_resized = image_array_resized / 255.0  # Normalize pixel values to [0, 1]

    # Expand the dimensions of the image array to match the input shape expected by the model
    input_data = np.expand_dims(image_array_resized, axis=0)

    return input_data


# Este nos transforma una predicción a la tecla que es
def num2key(num, white_keys_label, black_keys_label):
  if num < 8:
    # cv2.drawContours(img, [white_keys[num-1]], -1, (0, 255, 0), 2)
    note = white_keys_label[num-1]
  else:
    # cv2.drawContours(img, [black_keys[num-8]], -1, (255, 0, 0), 2)
    note = black_keys_label[num-8]

  return note 


'''
---------------------------
-----------MAIN------------
---------------------------
'''

piano = Piano('Piano\PianoNotes\\')

print('starting...')
model = load_model(os.path.join('model','pianoHand_sigmoid.h5'))
print('\nmodel loaded!')

class_mapping = {
        0: 1,   # Original class 0 is now class 1
        1: 10,  # Original class 1 is now class 10
        2: 11,  # Original class 2 is now class 11
        3: 12,  # Original class 3 is now class 12
        4: 2,   # Original class 4 is now class 2
        5: 3,   # Original class 5 is now class 3
        6: 4,   # Original class 6 is now class 4
        7: 5,   # Original class 7 is now class 5
        8: 6,   # Original class 8 is now class 6
        9: 7,   # Original class 9 is now class 7
        10: 8,  # Original class 10 is now class 8
        11: 9   # Original class 11 is now class 9
    }



piano_detected = False

cap = cv2.VideoCapture(1)  # El argumento 0 indica que se utilizará la primera cámara disponible
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

    # Esperar 1 milisegundo y verificar si se presionó la tecla 'q' para salir del bucle
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break
    
    
    #Primera parte: segmentación de piano
    if not piano_detected: 
      
      # Mostrar el fotograma en una ventana
      cv2.imshow('Frame', frame)
      
      # Esperar 1 milisegundo y verificar si se presionó la tecla 's' para comenzar a detectar teclas del piano
      key = cv2.waitKey(1) & 0xFF
      
      # Verificar si la tecla presionada es 's'
      if key == ord('s'):
        img = reduceResolution(frame)
        contour_ext, white_keys, black_keys, white_keys_centroid, black_keys_centroid, white_keys_label, black_keys_label, img_piano = getKeys(img)
        cv2.imshow('Piano', img_piano) #mostramos teclas detectadas
        
        #bucle para confirmar que las teclas se hayan detectado correctamente
        while True:
          # Esperar 1 milisegundo y verificar si se presionó la tecla 'c' para confirmar que se detectó bien
          key = cv2.waitKey(1) & 0xFF
      
          if key == ord('c'):
            piano_detected = True
            break
            
          elif key == ord('x'): #si se presionó la tecla 'x' para volverlo a detectar
            ret, frame = cap.read()
            img = reduceResolution(frame)
            contour_ext, white_keys, black_keys, white_keys_centroid, black_keys_centroid, white_keys_label, black_keys_label, img_piano = getKeys(img)
            cv2.imshow('Piano', img_piano)
            
            
    else: #Segunda parte: deteccion de teclas presionadas
      img = reduceResolution(frame) #reduce resolution 
      img = cropPiano(contour_ext, img) #crop using piano bounding box obtained
      img2 = np.copy(img)
      
      #como tal esta no detecta que se estén presionando las teclas si no que haya una mano/dedos en el frame
      pressing_keys, indexes = findFingers(img2, white_keys_centroid, white_keys_label, black_keys_centroid, black_keys_label)

      if pressing_keys == 0:
        cv2.imshow('Piano', img2)

      else:
        #esta determina si se tocan teclas o no, pero podríamos evitarlo para mayor velocidad de procesamiento y solo usar la de findFingers
        processed_image, notes = arise(img, pressing_keys, indexes, white_keys_label, black_keys_label)  # Extracción de sombras

        cv2.imshow('Piano', processed_image)
        
        for note in notes:
          piano.PlayNote(note)
        #-------------AQUÍ DEBE HACER LA PREDICCIÓN EL MODELO----------------
        
      

# Liberar el objeto de captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()



'''
#Este es para aplicarlo a un video
piano_detected = False
last_notes = []
last_frame = None

window_size = 5
buffer = deque(maxlen=window_size)

# Define the target input dimensions expected by your model
input_width, input_height = 256, 256

# Cambiar la fuente del video a un archivo en lugar de la cámara
video_path = os.path.join('videos_piano', 'lamb_3.mp4')
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

# Leer el primer fotograma
ret, frame = cap.read()
if not ret:
    print("No se pudo capturar el primer fotograma.")
    cap.release()
    exit()

# Primera parte: segmentación de piano
if not piano_detected:
    img = reduceResolution(frame)
    contour_ext, white_keys, black_keys, white_keys_centroid, black_keys_centroid, white_keys_label, black_keys_label, img_piano = getKeys(img)
    cv2.imshow('Piano', img_piano) # Mostrar teclas detectadas

    while True:
      key = cv2.waitKey(1) & 0xFF
      
      # Verificar si la tecla presionada es 's'
      if key == ord('c'):
        piano_detected = True
        break
      
      elif key == ord('x'): #si se presionó la tecla 'x' para volverlo a detectar
        ret, frame = cap.read()
        img = reduceResolution(frame)
        contour_ext, white_keys, black_keys, white_keys_centroid, black_keys_centroid, white_keys_label, black_keys_label, img_piano = getKeys(img)
        cv2.imshow('Piano', img_piano)

# Segunda parte: detección de teclas presionadas
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img3 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = reduceResolution(frame)
    img = cropPiano(contour_ext, img)  # Recortar usando el bounding box del piano obtenido
    img2 = np.copy(img)
    
    c_frame = np.copy(img)
    
    # Detectar dedos en el frame
    pressing_keys, indexes = findFingers(img2, white_keys_centroid, white_keys_label, black_keys_centroid, black_keys_label)


    if last_frame is None:
      last_frame = np.zeros_like(c_frame)
    
    De = np.sqrt(np.sum((last_frame - c_frame)**2))
    # print(De)
    
    if pressing_keys == 0 or frame_count % 9 != 0:
        # if De < 3760:
        #   buffer.append('sk')
        cv2.imshow('Piano', img2)
    else:
        processed_image, notes = arise(img, pressing_keys, indexes, white_keys_label, black_keys_label)  # Extracción de sombras
        cv2.imshow('Piano', processed_image)

        model_input = load_and_preprocess_image(img3, input_width, input_height)
        
        # Perform inference using the loaded model
        predictions = model.predict(model_input)

        # Assuming a classification task with softmax output, get the predicted class
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        if np.amax(predictions) > 0.9:
          # Map the predicted class index to the correct class    
          predicted_class_label = class_mapping.get(predicted_class_index, 'Unknown')
          
          # Convert prediction to associated key
          key = num2key(predicted_class_label, white_keys_label, black_keys_label)

          buffer.append(key)
          note = mode(buffer)

          fingerUp = (buffer.count(note) > 4 and note != 'sk' and De > 2260 and De < 3400)
          newnote = (buffer.count(note) > 1 and note not in last_notes)

          if newnote or fingerUp:
            if note != 'sk': piano.PlayNote(note)
            # print(newnote, fingerUp)
            print(note)
            print(De)
            last_notes = [note]
        else:
          buffer.append('sk')
        # print(buffer)
    last_frame = c_frame
    frame_count+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
'''