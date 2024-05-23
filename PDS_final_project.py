import cv2
import numpy as np
import os

#reduce resolution function ---OPTIONAL---
def reduceResolution(img):
    while img.shape[1] > 2500:
        h, w, _ = img.shape
        img = cv2.resize(img, (w//2, h//2))
    return img

#functions to identify piano keys
def getBinaryImage(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_threshold_value = 100
    _, b_binary_image = cv2.threshold(gray_image, black_threshold_value, 255, cv2.THRESH_BINARY)
    return b_binary_image

def thickEdges(im_gray):
    edges = cv2.Canny(im_gray, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thick_edges = cv2.dilate(edges, kernel, iterations=12)
    return thick_edges

def calculateDistance(point1, point2):
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
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = im_color[y:y+h, x:x+w]
    return cropped_image

def getKeys(img_c):
    white_key_order = np.array(['C', 'D', 'E', 'F', 'G', 'A', 'B'])
    black_key_order = np.array(['C#', 'D#', 'F#', 'G#', 'A#'])
    white_contours, black_contours = [], []
    white_area, black_area = [], []
    white_keys, black_keys = [], []

    img_b = getBinaryImage(img_c)
    edges = thickEdges(img_b)
    contour_ext, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contour_ext) > 1:
        temp = 0
        for c in contour_ext:
            area = cv2.contourArea(c)
            if area > temp:
                temp = area
                contour_ext = [c]

    img_c = cropPiano(contour_ext[0], img_c)
    img_b = cropPiano(contour_ext[0], img_b)
    edges = cropPiano(contour_ext[0], edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(1, len(contours)):
        mask = np.zeros_like(img_b)
        cv2.drawContours(mask, [contours[i]], -1, (255), thickness=cv2.FILLED)
        mean_value = cv2.mean(img_c, mask=mask)[0]
        if mean_value < 100:
            black_contours.append([np.max(contours[i][:, :, 0]), i])
        else:
            white_contours.append([np.max(contours[i][:, :, 0]), i])

    black_contours = np.array(black_contours)
    black_contours = black_contours[black_contours[:, 0].argsort()]
    white_contours = np.array(white_contours)
    white_contours = white_contours[white_contours[:, 0].argsort()]

    for bc in black_contours:
        black_keys.append(contours[bc[1]])

    for wc in white_contours:
        white_keys.append(contours[wc[1]])

    final_white_keys = []
    for k in white_keys:
        area = cv2.contourArea(k)
        white_area.append(area)
    white_median = np.median(np.sort(white_area))
    w_threshold = 3*(white_median/4)
    upper = white_median + w_threshold
    under = white_median - w_threshold

    for i in range(0, len(white_area)):
        if white_area[i] < upper and white_area[i] > under:
            final_white_keys.append(white_keys[i])

    final_black_keys = []
    for k in black_keys:
        area = cv2.contourArea(k)
        black_area.append(area)
    black_median = np.median(np.sort(black_area))
    b_threshold = 2*(black_median/3)
    upper = black_median + b_threshold
    under = black_median - b_threshold

    for i in range(0, len(black_area)):
        if black_area[i] < upper and black_area[i] > under:
            final_black_keys.append(black_keys[i])

    white_keys = final_white_keys
    black_keys = final_black_keys

    is_middle_key = []
    white_keys_centroid = []
    black_keys_centroid = []

    for bk in black_keys:
        centroid2 = calculateCentroid(bk)
        cv2.circle(img_c, centroid2, 5, (0, 0, 255), -1)
        black_keys_centroid.append(centroid2)

    for wk in white_keys:
        wkc = calculateCentroid(wk)
        white_keys_centroid.append(wkc)
        cv2.circle(img_c, wkc, 5, (0, 255, 0), -1)

        distances = []
        for bkc in black_keys_centroid:
            distance = calculateDistance(wkc, bkc)
            distances.append(distance)

        min = np.min(distances)
        limit = 30
        count = 0

        for d in distances:
            if d <= min+limit and d >= min-limit:
                count += 1

        if count == 2:
            is_middle_key.append(1)
        else:
            is_middle_key.append(0)

    white_key_label = []
    black_key_label = []

    for wk in white_keys:
        white_key_label.append('nn')

    for bk in black_keys:
        black_key_label.append('nn')

    for i in range(0, len(is_middle_key)):
        if is_middle_key[i] == 1 and is_middle_key[i+1] == 1:
            white_key_label[i] = 'G'
            index = i
            break

    m = np.where(white_key_order == white_key_label[index])[0][0] - 1
    for j in range(index-1, -1, -1):
        white_key_label[j] = white_key_order[np.abs(m % len(white_key_order))]
        m-=1

    m = np.where(white_key_order == white_key_label[index])[0][0] + 1
    for j in range(index+1, len(white_key_label)):
        white_key_label[j] = white_key_order[m % len(white_key_order)]
        m+=1

    if white_key_label[0] == 'C':
        m = 0
    elif white_key_label[0] == 'F':
        m = 2

    for j in range(0, len(black_key_label)):
        black_key_label[j] = black_key_order[m % len(black_key_order)]
        m+=1

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(0, len(white_keys_centroid)):
        text = white_key_label[i]
        position = white_keys_centroid[i]
        cv2.putText(img_c, text, position, font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    for i in range(0, len(black_keys_centroid)):
        text = black_key_label[i]
        position = black_keys_centroid[i]
        cv2.putText(img_c, text, position, font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return contour_ext[0], white_keys, black_keys, white_keys_centroid, black_keys_centroid, white_key_label, black_key_label, img_c

def findFingers(image, white_keys_centroid, white_keys_label, black_keys_centroid, black_keys_label):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 50), (255,255,150))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.erode(mask, kernel, iterations=4)
    mask = cv2.dilate(mask, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0
    else:
        areas = [cv2.contourArea(contour) for contour in contours]
        max_area = np.max(areas)
        new_contours = [contours[i] for i, area in enumerate(areas) if area > 0.3 * max_area]

    fingers = []
    finger_index = []

    for contour in new_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        centroid = (cX, cY)
        distance_w = [calculateDistance(centroid, white_keys_centroid[i]) for i in range(len(white_keys_centroid))]
        distance_b = [calculateDistance(centroid, black_keys_centroid[i]) for i in range(len(black_keys_centroid))]
        min_distance_w = np.min(distance_w)
        min_distance_b = np.min(distance_b)

        if min_distance_w < min_distance_b:
            key_index = np.argmin(distance_w)
            key = white_keys_label[key_index]
        else:
            key_index = np.argmin(distance_b)
            key = black_keys_label[key_index]

        fingers.append(centroid)
        finger_index.append(key_index)
        cv2.circle(image, centroid, 5, (0, 0, 255), -1)
        cv2.putText(image, key, centroid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return fingers, finger_index

# Load video

video_path = os.path.join('model', 'vid1.mp4')
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # Resize frame if necessary
    frame = reduceResolution(frame)

    # Process frame
    try:
        _, white_keys, black_keys, white_keys_centroid, black_keys_centroid, white_keys_label, black_keys_label, processed_frame = getKeys(frame)
        fingers, finger_index = findFingers(frame, white_keys_centroid, white_keys_label, black_keys_centroid, black_keys_label)

        # Display the resulting frame
        cv2.imshow('Processed Frame', processed_frame)

    except Exception as e:
        print(f"Error processing frame {frame_count}: {e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
