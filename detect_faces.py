import cv2

# Load the DNN-based face detector (deep learning model)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load your image
img = cv2.imread('face.png')
(h, w) = img.shape[:2]

# Prepare the image for the neural network
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

confidence_threshold = 0.3  # only keep detections the model is at least 50% confident about
face_count = 0

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > confidence_threshold:
        face_count += 1
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

print(f"Number of faces detected: {face_count}")

cv2.imwrite('face_detected.png', img)
print("Done! Check 'face_detected.png' in this folder.")