import cv2
from faced import FaceDetector
from faced.utils import annotate_image
import logging
import numpy as np
from keras.models import model_from_json
from draw_line import line
logging.getLogger('tensorflow').disabled = True

json_file = open('model2019-12-17.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model2019-12-17.h5")

face_detector = FaceDetector()
cam = cv2.VideoCapture(0)
while True:
    _, img = cam.read()
    img = cv2.resize(img, (228, 228), interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    bboxes = face_detector.predict(rgb_img)
    ann_img = annotate_image(img, bboxes)
    for x, y, w, h, p in bboxes:
        cropped_img = img[y - h // 2: y + h // 2, x - w // 2: x + w // 2]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        cropped_img = cv2.resize(cropped_img, (100, 100), interpolation=cv2.INTER_AREA)
        cropped_img = np.expand_dims(cropped_img, axis=-1)
        cropped_img = np.expand_dims(cropped_img, axis=0)
        cropped_img = cropped_img.astype('float32') / 255
        predictions = loaded_model.predict(cropped_img)
        tilt = predictions[0][0] * 90  # move the head up and down (x axis rotates)
        pan = predictions[0][1] * 90  # move the head from left to right (y axis rotates)
        ann_img = line(ann_img, (x, y), pan, tilt)
        # print("Tilt:", tilt, "\tPan:", pan)

    cv2.imshow('image', ann_img)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cam.release()
