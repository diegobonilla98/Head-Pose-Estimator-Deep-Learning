import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def process_image(file, face_pos):
    face_mid_x, face_mid_y, face_w, face_h = face_pos
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = img[face_mid_y-face_h//2: face_mid_y+face_h//2, face_mid_x-face_w//2: face_mid_x+face_w//2]
    face = cv2.resize(face, (100, 100), interpolation=cv2.INTER_AREA)
    face = np.expand_dims(face, axis=-1)
    face = face.astype('float32') / 255
    return face


def parse_name(file):
    data_str = file[11:file.find('.jpg')]
    sep = data_str[1:].find('-')
    if sep == -1:
        sep = data_str[1:].find('+')
    tilt = int(data_str[:sep + 1])
    pan = int(data_str[sep + 1:])
    return np.array([tilt, pan])


root_dir = "D:\\datasets_AI\\HeadPoseImageDatabase\\"
people_images = []
people_data = []
for face_idx in range(1, 16):
    print("Reading person", face_idx, "/", 15, "...")
    person_img = []
    person_data = []
    curr_dir = root_dir + "Person" + str(face_idx).zfill(2) + "\\"
    data_files = glob.glob(curr_dir + "*.txt")
    for data in data_files:
        with open(data, 'r') as data_file:
            data_arr = data_file.read().split('\n')
        image_file = data_arr[0]
        face_info = (int(data_arr[3]),
                     int(data_arr[4]),
                     int(data_arr[5]),
                     int(data_arr[6]))

        info = parse_name(image_file)
        person_img.append(process_image(curr_dir + image_file, face_info))
        person_data.append(info)
    people_images.append(np.array(person_img))
    people_data.append(np.array(person_data))
print("Done reading!")
people_images = np.array(people_images)
people_data = np.array(people_data)

np.save('people_images', people_images)
np.save('people_head_data', people_data)
