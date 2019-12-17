import numpy as np
from keras import layers
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
import datetime

root_dir = "D:\\datasets_AI\\HeadPoseImageDatabase\\"
people_images = np.load(root_dir + 'people_images.npy')
people_data = np.load(root_dir + 'people_head_data.npy')

people_imgs = np.array([img for person in people_images for img in person]).astype('float32')
print(people_imgs.shape)
people_info = np.array([data for person in people_data for data in person]).astype('float32') / 90
print(people_info.shape)

X, y = shuffle(people_imgs, people_info, random_state=42)

model = Sequential()
regularization = l2(0.001)

model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularization, kernel_initializer='normal'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, kernel_initializer='normal'))

model.summary()
early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=4, mode='min')
optimizer = Adam(lr=0.0001, decay=10e-6)
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
model.fit(X, y, epochs=12, validation_split=0.1, batch_size=32, callbacks=[early_stopping])

model_json = model.to_json()
date = str(datetime.date.today())
with open("model" + date + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model" + date + ".h5")
print("Saved model to disk")
