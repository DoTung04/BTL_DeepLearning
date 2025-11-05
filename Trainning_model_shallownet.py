# Trainning_model_shallownet.py
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from conv.shallownet import ShallowNet

# --- Bước 1: Chuẩn bị dữ liệu ---
print("[INFO] Nạp ảnh...")
imagePaths = list(paths.list_images("datasets"))  # datasets/cat/, datasets/dog/, ...

sp = SimplePreprocessor(64, 64)  # Resize ảnh 64x64 (nên hơn 32x32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])

(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Chia dữ liệu train/test
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

# One-hot encode nhãn
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# --- Bước 2: Data Augmentation ---
aug = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# --- Bước 3: Xây dựng và compile mô hình ---
print("[INFO] Tạo mô hình ShallowNet cải tiến...")
model = ShallowNet.build(width=64, height=64, depth=3, classes=4)

opt = Adam(learning_rate=1e-3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# --- Bước 4: Train mô hình ---
print("[INFO] Đang training mạng...")
EPOCHS = 50
BS = 32

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    epochs=EPOCHS,
    verbose=1,
)

# --- Bước 5: Lưu model ---
print("[INFO] Lưu model thành 'model_animals.hdf5'")
model.save("model_animals.hdf5")

# --- Bước 6: Đánh giá ---
print("[INFO] Đánh giá mô hình...")
predictions = model.predict(testX, batch_size=BS)
print(
    classification_report(
        testY.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=["cat", "chicken", "dog", "pig"],
    )
)

# --- Bước 7: Vẽ biểu đồ ---
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="Training loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="Validation loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="Training accuracy")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="Validation accuracy")
plt.title("Training Loss and Accuracy on Animal Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
