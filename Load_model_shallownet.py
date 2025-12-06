# ======================================================
# TEST MODEL PHÂN LOẠI ĐỘNG VẬT (cat, chicken, dog, pig)
# ======================================================

from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2
import os
import random

# --- Danh sách nhãn đúng theo thứ tự lúc train ---
classLabels = ["cat", "chicken", "dog", "pig"]

print("[INFO] Đang nạp ảnh test để phân lớp...")
imagePaths = list(paths.list_images("image"))  # Tự động lấy toàn bộ ảnh trong các thư mục con
print(f"[INFO] Tổng số ảnh test: {len(imagePaths)}")

# --- Lấy nhãn thật từ tên thư mục cha (ground truth) ---
trueLabels = [os.path.basename(os.path.dirname(p)) for p in imagePaths]

# --- Tiền xử lý ảnh ---
sp = SimplePreprocessor(64, 64)   # phải cùng kích thước với ảnh train
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])

(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0

# --- Nạp model đã train ---
print("[INFO] Nạp model mạng pre-trained ...")
model = load_model("model_animals.hdf5")

# --- Dự đoán ---
print("[INFO] Đang dự đoán...")
preds = model.predict(data, batch_size=32).argmax(axis=1)
predLabels = [classLabels[p] for p in preds]

# --- Thống kê kết quả ---
correct = np.sum(np.array(predLabels) == np.array(trueLabels))
total = len(imagePaths)
accuracy = correct / total * 100

MAX_SHOW = 50
show_indexes = set(random.sample(range(len(imagePaths)), MAX_SHOW))
print("\n=== KẾT QUẢ PHÂN LOẠI ===")
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    trueLabel = trueLabels[i]
    predLabel = predLabels[i]
    color = (0, 255, 0) if trueLabel == predLabel else (0, 0, 255)

    text = f"Pred: {predLabel} | True: {trueLabel}"
    print(f"[{i+1:02d}] {os.path.basename(imagePath)} --> {text}")

    # Chỉ hiển thị 50 ảnh ngẫu nhiên
    if i in show_indexes:
        cv2.putText(image, text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Prediction", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()

# --- In kết quả tổng ---
print(f"\n✅ Tổng số ảnh kiểm tra: {total}")
print(f"✅ Phân loại đúng: {correct}")
print(f"✅ Độ chính xác: {accuracy:.2f}%")
