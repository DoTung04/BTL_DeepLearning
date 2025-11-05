# import the necessary packages
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # Lưu danh sách các bộ tiền xử lý (nếu có)
        self.preprocessors = preprocessors if preprocessors is not None else []

    def load(self, imagePaths, verbose=-1):
        # Khởi tạo danh sách dữ liệu và nhãn
        data = []
        labels = []

        # Lặp qua tất cả ảnh đầu vào
        for (i, imagePath) in enumerate(imagePaths):
            # Nạp ảnh từ ổ đĩa
            image = cv2.imread(imagePath)

            # Nếu ảnh không đọc được (None), bỏ qua
            if image is None:
                print(f"[WARNING] Không thể đọc ảnh: {imagePath}, bỏ qua...")
                continue

            # Lấy nhãn từ đường dẫn: /path/to/dataset/{class}/{image}.jpg
            label = os.path.basename(os.path.dirname(imagePath))

            # Áp dụng tất cả các bộ tiền xử lý (nếu có)
            for p in self.preprocessors:
                image = p.preprocess(image)

            # Lưu ảnh và nhãn
            data.append(image)
            labels.append(label)

            # In tiến độ
            if verbose > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] Đã xử lý {i + 1}/{len(imagePaths)} ảnh...")

        # Trả về tuple gồm dữ liệu và nhãn (mảng NumPy)
        return (np.array(data), np.array(labels))
